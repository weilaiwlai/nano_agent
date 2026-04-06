"""NanoAgent 多智能体（Supervisor）工作流。

流程：
START -> retrieve_memory_node -> supervisor_node
supervisor_node -> (DataScientist | Reporter | Assistant | END)
worker_node -> (tools_node | END)
tools_node -> (back to sender worker)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from uuid import uuid4
from typing import Annotated, Any, Literal, TypedDict

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode

from memory import UserMemoryManager

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger("nanoagent.agent_service.graph")

MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://mcp_server:8000").rstrip("/")
MCP_SERVICE_TOKEN = os.getenv("MCP_SERVICE_TOKEN", "").strip()
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0").strip()
GRAPH_CHECKPOINTER_BACKEND = os.getenv("GRAPH_CHECKPOINTER_BACKEND", "postgres").strip().lower()
GRAPH_CHECKPOINTER_REDIS_URL = os.getenv("GRAPH_CHECKPOINTER_REDIS_URL", REDIS_URL).strip() or REDIS_URL
GRAPH_CHECKPOINTER_POSTGRES_URL = os.getenv("GRAPH_CHECKPOINTER_POSTGRES_URL", "").strip()
GRAPH_CHECKPOINTER_PREFIX = os.getenv("GRAPH_CHECKPOINTER_PREFIX", "nanoagent:graph").strip() or "nanoagent:graph"
GRAPH_CHECKPOINTER_ALLOW_MEMORY_FALLBACK = (
    os.getenv("GRAPH_CHECKPOINTER_ALLOW_MEMORY_FALLBACK", "true").strip().lower() == "true"
)
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", os.getenv("QWEN_MODEL", "qwen3.5-plus"))
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("QWEN_API_KEY"))
DEFAULT_OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
).rstrip("/")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

if not MCP_SERVICE_TOKEN:
    logger.warning("未配置 MCP_SERVICE_TOKEN，访问受保护的 MCP 工具端点将失败。")

_memory_manager: UserMemoryManager | None = None
_llm_cache: dict[tuple[str, str, str], ChatOpenAI] = {}
_non_stream_llm_cache: dict[tuple[str, str, str], ChatOpenAI] = {}
_bound_llm_cache: dict[tuple[str, str, str, str], Any] = {}
_checkpointer_cm: Any | None = None
_checkpointer_backend_in_use: str = "memory"
app_graph: Any | None = None
DEBUG_MODE = os.getenv("DEBUG", "false").strip().lower() == "true"
TOOL_LOG_MAX_TEXT_LENGTH = 120
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SQL_SNIPPET_PATTERN = re.compile(
    r"\b(select|with|from|where|join|group\s+by|order\s+by|limit|count\s*\(|information_schema|pg_database)\b",
    re.IGNORECASE,
)
DATABASE_HELP_KEYWORDS = (
    "数据库",
    "查库",
    "sql",
    "postgres",
    "postgre",
    "table",
    "表",
    "字段",
    "schema",
    "库里",
    "db",
)
DSML_CONTROL_TOKEN_PATTERN = re.compile(r"<\s*/?\s*\|\s*DSML\s*\|[^>]*>", re.IGNORECASE)
CONTROL_MARKUP_FRAGMENT_PATTERN = re.compile(
    r"<[^>\n]{0,220}(?:tool_[a-z_]+|function_calls?|invoke|parameter|dsml|\"?email\"?\s+string|\"?content\"?\s+string|string\s*=\s*\"?true\"?)[^>\n]*>",
    re.IGNORECASE,
)
CONTROL_EMAIL_ARG_PATTERN = re.compile(
    r"<\s*\"?email\"?[^>]*>[^<>\n]{0,320}(?:</\s*parameter\s*>)?",
    re.IGNORECASE,
)
CONTROL_CONTENT_ARG_OPEN_PATTERN = re.compile(r"<\s*\"?content\"?[^>]*>", re.IGNORECASE)
CONTROL_MARKUP_KEYWORDS = (
    "dsml",
    "function_calls",
    "function_call",
    "invoke",
    "parameter",
    "tool_send_report",
    "tool_query_database",
    "string=\"true\"",
)
MAX_MODEL_HISTORY_MESSAGES = int(os.getenv("MAX_MODEL_HISTORY_MESSAGES", "60"))
REPORT_CONTENT_SOFT_LIMIT = int(os.getenv("REPORT_CONTENT_SOFT_LIMIT", "8000"))
EMAIL_DRAFT_TARGET_CHARS = int(os.getenv("EMAIL_DRAFT_TARGET_CHARS", "2000"))
SUPERVISOR_ROUTER_PROMPT = (
    "你是多智能体系统的极速语义路由器（Supervisor Router）。\n"
    "你只能输出一个词：DataScientist / Reporter / Assistant / FINISH。\n"
    "不要输出任何解释、标点、JSON 或多余文本。\n\n"
    "路由原则：\n"
    "1) DataScientist：只有当用户需要数据库查询、数据统计、SQL/表数据验证时。\n"
    "2) Reporter：只有当用户明确要求“立即执行外部动作”，当前仅包括发送邮件。\n"
    "   注意：仅要求“写邮件草稿/润色/总结内容”属于 Assistant，不属于 Reporter。\n"
    "3) Assistant：普通问答、解释、总结、建议、邮件草稿撰写、改写等无外部副作用场景。\n"
    "4) FINISH：用户明确表示结束对话时。\n"
)
NO_TOOL_INTENT_PROMPT = (
    "你是 Assistant 智能体，负责普通问答与文本生成。\n"
    "你不能调用任何外部工具；如果用户想发送邮件，先帮用户生成草稿并提示用户明确确认发送。\n"
    f"当你在生成邮件正文/报告草稿时，必须先提炼再输出，目标长度不超过 {EMAIL_DRAFT_TARGET_CHARS} 字符。\n"
    "如果原始信息很长，只保留关键信息与结论，不要输出冗长铺陈。\n"
)
REPORT_EXECUTION_GUARD_PROMPT = (
    "你是外部动作执行闸门。\n"
    "请判断用户最后一条消息是否在明确要求“立刻发送邮件”。\n"
    "只输出 EXECUTE 或 DRAFT 两个词之一，不要输出其他任何内容。\n"
    "若只是让助手写草稿、总结、润色、准备内容，则输出 DRAFT。\n"
    "只有明确执行发送动作时才输出 EXECUTE。\n"
)


def _truncate_for_log(value: str, *, max_len: int = TOOL_LOG_MAX_TEXT_LENGTH) -> str:
    """截断工具日志文本，避免记录过长输入。"""
    text = value.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...(truncated)"


def _mask_email_for_log(value: str) -> str:
    """邮箱脱敏，仅用于日志。"""
    email = value.strip()
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    if not local:
        return f"***@{domain}"
    keep = 1 if len(local) < 4 else 2
    return f"{local[:keep]}***@{domain}"


def _short_text_digest(value: str) -> str:
    """生成固定长度摘要，便于在日志中定位请求而不暴露原文。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _build_workflow() -> StateGraph:
    """创建并返回状态图构建器。"""
    return workflow


async def _build_persistent_checkpointer() -> tuple[Any, Any | None, str]:
    """构建持久化 checkpointer（异步版本）。"""
    backend = GRAPH_CHECKPOINTER_BACKEND or "postgres"

    if backend == "memory":
        logger.warning("checkpointer 使用内存模式：服务重启后审批上下文会丢失。")
        return MemorySaver(), None, "memory"

    if backend == "redis":
        try:
            from langgraph.checkpoint.redis import AsyncRedisSaver
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("未安装 Redis checkpointer 依赖：langgraph-checkpoint-redis") from exc

        cm = AsyncRedisSaver.from_conn_string(
            GRAPH_CHECKPOINTER_REDIS_URL,
            checkpoint_prefix=f"{GRAPH_CHECKPOINTER_PREFIX}:checkpoint",
            checkpoint_write_prefix=f"{GRAPH_CHECKPOINTER_PREFIX}:checkpoint_write",
        )
        saver = await cm.__aenter__()
        await saver.setup()
        logger.info(
            "checkpointer 初始化成功 | backend=redis | redis_url=%s | prefix=%s",
            GRAPH_CHECKPOINTER_REDIS_URL,
            GRAPH_CHECKPOINTER_PREFIX,
        )
        return saver, cm, "redis"

    if backend in {"postgres", "postgresql"}:
        if not GRAPH_CHECKPOINTER_POSTGRES_URL:
            raise RuntimeError("GRAPH_CHECKPOINTER_POSTGRES_URL 未配置，无法启用 Postgres checkpointer。")

        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "未安装 Postgres checkpointer 依赖：langgraph-checkpoint-postgres + psycopg[binary]"
            ) from exc

        cm = AsyncPostgresSaver.from_conn_string(GRAPH_CHECKPOINTER_POSTGRES_URL)
        saver = await cm.__aenter__()
        await saver.setup()
        logger.info("checkpointer 初始化成功 | backend=postgres")
        return saver, cm, "postgres"

    raise RuntimeError(f"未知 GRAPH_CHECKPOINTER_BACKEND 配置：{backend}")


async def init_graph_runtime() -> Any:
    """初始化图运行时（含持久化 checkpointer）并返回编译后的 graph。"""
    global app_graph, _checkpointer_cm, _checkpointer_backend_in_use
    if app_graph is not None:
        return app_graph

    try:
        checkpointer, cm, backend = await _build_persistent_checkpointer()
        _checkpointer_cm = cm
        _checkpointer_backend_in_use = backend
    except Exception as exc:  # noqa: BLE001
        if not GRAPH_CHECKPOINTER_ALLOW_MEMORY_FALLBACK:
            raise
        logger.warning(
            "持久化 checkpointer 初始化失败，回退 MemorySaver | backend=%s | error=%s",
            GRAPH_CHECKPOINTER_BACKEND,
            exc,
        )
        checkpointer = MemorySaver()
        _checkpointer_cm = None
        _checkpointer_backend_in_use = "memory"

    app_graph = _build_workflow().compile(checkpointer=checkpointer, interrupt_before=["tools_node"])
    logger.info("graph runtime 初始化完成 | checkpointer_backend=%s", _checkpointer_backend_in_use)
    return app_graph


async def shutdown_graph_runtime() -> None:
    """释放图运行时资源（主要是异步 checkpointer 连接）。"""
    global app_graph, _checkpointer_cm
    if _checkpointer_cm is not None:
        try:
            await _checkpointer_cm.__aexit__(None, None, None)
            logger.info("checkpointer 资源已释放 | backend=%s", _checkpointer_backend_in_use)
        except Exception as exc:  # noqa: BLE001
            logger.warning("释放 checkpointer 资源失败 | backend=%s | error=%s", _checkpointer_backend_in_use, exc)
        finally:
            _checkpointer_cm = None
    app_graph = None


def get_app_graph() -> Any:
    """返回已初始化的 app_graph，未初始化时抛出异常。"""
    if app_graph is None:
        raise RuntimeError("app_graph 尚未初始化，请先调用 init_graph_runtime()。")
    return app_graph


class AgentState(TypedDict):
    """NanoAgent 的图状态定义。"""

    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    memory_context: str
    sender: str


def _default_llm_profile() -> dict[str, str] | None:
    """读取环境变量默认 LLM 配置（兼容旧模式）。"""
    if not DEFAULT_OPENAI_API_KEY:
        return None
    return {
        "api_key": DEFAULT_OPENAI_API_KEY,
        "base_url": DEFAULT_OPENAI_BASE_URL,
        "model": DEFAULT_OPENAI_MODEL,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
    }


def _normalize_llm_profile(profile: Any) -> dict[str, str] | None:
    """标准化会话级 LLM 配置。"""
    if not isinstance(profile, dict):
        return None

    api_key = str(profile.get("api_key", "")).strip()
    base_url = str(profile.get("base_url", "")).strip().rstrip("/")
    model = str(profile.get("model", "")).strip()
    embedding_model = str(profile.get("embedding_model", "")).strip()
    has_embedding_model_field = "embedding_model" in profile

    if not api_key or not base_url or not model:
        return None

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        # 若上游显式传空字符串，表示禁用 embedding；否则再回退默认值。
        "embedding_model": embedding_model if has_embedding_model_field else (embedding_model or DEFAULT_EMBEDDING_MODEL),
    }


def _llm_profile_from_config(config: RunnableConfig | None) -> dict[str, str] | None:
    """优先从请求级 metadata 读取 llm_profile，缺失时回退到默认环境变量。"""
    if isinstance(config, dict):
        metadata = config.get("metadata", {})
        if isinstance(metadata, dict):
            profile = _normalize_llm_profile(metadata.get("llm_profile"))
            if profile is not None:
                return profile

    return _default_llm_profile()


def _chat_llm_from_profile(profile: dict[str, str]) -> ChatOpenAI:
    """按 profile 获取可复用 ChatOpenAI 客户端。"""
    cache_key = (profile["model"], profile["base_url"], profile["api_key"])
    cached = _llm_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        llm_client = ChatOpenAI(
            model=profile["model"],
            api_key=profile["api_key"],
            base_url=profile["base_url"],
            streaming=True,
            temperature=0,
            timeout=60,
            max_retries=2,
        )
    except TypeError:
        llm_client = ChatOpenAI(
            model=profile["model"],
            openai_api_key=profile["api_key"],
            openai_api_base=profile["base_url"],
            streaming=True,
            temperature=0,
            timeout=60,
            max_retries=2,
        )

    _llm_cache[cache_key] = llm_client
    return llm_client


def _non_stream_chat_llm_from_profile(profile: dict[str, str]) -> ChatOpenAI:
    """按 profile 获取非流式 ChatOpenAI 客户端（用于路由/判定器）。"""
    cache_key = (profile["model"], profile["base_url"], profile["api_key"])
    cached = _non_stream_llm_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        llm_client = ChatOpenAI(
            model=profile["model"],
            api_key=profile["api_key"],
            base_url=profile["base_url"],
            streaming=False,
            temperature=0,
            timeout=30,
            max_retries=1,
        )
    except TypeError:
        llm_client = ChatOpenAI(
            model=profile["model"],
            openai_api_key=profile["api_key"],
            openai_api_base=profile["base_url"],
            streaming=False,
            temperature=0,
            timeout=30,
            max_retries=1,
        )

    _non_stream_llm_cache[cache_key] = llm_client
    return llm_client


def _get_chat_llm(config: RunnableConfig | None) -> ChatOpenAI:
    """获取当前请求对应的基础 LLM 客户端。"""
    profile = _llm_profile_from_config(config)
    if profile is None:
        raise RuntimeError("未提供可用的 LLM 配置，请先创建会话或设置默认 OPENAI_API_KEY。")
    return _chat_llm_from_profile(profile)


def _get_non_stream_chat_llm(config: RunnableConfig | None) -> ChatOpenAI:
    """获取当前请求对应的非流式 LLM 客户端。"""
    profile = _llm_profile_from_config(config)
    if profile is None:
        raise RuntimeError("未提供可用的 LLM 配置，请先创建会话或设置默认 OPENAI_API_KEY。")
    return _non_stream_chat_llm_from_profile(profile)


def _get_bound_llm(config: RunnableConfig | None, worker: Literal["data_scientist", "reporter"]) -> Any:
    """获取绑定工具后的 Worker LLM。"""
    profile = _llm_profile_from_config(config)
    if profile is None:
        raise RuntimeError("未提供可用的 LLM 配置，请先创建会话或设置默认 OPENAI_API_KEY。")

    base_llm = _chat_llm_from_profile(profile)
    cache_key = (profile["model"], profile["base_url"], profile["api_key"], worker)
    cached = _bound_llm_cache.get(cache_key)
    if cached is not None:
        return cached

    if worker == "data_scientist":
        bound = base_llm.bind_tools([tool_query_database])
    else:
        # 预留：Reporter 若扩展受控写能力，可复用该分支。
        bound = base_llm.bind_tools([tool_upsert_user_setting])

    _bound_llm_cache[cache_key] = bound
    return bound


def _message_to_text(message: BaseMessage) -> str:
    """尽力将 LangChain 消息内容转换为纯文本。"""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return " ".join(parts).strip()
    return str(content)


def _latest_user_query(messages: list[BaseMessage]) -> str:
    """从消息列表中提取最新一条用户问题。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _message_to_text(msg).strip()
    return ""


def _has_database_intent(query: str) -> bool:
    """判断用户问题是否与数据库查询有关。"""
    normalized = query.strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in DATABASE_HELP_KEYWORDS)


def _has_sql_snippet(query: str) -> bool:
    """判断输入中是否包含可执行 SQL 片段。"""
    normalized = query.strip()
    if not normalized:
        return False
    return bool(SQL_SNIPPET_PATTERN.search(normalized))


def _build_database_help_answer() -> str:
    """生成数据库查询引导文案（面向非 SQL 用户）。"""
    return (
        "我可以帮你查数据库，但需要你给出更具体的查询目标或 SQL。下面是最快可用的提问方式：\n\n"
        "1. 你可以查什么\n"
        "- 当前有哪些数据库\n"
        "- 当前库有哪些表\n"
        "- 某张表有哪些字段\n"
        "- 简单统计（总数、分组计数、最近 N 条记录）\n\n"
        "2. 推荐提问模板（直接复制）\n"
        "- 请调用数据库工具执行 SQL：SELECT datname FROM pg_database;\n"
        "- 请调用数据库工具执行 SQL：SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;\n"
        "- 请调用数据库工具执行 SQL：SELECT column_name, data_type FROM information_schema.columns WHERE table_name='你的表名' ORDER BY ordinal_position;\n"
        "- 请调用数据库工具执行 SQL：SELECT COUNT(*) AS total FROM 你的表名;\n\n"
        "3. 常见失败原因（以及怎么改）\n"
        "- 只允许只读查询：仅支持 SELECT/CTE，不能 INSERT/UPDATE/DELETE/DDL\n"
        "- 不支持多语句：一条请求里只能有一条 SQL（不要写分号后第二条）\n"
        "- 表名/字段名不存在：先查 information_schema 确认结构\n"
        "- 查询超时或结果过大：加 WHERE / LIMIT，缩小范围\n\n"
        "你现在只要告诉我“要查哪张表、查什么字段、时间范围”，我就能帮你拼出可执行 SQL。"
    )


def _strip_dsml_control_tokens(text: str) -> str:
    """移除部分模型返回的 DSML 控制标记，避免污染前端可读输出。"""
    if not text:
        return text
    cleaned = DSML_CONTROL_TOKEN_PATTERN.sub("", text)
    cleaned = CONTROL_EMAIL_ARG_PATTERN.sub("", cleaned)
    cleaned = CONTROL_CONTENT_ARG_OPEN_PATTERN.sub("", cleaned)
    cleaned = CONTROL_MARKUP_FRAGMENT_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"</?\s*>", "", cleaned)
    cleaned_lines: list[str] = []
    for line in cleaned.splitlines():
        normalized_line = line.strip()
        lower_line = normalized_line.lower()
        if normalized_line and any(keyword in lower_line for keyword in CONTROL_MARKUP_KEYWORDS):
            if ("<" in normalized_line) or (">" in normalized_line) or ("string=" in lower_line):
                continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    # 收敛多余空白，但保留换行结构。
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _sanitize_ai_message_text(message: BaseMessage) -> BaseMessage:
    """清理 AIMessage 中的控制标记文本。"""
    if not isinstance(message, AIMessage):
        return message

    original_text = _message_to_text(message)
    cleaned_text = _strip_dsml_control_tokens(original_text)
    if cleaned_text == original_text:
        return message

    try:
        return message.model_copy(update={"content": cleaned_text})
    except AttributeError:
        return message.copy(update={"content": cleaned_text})


def _sanitize_history_for_model(messages: list[BaseMessage], *, max_messages: int = MAX_MODEL_HISTORY_MESSAGES) -> list[BaseMessage]:
    """清理对模型不友好的历史片段（未闭环 tool_calls / 孤立 ToolMessage）。"""
    if not messages:
        return messages

    sanitized: list[BaseMessage] = []
    dropped_count = 0
    index = 0
    total = len(messages)

    while index < total:
        message = messages[index]

        if isinstance(message, AIMessage) and message.tool_calls:
            tool_call_ids = [
                str(call.get("id", "")).strip()
                for call in message.tool_calls
                if isinstance(call, dict) and str(call.get("id", "")).strip()
            ]

            next_index = index + 1
            following_tools: list[ToolMessage] = []
            while next_index < total and isinstance(messages[next_index], ToolMessage):
                following_tools.append(messages[next_index])  # type: ignore[arg-type]
                next_index += 1

            if not tool_call_ids:
                dropped_count += 1
                index = next_index
                continue

            required_ids = set(tool_call_ids)
            found_ids = {
                str(getattr(tool_message, "tool_call_id", "")).strip()
                for tool_message in following_tools
                if str(getattr(tool_message, "tool_call_id", "")).strip()
            }

            if required_ids.issubset(found_ids):
                sanitized.append(message)
                for tool_message in following_tools:
                    tool_call_id = str(getattr(tool_message, "tool_call_id", "")).strip()
                    if tool_call_id in required_ids:
                        sanitized.append(tool_message)
                    else:
                        dropped_count += 1
                index = next_index
                continue

            # assistant/tool 闭环不完整：整段丢弃，避免 OpenAI-compatible 400。
            dropped_count += 1 + len(following_tools)
            index = next_index
            continue

        if isinstance(message, ToolMessage):
            dropped_count += 1
            index += 1
            continue

        sanitized.append(message)
        index += 1

    if max_messages > 0 and len(sanitized) > max_messages:
        tail = sanitized[-max_messages:]
        # 取尾部后再做一次清理，避免截断在 tool_calls 闭环中间。
        sanitized = _sanitize_history_for_model(tail, max_messages=0)

    if dropped_count > 0:
        logger.info("历史消息清理完成 | dropped=%d | kept=%d", dropped_count, len(sanitized))

    return sanitized


def _latest_human_index(messages: list[BaseMessage]) -> int:
    """返回最近一条 HumanMessage 的索引，不存在则返回 -1。"""
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            return idx
    return -1


def _has_recent_send_report_tool_result(messages: list[BaseMessage]) -> bool:
    """判断当前轮次是否已出现 send_report 工具结果。"""
    if not messages:
        return False

    last_human_idx = _latest_human_index(messages)
    start_idx = last_human_idx + 1 if last_human_idx >= 0 else 0

    for message in messages[start_idx:]:
        if not isinstance(message, ToolMessage):
            continue

        text = _message_to_text(message).lower()
        if not text:
            continue
        message_name = str(getattr(message, "name", "")).strip().lower()
        if message_name in {"send_report", "tool_send_report"}:
            return True

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                tool_name = str(payload.get("tool", "")).strip().lower()
                if tool_name in {"send_report", "tool_send_report"}:
                    return True
        except Exception:  # noqa: BLE001
            pass

        if "send_report" in text or "tool_send_report" in text:
            return True

    return False


def _extract_first_email(text: str) -> str:
    """从文本中提取第一个邮箱地址。"""
    match = EMAIL_PATTERN.search(text)
    return match.group(0).strip() if match else ""


def _fallback_report_content_from_query(query: str) -> str:
    """当模型未给出可用正文时，从用户问题生成可发送的兜底正文。"""
    normalized = query.strip()
    if not normalized:
        return "请查收本次自动生成的报告。"

    # 去掉邮箱后，尽量保留需求描述。
    without_email = EMAIL_PATTERN.sub("", normalized)
    without_email = re.sub(
        r"(请|帮我|麻烦)?\s*(发送|发到|发给|寄给|转发)\s*(到)?\s*(我的)?\s*(邮件|邮箱|电邮)",
        "",
        without_email,
        flags=re.IGNORECASE,
    )
    without_email = re.sub(r"(上述|上面|以上|这段|这份)\s*内容", "", without_email)
    without_email = re.sub(r"[，,。；;:：\\s]+", " ", without_email).strip()
    if not without_email:
        return "请查收本次自动生成的报告。"
    return without_email[:REPORT_CONTENT_SOFT_LIMIT]


def _latest_assistant_answer_before_last_user(messages: list[BaseMessage]) -> str:
    """提取最近一轮用户输入之前的最后一条 assistant 可读回答。"""
    last_human_idx = _latest_human_index(messages)
    if last_human_idx <= 0:
        return ""

    for idx in range(last_human_idx - 1, -1, -1):
        message = messages[idx]
        if not isinstance(message, AIMessage):
            continue

        text = _strip_dsml_control_tokens(_message_to_text(message)).strip()
        if not text:
            continue

        upper_text = text.upper().replace('"', "").replace("'", "").strip()
        if upper_text in {"DATASCIENTIST", "REPORTER", "ASSISTANT", "FINISH"}:
            continue

        return text

    return ""


def _derive_report_content(*, latest_query: str, model_response_text: str, history: list[BaseMessage]) -> str:
    """合成稳定可发送的邮件正文，优先模型正文，其次复用上一条助手回答。"""
    cleaned_model_text = _strip_dsml_control_tokens(model_response_text).strip()
    if cleaned_model_text and len(cleaned_model_text) >= 12:
        return cleaned_model_text[:REPORT_CONTENT_SOFT_LIMIT]

    if any(keyword in latest_query for keyword in ("上述内容", "上面内容", "以上内容", "刚才内容")):
        previous_answer = _latest_assistant_answer_before_last_user(history)
        if previous_answer:
            return previous_answer[:REPORT_CONTENT_SOFT_LIMIT]

    return _fallback_report_content_from_query(latest_query)


def _extract_report_content_from_query(query: str) -> str:
    """从用户输入中尽量提取“邮件正文”字段。"""
    normalized = query.strip()
    if not normalized:
        return ""

    patterns = [
        r"(?:内容|正文)\s*[：:]\s*(.+)$",
        r"(?:发送|发给|发到).{0,30}(?:内容|正文)\s*(?:是|为)\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        extracted = _strip_dsml_control_tokens(match.group(1)).strip()
        if extracted:
            return extracted[:REPORT_CONTENT_SOFT_LIMIT]

    return ""


def _build_reporter_success_message(tool_payload: str) -> str:
    """基于 send_report 工具结果生成稳定、可读的确认文案。"""
    parsed_payload: Any = None
    try:
        parsed_payload = json.loads(tool_payload)
    except Exception:  # noqa: BLE001
        parsed_payload = None

    if isinstance(parsed_payload, dict):
        status = str(parsed_payload.get("status", "")).strip().lower()
        message = str(parsed_payload.get("message", "")).strip()
        if status == "success":
            return message or "邮件已发送成功。"
        if status == "error":
            return message or "邮件发送失败，请稍后重试。"

    cleaned = _strip_dsml_control_tokens(tool_payload).strip()
    if cleaned:
        return cleaned[:300]
    return "邮件操作已执行完成。"


def _build_tool_call_message(tool_name: str, args: dict[str, Any]) -> AIMessage:
    """构造可被 ToolNode 执行的 AIMessage(tool_calls)。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": f"call_{uuid4().hex[:12]}",
                "type": "tool_call",
                "name": tool_name,
                "args": args,
            }
        ],
    )


def _normalize_send_report_args(raw_args: Any, *, latest_query: str, history: list[BaseMessage]) -> dict[str, Any] | None:
    """标准化 send_report 参数；无法构造有效参数时返回 None。"""
    args = raw_args if isinstance(raw_args, dict) else {}
    email = _extract_first_email(str(args.get("email", "")).strip()) or _extract_first_email(latest_query)
    if not email:
        return None

    raw_content = str(args.get("content", "")).strip()
    content = _derive_report_content(
        latest_query=latest_query,
        model_response_text=raw_content,
        history=history,
    )
    if not content:
        content = "请查收本次自动生成的报告。"

    return {"email": email, "content": content}


def _get_memory_manager() -> UserMemoryManager | None:
    """懒加载并返回记忆管理器。"""
    global _memory_manager
    if _memory_manager is not None:
        return _memory_manager
    try:
        _memory_manager = UserMemoryManager()
        return _memory_manager
    except Exception as exc:  # noqa: BLE001
        logger.exception("初始化 UserMemoryManager 失败：%s", exc)
        return None


async def _call_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    enforced_user_id: str | None = None,
) -> str:
    """通过 HTTP 代理风格端点调用 MCP 服务。"""
    timeout = httpx.Timeout(connect=5.0, read=25.0, write=25.0, pool=5.0)
    headers: dict[str, str] = {}
    if MCP_SERVICE_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_SERVICE_TOKEN}"
        headers["X-Service-Token"] = MCP_SERVICE_TOKEN
    normalized_user_id = (enforced_user_id or "").strip()
    if normalized_user_id:
        headers["X-NanoAgent-User-Id"] = normalized_user_id

    request_specs: list[tuple[str, dict[str, Any]]] = [
        (f"{MCP_BASE_URL}/tools/{tool_name}", arguments),
        # 移除对 /mcp/tools/{tool_name} 的尝试，因为这是SSE传输端点
    ]

    async with httpx.AsyncClient(timeout=timeout) as client:
        for url, payload in request_specs:
            try:
                logger.info(
                    "MCP 代理请求 | tool=%s | url=%s | payload_keys=%s",
                    tool_name,
                    url,
                    list(payload.keys()),
                )
                response = await client.post(url, json=payload, headers=headers or None)
                if response.is_success:
                    try:
                        body: Any = response.json()
                    except ValueError:
                        body = response.text
                    return json.dumps(
                        {
                            "status": "success",
                            "tool": tool_name,
                            "endpoint": url,
                            "response": body,
                        },
                        ensure_ascii=False,
                        default=str,
                    )

                logger.warning(
                    "MCP 代理响应非成功 | tool=%s | url=%s | status=%d",
                    tool_name,
                    url,
                    response.status_code,
                )
            except httpx.TimeoutException as exc:
                logger.warning(
                    "MCP 代理超时 | tool=%s | url=%s | error=%s",
                    tool_name,
                    url,
                    exc,
                )
            except httpx.RequestError as exc:
                logger.warning(
                    "MCP 代理请求错误 | tool=%s | url=%s | error=%s",
                    tool_name,
                    url,
                    exc,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "MCP 代理出现未知错误 | tool=%s | url=%s | error=%s",
                    tool_name,
                    url,
                    exc,
                )

    return json.dumps(
        {
            "status": "error",
            "tool": tool_name,
            "message": "通过 HTTP 代理端点调用 MCP 失败。",
            "tried_endpoints": [spec[0] for spec in request_specs],
        },
        ensure_ascii=False,
    )


@tool("tool_query_database")
async def tool_query_database(sql: str) -> str:
    """将数据库查询请求代理到 MCP 服务。"""
    normalized_sql = " ".join(sql.split()).strip()
    sql_digest = _short_text_digest(normalized_sql or "<empty>")
    if DEBUG_MODE:
        logger.info(
            "工具行动 | tool_query_database | sql_digest=%s | sql_len=%d | sql_preview=%s",
            sql_digest,
            len(normalized_sql),
            _truncate_for_log(normalized_sql),
        )
    else:
        logger.info(
            "工具行动 | tool_query_database | sql_digest=%s | sql_len=%d",
            sql_digest,
            len(normalized_sql),
        )
    return await _call_mcp_tool("query_database", {"sql": sql})


@tool("tool_send_report")
async def tool_send_report(email: str, content: str) -> str:
    """将报告发送请求代理到 MCP 服务。"""
    masked_email = _mask_email_for_log(email)
    if DEBUG_MODE:
        logger.info(
            "工具行动 | tool_send_report | email=%s | content_length=%d | content_preview=%s",
            masked_email,
            len(content),
            _truncate_for_log(content),
        )
    else:
        logger.info(
            "工具行动 | tool_send_report | email=%s | content_length=%d",
            masked_email,
            len(content),
        )
    return await _call_mcp_tool("send_report", {"email": email, "content": content})


@tool("tool_upsert_user_setting")
async def tool_upsert_user_setting(
    setting_key: str,
    setting_value: str,
    state: Annotated[dict[str, Any], InjectedState],
) -> str:
    """受控写入用户设置（白名单键 + 参数化 upsert）。"""
    effective_user_id = str(state.get("user_id", "")).strip()
    if not effective_user_id:
        logger.error("工具拒绝执行 | tool_upsert_user_setting | reason=missing_effective_user_id")
        return json.dumps(
            {
                "status": "error",
                "tool": "upsert_user_setting",
                "message": "工具执行缺少用户上下文，已拒绝写入。",
            },
            ensure_ascii=False,
        )

    logger.info(
        "工具行动 | tool_upsert_user_setting | user_id=%s | setting_key=%s",
        effective_user_id,
        setting_key,
    )
    return await _call_mcp_tool(
        "upsert_user_setting",
        {
            "user_id": effective_user_id,
            "setting_key": setting_key,
            "setting_value": setting_value,
        },
        enforced_user_id=effective_user_id,
    )


tools = [tool_query_database, tool_send_report, tool_upsert_user_setting]
tools_node = ToolNode(tools)


async def retrieve_memory_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, str]:
    """检索用户长期记忆并写回状态。"""
    user_id = state.get("user_id", "").strip()
    messages = state.get("messages", [])
    current_query = _latest_user_query(messages)

    logger.info(
        "节点开始 | retrieve_memory_node | user_id=%s | has_query=%s",
        user_id or "unknown",
        bool(current_query),
    )

    manager = _get_memory_manager()
    if manager is None or not user_id or not current_query:
        logger.info("节点结束 | retrieve_memory_node | memory_context_chars=0")
        return {"memory_context": "", "sender": ""}

    llm_profile = _llm_profile_from_config(config)
    memory_context = manager.retrieve_context(
        user_id=user_id,
        current_query=current_query,
        k=3,
        embedding_profile=llm_profile,
    )
    logger.info(
        "节点结束 | retrieve_memory_node | user_id=%s | memory_context_chars=%d",
        user_id,
        len(memory_context),
    )
    return {"memory_context": memory_context, "sender": ""}


def _normalize_supervisor_decision(
    raw_text: str,
) -> Literal["DataScientist", "Reporter", "Assistant", "FINISH"]:
    """将主管输出规范化为固定选项。"""
    text = raw_text.strip().replace('"', "").replace("'", "")
    upper_text = text.upper()

    if "DATASCIENTIST" in upper_text:
        return "DataScientist"
    if "REPORTER" in upper_text:
        return "Reporter"
    if "ASSISTANT" in upper_text:
        return "Assistant"
    if "FINISH" in upper_text:
        return "FINISH"
    return "FINISH"


def _friendly_supervisor_error_message(exc: Exception) -> str:
    """将 Supervisor 调用异常转换为面向用户的友好提示。"""
    text = str(exc).lower()
    if (
        "invalid_api_key" in text
        or "incorrect api key" in text
        or "authentication" in text
        or "401" in text
    ):
        return "模型调用失败：当前 AI 会话的 API Key 无效或已过期，请在左侧 AI 配置中重新设置后重试。"
    if "timeout" in text:
        return "模型调用超时，请稍后重试。"
    return "模型调用失败，请检查会话配置后重试。"


async def _is_explicit_send_execution_intent(
    history: list[BaseMessage],
    config: RunnableConfig,
) -> bool:
    """语义判断是否是“立刻执行发送邮件”意图。"""
    if not history:
        return False
    latest_query = _latest_user_query(history)
    if not latest_query:
        return False

    guard_input: list[BaseMessage] = [
        SystemMessage(content=REPORT_EXECUTION_GUARD_PROMPT),
        HumanMessage(content=latest_query),
    ]
    try:
        response = await _get_non_stream_chat_llm(config).ainvoke(guard_input, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("执行意图判定失败，按 DRAFT 处理 | error=%s", exc)
        return False

    decision = _message_to_text(response).strip().upper()
    if "EXECUTE" in decision:
        return True
    return False


async def supervisor_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage] | str]:
    """主管节点：语义路由到 DataScientist / Reporter / Assistant / FINISH。"""
    user_id = state.get("user_id", "").strip()
    history = _sanitize_history_for_model(state.get("messages", []))
    memory_context = state.get("memory_context", "")

    logger.info(
        "节点开始 | supervisor_node | user_id=%s | history_len=%d",
        user_id or "unknown",
        len(history),
    )

    supervisor_prompt = (
        f"{SUPERVISOR_ROUTER_PROMPT}\n\n"
        f"长期记忆上下文：\n{memory_context or '（无）'}"
    )

    model_input: list[BaseMessage] = [SystemMessage(content=supervisor_prompt), *history]

    try:
        response = await _get_chat_llm(config).ainvoke(model_input, config=config)
        decision = _normalize_supervisor_decision(_message_to_text(response))
        logger.info(
            "节点结束 | supervisor_node | user_id=%s | decision=%s",
            user_id or "unknown",
            decision,
        )
        return {"messages": [AIMessage(content=decision)], "sender": "Supervisor"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("节点异常 | supervisor_node | user_id=%s | error=%s", user_id, exc)
        # 不再吞掉异常并静默 FINISH，而是写入可读错误文案，
        # 由流式层在“无 token 场景”作为最终回复下发前端。
        return {"messages": [AIMessage(content=_friendly_supervisor_error_message(exc))], "sender": "Supervisor"}


def _trim_supervisor_decision(messages: list[BaseMessage]) -> list[BaseMessage]:
    """避免把 supervisor 的路由词直接喂给 worker。"""
    if not messages:
        return messages

    last = messages[-1]
    if isinstance(last, AIMessage):
        decision = _normalize_supervisor_decision(_message_to_text(last))
        if decision in {"DataScientist", "Reporter", "Assistant", "FINISH"}:
            return messages[:-1]
    return messages


async def data_scientist_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage] | str]:
    """数据科学家节点：负责数据分析与数据库工具调用。"""
    user_id = state.get("user_id", "").strip()
    history = _sanitize_history_for_model(_trim_supervisor_decision(state.get("messages", [])))
    memory_context = state.get("memory_context", "")
    latest_query = _latest_user_query(history)

    logger.info(
        "节点开始 | data_scientist_node | user_id=%s | history_len=%d",
        user_id or "unknown",
        len(history),
    )

    # 新手友好模式：问“数据库相关”但没给 SQL 时，先返回操作指南而不是空泛回答。
    if _has_database_intent(latest_query) and not _has_sql_snippet(latest_query):
        logger.info(
            "节点结束 | data_scientist_node | user_id=%s | mode=db_help_without_sql",
            user_id or "unknown",
        )
        return {"messages": [AIMessage(content=_build_database_help_answer())], "sender": "DataScientist"}

    system_prompt = (
        "你是 DataScientist 智能体，负责数据分析与事实查询。\n"
        "如需读取数据库，请调用 tool_query_database；若无需查库可直接回答。\n"
        "回答应准确、结构化，并基于可验证信息。\n"
        "当用户表达数据库需求但未提供具体 SQL 时，请先给出清晰的查询引导和可复制 SQL 示例。\n"
        "请优先参考用户长期记忆。\n\n"
        f"长期记忆上下文：\n{memory_context or '（无）'}"
    )
    llm_runner = _get_bound_llm(config, "data_scientist")
    model_input: list[BaseMessage] = [SystemMessage(content=system_prompt), *history]

    try:
        response = await llm_runner.ainvoke(model_input, config=config)
        response = _sanitize_ai_message_text(response)
        tool_call_count = len(response.tool_calls) if isinstance(response, AIMessage) else 0
        logger.info(
            "节点结束 | data_scientist_node | user_id=%s | tool_calls=%d",
            user_id or "unknown",
            tool_call_count,
        )
        return {"messages": [response], "sender": "DataScientist"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("节点异常 | data_scientist_node | user_id=%s | error=%s", user_id, exc)
        fallback = AIMessage(content="数据分析节点处理失败，请稍后重试。")
        return {"messages": [fallback], "sender": "DataScientist"}


async def reporter_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage] | str]:
    """报告专家节点（执行阶段）：仅处理“发送邮件”动作。"""
    user_id = state.get("user_id", "").strip()
    history = _sanitize_history_for_model(_trim_supervisor_decision(state.get("messages", [])))
    latest_query = _latest_user_query(history)
    has_send_report_result = _has_recent_send_report_tool_result(history)

    logger.info(
        "节点开始 | reporter_node | user_id=%s | history_len=%d",
        user_id or "unknown",
        len(history),
    )

    if has_send_report_result:
        for message in reversed(history):
            if not isinstance(message, ToolMessage):
                continue
            name = str(getattr(message, "name", "")).strip().lower()
            if name not in {"tool_send_report", "send_report"}:
                continue
            summary = _build_reporter_success_message(_message_to_text(message))
            logger.info(
                "节点结束 | reporter_node | user_id=%s | mode=post_send_summary",
                user_id or "unknown",
            )
            return {"messages": [AIMessage(content=summary)], "sender": "Reporter"}

        logger.info(
            "节点结束 | reporter_node | user_id=%s | mode=post_send_summary_fallback",
            user_id or "unknown",
        )
        return {"messages": [AIMessage(content="邮件发送流程已结束。")], "sender": "Reporter"}

    try:
        execute_intent = await _is_explicit_send_execution_intent(history, config)
        if not execute_intent:
            logger.info(
                "节点结束 | reporter_node | user_id=%s | reason=not_explicit_execute_intent",
                user_id or "unknown",
            )
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "我已将本轮需求判定为“内容起草/普通对话”，不会直接发送邮件。"
                            "如果你确认要发送，请明确回复：确认发送到 xxx@xxx.com。"
                        )
                    )
                ],
                "sender": "Reporter",
            }

        email = _extract_first_email(latest_query)
        content = _extract_report_content_from_query(latest_query)
        if not content:
            content = _latest_assistant_answer_before_last_user(history)
        content = _strip_dsml_control_tokens(content).strip()

        if not email:
            logger.info("节点结束 | reporter_node | user_id=%s | reason=missing_email", user_id or "unknown")
            return {
                "messages": [AIMessage(content="我还没有拿到收件邮箱，请补充“发送到 xxx@xxx.com”。")],
                "sender": "Reporter",
            }

        if not content:
            logger.info("节点结束 | reporter_node | user_id=%s | reason=missing_content", user_id or "unknown")
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "当前没有可发送的正文。请先让我生成邮件草稿，"
                            "然后再回复“确认发送到 xxx@xxx.com”。"
                        )
                    )
                ],
                "sender": "Reporter",
            }

        normalized_args = _normalize_send_report_args(
            {"email": email, "content": content[:REPORT_CONTENT_SOFT_LIMIT]},
            latest_query=latest_query,
            history=history,
        )
        if normalized_args is None:
            return {
                "messages": [AIMessage(content="邮件参数不完整，请补充邮箱和发送内容后重试。")],
                "sender": "Reporter",
            }

        tool_call_msg = _build_tool_call_message("tool_send_report", normalized_args)
        logger.info(
            "节点结束 | reporter_node | user_id=%s | mode=prepare_send | email_masked=%s",
            user_id or "unknown",
            _mask_email_for_log(normalized_args["email"]),
        )
        return {"messages": [tool_call_msg], "sender": "Reporter"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("节点异常 | reporter_node | user_id=%s | error=%s", user_id, exc)
        return {"messages": [AIMessage(content="报告执行节点处理失败，请稍后重试。")], "sender": "Reporter"}


async def assistant_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage] | str]:
    """Assistant 节点：负责一般对话生成，不调用数据库和邮件工具。"""
    user_id = state.get("user_id", "").strip()
    history = _sanitize_history_for_model(_trim_supervisor_decision(state.get("messages", [])))
    memory_context = state.get("memory_context", "")

    logger.info(
        "节点开始 | assistant_node | user_id=%s | history_len=%d",
        user_id or "unknown",
        len(history),
    )

    system_prompt = (
        f"{NO_TOOL_INTENT_PROMPT}\n"
        "当用户提出“发邮件”诉求时，先生成可审阅草稿，不要直接执行发送。\n"
        "回答要自然、直接、可执行。\n\n"
        f"长期记忆上下文：\n{memory_context or '（无）'}"
    )
    model_input: list[BaseMessage] = [SystemMessage(content=system_prompt), *history]

    try:
        response = await _get_chat_llm(config).ainvoke(model_input, config=config)
        response = _sanitize_ai_message_text(response)
        logger.info("节点结束 | assistant_node | user_id=%s", user_id or "unknown")
        return {"messages": [response], "sender": "Assistant"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("节点异常 | assistant_node | user_id=%s | error=%s", user_id, exc)
        fallback = AIMessage(content="助手节点处理失败，请稍后重试。")
        return {"messages": [fallback], "sender": "Assistant"}


def _route_after_supervisor(
    state: AgentState,
) -> Literal["data_scientist_node", "reporter_node", "assistant_node", "__end__"]:
    """主管路由：根据主管最后输出决定下一跳。"""
    messages = state.get("messages", [])
    if not messages:
        logger.info("路由 | supervisor_node -> END | reason=no_messages")
        return "__end__"

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        logger.info("路由 | supervisor_node -> END | reason=last_not_ai")
        return "__end__"

    decision = _normalize_supervisor_decision(_message_to_text(last_message))
    if decision == "DataScientist":
        logger.info("路由 | supervisor_node -> data_scientist_node")
        return "data_scientist_node"
    if decision == "Reporter":
        logger.info("路由 | supervisor_node -> reporter_node")
        return "reporter_node"
    if decision == "Assistant":
        logger.info("路由 | supervisor_node -> assistant_node")
        return "assistant_node"

    logger.info("路由 | supervisor_node -> END")
    return "__end__"


def _route_after_data_scientist(
    state: AgentState,
) -> Literal["tools_node", "__end__"]:
    """数据科学家节点后路由。"""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("路由 | data_scientist_node -> tools_node | tool_calls=%d", len(last_message.tool_calls))
        return "tools_node"

    logger.info("路由 | data_scientist_node -> END | reason=no_tool_calls")
    return "__end__"


def _route_after_reporter(
    state: AgentState,
) -> Literal["tools_node", "__end__"]:
    """报告节点后路由。"""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("路由 | reporter_node -> tools_node | tool_calls=%d", len(last_message.tool_calls))
        return "tools_node"

    logger.info("路由 | reporter_node -> END | reason=no_tool_calls")
    return "__end__"


def _route_after_assistant(_: AgentState) -> Literal["__end__"]:
    """Assistant 节点后路由：本轮回答完成后直接结束，避免无效循环。"""
    return "__end__"


def _route_after_tools(state: AgentState) -> Literal["data_scientist_node", "reporter_node"]:
    """工具节点后路由：按 sender 回到对应 Worker。"""
    sender = (state.get("sender") or "").strip()
    if sender == "Reporter":
        logger.info("路由 | tools_node -> reporter_node | sender=%s", sender)
        return "reporter_node"

    logger.info("路由 | tools_node -> data_scientist_node | sender=%s", sender or "unknown")
    return "data_scientist_node"


workflow = StateGraph(AgentState)
workflow.add_node("retrieve_memory_node", retrieve_memory_node)
workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("data_scientist_node", data_scientist_node)
workflow.add_node("reporter_node", reporter_node)
workflow.add_node("assistant_node", assistant_node)
workflow.add_node("tools_node", tools_node)

workflow.add_edge(START, "retrieve_memory_node")
workflow.add_edge("retrieve_memory_node", "supervisor_node")

workflow.add_conditional_edges(
    "supervisor_node",
    _route_after_supervisor,
    {
        "data_scientist_node": "data_scientist_node",
        "reporter_node": "reporter_node",
        "assistant_node": "assistant_node",
        "__end__": END,
    },
)

workflow.add_conditional_edges(
    "data_scientist_node",
    _route_after_data_scientist,
    {
        "tools_node": "tools_node",
        "__end__": END,
    },
)

workflow.add_conditional_edges(
    "reporter_node",
    _route_after_reporter,
    {
        "tools_node": "tools_node",
        "__end__": END,
    },
)

workflow.add_conditional_edges(
    "assistant_node",
    _route_after_assistant,
    {
        "__end__": END,
    },
)

workflow.add_conditional_edges(
    "tools_node",
    _route_after_tools,
    {
        "data_scientist_node": "data_scientist_node",
        "reporter_node": "reporter_node",
    },
)