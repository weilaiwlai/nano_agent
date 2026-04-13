"""工具函数模块。"""

import asyncio
import ipaddress
import json
import logging
import os
import re
from typing import Any, AsyncIterator
import uuid

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import (
    ALLOWED_LLM_BASE_URLS,
    AUTO_MEMORY_MAX_LEN,
    DEBUG_MODE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FALLBACK_BASE_URL,
    ENVIRONMENT,
    MAX_TOOL_PAYLOAD_LENGTH,
    MEMORY_FACT_PATTERN,
    PRODUCTION_ENV_ALIASES,
    PROVIDER_PRESETS,
    SENSITIVE_KEYWORDS,
    STREAM_CONTROL_CONTENT_ARG_OPEN_PATTERN,
    STREAM_CONTROL_EMAIL_ARG_PATTERN,
    STREAM_CONTROL_FRAGMENT_PATTERN,
    STREAM_CONTROL_KEYWORDS,
    SUPPORTED_PROVIDERS,
    _SUPERVISOR_ROUTE_WORDS,
)
from memory import MemoryProviderError, UserMemoryManager
from session_store import LLMSessionStore
from graph.workflow import get_app_graph

logger = logging.getLogger("nanoagent.agent_service.utils")


_memory_manager: UserMemoryManager | None = None


def _get_memory_manager() -> UserMemoryManager:
    """懒加载并返回记忆管理器单例。"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = UserMemoryManager()
    return _memory_manager


def _should_auto_save_memory(query: str) -> bool:
    """判断用户输入是否应自动写入长期记忆。"""
    normalized_query = query.strip()
    if not normalized_query:
        return False
    if len(normalized_query) < 4 or len(normalized_query) > AUTO_MEMORY_MAX_LEN:
        return False
    return bool(MEMORY_FACT_PATTERN.search(normalized_query))


def _is_production_environment() -> bool:
    """当前是否处于生产环境。"""
    return ENVIRONMENT in PRODUCTION_ENV_ALIASES


def _is_disallowed_local_base_url(base_url: str) -> bool:
    """生产环境下禁止本地/内网地址作为模型网关。"""
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return True

    blocked_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}
    if host in blocked_hosts or host.endswith(".local"):
        return True

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False

    return any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_reserved,
            ip.is_multicast,
            ip.is_unspecified,
        )
    )


def _normalize_base_url(base_url: str | None, *, fallback_to_env: bool = True) -> str:
    """标准化并校验模型服务 Base URL。"""
    from urllib.parse import urlparse

    default_url = os.getenv("OPENAI_BASE_URL", os.getenv("QWEN_BASE_URL", DEFAULT_FALLBACK_BASE_URL))
    raw = (base_url or default_url if fallback_to_env else base_url or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="base_url 不能为空")

    normalized = raw.rstrip("/")

    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="base_url 非法，请传入有效的 http(s) URL")

    if _is_production_environment() and _is_disallowed_local_base_url(normalized):
        raise HTTPException(status_code=400, detail="生产环境禁止使用本地或内网 base_url")

    if ALLOWED_LLM_BASE_URLS and all(not normalized.startswith(prefix) for prefix in ALLOWED_LLM_BASE_URLS):
        raise HTTPException(status_code=400, detail="base_url 不在允许列表内")

    return normalized


def _infer_provider_from_base_url(base_url: str) -> str:
    """根据 base_url 推断 provider。"""
    from urllib.parse import urlparse

    host = (urlparse(base_url).netloc or "").lower()
    if "dashscope.aliyuncs.com" in host:
        return "qwen"
    if "api.openai.com" in host:
        return "openai"
    if "api.deepseek.com" in host:
        return "deepseek"
    if "api.groq.com" in host:
        return "groq"
    return "other"


def _normalize_provider(provider: str | None, base_url: str | None) -> str:
    """标准化 provider，缺失时按 base_url 自动推断。"""
    normalized_provider = (provider or "").strip().lower()
    if normalized_provider:
        if normalized_provider not in SUPPORTED_PROVIDERS:
            raise HTTPException(status_code=400, detail=f"provider 不支持，允许值: {', '.join(SUPPORTED_PROVIDERS)}")
        return normalized_provider

    if base_url and base_url.strip():
        return _infer_provider_from_base_url(_normalize_base_url(base_url, fallback_to_env=False))

    env_base_url = _normalize_base_url(
        os.getenv("OPENAI_BASE_URL", os.getenv("QWEN_BASE_URL", DEFAULT_FALLBACK_BASE_URL)),
        fallback_to_env=False,
    )
    return _infer_provider_from_base_url(env_base_url)


def _resolve_base_url_for_provider(provider: str, requested_base_url: str | None) -> str:
    """解析 provider 下最终使用的 base_url。"""
    preset = PROVIDER_PRESETS.get(provider, {})
    preset_base_url = str(preset.get("base_url", "")).strip()

    if provider == "other":
        if not requested_base_url or not requested_base_url.strip():
            raise HTTPException(status_code=400, detail="provider=other 时必须提供 base_url")
        return _normalize_base_url(requested_base_url, fallback_to_env=False)

    normalized_preset = _normalize_base_url(preset_base_url, fallback_to_env=False)
    if requested_base_url and requested_base_url.strip():
        normalized_requested = _normalize_base_url(requested_base_url, fallback_to_env=False)
        if normalized_requested != normalized_preset:
            raise HTTPException(
                status_code=400,
                detail=f"provider={provider} 使用平台预设 base_url，不支持自定义 URL",
            )

    return normalized_preset


def _resolve_embedding_model_for_provider(provider: str, requested_embedding_model: str | None) -> str:
    """解析 provider 下最终使用的 embedding 模型。"""
    if requested_embedding_model is not None:
        return requested_embedding_model.strip()

    preset = PROVIDER_PRESETS.get(provider, {})
    if "embedding_model" in preset:
        return str(preset.get("embedding_model", "")).strip()

    preset_embedding_model = str(preset.get("embedding_model", "")).strip()
    if preset_embedding_model:
        return preset_embedding_model

    return DEFAULT_EMBEDDING_MODEL


def _default_llm_profile() -> dict[str, str] | None:
    """读取默认环境变量模型配置。"""
    api_key = os.getenv("OPENAI_API_KEY", os.getenv("QWEN_API_KEY", "")).strip()
    if not api_key:
        return None

    env_base_url = _normalize_base_url(
        os.getenv("OPENAI_BASE_URL", os.getenv("QWEN_BASE_URL", DEFAULT_FALLBACK_BASE_URL))
    )
    provider = _normalize_provider(None, env_base_url)
    embedding_model = (os.getenv("EMBEDDING_MODEL", "").strip() or _resolve_embedding_model_for_provider(provider, None))

    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": env_base_url,
        "model": os.getenv("OPENAI_MODEL", os.getenv("QWEN_MODEL", "qwen3.5-plus")).strip(),
        "embedding_model": embedding_model,
    }


def _safe_error_text(exc: Exception, *, max_len: int = 220) -> str:
    """生成适合给前端展示的短错误信息。"""
    text = str(exc).replace("\n", " ").strip()
    if not text:
        text = exc.__class__.__name__
    if len(text) > max_len:
        return text[:max_len] + "...(truncated)"
    return text


def _truncate_text(value: str, *, max_len: int = MAX_TOOL_PAYLOAD_LENGTH) -> str:
    """按最大长度截断字符串。"""
    text = value.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...(truncated)"


def _mask_email(value: str) -> str:
    """对邮箱做部分脱敏。"""
    email = value.strip()
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    if not local:
        return f"***@{domain}"
    keep = 1 if len(local) < 4 else 2
    return f"{local[:keep]}***@{domain}"


def _looks_sensitive_field(field_name: str) -> bool:
    """判断字段名是否可能包含敏感信息。"""
    normalized = field_name.strip().lower()
    return any(keyword in normalized for keyword in SENSITIVE_KEYWORDS)


def _redact_value_by_key(key: str, value: Any) -> Any:
    """按字段名脱敏值。"""
    if not _looks_sensitive_field(key):
        return value
    if isinstance(value, str) and "@" in value:
        return _mask_email(value)
    return "[REDACTED]"


def _safe_tool_payload(payload: Any, max_length: int = MAX_TOOL_PAYLOAD_LENGTH) -> Any:
    """工具输入输出默认脱敏并截断，避免泄露敏感数据。"""
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for raw_key, raw_value in payload.items():
            key = str(raw_key)
            value = _redact_value_by_key(key, raw_value)
            if value == "[REDACTED]":
                sanitized[key] = value
            else:
                sanitized[key] = _safe_tool_payload(value, max_length=max_length)
        return sanitized

    if isinstance(payload, list):
        return [_safe_tool_payload(item, max_length=max_length) for item in payload[:20]]

    if isinstance(payload, str):
        return _truncate_text(payload, max_len=max_length)

    return payload


def _tool_payload_summary(payload: Any) -> dict[str, Any]:
    """生产默认仅返回工具事件摘要，不返回细节入参/出参。"""
    if isinstance(payload, dict):
        return {"kind": "object", "keys": sorted(str(key) for key in payload.keys())[:20]}
    if isinstance(payload, list):
        return {"kind": "array", "size": len(payload)}
    if isinstance(payload, str):
        return {"kind": "text", "chars": len(payload)}
    return {"kind": type(payload).__name__}


def _sanitize_tool_call_for_client(call: dict[str, Any]) -> dict[str, Any]:
    """SSE 对前端下发的 tool_call 安全视图。"""
    sanitized = {
        "id": str(call.get("id", "")).strip(),
        "name": str(call.get("name", "")).strip(),
        "type": str(call.get("type", "tool_call")).strip() or "tool_call",
    }
    args = call.get("args")
    if DEBUG_MODE:
        sanitized["args"] = _safe_tool_payload(args)
    else:
        sanitized["args"] = _tool_payload_summary(args)
        sanitized["redacted"] = True
    return sanitized


def _sanitize_tool_calls_for_client(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """批量脱敏 tool_calls。"""
    return [_sanitize_tool_call_for_client(call) for call in tool_calls if isinstance(call, dict)]


def _build_validation_chat_client(llm_profile: dict[str, str]) -> ChatOpenAI:
    """为连接校验构建轻量 Chat 模型客户端。"""
    common_kwargs: dict[str, Any] = {
        "model": llm_profile["model"],
        "temperature": 0,
        "streaming": False,
        "timeout": 20,
        "max_retries": 0,
        "max_tokens": 8,
    }
    try:
        return ChatOpenAI(
            api_key=llm_profile["api_key"],
            base_url=llm_profile["base_url"],
            **common_kwargs,
        )
    except TypeError:
        return ChatOpenAI(
            openai_api_key=llm_profile["api_key"],
            openai_api_base=llm_profile["base_url"],
            **common_kwargs,
        )


def _build_validation_embeddings_client(llm_profile: dict[str, str]) -> OpenAIEmbeddings:
    """为连接校验构建 Embedding 客户端。"""
    try:
        return OpenAIEmbeddings(
            model=llm_profile["embedding_model"],
            api_key=llm_profile["api_key"],
            base_url=llm_profile["base_url"],
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )
    except TypeError:
        return OpenAIEmbeddings(
            model=llm_profile["embedding_model"],
            openai_api_key=llm_profile["api_key"],
            openai_api_base=llm_profile["base_url"],
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )


async def _validate_llm_profile_connectivity(
    llm_profile: dict[str, str],
    *,
    validate_chat: bool,
    validate_embedding: bool,
) -> tuple[bool, bool, list[str]]:
    """对配置执行轻量连通性验证。"""
    chat_ok = True
    embedding_ok = True
    errors: list[str] = []

    if validate_chat:
        try:
            chat_client = _build_validation_chat_client(llm_profile)
            await chat_client.ainvoke([HumanMessage(content="ping")])
        except Exception as exc:  # noqa: BLE001
            chat_ok = False
            errors.append(f"chat 校验失败: {_safe_error_text(exc)}")

    if validate_embedding:
        if not llm_profile.get("embedding_model", "").strip():
            embedding_ok = True
            errors.append("embedding 校验已跳过：当前会话未配置 embedding 模型。")
        else:
            try:
                embedding_client = _build_validation_embeddings_client(llm_profile)
                await asyncio.to_thread(embedding_client.embed_query, "ping")
            except Exception as exc:  # noqa: BLE001
                embedding_ok = False
                errors.append(f"embedding 校验失败: {_safe_error_text(exc)}")

    return chat_ok, embedding_ok, errors


async def _resolve_llm_profile(session_id: str | None, *, owner_id: str, session_store: LLMSessionStore, session_store_ready: bool) -> dict[str, str]:
    """解析会话级 LLM profile（优先 session，其次默认环境）。"""
    normalized_session_id = (session_id or "").strip()
    if normalized_session_id:
        if not session_store_ready:
            raise HTTPException(status_code=503, detail="LLM 会话存储未就绪，请稍后重试")
        profile = await session_store.get_session(normalized_session_id, owner_id=owner_id)
        if profile is None:
            raise HTTPException(status_code=401, detail="LLM 会话不存在、已过期或不属于当前用户")
        if "provider" not in profile:
            profile["provider"] = _infer_provider_from_base_url(profile["base_url"])
        return profile

    from config import REQUIRE_LLM_SESSION

    if REQUIRE_LLM_SESSION:
        raise HTTPException(status_code=401, detail="请先创建 LLM 会话并携带 session_id")

    default_profile = _default_llm_profile()
    if default_profile is not None:
        return default_profile

    raise HTTPException(status_code=401, detail="未配置默认模型，请先创建 LLM 会话并携带 session_id")


def _try_auto_save_memory(
    user_id: str,
    query: str,
    llm_profile: dict[str, str],
) -> str | None:
    """尝试将用户自述背景自动写入长期记忆。"""
    if not _should_auto_save_memory(query):
        return None
    if not llm_profile.get("embedding_model", "").strip():
        logger.info("自动记忆写入跳过 | user_id=%s | reason=embedding_disabled", user_id)
        return None

    try:
        manager = _get_memory_manager()
        memory_text = f"用户自述背景：{query.strip()}"
        memory_id = manager.save_preference(
            user_id=user_id,
            preference_text=memory_text,
            embedding_profile=llm_profile,
        )
        if memory_id:
            logger.info("自动记忆写入成功 | user_id=%s | memory_id=%s", user_id, memory_id)
        return memory_id
    except MemoryProviderError as exc:
        logger.warning(
            "自动记忆写入失败（模型侧） | user_id=%s | reason=%s | error=%s",
            user_id,
            exc.reason,
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("自动记忆写入失败（不中断主流程） | user_id=%s | error=%s", user_id, exc)
        return None


def _to_sse(payload: dict[str, Any]) -> str:
    """将字典消息包装为 SSE data 帧。"""
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _extract_token_content(event: dict[str, Any]) -> str:
    """从 on_chat_model_stream 事件中提取 token 文本。"""
    data = event.get("data", {}) if isinstance(event, dict) else {}
    chunk = data.get("chunk")
    if chunk is None:
        return ""

    content = getattr(chunk, "content", None)
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
        return "".join(parts)

    if isinstance(chunk, dict):
        dict_content = chunk.get("content")
        if isinstance(dict_content, str):
            return dict_content

    return ""


def _sanitize_stream_token_text(token_text: str) -> str:
    """清理流式 token 中的控制标记，避免污染前端展示。"""
    if not token_text:
        return token_text
    cleaned = STREAM_CONTROL_EMAIL_ARG_PATTERN.sub("", token_text)
    cleaned = STREAM_CONTROL_CONTENT_ARG_OPEN_PATTERN.sub("", cleaned)
    cleaned = STREAM_CONTROL_FRAGMENT_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"</?\s*>", "", cleaned)
    lowered = cleaned.lower()
    if any(keyword in lowered for keyword in STREAM_CONTROL_KEYWORDS):
        if ("<" in cleaned) or (">" in cleaned) or ("string=" in lowered):
            return ""
    return cleaned


def _graph_config(user_id: str, llm_profile: dict[str, str], thread_id: str) -> dict[str, Any]:
    """构建图执行配置。"""
    logger.info("创建新对话：当前thread_id=%s", f"{thread_id}")
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "llm_api_key": llm_profile["api_key"],
            "llm_base_url": llm_profile["base_url"],
            "llm_model": llm_profile["model"],
            "llm_embedding_model": llm_profile.get("embedding_model", ""),
        }
    }


def _is_waiting_for_tools_node(state: Any) -> bool:
    """判断图是否在等待工具节点审批。"""
    values = getattr(state, "values", {}) or {}
    if not isinstance(values, dict):
        return False
    next_node = values.get("__interrupt__")
    if not next_node:
        return False
    if isinstance(next_node, dict):
        next_node = next_node.get("value")
    return str(next_node) == "tools_node"


def _extract_pending_tool_calls(state: Any) -> list[dict[str, Any]]:
    """从图状态中提取待审批的工具调用。"""
    values = getattr(state, "values", {}) or {}
    if not isinstance(values, dict):
        return []

    messages = values.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return []

    tool_calls: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        calls = getattr(message, "tool_calls", [])
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            tool_calls.append(call)

    return tool_calls


def _message_content_to_text(content: Any) -> str:
    """将消息内容转换为纯文本。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts).strip()
    return str(content).strip()


def _extract_final_ai_answer_text(state: Any) -> str:
    """当流式 token 为空时，从最终状态提取可展示的 AI 文本。"""
    values = getattr(state, "values", {}) or {}
    if not isinstance(values, dict):
        return ""

    messages = values.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return ""

    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        text = _message_content_to_text(getattr(message, "content", ""))
        if not text:
            continue
        normalized = text.replace('"', "").replace("'", "").strip().upper()
        if normalized in _SUPERVISOR_ROUTE_WORDS:
            continue
        return text
    return ""


def _build_reject_tool_messages(tool_calls: list[dict[str, Any]]) -> list[ToolMessage]:
    """构造"用户拒绝执行工具"的 ToolMessage 列表。"""
    tool_messages: list[ToolMessage] = []

    for call in tool_calls:
        tool_call_id = str(call.get("id", "")).strip()
        if not tool_call_id:
            continue
        tool_messages.append(
            ToolMessage(
                content="用户拒绝了此操作",
                tool_call_id=tool_call_id,
            )
        )

    return tool_messages


def _is_email_delivery_query(query: str) -> bool:
    """轻量判断当前问题是否明确要求"发邮件"。"""
    import re

    normalized = query.strip()
    if not normalized:
        return False
    lower_text = normalized.lower()
    has_send_action = any(keyword in normalized for keyword in ("发送", "发到", "发给", "寄给", "转发"))
    has_mail_target = any(keyword in normalized for keyword in ("邮件", "邮箱", "电邮")) or ("email" in lower_text)
    has_email_address = bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", normalized))
    return (has_send_action and has_mail_target) or (has_send_action and has_email_address)


def _all_pending_tool_calls_are_send_report(tool_calls: list[dict[str, Any]]) -> bool:
    """判断待审批工具是否全部为 send_report。"""
    if not tool_calls:
        return False
    for call in tool_calls:
        if not isinstance(call, dict):
            return False
        name = str(call.get("name", "")).strip()
        if name not in {"tool_send_report", "send_report"}:
            return False
    return True


async def _stream_pending_interrupt_only(
    *,
    user_id: str,
    pending_tool_calls: list[dict[str, Any]],
) -> AsyncIterator[str]:
    """仅回放待审批中断事件，帮助前端在漏事件后恢复审批状态。"""
    logger.info(
        "回放 HITL 待审批事件 | user_id=%s | pending_tools=%d",
        user_id,
        len(pending_tool_calls),
    )
    yield _to_sse(
        {
            "type": "interrupt",
            "tool_calls": _sanitize_tool_calls_for_client(pending_tool_calls),
        }
    )
    yield _to_sse(
        {
            "type": "error",
            "message": "检测到上一轮存在待审批工具调用，请先审批后再继续聊天。",
        }
    )
    yield "data: [DONE]\n\n"


async def _stream_resume_no_pending(*, user_id: str) -> AsyncIterator[str]:
    """resume 幂等兜底：无待审批时返回可读提示而非报错。"""
    logger.info("resume 幂等返回 | user_id=%s | reason=no_pending_interrupt", user_id)
    yield _to_sse({"type": "token", "content": "审批状态已更新：当前没有待审批工具调用。"})
    yield "data: [DONE]\n\n"


async def _stream_graph_events(
    initial_input: Any,
    *,
    config: dict[str, Any],
    user_id: str,
    emit_interrupt: bool,
) -> AsyncIterator[str]:
    """复用的图事件流 -> SSE 转换器。"""

    active_node = ""
    emitted_worker_token = False
    worker_nodes = {"knowledge_worker_node", "reporter_node", "assistant_node"}
    broadcast_nodes = {"supervisor_node", "knowledge_worker_node", "reporter_node", "assistant_node"}

    try:
        async for event in get_app_graph().astream_events(initial_input, config=config, version="v1"):
            event_name = event.get("event", "")
            node_name = str(event.get("name", "")).strip()

            if event_name == "on_chain_start" and node_name in broadcast_nodes:
                active_node = node_name
                yield _to_sse({"type": "agent_switch", "agent": node_name})

            if event_name == "on_chat_model_stream":
                if active_node not in worker_nodes:
                    continue
                token_text = _extract_token_content(event)
                token_text = _sanitize_stream_token_text(token_text)
                if token_text:
                    emitted_worker_token = True
                    yield _to_sse({"type": "token", "content": token_text})

            elif event_name == "on_tool_start":
                data = event.get("data", {}) if isinstance(event, dict) else {}
                tool_name = event.get("name", "unknown_tool")
                raw_tool_input = data.get("input")
                tool_input = _safe_tool_payload(raw_tool_input)
                logger.info("流式事件 | tool_start | user_id=%s | tool=%s", user_id, tool_name)
                payload: dict[str, Any] = {"type": "tool_start", "tool": tool_name}
                if DEBUG_MODE:
                    payload["input"] = tool_input
                else:
                    payload["input"] = _tool_payload_summary(tool_input)
                    payload["redacted"] = True
                yield _to_sse(payload)

            elif event_name == "on_tool_end":
                data = event.get("data", {}) if isinstance(event, dict) else {}
                tool_name = event.get("name", "unknown_tool")
                raw_tool_output = data.get("output")
                tool_output = _safe_tool_payload(raw_tool_output)
                logger.info("流式事件 | tool_end | user_id=%s | tool=%s", user_id, tool_name)
                payload = {"type": "tool_end", "tool": tool_name}
                if DEBUG_MODE:
                    payload["output"] = tool_output
                else:
                    payload["output"] = _tool_payload_summary(tool_output)
                    payload["redacted"] = True
                yield _to_sse(payload)

        state: Any | None = None
        if emit_interrupt or not emitted_worker_token:
            state = await get_app_graph().aget_state(config)

        if emit_interrupt and state is not None:
            if _is_waiting_for_tools_node(state):
                pending_tool_calls = _extract_pending_tool_calls(state)
                if pending_tool_calls:
                    logger.info(
                        "命中 HITL 中断 | user_id=%s | pending_tools=%d",
                        user_id,
                        len(pending_tool_calls),
                    )
                    yield _to_sse(
                        {
                            "type": "interrupt",
                            "tool_calls": _sanitize_tool_calls_for_client(pending_tool_calls),
                        }
                    )
            elif not emitted_worker_token:
                fallback_answer = _extract_final_ai_answer_text(state)
                if fallback_answer:
                    yield _to_sse({"type": "token", "content": fallback_answer})
        elif not emitted_worker_token and state is not None:
            fallback_answer = _extract_final_ai_answer_text(state)
            if fallback_answer:
                yield _to_sse({"type": "token", "content": fallback_answer})

        yield "data: [DONE]\n\n"
    except Exception as exc:  # noqa: BLE001
        if DEBUG_MODE:
            logger.exception("流式聊天执行失败 | user_id=%s | error=%s", user_id, exc)
            message = _safe_error_text(exc, max_len=300)
        else:
            logger.exception("流式聊天执行失败 | user_id=%s", user_id)
            message = "流式处理出现异常，请稍后重试。"
        yield _to_sse({"type": "error", "message": message})
        yield "data: [DONE]\n\n"