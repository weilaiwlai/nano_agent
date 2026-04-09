"""NanoAgent 节点函数模块。

定义工作流中的所有节点函数。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from .config import REPORT_CONTENT_SOFT_LIMIT, logger
from .llm import _get_bound_llm, _get_chat_llm, _get_non_stream_chat_llm, _llm_profile_from_config
from .prompts import (
    NO_TOOL_INTENT_PROMPT,
    REPORT_EXECUTION_GUARD_PROMPT,
    SUPERVISOR_ROUTER_PROMPT,
)
from .state import AgentState
from .tools import _get_memory_manager
from .utils import _build_tool_call_message, _build_reporter_success_message, _normalize_send_report_args
from .utils import (
    _build_database_help_answer,
    _derive_report_content,
    _extract_first_email,
    _extract_report_content_from_query,
    _has_database_intent,
    _has_recent_send_report_tool_result,
    _has_sql_snippet,
    _latest_assistant_answer_before_last_user,
    _latest_user_query,
    _mask_email_for_log,
    _message_to_text,
    _sanitize_ai_message_text,
    _sanitize_history_for_model,
    _strip_dsml_control_tokens,
)


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
    """语义判断是否是"立刻执行发送邮件"意图。"""
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

    if _has_database_intent(latest_query) and not _has_sql_snippet(latest_query):
        logger.info(
            "节点结束 | data_scientist_node | user_id=%s | mode=db_help_without_sql",
            user_id or "unknown",
        )
        return {"messages": [AIMessage(content=_build_database_help_answer())], "sender": "DataScientist"}

    system_prompt = (
        "你是 DataScientist 智能体，负责数据分析与事实查询。\n"
        "如需读取数据库，请调用 tool_query_database；若无需查库可直接回答。\n"
        "如需查询当前时间，请调用 tool_get_current_time。\n"
        "如需查询网络信息，请调用 tool_search。\n"
        "如需查询允许目录，请调用 tool_list_allowed_directories。\n"
        "如需检查路径是否被允许，请调用 tool_is_path_allowed。\n"
        "如需读取文件，请调用 tool_read_file。\n"
        "如需写入文件，请调用 tool_write_file。\n"
        "如需创建目录，请调用 tool_create_directory。\n"
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
    """报告专家节点（执行阶段）：仅处理"发送邮件"动作。"""
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
                            "我已将本轮需求判定为'内容起草/普通对话'，不会直接发送邮件。"
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
                "messages": [AIMessage(content="我还没有拿到收件邮箱，请补充'发送到 xxx@xxx.com'。")],
                "sender": "Reporter",
            }

        if not content:
            logger.info("节点结束 | reporter_node | user_id=%s | reason=missing_content", user_id or "unknown")
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "当前没有可发送的正文。请先让我生成邮件草稿，"
                            "然后再回复'确认发送到 xxx@xxx.com'。"
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
        "当用户提出'发邮件'诉求时，先生成可审阅草稿，不要直接执行发送。\n"
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