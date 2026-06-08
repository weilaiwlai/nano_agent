from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from typing import Any, Literal

from .config import REPORT_CONTENT_SOFT_LIMIT, logger, MAX_MODEL_HISTORY_MESSAGES
from .llm import _get_bound_llm, _get_chat_llm, _get_non_stream_chat_llm, _llm_profile_from_config
from .prompts import (
    ANALYST_PROMPT,
    ASSISTANT_PROMPT,
    ORCHESTRATOR_PROMPT,
    REPORT_EXECUTION_GUARD_PROMPT,
    REPORT_PROMPT,
)
from .state import AgentState
from .tools import _get_memory_manager, SAFE_TOOLS
from .utils import (
    _build_database_help_answer,
    _build_reporter_success_message,
    _extract_first_email,
    _extract_report_content_from_query,
    _has_database_intent,
    _has_recent_send_report_tool_result,
    _has_sql_snippet,
    _latest_assistant_answer_before_last_user,
    _latest_user_query,
    _mask_email_for_log,
    _message_to_text,
    _normalize_send_report_args,
    _sanitize_ai_message_text,
    _sanitize_history_for_model,
    _strip_dsml_control_tokens,
    get_messages_info_from_redis,
    store_messages_info_to_redis,
    summaries_messages,
)
from .skills.loader import SkillRegistry
from .skills.tools import set_active_path


# ── 辅助函数 ──────────────────────────────────────────────────────────


def _friendly_error_message(exc: Exception) -> str:
    """将异常转换为面向用户的友好提示。"""
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


def _handle_message_summarization(messages: list, config: RunnableConfig) -> list:
    """处理消息摘要逻辑，返回处理后的消息列表。"""
    if len(messages) > MAX_MODEL_HISTORY_MESSAGES:
        messages = messages[:-MAX_MODEL_HISTORY_MESSAGES]
        message_info = get_messages_info_from_redis(config)
        sum_messages = None
        if message_info:
            sum_messages = message_info['sum_messages']
            message_len = message_info['message_len']
            new_messages = messages[message_len - 1:]
        else:
            new_messages = messages
        sum_messages = summaries_messages(new_messages, config=config, sum_messages=sum_messages)
        message_len = len(messages)
        store_messages_info_to_redis(message_len, config, sum_messages)
    return messages


def _inject_summary_if_available(history: list, config: RunnableConfig) -> list:
    """如果 Redis 中有摘要，注入到历史消息前面。"""
    summary = get_messages_info_from_redis(config)
    if summary:
        return summary['sum_messages'] + history
    return history


def _parse_orchestrator_route(raw_text: str) -> Literal["data_analyst", "reporter", "assistant", "FINISH"]:
    """解析 orchestrator 输出的 JSON，提取路由目标。"""
    logger.info("原始编排器输出 | %s", raw_text)
    text = raw_text.strip()

    # 尝试解析 JSON
    try:
        # 处理可能的 markdown 代码块包裹
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(json_lines)

        data = json.loads(text)
        route = str(data.get("route", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        # JSON 解析失败，降级为关键词匹配
        upper_text = text.upper()
        if "DATA_ANALYST" in upper_text or "DATAANALYST" in upper_text:
            return "data_analyst"
        if "REPORTER" in upper_text:
            return "reporter"
        if "ASSISTANT" in upper_text:
            return "assistant"
        if "FINISH" in upper_text:
            return "FINISH"
        return "FINISH"

    # JSON 解析成功
    route_upper = route.upper()
    if "DATA_ANALYST" in route_upper or "DATAANALYST" in route_upper:
        return "data_analyst"
    if "REPORTER" in route_upper:
        return "reporter"
    if "ASSISTANT" in route_upper:
        return "assistant"
    if "FINISH" in route_upper:
        return "FINISH"
    return "FINISH"


# ── 节点定义 ──────────────────────────────────────────────────────────


async def memory_retriever_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, str]:
    """检索用户长期记忆并写回状态。"""
    user_id = state.get("user_id", "").strip()
    messages = state.get("messages", [])
    current_query = _latest_user_query(messages)

    logger.info(
        "节点开始 | memory_retriever | user_id=%s | has_query=%s",
        user_id or "unknown",
        bool(current_query),
    )

    manager = _get_memory_manager()
    if manager is None or not user_id or not current_query:
        logger.info("节点结束 | memory_retriever | memory_context_chars=0")
        return {"memory_context": "", "current_agent": ""}

    llm_profile = _llm_profile_from_config(config)
    memory_context = manager.retrieve_context(
        user_id=user_id,
        current_query=current_query,
        k=3,
        embedding_profile=llm_profile,
    )
    logger.info(
        "节点结束 | memory_retriever | user_id=%s | memory_context_chars=%d",
        user_id,
        len(memory_context),
    )
    return {"memory_context": memory_context, "current_agent": ""}


async def orchestrator_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage] | str]:
    """编排器节点：分析用户意图，决定路由 + 传递任务上下文。"""
    user_id = state.get("user_id", "").strip()
    history = _sanitize_history_for_model(state.get("messages", []), config=config)
    memory_context = state.get("memory_context", "")
    messages = state.get("messages", [])

    # 处理消息摘要
    _handle_message_summarization(messages, config)

    logger.info(
        "节点开始 | orchestrator | user_id=%s | history_len=%d",
        user_id or "unknown",
        len(history),
    )

    orchestrator_prompt = (
        f"{ORCHESTRATOR_PROMPT}\n\n"
        f"长期记忆上下文：\n{memory_context or '（无）'}"
    )

    model_input: list[BaseMessage] = [SystemMessage(content=orchestrator_prompt), *history]

    try:
        response = await _get_chat_llm(config).ainvoke(model_input, config=config)
        raw_text = _message_to_text(response)
        route = _parse_orchestrator_route(raw_text)
        logger.info(
            "节点结束 | orchestrator | user_id=%s | route=%s",
            user_id or "unknown",
            route,
        )
        # 保存任务上下文到 state
        try:
            data = json.loads(raw_text.strip().strip("`").split("\n")[-1] if "```" in raw_text else raw_text)
            task_summary = data.get("task_summary", "")
        except (json.JSONDecodeError, AttributeError):
            task_summary = ""

        return {
            "messages": [AIMessage(content=raw_text)],
            "current_agent": route if route != "FINISH" else "",
            "orchestrator_context": task_summary,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("节点异常 | orchestrator | user_id=%s | error=%s", user_id, exc)
        return {
            "messages": [AIMessage(content=_friendly_error_message(exc))],
            "current_agent": "",
            "orchestrator_context": "",
        }


async def data_analyst_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage] | str]:
    """数据分析专家节点：负责数据库查询与经营分析。"""
    user_id = state.get("user_id", "").strip()
    history = _sanitize_history_for_model(state.get("messages", []), config=config)
    memory_context = state.get("memory_context", "")
    orchestrator_context = state.get("orchestrator_context", "")
    latest_query = _latest_user_query(history)

    # 注入摘要
    history = _inject_summary_if_available(history, config)

    logger.info(
        "节点开始 | data_analyst | user_id=%s | history_len=%d",
        user_id or "unknown",
        len(history),
    )

    # 如果用户只是问数据库相关问题但没给 SQL，给出帮助
    if _has_database_intent(latest_query) and not _has_sql_snippet(latest_query):
        logger.info(
            "节点结束 | data_analyst | user_id=%s | mode=db_help_without_sql",
            user_id or "unknown",
        )
        return {
            "messages": [AIMessage(content=_build_database_help_answer())],
            "current_agent": "data_analyst",
        }

    system_prompt = f"{ANALYST_PROMPT}\n\n长期记忆上下文：\n{memory_context or '（无）'}"
    if orchestrator_context:
        system_prompt += f"\n\n任务描述：\n{orchestrator_context}"

    llm_runner = _get_bound_llm(config, "data_analyst")
    model_input: list[BaseMessage] = [SystemMessage(content=system_prompt), *history]

    try:
        response = await llm_runner.ainvoke(model_input, config=config)
        response = _sanitize_ai_message_text(response)
        tool_call_count = len(response.tool_calls) if isinstance(response, AIMessage) else 0
        logger.info(
            "节点结束 | data_analyst | user_id=%s | tool_calls=%d",
            user_id or "unknown",
            tool_call_count,
        )
        return {"messages": [response], "current_agent": "data_analyst"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("节点异常 | data_analyst | user_id=%s | error=%s", user_id, exc)
        fallback = AIMessage(content="数据分析节点处理失败，请稍后重试。")
        return {"messages": [fallback], "current_agent": "data_analyst"}


async def reporter_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage] | str]:
    """邮件报告专家节点：专注于邮件发送流程。"""
    user_id = state.get("user_id", "").strip()
    history = _sanitize_history_for_model(state.get("messages", []), config=config)
    latest_query = _latest_user_query(history)
    has_send_report_result = _has_recent_send_report_tool_result(history)

    # 注入摘要
    history = _inject_summary_if_available(history, config)

    logger.info(
        "节点开始 | reporter | user_id=%s | history_len=%d",
        user_id or "unknown",
        len(history),
    )

    # 如果已经有发送结果，生成汇总消息
    if has_send_report_result:
        for message in reversed(history):
            if not isinstance(message, ToolMessage):
                continue
            name = str(getattr(message, "name", "")).strip().lower()
            if name not in {"tool_send_report", "send_report"}:
                continue
            summary = _build_reporter_success_message(_message_to_text(message))
            logger.info(
                "节点结束 | reporter | user_id=%s | mode=post_send_summary",
                user_id or "unknown",
            )
            return {
                "messages": [AIMessage(content=summary)],
                "current_agent": "reporter",
            }

        logger.info(
            "节点结束 | reporter | user_id=%s | mode=post_send_summary_fallback",
            user_id or "unknown",
        )
        return {
            "messages": [AIMessage(content="邮件发送流程已结束。")],
            "current_agent": "reporter",
        }

    try:
        # 判断是否为执行意图
        execute_intent = await _is_explicit_send_execution_intent(history, config)
        if not execute_intent:
            logger.info(
                "节点结束 | reporter | user_id=%s | reason=not_explicit_execute_intent",
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
                "current_agent": "reporter",
            }

        email = _extract_first_email(latest_query)
        content = _extract_report_content_from_query(latest_query)
        if not content:
            content = _latest_assistant_answer_before_last_user(history)
        content = _strip_dsml_control_tokens(content).strip()

        if not email:
            logger.info("节点结束 | reporter | user_id=%s | reason=missing_email", user_id or "unknown")
            return {
                "messages": [AIMessage(content="我还没有拿到收件邮箱，请补充'发送到 xxx@xxx.com'。")],
                "current_agent": "reporter",
            }

        if not content:
            logger.info("节点结束 | reporter | user_id=%s | reason=missing_content", user_id or "unknown")
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "当前没有可发送的正文。请先让我生成邮件草稿，"
                            "然后再回复'确认发送到 xxx@xxx.com'。"
                        )
                    )
                ],
                "current_agent": "reporter",
            }

        normalized_args = _normalize_send_report_args(
            {"email": email, "content": content[:REPORT_CONTENT_SOFT_LIMIT]},
            latest_query=latest_query,
            history=history,
        )
        if normalized_args is None:
            return {
                "messages": [AIMessage(content="邮件参数不完整，请补充邮箱和发送内容后重试。")],
                "current_agent": "reporter",
            }

        from .utils import _build_tool_call_message
        tool_call_msg = _build_tool_call_message("tool_send_report", normalized_args)
        logger.info(
            "节点结束 | reporter | user_id=%s | mode=prepare_send | email_masked=%s",
            user_id or "unknown",
            _mask_email_for_log(normalized_args["email"]),
        )
        return {"messages": [tool_call_msg], "current_agent": "reporter"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("节点异常 | reporter | user_id=%s | error=%s", user_id, exc)
        return {
            "messages": [AIMessage(content="报告执行节点处理失败，请稍后重试。")],
            "current_agent": "reporter",
        }


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


# 技能注册表（全局单例）
_skill_registry = SkillRegistry()


async def assistant_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage] | str]:
    """Assistant 节点：负责一般对话和技能调度。"""
    user_id = state.get("user_id", "").strip()
    history = _sanitize_history_for_model(state.get("messages", []), config=config)
    memory_context = state.get("memory_context", "")
    orchestrator_context = state.get("orchestrator_context", "")

    # 注入摘要
    history = _inject_summary_if_available(history, config)

    logger.info(
        "节点开始 | assistant | user_id=%s | history_len=%d",
        user_id or "unknown",
        len(history),
    )

    # 刷新技能列表
    _skill_registry.refresh()
    skills = _skill_registry.list_skills()
    skill_list_str = "\n".join([f"- {s['name']}: {s['description']}" for s in skills]) or "（暂无可用技能）"

    system_prompt = (
        f"{ASSISTANT_PROMPT}\n\n"
        f"可用的专家技能团队：\n{skill_list_str}\n\n"
        f"长期记忆上下文：\n{memory_context or '（无）'}"
    )
    if orchestrator_context:
        system_prompt += f"\n\n任务描述：\n{orchestrator_context}"

    # 绑定安全工具（含技能工具）
    from .skills.tools import run_skill_script, read_reference
    llm = _get_chat_llm(config).bind_tools(SAFE_TOOLS)
    model_input: list[BaseMessage] = [SystemMessage(content=system_prompt), *history]

    try:
        response = await llm.ainvoke(model_input, config=config)
        response = _sanitize_ai_message_text(response)

        # 如果返回的是技能名称（纯文本，无工具调用），设置活跃技能路径
        if isinstance(response, AIMessage) and not response.tool_calls:
            content = response.content.strip()
            skill = _skill_registry.get_skill(content)
            if skill:
                set_active_path(skill.root_path)
                logger.info("节点结束 | assistant | user_id=%s | skill_activated=%s", user_id or "unknown", content)
            else:
                set_active_path(None)
        else:
            set_active_path(None)

        logger.info("节点结束 | assistant | user_id=%s", user_id or "unknown")
        return {"messages": [response], "current_agent": "assistant"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("节点异常 | assistant | user_id=%s | error=%s", user_id, exc)
        fallback = AIMessage(content="助手节点处理失败，请稍后重试。")
        return {"messages": [fallback], "current_agent": "assistant"}
