from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from .utils import get_messages_info_from_redis, summaries_messages, store_messages_info_to_redis

from .config import REPORT_CONTENT_SOFT_LIMIT, logger,MAX_MODEL_HISTORY_MESSAGES
from .llm import _get_bound_llm, _get_chat_llm, _get_non_stream_chat_llm, _llm_profile_from_config
from .prompts import (
    ASSISTANT_PROMPT,
    REPORT_EXECUTION_GUARD_PROMPT,
    SUPERVISOR_ROUTER_PROMPT,
    KNOWLEDGE_WORKER_PROMPT,
)
from .state import AgentState
from .tools import _get_memory_manager
from .utils import _build_tool_call_message, _build_reporter_success_message, _normalize_send_report_args
from .utils import (
    _build_database_help_answer,
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
from .skills.loader import SkillRegistry
from .skills.tools import DEFAULT_TOOLS, set_active_path


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
) -> Literal["KnowledgeWorker", "Reporter", "Assistant", "FINISH"]:
    """将主管输出规范化为固定选项。"""
    logger.info("原始主管输出 | %s", raw_text)
    text = raw_text.strip().replace('"', "").replace("'", "")
    upper_text = text.upper()

    if "KNOWLEDGEWORKER" in upper_text:
        return "KnowledgeWorker"
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
    """主管节点：语义路由到 KnowledgeWorker / Reporter / Assistant / FINISH。"""
    user_id = state.get("user_id", "").strip()
    history = _sanitize_history_for_model(state.get("messages", []),config=config)
    memory_context = state.get("memory_context", "")
    messages = state.get("messages", [])
    if len(messages) > MAX_MODEL_HISTORY_MESSAGES:
        messages = messages[:-MAX_MODEL_HISTORY_MESSAGES]
        sum_messages = None
        message_info = get_messages_info_from_redis(config)
        if message_info:
            sum_messages = message_info['sum_messages']
            message_len = message_info['message_len']
            new_messages=messages[message_len-1:]
        else:
            new_messages=messages
        sum_messages = summaries_messages(new_messages,config=config,sum_messages=sum_messages)
        # 将消息长度和摘要消息存储到Redis
        message_len = len(messages)
        store_messages_info_to_redis(message_len, config, sum_messages)
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
        if decision in {"KnowledgeWorker", "Reporter", "Assistant", "FINISH"}:
            return messages[:-1]
    return messages

async def knowledge_worker_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage] | str]:
    """数据科学家节点：负责数据分析与数据库工具调用。"""
    user_id = state.get("user_id", "").strip()
    history = _sanitize_history_for_model(_trim_supervisor_decision(state.get("messages", [])),config=config)
    memory_context = state.get("memory_context", "")
    latest_query = _latest_user_query(history)
    summary=get_messages_info_from_redis(config)
    if summary:
        history=summary['sum_messages']+history
    logger.info(
        "节点开始 | knowledge_worker_node | user_id=%s | history_len=%d",
        user_id or "unknown",
        len(history),
    )

    if _has_database_intent(latest_query) and not _has_sql_snippet(latest_query):
        logger.info(
            "节点结束 | knowledge_worker_node | user_id=%s | mode=db_help_without_sql",
            user_id or "unknown",
        )
        return {"messages": [AIMessage(content=_build_database_help_answer())], "sender": "KnowledgeWorker"}

    system_prompt = f"{KNOWLEDGE_WORKER_PROMPT}\n\n长期记忆上下文：\n{memory_context or '（无）'}"
    llm_runner = _get_bound_llm(config, "knowledge_worker")
    model_input: list[BaseMessage] = [SystemMessage(content=system_prompt), *history]

    try:
        response = await llm_runner.ainvoke(model_input, config=config)
        response = _sanitize_ai_message_text(response)
        tool_call_count = len(response.tool_calls) if isinstance(response, AIMessage) else 0
        logger.info(
            "节点结束 | knowledge_worker_node | user_id=%s | tool_calls=%d",
            user_id or "unknown",
            tool_call_count,
        )
        return {"messages": [response], "sender": "KnowledgeWorker"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("节点异常 | knowledge_worker_node | user_id=%s | error=%s", user_id, exc)
        fallback = AIMessage(content="数据分析节点处理失败，请稍后重试。")
        return {"messages": [fallback], "sender": "KnowledgeWorker"}


async def reporter_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage] | str]:
    """报告专家节点（执行阶段）：仅处理"发送邮件"动作。"""
    user_id = state.get("user_id", "").strip()
    history = _sanitize_history_for_model(_trim_supervisor_decision(state.get("messages", [])),config=config)
    latest_query = _latest_user_query(history)
    has_send_report_result = _has_recent_send_report_tool_result(history)
    summary=get_messages_info_from_redis(config)
    if summary:
        history=summary['sum_messages']+history
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

registry = SkillRegistry() 
async def assistant_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage] | str]:
    """Assistant 节点：负责一般对话生成，可以使用所有skill工具。"""
    user_id = state.get("user_id", "").strip()
    history = _sanitize_history_for_model(_trim_supervisor_decision(state.get("messages", [])),config=config)
    memory_context = state.get("memory_context", "")
    
    logger.info(
        "节点开始 | assistant_node | user_id=%s | history_len=%d",
        user_id or "unknown",
        len(history),
    )

    registry.refresh()
    skills = registry.list_skills()
    if not skills:
        logger.info("当前没有可用的技能")
    else:
        logger.info("当前可用技能列表：%s", ", ".join([s["name"] for s in skills]))

    skill_list_str = "\n".join([f"- {s['name']}: {s['description']}" for s in skills])
    summary=get_messages_info_from_redis(config)
    if summary:
        history=summary['sum_messages']+history
    system_prompt = (
        f"{ASSISTANT_PROMPT}\n"
        "你是一个智能助手，拥有专业的技能团队来帮助你解决问题。\n\n"
        f"可用的专家技能团队：\n{skill_list_str}\n\n"
        "重要规则：\n"
        "1. 当用户的问题适合使用特定技能时，必须只返回要激活的技能的确切名称，不要包含任何其他文字。\n"
        "2. 例如：如果需要旅行规划技能，只返回'travel-planning'，不要返回'生成的是 travel-planning'或类似文本。\n"
        "3. 如果不需要特定技能，请直接回答用户的问题。\n\n"
        "当用户提出'发邮件'诉求时，先生成可审阅草稿，不要直接执行发送\n\n"
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

async def skills_tools_node(state: AgentState, config: RunnableConfig) -> dict[str, list[BaseMessage] | str]:
    skill_name = state["messages"][-1].content.strip() if state.get("messages") else None
    skill = registry.get_skill(skill_name)
    
    logger.info(
        "节点开始 | skills_tools_node | skill=%s",
        skill_name or "none",
    )

    system_text = "You are a helpful AI assistant."
    if skill:
        set_active_path(skill.root_path)
        
        system_text += f"\n\n=== ACTIVE SKILL: {skill.name} ===\n{skill.instructions}"
        
        ref_dir = skill.root_path / "references"
        if ref_dir.exists():
            files = [f.name for f in ref_dir.glob("*") if f.is_file() and not f.name.startswith(".")]
            if files:
                system_text += "\n\n=== AVAILABLE REFERENCES (Knowledge Base) ===\n"
                system_text += "You have access to the following files in the 'references' folder:\n"
                for f in files:
                    system_text += f"- {f}\n"
                system_text += "Use the `read_reference` tool to read their content if needed.\n"
        
        logger.debug(f"Injecting instructions for {skill.name}")
    else:
        set_active_path(None)

    try:
        llm = _get_chat_llm(config)
        model = llm.bind_tools(DEFAULT_TOOLS)
        full_messages = [SystemMessage(content=system_text)] + state["messages"]
        
        logger.debug("Invoking LLM...")
        response = await model.ainvoke(full_messages)
        
        if response.tool_calls:
            logger.info(f"🛠️ Agent requested tools: {response.tool_calls}")
        else:
            logger.info("🗣️ Agent responded with text.")
            
        logger.info(
            "节点结束 | skills_tools_node | skill=%s | tool_calls=%d",
            skill_name or "none",
            len(response.tool_calls) if response.tool_calls else 0,
        )
        
        return {"messages": [response]}
        
    except Exception as exc:  # noqa: BLE001
        logger.exception("节点异常 | skills_tools_node | skill=%s | error=%s", skill_name, exc)
        fallback = AIMessage(content="Skills工具节点处理失败，请稍后重试。")
        return {"messages": [fallback]}