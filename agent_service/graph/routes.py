"""NanoAgent 路由函数模块。

定义工作流中的所有条件路由函数。
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage

from typing import Literal

from langchain_core.messages import AIMessage

from .config import logger, REPORT_INTERNAL_EMAIL_DOMAINS, SENSITIVE_REPORT_KEYWORDS
from .nodes import _normalize_supervisor_decision
from .state import AgentState
from .utils import _message_to_text


def _route_after_supervisor(
    state: AgentState,
) -> Literal["knowledge_worker_node", "reporter_node", "assistant_node", "__end__"]:
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
    if decision == "KnowledgeWorker":
        logger.info("路由 | supervisor_node -> knowledge_worker_node")
        return "knowledge_worker_node"
    if decision == "Reporter":
        logger.info("路由 | supervisor_node -> reporter_node")
        return "reporter_node"
    if decision == "Assistant":
        logger.info("路由 | supervisor_node -> assistant_node")
        return "assistant_node"

    logger.info("路由 | supervisor_node -> END")
    return "__end__"


def _route_after_knowledge_worker(
    state: AgentState,
) -> Literal["tools_node", "__end__"]:
    """数据科学家节点后路由。"""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("路由 | knowledge_worker_node -> tools_node | tool_calls=%d", len(last_message.tool_calls))
        return "tools_node"

    logger.info("路由 | knowledge_worker_node -> END | reason=no_tool_calls")
    return "__end__"


def _extract_email_from_tool_calls(messages: list) -> str:
    """从 tool_calls 中提取目标邮箱地址。"""
    import re
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        for call in msg.tool_calls:
            args = call.get("args", call.get("arguments", {}))
            if isinstance(args, dict):
                email = str(args.get("email", "")).strip()
                if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
                    return email.lower()
    return ""


def _email_domain(email: str) -> str:
    """提取邮箱域名。"""
    if "@" not in email:
        return ""
    return email.split("@", 1)[1].lower()


def _has_sensitive_content(messages: list) -> bool:
    """检查最近消息中是否包含敏感业务关键词。"""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        text = _message_to_text(msg).lower()
        return any(kw in text for kw in SENSITIVE_REPORT_KEYWORDS)
    return False


def _route_after_reporter(
    state: AgentState,
) -> Literal["permission_tools_node", "__end__"]:
    """报告节点后路由（HITL 权限分级）。

    分级策略：
    - 发送到内部域名的普通报告 → 自动放行（跳过 permission_tools_node）
    - 发送到外部域名的报告 → 需要 HITL 审批
    - 包含敏感财务关键词的报告 → 强制 HITL 审批（无论内外部）
    - 无工具调用 → 结束
    """
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.info("路由 | reporter_node -> END | reason=no_tool_calls")
        return "__end__"

    # HITL 权限分级判断
    target_email = _extract_email_from_tool_calls(messages)
    target_domain = _email_domain(target_email)
    has_sensitive = _has_sensitive_content(messages)

    # 包含敏感内容 → 强制审批
    if has_sensitive:
        logger.info(
            "路由 | reporter_node -> permission_tools_node | reason=sensitive_content | email=%s",
            target_email,
        )
        return "permission_tools_node"

    # 内部域名 + 配置了内部域名白名单 → 自动放行
    if target_domain and REPORT_INTERNAL_EMAIL_DOMAINS and target_domain in REPORT_INTERNAL_EMAIL_DOMAINS:
        logger.info(
            "路由 | reporter_node -> END | reason=internal_email_auto_approve | domain=%s",
            target_domain,
        )
        return "__end__"

    # 其他情况（外部域名或未配置内部域名白名单）→ 需要审批
    logger.info(
        "路由 | reporter_node -> permission_tools_node | reason=external_or_unknown | email=%s | domain=%s",
        target_email,
        target_domain,
    )
    return "permission_tools_node"


def _route_after_assistant(state: AgentState) -> Literal["skills_tools_node", "__end__"]:
    """Assistant 节点后路由：如果有工具调用或skill名称则进入技能工具节点，否则结束。"""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    
    # 检查是否有工具调用
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("路由 | assistant_node -> skills_tools_node | tool_calls=%d", len(last_message.tool_calls))
        return "skills_tools_node"
    
    # 检查是否是skill名称（文本内容）
    if isinstance(last_message, AIMessage):
        content = last_message.content.strip()
        # 检查内容是否是有效的skill名称
        from .skills.loader import SkillRegistry
        registry = SkillRegistry()
        skills = registry.list_skills()
        skill_names = [s["name"] for s in skills]
        
        if content in skill_names:
            logger.info("路由 | assistant_node -> skills_tools_node | skill_name=%s", content)
            return "skills_tools_node"

    logger.info("路由 | assistant_node -> END | reason=no_tool_calls_or_skill_name")
    return "__end__"


def _route_after_tools(state: AgentState) -> Literal["knowledge_worker_node", "reporter_node"]:
    """工具节点后路由：按 sender 回到对应 Worker。"""
    sender = (state.get("sender") or "").strip()
    # if sender == "Reporter":
    #     logger.info("路由 | tools_node -> reporter_node | sender=%s", sender)
    #     return "reporter_node"
    if sender == "KnowledgeWorker":
        logger.info("路由 | tools_node -> knowledge_worker_node | sender=%s", sender or "unknown")
        return "knowledge_worker_node"
    return "knowledge_worker_node"
def _route_after_permission_tools(state: AgentState) -> Literal["reporter_node"]:
    """工具节点后路由：按 sender 回到对应 Worker。"""
    sender = (state.get("sender") or "").strip()
    if sender == "Reporter":
        logger.info("路由 | permission_tools_node -> reporter_node | sender=%s", sender)
        return "reporter_node"

def _route_after_skills_tools(state: AgentState) -> Literal["assistant_node"]:
    """工具节点后路由：按 sender 回到 Assistant 节点。"""
    sender = (state.get("sender") or "").strip()
    logger.info("路由 | skills_tools_node -> assistant_node | sender=%s", sender or "unknown")
    return "assistant_node"