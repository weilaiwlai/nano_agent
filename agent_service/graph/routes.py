"""NanoAgent 路由函数模块。

定义工作流中的所有条件路由函数。
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage

from typing import Literal

from langchain_core.messages import AIMessage

from .config import logger
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


def _route_after_reporter(
    state: AgentState,
) -> Literal["permission_tools_node", "__end__"]:
    """报告节点后路由。"""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("路由 | reporter_node -> permission_tools_node | tool_calls=%d", len(last_message.tool_calls))
        return "permission_tools_node"

    logger.info("路由 | reporter_node -> END | reason=no_tool_calls")
    return "__end__"


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