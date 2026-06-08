"""NanoAgent 路由函数模块。

定义工作流中的所有条件路由函数。
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage

from .config import logger
from .nodes import _parse_orchestrator_route
from .skills.loader import SkillRegistry
from .state import AgentState
from .utils import _message_to_text


# ── 技能注册表（用于 assistant 路由判断） ──────────────────────────────

_skill_registry = SkillRegistry()


def _get_skill_names() -> list[str]:
    """获取当前可用的技能名称列表。"""
    _skill_registry.refresh()
    return [s["name"] for s in _skill_registry.list_skills()]


# ── 路由函数 ──────────────────────────────────────────────────────────


def route_after_orchestrator(
    state: AgentState,
) -> Literal["data_analyst", "reporter", "assistant", "__end__"]:
    """编排器路由：根据 orchestrator 输出路由到对应 Agent。"""
    messages = state.get("messages", [])
    if not messages:
        logger.info("路由 | orchestrator -> END | reason=no_messages")
        return "__end__"

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        logger.info("路由 | orchestrator -> END | reason=last_not_ai")
        return "__end__"

    decision = _parse_orchestrator_route(_message_to_text(last_message))
    if decision == "data_analyst":
        logger.info("路由 | orchestrator -> data_analyst")
        return "data_analyst"
    if decision == "reporter":
        logger.info("路由 | orchestrator -> reporter")
        return "reporter"
    if decision == "assistant":
        logger.info("路由 | orchestrator -> assistant")
        return "assistant"

    logger.info("路由 | orchestrator -> END | decision=%s", decision)
    return "__end__"


def route_after_analyst(
    state: AgentState,
) -> Literal["high_risk_tools", "__end__"]:
    """数据分析节点后路由：有工具调用 → high_risk_tools，否则 → END。"""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("路由 | data_analyst -> high_risk_tools | tool_calls=%d", len(last_message.tool_calls))
        return "high_risk_tools"

    logger.info("路由 | data_analyst -> END | reason=no_tool_calls")
    return "__end__"


def route_after_reporter(
    state: AgentState,
) -> Literal["high_risk_tools", "__end__"]:
    """邮件报告节点后路由：有工具调用 → high_risk_tools，否则 → END。"""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("路由 | reporter -> high_risk_tools | tool_calls=%d", len(last_message.tool_calls))
        return "high_risk_tools"

    logger.info("路由 | reporter -> END | reason=no_tool_calls")
    return "__end__"


def route_after_assistant(
    state: AgentState,
) -> Literal["safe_tools", "__end__"]:
    """Assistant 节点后路由：有工具调用或技能名称 → safe_tools，否则 → END。"""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]

    # 检查是否有工具调用
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("路由 | assistant -> safe_tools | tool_calls=%d", len(last_message.tool_calls))
        return "safe_tools"

    # 检查是否是技能名称（文本内容）
    if isinstance(last_message, AIMessage):
        content = last_message.content.strip()
        skill_names = _get_skill_names()
        if content in skill_names:
            logger.info("路由 | assistant -> safe_tools | skill_name=%s", content)
            return "safe_tools"

    logger.info("路由 | assistant -> END | reason=no_tool_calls_or_skill_name")
    return "__end__"


def route_after_high_risk_tools(
    state: AgentState,
) -> Literal["data_analyst", "reporter"]:
    """高危工具执行后路由：根据 current_agent 回跳到对应 Worker。"""
    current_agent = (state.get("current_agent") or "").strip()

    if current_agent == "reporter":
        logger.info("路由 | high_risk_tools -> reporter | current_agent=%s", current_agent)
        return "reporter"

    # 默认回跳到 data_analyst
    logger.info("路由 | high_risk_tools -> data_analyst | current_agent=%s", current_agent or "unknown")
    return "data_analyst"


def route_after_safe_tools(
    state: AgentState,
) -> Literal["assistant"]:
    """安全工具执行后路由：回到 assistant。"""
    logger.info("路由 | safe_tools -> assistant")
    return "assistant"
