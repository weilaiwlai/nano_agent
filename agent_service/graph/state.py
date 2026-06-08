from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """NanoAgent 的图状态定义。"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    memory_context: str
    current_agent: str          # 当前活跃的 agent："data_analyst" | "reporter" | "assistant"
    orchestrator_context: str   # orchestrator 输出的任务描述，传递给下游 Worker
