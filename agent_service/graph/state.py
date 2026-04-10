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
    sender: str
