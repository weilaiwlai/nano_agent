"""NanoAgent 工作流构建模块。

构建和编译状态图工作流。
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import (
    GRAPH_CHECKPOINTER_ALLOW_MEMORY_FALLBACK,  #
    GRAPH_CHECKPOINTER_BACKEND,
    GRAPH_CHECKPOINTER_POSTGRES_URL,
    GRAPH_CHECKPOINTER_PREFIX,
    GRAPH_CHECKPOINTER_REDIS_URL,
    _graph_runtime_globals,
    logger,
)
from .nodes import (
    assistant_node,
    data_scientist_node,
    reporter_node,
    retrieve_memory_node,
    supervisor_node,
)
from .routes import (
    _route_after_assistant,
    _route_after_data_scientist,
    _route_after_reporter,
    _route_after_supervisor,
    _route_after_tools,
)
from .state import AgentState
from .tools import tools_node

app_graph = _graph_runtime_globals["app_graph"]
_checkpointer_cm = _graph_runtime_globals["_checkpointer_cm"]
_checkpointer_backend_in_use = _graph_runtime_globals["_checkpointer_backend_in_use"]


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


def _build_workflow() -> StateGraph:
    """创建并返回状态图构建器。"""
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
    return workflow


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

    app_graph = _build_workflow().compile(checkpointer=checkpointer)
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


def save_graph_visualization(graph: Any, filename: str = "graph.png") -> None:
    """保存状态图的可视化表示。

    Args:
        graph: 状态图实例。
        filename: 保存文件路径。
    """
    try:
        with open(filename, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        logger.info(f"Graph visualization saved as {filename}")
    except IOError as e:
        logger.warning(f"Failed to save graph visualization: {e}")


async def visualize_graph() -> None:
    """可视化并保存工作流图。"""
    graph = await init_graph_runtime()
    save_graph_visualization(graph, "nano_agent_graph.png")


if __name__ == "__main__":
    asyncio.run(visualize_graph())