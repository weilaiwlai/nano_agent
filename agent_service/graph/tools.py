"""NanoAgent 工具定义模块。

定义所有 LangChain 工具和 MCP 工具调用逻辑。
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, ToolNode

from .config import (
    DEBUG_MODE,
    MCP_BASE_URL,
    MCP_SERVICE_TOKEN,
    _graph_runtime_globals,
    logger,
)
from memory import UserMemoryManager
from .utils import _mask_email_for_log, _short_text_digest, _truncate_for_log

_memory_manager: UserMemoryManager | None = None


def _get_memory_manager() -> UserMemoryManager | None:
    """懒加载并返回记忆管理器。"""
    global _memory_manager
    if _memory_manager is not None:
        return _memory_manager
    try:
        _memory_manager = UserMemoryManager()
        return _memory_manager
    except Exception as exc:  # noqa: BLE001
        logger.exception("初始化 UserMemoryManager 失败：%s", exc)
        return None


async def _call_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    enforced_user_id: str | None = None,
) -> str:
    """通过 HTTP 代理风格端点调用 MCP 服务。"""
    timeout = httpx.Timeout(connect=5.0, read=25.0, write=25.0, pool=5.0)
    headers: dict[str, str] = {}
    if MCP_SERVICE_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_SERVICE_TOKEN}"
        headers["X-Service-Token"] = MCP_SERVICE_TOKEN
    normalized_user_id = (enforced_user_id or "").strip()
    if normalized_user_id:
        headers["X-NanoAgent-User-Id"] = normalized_user_id

    request_specs: list[tuple[str, dict[str, Any]]] = [
        (f"{MCP_BASE_URL}/tools/{tool_name}", arguments),
    ]

    async with httpx.AsyncClient(timeout=timeout) as client:
        for url, payload in request_specs:
            try:
                logger.info(
                    "MCP 代理请求 | tool=%s | url=%s | payload_keys=%s",
                    tool_name,
                    url,
                    list(payload.keys()),
                )
                response = await client.post(url, json=payload, headers=headers or None)
                if response.is_success:
                    try:
                        body: Any = response.json()
                    except ValueError:
                        body = response.text
                    return json.dumps(
                        {
                            "status": "success",
                            "tool": tool_name,
                            "endpoint": url,
                            "response": body,
                        },
                        ensure_ascii=False,
                        default=str,
                    )

                logger.warning(
                    "MCP 代理响应非成功 | tool=%s | url=%s | status=%d",
                    tool_name,
                    url,
                    response.status_code,
                )
            except httpx.TimeoutException as exc:
                logger.warning(
                    "MCP 代理超时 | tool=%s | url=%s | error=%s",
                    tool_name,
                    url,
                    exc,
                )
            except httpx.RequestError as exc:
                logger.warning(
                    "MCP 代理请求错误 | tool=%s | url=%s | error=%s",
                    tool_name,
                    url,
                    exc,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "MCP 代理出现未知错误 | tool=%s | url=%s | error=%s",
                    tool_name,
                    url,
                    exc,
                )

    return json.dumps(
        {
            "status": "error",
            "tool": tool_name,
            "message": "通过 HTTP 代理端点调用 MCP 失败。",
            "tried_endpoints": [spec[0] for spec in request_specs],
        },
        ensure_ascii=False,
    )


@tool("tool_search")
async def tool_search(query: str) -> str:
    """网络搜索关键字查询信息"""
    result = await _call_mcp_tool("search", {"query": query})
    logging.info(f"搜索 | tool=search | query={query} | result={result}")
    return result


@tool("tool_get_current_time")
async def tool_get_current_time() -> str:
    """获取当前时间"""
    result = await _call_mcp_tool("get_current_time", {})
    logging.info(f"获取时间 | tool=get_current_time | result={result}")
    return result


@tool("tool_query_database")
async def tool_query_database(sql: str) -> str:
    """将数据库查询请求代理到 MCP 服务。"""
    normalized_sql = " ".join(sql.split()).strip()
    sql_digest = _short_text_digest(normalized_sql or "<empty>")
    if DEBUG_MODE:
        logger.info(
            "工具行动 | tool_query_database | sql_digest=%s | sql_len=%d | sql_preview=%s",
            sql_digest,
            len(normalized_sql),
            _truncate_for_log(normalized_sql),
        )
    else:
        logger.info(
            "工具行动 | tool_query_database | sql_digest=%s | sql_len=%d",
            sql_digest,
            len(normalized_sql),
        )
    return await _call_mcp_tool("query_database", {"sql": sql})


@tool("tool_send_report")
async def tool_send_report(email: str, content: str) -> str:
    """将报告发送请求代理到 MCP 服务。"""
    masked_email = _mask_email_for_log(email)
    if DEBUG_MODE:
        logger.info(
            "工具行动 | tool_send_report | email=%s | content_length=%d | content_preview=%s",
            masked_email,
            len(content),
            _truncate_for_log(content),
        )
    else:
        logger.info(
            "工具行动 | tool_send_report | email=%s | content_length=%d",
            masked_email,
            len(content),
        )
    return await _call_mcp_tool("send_report", {"email": email, "content": content})


@tool("tool_upsert_user_setting")
async def tool_upsert_user_setting(
    setting_key: str,
    setting_value: str,
    state: Annotated[dict[str, Any], InjectedState],
) -> str:
    """受控写入用户设置（白名单键 + 参数化 upsert）。"""
    effective_user_id = str(state.get("user_id", "")).strip()
    if not effective_user_id:
        logger.error("工具拒绝执行 | tool_upsert_user_setting | reason=missing_effective_user_id")
        return json.dumps(
            {
                "status": "error",
                "tool": "upsert_user_setting",
                "message": "工具执行缺少用户上下文，已拒绝写入。",
            },
            ensure_ascii=False,
        )

    logger.info(
        "工具行动 | tool_upsert_user_setting | user_id=%s | setting_key=%s",
        effective_user_id,
        setting_key,
    )
    return await _call_mcp_tool(
        "upsert_user_setting",
        {
            "user_id": effective_user_id,
            "setting_key": setting_key,
            "setting_value": setting_value,
        },
        enforced_user_id=effective_user_id,
    )

@tool("tool_list_allowed_directories")
async def tool_list_allowed_directories() -> str:
    """获取允许的目录列表"""
    result = await _call_mcp_tool("list_allowed_directories", {})
    logging.info(f"获取目录 | tool=list_allowed_directories | result={result}")
    return result
@tool("tool_is_path_allowed")
async def tool_is_path_allowed(path: str) -> str:
    """检查路径是否被允许"""
    result = await _call_mcp_tool("is_path_allowed", {"path": path})
    logging.info(f"检查路径 | tool=is_path_allowed | path={path} | result={result}")
    return result
@tool("tool_read_file")
async def tool_read_file(path: str) -> str:
    """读取文件内容"""
    result = await _call_mcp_tool("read_file", {"path": path})
    logging.info(f"读取文件 | tool=read_file | path={path} | result={result}")
    return result
@tool("tool_write_file")
async def tool_write_file(path: str, content: str) -> str:
    """写入文件内容"""
    result = await _call_mcp_tool("write_file", {"path": path, "content": content})
    logging.info(f"写入文件 | tool=write_file | path={path} | content={content} | result={result}")
    return result
@tool("tool_create_directory")
async def tool_create_directory(path: str) -> str:
    """创建目录"""
    result = await _call_mcp_tool("create_directory", {"path": path})
    logging.info(f"创建目录 | tool=create_directory | path={path} | result={result}")
    return result
@tool("tool_move_file")
async def tool_move_file(src: str, dst: str) -> str:
    """移动文件"""
    result = await _call_mcp_tool("move_file", {"src": src, "dst": dst})
    logging.info(f"移动文件 | tool=move_file | src={src} | dst={dst} | result={result}")
    return result
@tool("tool_edit_file")
async def tool_edit_file(path: str, edits: list) -> str:
    """编辑文件内容"""
    result = await _call_mcp_tool("edit_file", {"path": path, "edits": edits})
    logging.info(f"编辑文件 | tool=edit_file | path={path} | edits={edits} | result={result}")
    return result
tools = [tool_query_database, tool_send_report, tool_upsert_user_setting, tool_get_current_time, tool_search, tool_list_allowed_directories,
         tool_is_path_allowed, tool_read_file, tool_write_file, tool_create_directory, tool_move_file, tool_edit_file]



tools_node = ToolNode(tools)