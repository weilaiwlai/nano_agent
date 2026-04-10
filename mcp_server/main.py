"""NanoAgent MCP 服务入口。

该模块提供：
1) FastAPI HTTP 接口（例如 /health）
2) 基于官方 MCP SSE transport 的工具挂载（/mcp）
3) SQLAlchemy 异步连接池的生命周期管理
"""

from __future__ import annotations

import logging
import os
import uvicorn
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse, Response
from mcp.server.fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from config import (
    DEBUG_MODE, MCP_REQUIRE_AUTH, MCP_SERVICE_TOKEN, DB_WRITE_URL, 
    logger, ALLOWED_SETTING_KEYS
)
from database import _db_engine, _db_write_engine, _ensure_write_schema
from security import _is_authorized_service_request, _is_protected_path, _resolve_effective_setting_user_id
from tools import register_tools
from utils import _parse_tool_output
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP(name="NanoAgent MCP Server")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理。"""
    # 启动时初始化数据库连接
    global _db_engine, _db_write_engine
    db_url = os.getenv("DB_URL", "").strip()
    if db_url:
        _db_engine = create_async_engine(db_url, pool_pre_ping=True)
        logger.info("数据库引擎初始化完成")
    
    if DB_WRITE_URL:
        _db_write_engine = create_async_engine(DB_WRITE_URL, pool_pre_ping=True)
        await _ensure_write_schema()
        logger.info("写入引擎初始化完成")
    
    yield
    
    if _db_engine:
        await _db_engine.dispose()
        logger.info("数据库引擎已关闭")
    
    if _db_write_engine:
        await _db_write_engine.dispose()
        logger.info("写入引擎已关闭")


# 注册所有工具


# 创建FastAPI应用
app = FastAPI(title="NanoAgent MCP Server", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def service_auth_middleware(request: Request, call_next):
    """服务间鉴权中间件。"""
    if MCP_REQUIRE_AUTH and _is_protected_path(request.url.path):
        if not _is_authorized_service_request(request):
            return JSONResponse(
                status_code=401,
                content={"status": "error", "message": "服务间鉴权失败"},
            )
    
    response = await call_next(request)
    return response

tool_functions = register_tools(mcp)
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "mcp_server"}


@app.post("/tools/{tool_name}")
async def proxy_tool_call(tool_name: str, request: Request) -> dict[str, Any]:
    """HTTP代理端点：将工具调用转发到MCP服务。"""
    try:
        body = await request.json()
    except Exception:
        return {"status": "error", "message": "请求体必须是有效的JSON"}
    
    # 特殊处理受控写工具的用户上下文
    if tool_name == "upsert_user_setting":
        user_id = body.get("user_id", "")
        try:
            effective_user_id = _resolve_effective_setting_user_id(request, user_id)
            body["user_id"] = effective_user_id
        except HTTPException as exc:
            return {"status": "error", "message": exc.detail}
    
    # 调用MCP工具
    try:
        if tool_name == "query_database":
            result = await tool_functions["query_database"](body.get("sql", ""))
        elif tool_name == "send_report":
            result = await tool_functions["send_report"](body.get("email", ""), body.get("content", ""))
        elif tool_name == "upsert_user_setting":
            result = await tool_functions["upsert_user_setting"](
                body.get("user_id", ""),
                body.get("setting_key", ""),
                body.get("setting_value", "")
            )
        elif tool_name == "get_current_time":
            result = await tool_functions["get_current_time"]()
        elif tool_name == "search":
            result = await tool_functions["search"](body.get("query", ""))
        elif tool_name == "is_path_allowed":
            result = await tool_functions["is_path_allowed"](body.get("path", ""))
        elif tool_name == "read_file":
            result = await tool_functions["read_file"](body.get("path", ""))
        elif tool_name == "write_file":
            result = await tool_functions["write_file"](body.get("path", ""), body.get("content", ""))
        elif tool_name == "create_directory":
            result = await tool_functions["create_directory"](body.get("path", ""))
        elif tool_name == "move_file":
            result = await tool_functions["move_file"](body.get("path", ""), body.get("new_path", ""))
        elif tool_name == "edit_file":
            result = await tool_functions["edit_file"](body.get("path", ""), body.get("content", ""))
        elif tool_name == "list_allowed_directories":
            result = await tool_functions["list_allowed_directories"]()
        else:
            return {"status": "error", "message": f"未知工具: {tool_name}"}

        
        return _parse_tool_output(result)
    except Exception as exc:
        logger.exception("工具调用异常 | tool=%s | error=%s", tool_name, exc)
        return {"status": "error", "message": f"工具调用失败: {str(exc)}"}



# 挂载MCP SSE传输
app.mount("/mcp", mcp.sse_app)


if __name__ == "__main__":
    """MCP服务启动入口"""
    
    # 获取配置
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"🚀 启动MCP服务: http://{host}:{port}")
    print(f"📊 调试模式: {debug}")
    print("💡 健康检查: http://localhost:8000/health")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 MCP服务已停止")
    except Exception as e:
        print(f"❌ MCP服务启动失败: {e}")
        raise