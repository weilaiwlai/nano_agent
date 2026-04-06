"""MCP 服务数据库连接和工具模块。"""

import asyncio
from typing import Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from config import DB_WRITE_URL, logger
from security import _sql_safety_error, _build_limited_select_sql
from utils import _json_response, _truncate_text

# 全局数据库引擎实例
_db_engine: Optional[AsyncEngine] = None
_db_write_engine: Optional[AsyncEngine] = None


def _get_engine() -> Optional[AsyncEngine]:
    """返回已初始化的异步数据库引擎。"""
    return _db_engine


def _get_write_engine() -> Optional[AsyncEngine]:
    """返回写入引擎（可选，未配置时为 None）。"""
    return _db_write_engine


async def _ensure_write_schema() -> None:
    """初始化受控写工具所需表结构。"""
    engine = _get_write_engine()
    if engine is None:
        return

    create_sql = text(
        """
        CREATE TABLE IF NOT EXISTS agent_user_settings (
            user_id VARCHAR(128) NOT NULL,
            setting_key VARCHAR(64) NOT NULL,
            setting_value TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, setting_key)
        )
        """
    )
    async with engine.begin() as conn:
        await conn.execute(create_sql)


async def query_database_tool(sql: str) -> str:
    """数据库查询工具实现。"""
    from config import MCP_QUERY_ROW_LIMIT, MCP_QUERY_TIMEOUT_MS, DEBUG_MODE
    
    engine = _get_engine()
    if engine is None:
        return _json_response(
            {
                "status": "error",
                "message": "数据库引擎尚未初始化。",
            }
        )

    normalized_sql = sql.strip()
    safety_error = _sql_safety_error(normalized_sql)
    if safety_error:
        return _json_response(
            {
                "status": "error",
                "message": safety_error,
            }
        )

    limited_sql = _build_limited_select_sql(normalized_sql, row_limit=MCP_QUERY_ROW_LIMIT)

    try:
        async with engine.begin() as conn:
            # Postgres 侧超时保险，避免复杂查询长期占用连接。
            if conn.dialect.name.startswith("postgres"):
                await conn.execute(text(f"SET LOCAL statement_timeout = {MCP_QUERY_TIMEOUT_MS}"))

            result = await conn.execute(text(limited_sql))
            rows = [dict(row) for row in result.mappings().all()]

        truncated = len(rows) >= MCP_QUERY_ROW_LIMIT
        return _json_response(
            {
                "status": "success",
                "row_count": len(rows),
                "row_limit": MCP_QUERY_ROW_LIMIT,
                "truncated": truncated,
                "rows": rows,
            }
        )
    except SQLAlchemyError as exc:
        lowered_error = str(exc).lower()
        if "statement timeout" in lowered_error or "canceling statement due to statement timeout" in lowered_error:
            return _json_response(
                {
                    "status": "error",
                    "message": f"数据库查询超时（>{MCP_QUERY_TIMEOUT_MS}ms），请缩小查询范围后重试。",
                }
            )
        if DEBUG_MODE:
            logger.exception("query_database 触发 SQLAlchemyError")
            return _json_response(
                {
                    "status": "error",
                    "message": "数据库查询失败，请检查 SQL 语法和表结构。",
                    "details": _truncate_text(str(exc)),
                }
            )
        logger.error("query_database 执行失败（SQLAlchemyError）。")
        return _json_response(
            {
                "status": "error",
                "message": "数据库查询失败，请检查 SQL 语法和表结构。",
            }
        )
    except Exception as exc:  # noqa: BLE001
        if DEBUG_MODE:
            logger.exception("query_database 出现未知错误")
            return _json_response(
                {
                    "status": "error",
                    "message": "执行数据库查询时发生未知错误。",
                    "details": _truncate_text(str(exc)),
                }
            )
        logger.error("query_database 出现未知错误。")
        return _json_response(
            {
                "status": "error",
                "message": "执行数据库查询时发生未知错误。",
            }
        )


async def upsert_user_setting_tool(user_id: str, setting_key: str, setting_value: str) -> str:
    """受控写工具实现。"""
    from config import ALLOWED_SETTING_KEYS, DEBUG_MODE
    
    engine = _get_write_engine()
    if engine is None:
        return _json_response(
            {
                "status": "error",
                "message": "写入引擎未配置，请设置 DB_WRITE_URL。",
            }
        )

    normalized_user_id = user_id.strip()
    normalized_key = setting_key.strip()
    normalized_value = setting_value.strip()

    if not normalized_user_id:
        return _json_response(
            {
                "status": "error",
                "message": "user_id 不能为空。",
            }
        )

    if not normalized_key:
        return _json_response(
            {
                "status": "error",
                "message": "setting_key 不能为空。",
            }
        )

    if normalized_key not in ALLOWED_SETTING_KEYS:
        return _json_response(
            {
                "status": "error",
                "message": f"setting_key '{normalized_key}' 不在允许的键列表中。",
            }
        )

    try:
        async with engine.begin() as conn:
            upsert_sql = text(
                """
                INSERT INTO agent_user_settings (user_id, setting_key, setting_value)
                VALUES (:user_id, :setting_key, :setting_value)
                ON CONFLICT (user_id, setting_key)
                DO UPDATE SET
                    setting_value = EXCLUDED.setting_value,
                    updated_at = NOW()
                """
            )
            await conn.execute(
                upsert_sql,
                {
                    "user_id": normalized_user_id,
                    "setting_key": normalized_key,
                    "setting_value": normalized_value,
                },
            )

        logger.info(
            "用户设置更新成功 | user_id=%s | key=%s | value_length=%d",
            normalized_user_id,
            normalized_key,
            len(normalized_value),
        )
        return _json_response(
            {
                "status": "success",
                "user_id": normalized_user_id,
                "setting_key": normalized_key,
                "message": "用户设置更新成功。",
            }
        )
    except SQLAlchemyError as exc:
        if DEBUG_MODE:
            logger.exception("upsert_user_setting 触发 SQLAlchemyError")
            return _json_response(
                {
                    "status": "error",
                    "message": "用户设置更新失败，请稍后重试。",
                    "details": _truncate_text(str(exc)),
                }
            )
        logger.error("upsert_user_setting 执行失败（SQLAlchemyError）。")
        return _json_response(
            {
                "status": "error",
                "message": "用户设置更新失败，请稍后重试。",
            }
        )
    except Exception as exc:  # noqa: BLE001
        if DEBUG_MODE:
            logger.exception("upsert_user_setting 出现未知错误")
            return _json_response(
                {
                    "status": "error",
                    "message": "更新用户设置时发生未知错误。",
                    "details": _truncate_text(str(exc)),
                }
            )
        logger.error("upsert_user_setting 出现未知错误。")
        return _json_response(
            {
                "status": "error",
                "message": "更新用户设置时发生未知错误。",
            }
        )