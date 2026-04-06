"""NanoAgent MCP 服务入口。

该模块提供：
1) FastAPI HTTP 接口（例如 /health）
2) 基于官方 MCP SSE transport 的工具挂载（/mcp）
3) SQLAlchemy 异步连接池的生命周期管理
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import smtplib
import ssl
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.message import EmailMessage
from hmac import compare_digest
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from mcp.server.fastmcp import FastMCP
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("nanoagent.mcp_server")

_db_engine: AsyncEngine | None = None
_db_write_engine: AsyncEngine | None = None

DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
MCP_REQUIRE_AUTH = os.getenv("MCP_REQUIRE_AUTH", "true").lower() == "true"
MCP_SERVICE_TOKEN = os.getenv("MCP_SERVICE_TOKEN", "").strip()
DB_WRITE_URL = os.getenv("DB_WRITE_URL", "").strip()
MAX_LOG_TEXT_LENGTH = int(os.getenv("MCP_LOG_MAX_LENGTH", "240"))
ALLOWED_SETTING_KEYS = {
    "report_language",
    "career_direction",
    "timezone",
    "notification_channel",
}


def _read_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """读取并约束整型环境变量，非法值自动回退默认值。"""
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning("环境变量 %s=%s 非法，回退默认值 %d。", name, raw, default)
        return default
    clamped = max(minimum, min(parsed, maximum))
    if clamped != parsed:
        logger.warning(
            "环境变量 %s=%d 超出范围[%d,%d]，已钳制为 %d。",
            name,
            parsed,
            minimum,
            maximum,
            clamped,
        )
    return clamped


def _read_bool_env(name: str, default: bool) -> bool:
    """读取布尔环境变量。"""
    raw = (os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _read_csv_env(name: str) -> set[str]:
    """读取逗号分隔字符串环境变量。"""
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


MCP_QUERY_ROW_LIMIT = _read_int_env("MCP_QUERY_ROW_LIMIT", 200, minimum=1, maximum=2000)
MCP_QUERY_TIMEOUT_MS = _read_int_env("MCP_QUERY_TIMEOUT_MS", 3000, minimum=100, maximum=60000)
MCP_SQL_MAX_LENGTH = _read_int_env("MCP_SQL_MAX_LENGTH", 4000, minimum=128, maximum=20000)
FORBIDDEN_SQL_FUNCTIONS = (
    "pg_sleep",
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "dblink_connect",
    "dblink_exec",
    "lo_import",
    "lo_export",
)
REPORT_PROVIDER = (os.getenv("REPORT_PROVIDER", "mock") or "mock").strip().lower()
REPORT_ALLOWED_EMAIL_DOMAINS = _read_csv_env("REPORT_ALLOWED_EMAIL_DOMAINS")
REPORT_MAX_CONTENT_CHARS = _read_int_env("REPORT_MAX_CONTENT_CHARS", 12000, minimum=100, maximum=200000)
REPORT_SUBJECT = (os.getenv("REPORT_SUBJECT", "NanoAgent 自动报告") or "NanoAgent 自动报告").strip()
REPORT_SOFT_BODY_CHARS = _read_int_env("REPORT_SOFT_BODY_CHARS", 2000, minimum=200, maximum=50000)
REPORT_SUMMARY_PREVIEW_CHARS = _read_int_env("REPORT_SUMMARY_PREVIEW_CHARS", 500, minimum=100, maximum=10000)
REPORT_ATTACH_OVERFLOW = _read_bool_env("REPORT_ATTACH_OVERFLOW", True)
REPORT_ATTACHMENT_PREFIX = (os.getenv("REPORT_ATTACHMENT_PREFIX", "nanoagent_report") or "nanoagent_report").strip()
SMTP_HOST = (os.getenv("SMTP_HOST", "") or "").strip()
SMTP_PORT = _read_int_env("SMTP_PORT", 587, minimum=1, maximum=65535)
SMTP_USERNAME = (os.getenv("SMTP_USERNAME", "") or "").strip()
SMTP_PASSWORD = (os.getenv("SMTP_PASSWORD", "") or "").strip()
SMTP_FROM = (os.getenv("SMTP_FROM", "") or "").strip()
SMTP_USE_TLS = _read_bool_env("SMTP_USE_TLS", True)
SMTP_USE_SSL = _read_bool_env("SMTP_USE_SSL", False)
SMTP_TIMEOUT_SECONDS = _read_int_env("SMTP_TIMEOUT_SECONDS", 10, minimum=1, maximum=120)

mcp = FastMCP(name="NanoAgent MCP Server")


def _json_response(payload: dict[str, Any]) -> str:
    """将工具返回结果序列化为 JSON 文本。"""
    return json.dumps(payload, ensure_ascii=False, default=str)


def _truncate_text(value: str, *, max_len: int = MAX_LOG_TEXT_LENGTH) -> str:
    """截断字符串，避免日志中出现超长敏感文本。"""
    text = value.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...(truncated)"


def _mask_email(value: str) -> str:
    """邮件地址脱敏。"""
    email = value.strip()
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    if not local:
        return f"***@{domain}"
    keep = 1 if len(local) < 4 else 2
    return f"{local[:keep]}***@{domain}"


def _is_valid_email(value: str) -> bool:
    """执行基础邮箱格式校验。"""
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()))


def _email_domain(value: str) -> str:
    """提取邮箱域名（小写）。"""
    email = value.strip().lower()
    if "@" not in email:
        return ""
    return email.split("@", 1)[1]


def _is_report_email_allowed(value: str) -> bool:
    """基于域名白名单校验是否允许发送。"""
    if not REPORT_ALLOWED_EMAIL_DOMAINS:
        return True
    return _email_domain(value) in REPORT_ALLOWED_EMAIL_DOMAINS


def _build_report_attachment_filename(timestamp: str) -> str:
    """构造稳定、可读的报告附件名。"""
    # 例：2026-03-27T18:30:25.123456+00:00 -> 20260327_183025
    compact = re.sub(r"[^0-9]", "", timestamp)
    date_part = compact[:8] if len(compact) >= 8 else datetime.now(timezone.utc).strftime("%Y%m%d")
    time_part = compact[8:14] if len(compact) >= 14 else datetime.now(timezone.utc).strftime("%H%M%S")
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]", "_", REPORT_ATTACHMENT_PREFIX) or "nanoagent_report"
    return f"{safe_prefix}_{date_part}_{time_part}.txt"


def _prepare_report_email_payload(content: str, *, timestamp: str) -> tuple[str, bytes | None, str | None, bool]:
    """对超长报告做优雅降级：正文摘要 + 附件保留全文。"""
    normalized_content = content.strip()
    if len(normalized_content) <= REPORT_SOFT_BODY_CHARS:
        return normalized_content, None, None, False

    preview = normalized_content[:REPORT_SUMMARY_PREVIEW_CHARS].strip()
    summary_notice = (
        "系统提示：本次报告内容较长，完整内容已作为附件发送；"
        "正文保留摘要以提升阅读体验。\n\n"
    )
    summary_body = (
        f"{summary_notice}"
        f"原始长度：{len(normalized_content)} 字符\n"
        f"摘要预览：\n{preview}\n\n"
        "请查看附件中的完整报告。"
    )

    attachment_filename = _build_report_attachment_filename(timestamp)
    if REPORT_ATTACH_OVERFLOW:
        attachment_bytes = normalized_content.encode("utf-8")
        return summary_body, attachment_bytes, attachment_filename, True

    # 未启用附件时退化为正文截断，仍显式告知用户。
    fallback = (
        f"{summary_notice}"
        f"原始长度：{len(normalized_content)} 字符\n"
        "当前环境未启用附件发送，以下为截断正文：\n"
        f"{normalized_content[:REPORT_SOFT_BODY_CHARS]}"
    )
    return fallback, None, None, True


def _smtp_send_report_sync(
    *,
    to_email: str,
    subject: str,
    content: str,
    attachment_bytes: bytes | None = None,
    attachment_filename: str | None = None,
) -> None:
    """同步 SMTP 发送实现（由 asyncio.to_thread 调用）。"""
    message = EmailMessage()
    message["From"] = SMTP_FROM
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(content)
    if attachment_bytes:
        filename = (attachment_filename or "report.txt").strip() or "report.txt"
        message.add_attachment(
            attachment_bytes,
            maintype="text",
            subtype="plain",
            filename=filename,
        )

    if SMTP_USE_SSL:
        with smtplib.SMTP_SSL(
            host=SMTP_HOST,
            port=SMTP_PORT,
            timeout=SMTP_TIMEOUT_SECONDS,
            context=ssl.create_default_context(),
        ) as client:
            if SMTP_USERNAME:
                client.login(SMTP_USERNAME, SMTP_PASSWORD)
            client.send_message(message)
        return

    with smtplib.SMTP(host=SMTP_HOST, port=SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS) as client:
        client.ehlo()
        if SMTP_USE_TLS:
            client.starttls(context=ssl.create_default_context())
            client.ehlo()
        if SMTP_USERNAME:
            client.login(SMTP_USERNAME, SMTP_PASSWORD)
        client.send_message(message)


def _extract_service_token(request: Request) -> str:
    """从请求头提取服务间鉴权令牌。"""
    header_token = (request.headers.get("X-Service-Token") or "").strip()
    if header_token:
        return header_token

    authorization = (request.headers.get("Authorization") or "").strip()
    if not authorization:
        return ""
    parts = authorization.split(" ", 1)
    if len(parts) != 2:
        return ""
    if parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()


def _is_authorized_service_request(request: Request) -> bool:
    """校验服务间调用令牌。"""
    token = _extract_service_token(request)
    if not token:
        return False
    return compare_digest(token, MCP_SERVICE_TOKEN)


def _service_user_id_from_header(request: Request) -> str:
    """读取服务侧透传的用户标识。"""
    return (request.headers.get("X-NanoAgent-User-Id") or "").strip()


def _resolve_effective_setting_user_id(request: Request, payload_user_id: str) -> str:
    """解析受控写工具的最终 user_id，并做服务头与请求体一致性校验。"""
    header_user_id = _service_user_id_from_header(request)
    normalized_payload_user_id = payload_user_id.strip()

    if not header_user_id:
        raise HTTPException(status_code=422, detail="upsert_user_setting 缺少服务侧用户上下文")

    if header_user_id and normalized_payload_user_id and header_user_id != normalized_payload_user_id:
        logger.warning(
            "拒绝跨租户写入 | header_user_id=%s | payload_user_id=%s",
            header_user_id,
            normalized_payload_user_id,
        )
        raise HTTPException(status_code=403, detail="用户标识不一致，拒绝写入。")

    return header_user_id


def _is_protected_path(path: str) -> bool:
    """判断当前路径是否需要服务间鉴权。"""
    normalized_path = path.rstrip("/") or "/"
    return normalized_path.startswith("/tools/") or normalized_path.startswith("/mcp")


def _normalize_sql_for_check(sql: str) -> str:
    """标准化 SQL 文本，便于白名单校验。"""
    return " ".join(sql.strip().split()).lower()


def _sql_safety_error(sql: str) -> str | None:
    """返回 SQL 安全检查错误，安全时返回 None。"""
    normalized_sql = _normalize_sql_for_check(sql)
    if not normalized_sql:
        return "sql 不能为空。"

    if len(normalized_sql) > MCP_SQL_MAX_LENGTH:
        return f"sql 过长（>{MCP_SQL_MAX_LENGTH} 字符），请缩短后重试。"

    # 禁止多语句执行（简单且有效）。
    if ";" in normalized_sql:
        return "仅允许单条只读 SELECT 查询，不支持多语句。"

    if not (normalized_sql.startswith("select ") or normalized_sql.startswith("with ")):
        return "仅允许只读 SELECT/CTE 查询。"

    forbidden_tokens = (
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " alter ",
        " create ",
        " truncate ",
        " grant ",
        " revoke ",
        " merge ",
        " call ",
        " execute ",
        " do ",
        " copy ",
    )
    padded = f" {normalized_sql} "
    for token in forbidden_tokens:
        if token in padded:
            return "仅允许只读 SELECT 查询；增删改和 DDL 已被禁用。"

    for func_name in FORBIDDEN_SQL_FUNCTIONS:
        pattern = rf"\b{re.escape(func_name)}\s*\("
        if re.search(pattern, normalized_sql):
            return f"SQL 包含高风险函数 `{func_name}`，已拒绝执行。"

    return None


def _build_limited_select_sql(sql: str, *, row_limit: int) -> str:
    """将任意只读查询包裹为外层 LIMIT，控制结果集规模。"""
    return f"SELECT * FROM ({sql}) AS nanoagent_safe_query LIMIT {row_limit}"


def _get_engine() -> AsyncEngine | None:
    """返回已初始化的异步数据库引擎。"""
    return _db_engine


def _get_write_engine() -> AsyncEngine | None:
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


def _parse_tool_output(raw_output: str) -> dict[str, Any]:
    """将工具返回的 JSON 字符串转换为字典，便于 HTTP 代理端点复用。"""
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": "工具返回了非 JSON 文本。",
            "raw_output": raw_output,
        }

    if isinstance(parsed, dict):
        return parsed

    return {
        "status": "error",
        "message": "工具返回了非对象结构。",
        "raw_output": parsed,
    }


@mcp.tool()
async def query_database(sql: str) -> str:
    """异步执行 SQL，并将结果集以 JSON 返回。"""
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


@mcp.tool()
async def send_report(email: str, content: str) -> str:
    """发送报告邮件（支持 mock/smtp 双模式），并返回执行结果。"""
    timestamp = datetime.now(timezone.utc).isoformat()
    normalized_email = email.strip()
    normalized_content = content.strip()
    masked_email = _mask_email(normalized_email)
    content_preview = _truncate_text(normalized_content)

    if not _is_valid_email(normalized_email):
        return _json_response(
            {
                "status": "error",
                "message": "邮箱格式无效，请提供合法邮箱地址。",
            }
        )

    if not normalized_content:
        return _json_response(
            {
                "status": "error",
                "message": "content 不能为空。",
            }
        )

    if len(normalized_content) > REPORT_MAX_CONTENT_CHARS:
        return _json_response(
            {
                "status": "error",
                "message": f"content 过长（>{REPORT_MAX_CONTENT_CHARS} 字符），请缩短后重试。",
            }
        )

    mail_body, attachment_bytes, attachment_filename, downgraded = _prepare_report_email_payload(
        normalized_content,
        timestamp=timestamp,
    )

    if not _is_report_email_allowed(normalized_email):
        return _json_response(
            {
                "status": "error",
                "message": "目标邮箱域名不在白名单内，已拒绝发送。",
            }
        )

    if REPORT_PROVIDER not in {"mock", "smtp"}:
        logger.warning("REPORT_PROVIDER=%s 不受支持，回退为 mock。", REPORT_PROVIDER)

    provider = REPORT_PROVIDER if REPORT_PROVIDER in {"mock", "smtp"} else "mock"

    if provider == "smtp":
        missing_fields: list[str] = []
        if not SMTP_HOST:
            missing_fields.append("SMTP_HOST")
        if not SMTP_FROM:
            missing_fields.append("SMTP_FROM")
        if SMTP_USERNAME and not SMTP_PASSWORD:
            missing_fields.append("SMTP_PASSWORD")
        if missing_fields:
            return _json_response(
                {
                    "status": "error",
                    "message": f"SMTP 配置不完整，缺少: {', '.join(missing_fields)}",
                }
            )

        try:
            await asyncio.to_thread(
                _smtp_send_report_sync,
                to_email=normalized_email,
                subject=REPORT_SUBJECT,
                content=mail_body,
                attachment_bytes=attachment_bytes,
                attachment_filename=attachment_filename,
            )
            logger.info(
                "SMTP 发送报告成功 | email=%s | timestamp=%s | content_length=%d | downgraded=%s | attachment=%s",
                masked_email,
                timestamp,
                len(normalized_content),
                downgraded,
                bool(attachment_bytes),
            )
            return _json_response(
                {
                    "status": "success",
                    "provider": "smtp",
                    "delivery": "sent",
                    "email": masked_email,
                    "timestamp": timestamp,
                    "downgraded": downgraded,
                    "attachment_filename": attachment_filename,
                    "message": (
                        "报告发送成功。正文已摘要，完整内容作为附件发送。"
                        if downgraded and attachment_bytes
                        else "报告发送成功。"
                    ),
                }
            )
        except (smtplib.SMTPException, OSError, ssl.SSLError) as exc:
            if DEBUG_MODE:
                logger.exception("SMTP 发送失败 | email=%s | error=%s", masked_email, exc)
                return _json_response(
                    {
                        "status": "error",
                        "message": "SMTP 发送失败，请检查邮件服务配置或网络连通性。",
                        "details": _truncate_text(str(exc)),
                    }
                )
            logger.error("SMTP 发送失败 | email=%s", masked_email)
            return _json_response(
                {
                    "status": "error",
                    "message": "SMTP 发送失败，请检查邮件服务配置或网络连通性。",
                }
            )
        except Exception as exc:  # noqa: BLE001
            if DEBUG_MODE:
                logger.exception("send_report 出现未知错误 | email=%s | error=%s", masked_email, exc)
                return _json_response(
                    {
                        "status": "error",
                        "message": "发送报告时发生未知错误。",
                        "details": _truncate_text(str(exc)),
                    }
                )
            logger.error("send_report 出现未知错误 | email=%s", masked_email)
            return _json_response(
                {
                    "status": "error",
                    "message": "发送报告时发生未知错误。",
                }
            )

    if DEBUG_MODE:
        logger.info(
            "模拟发送报告 | email=%s | timestamp=%s | content_length=%d | downgraded=%s | attachment=%s | content_preview=%s",
            masked_email,
            timestamp,
            len(normalized_content),
            downgraded,
            bool(attachment_bytes),
            content_preview,
        )
    else:
        logger.info(
            "模拟发送报告 | email=%s | timestamp=%s | content_length=%d | downgraded=%s | attachment=%s",
            masked_email,
            timestamp,
            len(normalized_content),
            downgraded,
            bool(attachment_bytes),
        )
    return _json_response(
        {
            "status": "success",
            "provider": "mock",
            "delivery": "simulated",
            "email": masked_email,
            "timestamp": timestamp,
            "downgraded": downgraded,
            "attachment_filename": attachment_filename,
            "message": (
                "报告发送成功（模拟）。正文已摘要，完整内容作为附件发送（模拟）。"
                if downgraded and attachment_bytes
                else "报告发送成功（模拟）。"
            ),
        }
    )


@mcp.tool()
async def upsert_user_setting(user_id: str, setting_key: str, setting_value: str) -> str:
    """受控写工具：仅允许更新白名单用户设置键。"""
    engine = _get_write_engine()
    if engine is None:
        return _json_response(
            {
                "status": "error",
                "message": "写入引擎未配置，请设置 DB_WRITE_URL。",
            }
        )

    normalized_user_id = user_id.strip()
    normalized_key = setting_key.strip().lower()
    normalized_value = setting_value.strip()

    if not normalized_user_id or not normalized_key or not normalized_value:
        return _json_response(
            {
                "status": "error",
                "message": "user_id/setting_key/setting_value 均不能为空。",
            }
        )

    if normalized_key not in ALLOWED_SETTING_KEYS:
        return _json_response(
            {
                "status": "error",
                "message": f"setting_key 不允许，允许值: {', '.join(sorted(ALLOWED_SETTING_KEYS))}",
            }
        )

    upsert_sql = text(
        """
        INSERT INTO agent_user_settings (user_id, setting_key, setting_value, updated_at)
        VALUES (:user_id, :setting_key, :setting_value, NOW())
        ON CONFLICT (user_id, setting_key)
        DO UPDATE SET
            setting_value = EXCLUDED.setting_value,
            updated_at = NOW()
        """
    )

    try:
        async with engine.begin() as conn:
            await conn.execute(
                upsert_sql,
                {
                    "user_id": normalized_user_id,
                    "setting_key": normalized_key,
                    "setting_value": normalized_value,
                },
            )

        return _json_response(
            {
                "status": "success",
                "user_id": normalized_user_id,
                "setting_key": normalized_key,
                "message": "用户设置已保存。",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except SQLAlchemyError as exc:
        if DEBUG_MODE:
            logger.exception("upsert_user_setting 触发 SQLAlchemyError")
            return _json_response(
                {
                    "status": "error",
                    "message": "写入用户设置失败，请检查数据库配置与表结构。",
                    "details": _truncate_text(str(exc)),
                }
            )
        logger.error("upsert_user_setting 执行失败（SQLAlchemyError）。")
        return _json_response(
            {
                "status": "error",
                "message": "写入用户设置失败，请检查数据库配置与表结构。",
            }
        )
    except Exception as exc:  # noqa: BLE001
        if DEBUG_MODE:
            logger.exception("upsert_user_setting 出现未知错误")
            return _json_response(
                {
                    "status": "error",
                    "message": "写入用户设置时发生未知错误。",
                    "details": _truncate_text(str(exc)),
                }
            )
        logger.error("upsert_user_setting 出现未知错误。")
        return _json_response(
            {
                "status": "error",
                "message": "写入用户设置时发生未知错误。",
            }
        )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """初始化并优雅释放共享异步资源。"""
    global _db_engine, _db_write_engine

    db_url = os.getenv("DB_URL")
    if not db_url:
        logger.warning("未配置 DB_URL，query_database 工具将不可用。")
        _db_engine = None
    else:
        _db_engine = create_async_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        logger.info("异步数据库引擎初始化完成。")

    if DB_WRITE_URL:
        if db_url and DB_WRITE_URL == db_url:
            logger.warning("DB_WRITE_URL 与 DB_URL 相同；生产环境建议拆分只读/写入账户。")
        _db_write_engine = create_async_engine(
            DB_WRITE_URL,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        try:
            await _ensure_write_schema()
            logger.info("写入数据库引擎初始化完成。")
        except Exception as exc:  # noqa: BLE001
            logger.exception("初始化受控写表失败，写工具将不可用。error=%s", exc)
            await _db_write_engine.dispose()
            _db_write_engine = None
    else:
        _db_write_engine = None
        logger.warning("未配置 DB_WRITE_URL，upsert_user_setting 工具将不可用。")

    if REPORT_PROVIDER == "smtp":
        if not SMTP_HOST or not SMTP_FROM:
            logger.warning("REPORT_PROVIDER=smtp 但 SMTP_HOST/SMTP_FROM 未完整配置，send_report 将返回错误。")
        if SMTP_USE_SSL and SMTP_USE_TLS:
            logger.warning("SMTP_USE_SSL=true 时将忽略 SMTP_USE_TLS。")
        logger.info(
            "send_report 已启用 SMTP 模式 | smtp_host=%s | smtp_port=%d | soft_body_chars=%d | hard_max_chars=%d | attach_overflow=%s",
            SMTP_HOST or "(empty)",
            SMTP_PORT,
            REPORT_SOFT_BODY_CHARS,
            REPORT_MAX_CONTENT_CHARS,
            REPORT_ATTACH_OVERFLOW,
        )
    else:
        logger.info(
            "send_report 运行在 mock 模式（默认安全模式，不实际发信）| soft_body_chars=%d | hard_max_chars=%d | attach_overflow=%s",
            REPORT_SOFT_BODY_CHARS,
            REPORT_MAX_CONTENT_CHARS,
            REPORT_ATTACH_OVERFLOW,
        )

    try:
        yield
    finally:
        if _db_engine is not None:
            await _db_engine.dispose()
            logger.info("异步数据库引擎已释放。")
            _db_engine = None
        if _db_write_engine is not None:
            await _db_write_engine.dispose()
            logger.info("写入数据库引擎已释放。")
            _db_write_engine = None


app = FastAPI(
    title="NanoAgent MCP Server",
    version="0.2.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def service_auth_middleware(request: Request, call_next: Any) -> Response:
    """保护 MCP 工具入口，限制为服务间调用。"""
    if not MCP_REQUIRE_AUTH or not _is_protected_path(request.url.path):
        return await call_next(request)

    if not MCP_SERVICE_TOKEN:
        logger.error("MCP_REQUIRE_AUTH=true 但 MCP_SERVICE_TOKEN 为空。")
        return JSONResponse(
            status_code=503,
            content={"detail": "服务鉴权未完成配置，请联系管理员。"},
        )

    if not _is_authorized_service_request(request):
        return JSONResponse(status_code=401, content={"detail": "未授权的服务调用。"})

    return await call_next(request)


async def _invoke_tool_by_name(request: Request, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """兼容 HTTP Proxy 风格调用，将请求路由到本地 MCP 工具实现。"""
    normalized_name = tool_name.strip()

    if normalized_name in {"query_database", "tool_query_database"}:
        sql = str(payload.get("sql", "")).strip()
        if not sql:
            raise HTTPException(status_code=422, detail="query_database 需要字段 sql")
        return _parse_tool_output(await query_database(sql))

    if normalized_name in {"send_report", "tool_send_report"}:
        email = str(payload.get("email", "")).strip()
        content = str(payload.get("content", "")).strip()
        if not email or not content:
            raise HTTPException(status_code=422, detail="send_report 需要字段 email 与 content")
        return _parse_tool_output(await send_report(email=email, content=content))

    if normalized_name in {"upsert_user_setting", "tool_upsert_user_setting"}:
        payload_user_id = str(payload.get("user_id", "")).strip()
        setting_key = str(payload.get("setting_key", "")).strip()
        setting_value = str(payload.get("setting_value", "")).strip()
        if not setting_key or not setting_value:
            raise HTTPException(
                status_code=422,
                detail="upsert_user_setting 需要字段 setting_key 与 setting_value",
            )
        user_id = _resolve_effective_setting_user_id(request, payload_user_id)
        return _parse_tool_output(
            await upsert_user_setting(
                user_id=user_id,
                setting_key=setting_key,
                setting_value=setting_value,
            )
        )

    raise HTTPException(status_code=404, detail=f"未知工具：{normalized_name}")


@app.post("/tools/{tool_name}")
async def invoke_tool_http_proxy(
    request: Request,
    tool_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """与 agent_service 的 HTTP 代理约定兼容的工具调用入口。"""
    return await _invoke_tool_by_name(request=request, tool_name=tool_name, payload=payload)


@app.post("/mcp/tools/{tool_name}")
async def invoke_tool_http_proxy_legacy(
    request: Request,
    tool_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """兼容旧版代理路径，避免不同调用方路径不一致。"""
    return await _invoke_tool_by_name(request=request, tool_name=tool_name, payload=payload)


@app.get("/health")
async def health() -> dict[str, str]:
    """容器编排与上游服务使用的存活检查端点。"""
    return {"status": "ok", "service": "mcp_server"}


# 将官方 MCP SSE transport 挂载到 /mcp。
# 常见端点：
# - /mcp/sse
# - /mcp/messages/
app.mount("/mcp", mcp.sse_app())


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