"""MCP 服务配置管理模块。"""

import os
import logging
from typing import Set
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("nanoagent.mcp_server.config")

# 基础配置
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
MCP_REQUIRE_AUTH = os.getenv("MCP_REQUIRE_AUTH", "true").lower() == "true"
MCP_SERVICE_TOKEN = os.getenv("MCP_SERVICE_TOKEN", "").strip()
DB_WRITE_URL = os.getenv("DB_WRITE_URL", "").strip()
MAX_LOG_TEXT_LENGTH = int(os.getenv("MCP_LOG_MAX_LENGTH", "240"))

# 允许的设置键
ALLOWED_SETTING_KEYS = {
    "report_language",
    "career_direction", 
    "timezone",
    "notification_channel",
}

# 数据库查询配置
MCP_QUERY_ROW_LIMIT = 200
MCP_QUERY_TIMEOUT_MS = 3000
MCP_SQL_MAX_LENGTH = 4000

# 邮件配置
REPORT_PROVIDER = (os.getenv("REPORT_PROVIDER", "mock") or "mock").strip().lower()
REPORT_MAX_CONTENT_CHARS = 12000
REPORT_SOFT_BODY_CHARS = 2000
REPORT_SUMMARY_PREVIEW_CHARS = 500
REPORT_ATTACH_OVERFLOW = True
REPORT_ATTACHMENT_PREFIX = (os.getenv("REPORT_ATTACHMENT_PREFIX", "nanoagent_report") or "nanoagent_report").strip()
REPORT_SUBJECT = (os.getenv("REPORT_SUBJECT", "NanoAgent 自动报告") or "NanoAgent 自动报告").strip()

# SMTP配置
SMTP_HOST = (os.getenv("SMTP_HOST", "") or "").strip()
SMTP_PORT = 587
SMTP_USERNAME = (os.getenv("SMTP_USERNAME", "") or "").strip()
SMTP_PASSWORD = (os.getenv("SMTP_PASSWORD", "") or "").strip()
SMTP_FROM = (os.getenv("SMTP_FROM", "") or "").strip()
SMTP_USE_TLS = True
SMTP_USE_SSL = False
SMTP_TIMEOUT_SECONDS = 10

# 禁止的SQL函数
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


def _read_csv_env(name: str) -> Set[str]:
    """读取逗号分隔字符串环境变量。"""
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


# 初始化配置值
REPORT_ALLOWED_EMAIL_DOMAINS = _read_csv_env("REPORT_ALLOWED_EMAIL_DOMAINS")
MCP_QUERY_ROW_LIMIT = _read_int_env("MCP_QUERY_ROW_LIMIT", 200, minimum=1, maximum=2000)
MCP_QUERY_TIMEOUT_MS = _read_int_env("MCP_QUERY_TIMEOUT_MS", 3000, minimum=100, maximum=60000)
MCP_SQL_MAX_LENGTH = _read_int_env("MCP_SQL_MAX_LENGTH", 4000, minimum=128, maximum=20000)
REPORT_MAX_CONTENT_CHARS = _read_int_env("REPORT_MAX_CONTENT_CHARS", 12000, minimum=100, maximum=200000)
REPORT_SOFT_BODY_CHARS = _read_int_env("REPORT_SOFT_BODY_CHARS", 2000, minimum=200, maximum=50000)
REPORT_SUMMARY_PREVIEW_CHARS = _read_int_env("REPORT_SUMMARY_PREVIEW_CHARS", 500, minimum=100, maximum=10000)
REPORT_ATTACH_OVERFLOW = _read_bool_env("REPORT_ATTACH_OVERFLOW", True)
SMTP_PORT = _read_int_env("SMTP_PORT", 587, minimum=1, maximum=65535)
SMTP_USE_TLS = _read_bool_env("SMTP_USE_TLS", True)
SMTP_USE_SSL = _read_bool_env("SMTP_USE_SSL", False)
SMTP_TIMEOUT_SECONDS = _read_int_env("SMTP_TIMEOUT_SECONDS", 10, minimum=1, maximum=120)