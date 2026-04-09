"""NanoAgent 配置和环境变量模块。

提供所有配置常量、环境变量读取和日志配置。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger("nanoagent.agent_service.graph")

MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:8000").rstrip("/")
MCP_SERVICE_TOKEN = os.getenv("MCP_SERVICE_TOKEN", "").strip()
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0").strip()
GRAPH_CHECKPOINTER_BACKEND = os.getenv("GRAPH_CHECKPOINTER_BACKEND", "postgres").strip().lower()
GRAPH_CHECKPOINTER_REDIS_URL = os.getenv("GRAPH_CHECKPOINTER_REDIS_URL", REDIS_URL).strip() or REDIS_URL
GRAPH_CHECKPOINTER_POSTGRES_URL = os.getenv("GRAPH_CHECKPOINTER_POSTGRES_URL", "").strip()
GRAPH_CHECKPOINTER_PREFIX = os.getenv("GRAPH_CHECKPOINTER_PREFIX", "nanoagent:graph").strip() or "nanoagent:graph"
GRAPH_CHECKPOINTER_ALLOW_MEMORY_FALLBACK = (
    os.getenv("GRAPH_CHECKPOINTER_ALLOW_MEMORY_FALLBACK", "true").strip().lower() == "true"
)
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", os.getenv("QWEN_MODEL", "qwen3.5-plus"))
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("QWEN_API_KEY"))
DEFAULT_OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
).rstrip("/")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

if not MCP_SERVICE_TOKEN:
    logger.warning("未配置 MCP_SERVICE_TOKEN，访问受保护的 MCP 工具端点将失败。")

DEBUG_MODE = os.getenv("DEBUG", "false").strip().lower() == "true"
TOOL_LOG_MAX_TEXT_LENGTH = 120

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SQL_SNIPPET_PATTERN = re.compile(
    r"\b(select|with|from|where|join|group\s+by|order\s+by|limit|count\s*\(|information_schema|pg_database)\b",
    re.IGNORECASE,
)
DATABASE_HELP_KEYWORDS = (
    "数据库",
    "查库",
    "sql",
    "postgres",
    "postgre",
    "table",
    "表",
    "字段",
    "schema",
    "库里",
    "db",
)
DSML_CONTROL_TOKEN_PATTERN = re.compile(r"<\s*/?\s*\|\s*DSML\s*\|[^>]*>", re.IGNORECASE)
CONTROL_MARKUP_FRAGMENT_PATTERN = re.compile(
    r"<[^>\n]{0,220}(?:tool_[a-z_]+|function_calls?|invoke|parameter|dsml|\"?email\"?\s+string|\"?content\"?\s+string|string\s*=\s*\"?true\"?)[^>\n]*>",
    re.IGNORECASE,
)
CONTROL_EMAIL_ARG_PATTERN = re.compile(
    r"<\s*\"?email\"?[^>]*>[^<>\n]{0,320}(?:</\s*parameter\s*>)?",
    re.IGNORECASE,
)
CONTROL_CONTENT_ARG_OPEN_PATTERN = re.compile(r"<\s*\"?content\"?[^>]*>", re.IGNORECASE)
CONTROL_MARKUP_KEYWORDS = (
    "dsml",
    "function_calls",
    "function_call",
    "invoke",
    "parameter",
    "tool_send_report",
    "tool_query_database",
    "tool_get_current_time",
    "tool_search",
    "string=\"true\"",
)
MAX_MODEL_HISTORY_MESSAGES = int(os.getenv("MAX_MODEL_HISTORY_MESSAGES", "60"))
REPORT_CONTENT_SOFT_LIMIT = int(os.getenv("REPORT_CONTENT_SOFT_LIMIT", "8000"))
EMAIL_DRAFT_TARGET_CHARS = int(os.getenv("EMAIL_DRAFT_TARGET_CHARS", "2000"))

_graph_runtime_globals: dict[str, Any] = {
    "_llm_cache": {},
    "_non_stream_llm_cache": {},
    "_bound_llm_cache": {},
    "_checkpointer_cm": None,
    "_checkpointer_backend_in_use": "memory",
    "app_graph": None,
}
