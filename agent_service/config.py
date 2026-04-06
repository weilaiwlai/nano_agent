"""配置和常量模块。"""

import os
import re

from jwt import PyJWKClient
from dotenv import load_dotenv

load_dotenv()


def _parse_csv_env(value: str) -> list[str]:
    """解析逗号分隔环境变量，返回去重后的非空值列表。"""
    items = [item.strip() for item in value.split(",") if item.strip()]
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://mcp_server:8000").rstrip("/")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
AUTO_MEMORY_MAX_LEN = int(os.getenv("AUTO_MEMORY_MAX_LEN", "120"))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
LLM_SESSION_TTL_SECONDS = int(os.getenv("LLM_SESSION_TTL_SECONDS", "3600"))
LLM_SESSION_MAX_TTL_SECONDS = int(os.getenv("LLM_SESSION_MAX_TTL_SECONDS", "86400"))
REQUIRE_LLM_SESSION = os.getenv("REQUIRE_LLM_SESSION", "true").lower() == "true"
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "25"))
ALLOWED_LLM_BASE_URLS = [item.rstrip("/") for item in _parse_csv_env(os.getenv("ALLOWED_LLM_BASE_URLS", ""))]
REQUIRE_API_AUTH = os.getenv("REQUIRE_API_AUTH", "false").lower() == "true"
ALLOWED_API_KEYS = _parse_csv_env(
    os.getenv("AGENT_API_KEYS", "") or os.getenv("AGENT_API_TOKEN", "")
)
AUTH_REQUIRE_USER_SUB = os.getenv("AUTH_REQUIRE_USER_SUB", "true").lower() == "true"
AUTH_ALLOW_API_KEY_FALLBACK = os.getenv("AUTH_ALLOW_API_KEY_FALLBACK", "false").lower() == "true"
JWT_JWKS_URL = os.getenv("JWT_JWKS_URL", "").strip()
JWT_ISSUER = os.getenv("JWT_ISSUER", "").strip()
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "").strip()
JWT_HS256_SECRET = os.getenv("JWT_HS256_SECRET", "").strip()
LLM_SESSION_MASTER_KEY_RAW = os.getenv("LLM_SESSION_MASTER_KEY", "").strip()
LLM_SESSION_MASTER_KEY = (
    LLM_SESSION_MASTER_KEY_RAW
    or JWT_HS256_SECRET
    or os.getenv("AGENT_API_TOKEN", "").strip()
)
JWT_ALGORITHMS = _parse_csv_env(os.getenv("JWT_ALGORITHMS", "RS256,ES256,HS256"))
JWT_LEEWAY_SECONDS = int(os.getenv("JWT_LEEWAY_SECONDS", "30"))
CORS_ALLOW_ORIGINS = _parse_csv_env(
    os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
)
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
if "*" in CORS_ALLOW_ORIGINS:
    CORS_ALLOW_ORIGINS = ["*"]
    CORS_ALLOW_CREDENTIALS = False
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3").strip() or "text-embedding-v3"
DEFAULT_FALLBACK_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
STREAM_CONTROL_FRAGMENT_PATTERN = re.compile(
    r"<[^>\n]{0,220}(?:tool_[a-z_]+|function_calls?|invoke|parameter|dsml|\"?email\"?\s+string|\"?content\"?\s+string|string\s*=\s*\"?true\"?)[^>\n]*>",
    re.IGNORECASE,
)
STREAM_CONTROL_EMAIL_ARG_PATTERN = re.compile(
    r"<\s*\"?email\"?[^>]*>[^<>\n]{0,320}(?:</\s*parameter\s*>)?",
    re.IGNORECASE,
)
STREAM_CONTROL_CONTENT_ARG_OPEN_PATTERN = re.compile(r"<\s*\"?content\"?[^>]*>", re.IGNORECASE)
STREAM_CONTROL_KEYWORDS = (
    "dsml",
    "function_calls",
    "function_call",
    "invoke",
    "parameter",
    "tool_send_report",
    "tool_query_database",
    "string=\"true\"",
)

_jwt_jwk_client: PyJWKClient | None = PyJWKClient(JWT_JWKS_URL) if JWT_JWKS_URL else None

PROVIDER_PRESETS: dict[str, dict[str, str | bool]] = {
    "qwen": {
        "label": "Tongyi Qwen",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embedding_model": "",
        "requires_base_url": False,
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "embedding_model": "",
        "requires_base_url": False,
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "embedding_model": "",
        "requires_base_url": False,
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "embedding_model": "",
        "requires_base_url": False,
    },
    "other": {
        "label": "Other",
        "base_url": "",
        "embedding_model": "",
        "requires_base_url": True,
    },
}
SUPPORTED_PROVIDERS = tuple(PROVIDER_PRESETS.keys())
MEMORY_FACT_PATTERN = re.compile(
    r"^(我是|我是一名|我目前|我在读|我来自|我叫|我的偏好|请记住|记住|以后请|我喜欢|我不喜欢|我想从事)"
)
PRODUCTION_ENV_ALIASES = {"production", "prod"}
SENSITIVE_KEYWORDS = (
    "api_key",
    "authorization",
    "token",
    "secret",
    "password",
    "passwd",
    "cookie",
    "set-cookie",
    "content",
    "email",
)
MAX_TOOL_PAYLOAD_LENGTH = int(os.getenv("TOOL_PAYLOAD_MAX_LENGTH", "400"))
_SUPERVISOR_ROUTE_WORDS = {"CONTINUE", "END", "FINISH", "NEXT", "PROCEED", "ROUTE", "STEP"}