"""NanoAgent 主服务的 FastAPI 入口。

提供：
- 服务健康检查与 MCP 上游检查
- 基于 LangGraph 的流式聊天接口（SSE）
- 人机协同审批（HITL）恢复接口
- 长期记忆写入、查看、删除接口
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from hmac import compare_digest
import ipaddress
import json
import logging
import os
import re
from typing import Any, AsyncIterator, Literal, TypedDict
from urllib.parse import urlparse

import httpx
import jwt
from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from jwt import PyJWKClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from starlette.requests import Request

import graph as graph_runtime
from memory import MemoryProviderError, UserMemoryManager
from session_store import LLMSessionStore

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger("nanoagent.agent_service.main")


def _parse_csv_env(value: str) -> list[str]:
    """解析逗号分隔环境变量，返回去重后的非空值列表。"""
    items = [item.strip() for item in value.split(",") if item.strip()]
    # 保持原顺序去重，避免重复配置导致检查开销
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
    # 生产环境应明确列白名单；保留 "*" 仅用于快速联调。
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
        # DeepSeek 当前公开网关不提供与 text-embedding-v3 兼容的 embedding 端点，
        # 默认关闭 embedding（可由用户手动填写其他可用 embedding 模型）。
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

_memory_manager: UserMemoryManager | None = None
_session_store = LLMSessionStore(
    REDIS_URL,
    master_key=LLM_SESSION_MASTER_KEY,
    default_ttl_seconds=LLM_SESSION_TTL_SECONDS,
    max_ttl_seconds=LLM_SESSION_MAX_TTL_SECONDS,
)
_session_store_ready = False


class AuthContext(TypedDict):
    """请求认证上下文。"""

    auth_type: Literal["jwt", "api_key", "disabled"]
    subject: str | None


class ChatRequest(BaseModel):
    """聊天接口请求体。"""

    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    session_id: str | None = Field(default=None, min_length=8)


class ChatResumeRequest(BaseModel):
    """聊天续跑（审批）请求体。"""

    user_id: str = Field(..., min_length=1)
    action: Literal["approve", "reject"]
    session_id: str | None = Field(default=None, min_length=8)


class MemoryRequest(BaseModel):
    """长期记忆写入接口请求体。"""

    user_id: str = Field(..., min_length=1)
    preference_text: str = Field(..., min_length=1)
    session_id: str | None = Field(default=None, min_length=8)


class LLMSessionCreateRequest(BaseModel):
    """创建 LLM 会话请求体（BYOK）。"""

    provider: Literal["qwen", "openai", "deepseek", "groq", "other"] | None = None
    api_key: str = Field(..., min_length=10)
    model: str = Field(..., min_length=1)
    base_url: str | None = None
    embedding_model: str | None = None
    ttl_seconds: int | None = Field(default=None, ge=60, le=86400)


class LLMSessionCreateResponse(BaseModel):
    """创建 LLM 会话响应体。"""

    session_id: str
    provider: str
    expires_in: int
    model: str
    base_url: str
    embedding_model: str


class LLMSessionDeleteResponse(BaseModel):
    """删除 LLM 会话响应体。"""

    session_id: str
    status: str
    message: str


class LLMSessionValidateRequest(BaseModel):
    """校验 LLM 连接配置请求体。"""

    provider: Literal["qwen", "openai", "deepseek", "groq", "other"] | None = None
    api_key: str = Field(..., min_length=10)
    model: str = Field(..., min_length=1)
    base_url: str | None = None
    embedding_model: str | None = None
    validate_chat: bool = True
    validate_embedding: bool = True


class LLMSessionValidateResponse(BaseModel):
    """LLM 配置校验结果。"""

    provider: str
    model: str
    base_url: str
    embedding_model: str
    chat_ok: bool
    embedding_ok: bool
    errors: list[str]


class LLMProviderItem(BaseModel):
    """单个可选 Provider 配置。"""

    provider: str
    label: str
    requires_base_url: bool
    default_base_url: str | None = None
    default_embedding_model: str


class LLMProviderListResponse(BaseModel):
    """可选 Provider 列表。"""

    items: list[LLMProviderItem]


class MemoryResponse(BaseModel):
    """长期记忆写入接口响应体。"""

    user_id: str
    status: str
    message: str
    memory_id: str | None = None


class MemoryItem(BaseModel):
    """单条长期记忆记录。"""

    memory_id: str
    preference_text: str
    timestamp: str


class MemoryListResponse(BaseModel):
    """长期记忆列表响应体。"""

    user_id: str
    items: list[MemoryItem]


class MemoryDeleteResponse(BaseModel):
    """长期记忆删除响应体。"""

    user_id: str
    memory_id: str
    status: str
    message: str


def _get_memory_manager() -> UserMemoryManager:
    """懒加载并返回记忆管理器单例。"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = UserMemoryManager()
    return _memory_manager


def _should_auto_save_memory(query: str) -> bool:
    """判断用户输入是否应自动写入长期记忆。"""
    normalized_query = query.strip()
    if not normalized_query:
        return False
    if len(normalized_query) < 4 or len(normalized_query) > AUTO_MEMORY_MAX_LEN:
        return False
    return bool(MEMORY_FACT_PATTERN.search(normalized_query))


def _is_production_environment() -> bool:
    """当前是否处于生产环境。"""
    return ENVIRONMENT in PRODUCTION_ENV_ALIASES


def _is_disallowed_local_base_url(base_url: str) -> bool:
    """生产环境下禁止本地/内网地址作为模型网关。"""
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return True

    blocked_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}
    if host in blocked_hosts or host.endswith(".local"):
        return True

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # 域名不在本地保留名单内时放行，是否允许由白名单继续控制。
        return False

    return any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_reserved,
            ip.is_multicast,
            ip.is_unspecified,
        )
    )


def _normalize_base_url(base_url: str | None, *, fallback_to_env: bool = True) -> str:
    """标准化并校验模型服务 Base URL。"""
    default_url = os.getenv("OPENAI_BASE_URL", os.getenv("QWEN_BASE_URL", DEFAULT_FALLBACK_BASE_URL))
    raw = (base_url or default_url if fallback_to_env else base_url or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="base_url 不能为空")

    normalized = raw.rstrip("/")

    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="base_url 非法，请传入有效的 http(s) URL")

    if _is_production_environment() and _is_disallowed_local_base_url(normalized):
        raise HTTPException(status_code=400, detail="生产环境禁止使用本地或内网 base_url")

    if ALLOWED_LLM_BASE_URLS and all(not normalized.startswith(prefix) for prefix in ALLOWED_LLM_BASE_URLS):
        raise HTTPException(status_code=400, detail="base_url 不在允许列表内")

    return normalized


def _infer_provider_from_base_url(base_url: str) -> str:
    """根据 base_url 推断 provider。"""
    host = (urlparse(base_url).netloc or "").lower()
    if "dashscope.aliyuncs.com" in host:
        return "qwen"
    if "api.openai.com" in host:
        return "openai"
    if "api.deepseek.com" in host:
        return "deepseek"
    if "api.groq.com" in host:
        return "groq"
    return "other"


def _normalize_provider(provider: str | None, base_url: str | None) -> str:
    """标准化 provider，缺失时按 base_url 自动推断。"""
    normalized_provider = (provider or "").strip().lower()
    if normalized_provider:
        if normalized_provider not in SUPPORTED_PROVIDERS:
            raise HTTPException(status_code=400, detail=f"provider 不支持，允许值: {', '.join(SUPPORTED_PROVIDERS)}")
        return normalized_provider

    if base_url and base_url.strip():
        return _infer_provider_from_base_url(_normalize_base_url(base_url, fallback_to_env=False))

    env_base_url = _normalize_base_url(
        os.getenv("OPENAI_BASE_URL", os.getenv("QWEN_BASE_URL", DEFAULT_FALLBACK_BASE_URL)),
        fallback_to_env=False,
    )
    return _infer_provider_from_base_url(env_base_url)


def _resolve_base_url_for_provider(provider: str, requested_base_url: str | None) -> str:
    """解析 provider 下最终使用的 base_url。"""
    preset = PROVIDER_PRESETS.get(provider, {})
    preset_base_url = str(preset.get("base_url", "")).strip()

    if provider == "other":
        if not requested_base_url or not requested_base_url.strip():
            raise HTTPException(status_code=400, detail="provider=other 时必须提供 base_url")
        return _normalize_base_url(requested_base_url, fallback_to_env=False)

    normalized_preset = _normalize_base_url(preset_base_url, fallback_to_env=False)
    if requested_base_url and requested_base_url.strip():
        normalized_requested = _normalize_base_url(requested_base_url, fallback_to_env=False)
        if normalized_requested != normalized_preset:
            raise HTTPException(
                status_code=400,
                detail=f"provider={provider} 使用平台预设 base_url，不支持自定义 URL",
            )

    return normalized_preset


def _resolve_embedding_model_for_provider(provider: str, requested_embedding_model: str | None) -> str:
    """解析 provider 下最终使用的 embedding 模型。"""
    # 明确传了 embedding_model（即使是空字符串）时，以用户输入为准。
    if requested_embedding_model is not None:
        return requested_embedding_model.strip()

    preset = PROVIDER_PRESETS.get(provider, {})
    # 若 preset 显式给出空字符串，表示该 provider 默认禁用 embedding。
    if "embedding_model" in preset:
        return str(preset.get("embedding_model", "")).strip()

    preset_embedding_model = str(preset.get("embedding_model", "")).strip()
    if preset_embedding_model:
        return preset_embedding_model

    return DEFAULT_EMBEDDING_MODEL


def _default_llm_profile() -> dict[str, str] | None:
    """读取默认环境变量模型配置。"""
    api_key = os.getenv("OPENAI_API_KEY", os.getenv("QWEN_API_KEY", "")).strip()
    if not api_key:
        return None

    env_base_url = _normalize_base_url(
        os.getenv("OPENAI_BASE_URL", os.getenv("QWEN_BASE_URL", DEFAULT_FALLBACK_BASE_URL))
    )
    provider = _normalize_provider(None, env_base_url)
    embedding_model = (os.getenv("EMBEDDING_MODEL", "").strip() or _resolve_embedding_model_for_provider(provider, None))

    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": env_base_url,
        "model": os.getenv("OPENAI_MODEL", os.getenv("QWEN_MODEL", "qwen3.5-plus")).strip(),
        "embedding_model": embedding_model,
    }


def _build_llm_profile(request: LLMSessionCreateRequest | LLMSessionValidateRequest) -> dict[str, str]:
    """从请求构建标准化 LLM profile。"""
    normalized_api_key = request.api_key.strip()
    if not normalized_api_key:
        raise HTTPException(status_code=400, detail="api_key 不能为空")

    normalized_model = request.model.strip()
    if not normalized_model:
        raise HTTPException(status_code=400, detail="model 不能为空")

    provider = _normalize_provider(request.provider, request.base_url)
    normalized_base_url = _resolve_base_url_for_provider(provider, request.base_url)
    embedding_model = _resolve_embedding_model_for_provider(provider, request.embedding_model)

    return {
        "provider": provider,
        "api_key": normalized_api_key,
        "base_url": normalized_base_url,
        "model": normalized_model,
        "embedding_model": embedding_model,
    }


def _safe_error_text(exc: Exception, *, max_len: int = 220) -> str:
    """生成适合给前端展示的短错误信息。"""
    text = str(exc).replace("\n", " ").strip()
    if not text:
        text = exc.__class__.__name__
    if len(text) > max_len:
        return text[:max_len] + "...(truncated)"
    return text


def _truncate_text(value: str, *, max_len: int = MAX_TOOL_PAYLOAD_LENGTH) -> str:
    """按最大长度截断字符串。"""
    text = value.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...(truncated)"


def _mask_email(value: str) -> str:
    """对邮箱做部分脱敏。"""
    email = value.strip()
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    if not local:
        return f"***@{domain}"
    keep = 1 if len(local) < 4 else 2
    return f"{local[:keep]}***@{domain}"


def _looks_sensitive_field(field_name: str) -> bool:
    """判断字段名是否可能包含敏感信息。"""
    normalized = field_name.strip().lower()
    return any(keyword in normalized for keyword in SENSITIVE_KEYWORDS)


def _redact_value_by_key(key: str, value: Any) -> Any:
    """按字段名脱敏值。"""
    if not _looks_sensitive_field(key):
        return value
    if isinstance(value, str) and "@" in value:
        return _mask_email(value)
    return "[REDACTED]"


def _safe_tool_payload(payload: Any, max_length: int = MAX_TOOL_PAYLOAD_LENGTH) -> Any:
    """工具输入输出默认脱敏并截断，避免泄露敏感数据。"""
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for raw_key, raw_value in payload.items():
            key = str(raw_key)
            value = _redact_value_by_key(key, raw_value)
            if value == "[REDACTED]":
                sanitized[key] = value
            else:
                sanitized[key] = _safe_tool_payload(value, max_length=max_length)
        return sanitized

    if isinstance(payload, list):
        return [_safe_tool_payload(item, max_length=max_length) for item in payload[:20]]

    if isinstance(payload, str):
        return _truncate_text(payload, max_len=max_length)

    return payload


def _tool_payload_summary(payload: Any) -> dict[str, Any]:
    """生产默认仅返回工具事件摘要，不返回细节入参/出参。"""
    if isinstance(payload, dict):
        return {"kind": "object", "keys": sorted(str(key) for key in payload.keys())[:20]}
    if isinstance(payload, list):
        return {"kind": "array", "size": len(payload)}
    if isinstance(payload, str):
        return {"kind": "text", "chars": len(payload)}
    return {"kind": type(payload).__name__}


def _sanitize_tool_call_for_client(call: dict[str, Any]) -> dict[str, Any]:
    """SSE 对前端下发的 tool_call 安全视图。"""
    sanitized = {
        "id": str(call.get("id", "")).strip(),
        "name": str(call.get("name", "")).strip(),
        "type": str(call.get("type", "tool_call")).strip() or "tool_call",
    }
    args = call.get("args")
    if DEBUG_MODE:
        sanitized["args"] = _safe_tool_payload(args)
    else:
        sanitized["args"] = _tool_payload_summary(args)
        sanitized["redacted"] = True
    return sanitized


def _sanitize_tool_calls_for_client(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """批量脱敏 tool_calls。"""
    return [_sanitize_tool_call_for_client(call) for call in tool_calls if isinstance(call, dict)]


def _build_validation_chat_client(llm_profile: dict[str, str]) -> ChatOpenAI:
    """为连接校验构建轻量 Chat 模型客户端。"""
    common_kwargs: dict[str, Any] = {
        "model": llm_profile["model"],
        "temperature": 0,
        "streaming": False,
        "timeout": 20,
        "max_retries": 0,
        "max_tokens": 8,
    }
    try:
        return ChatOpenAI(
            api_key=llm_profile["api_key"],
            base_url=llm_profile["base_url"],
            **common_kwargs,
        )
    except TypeError:
        return ChatOpenAI(
            openai_api_key=llm_profile["api_key"],
            openai_api_base=llm_profile["base_url"],
            **common_kwargs,
        )


def _build_validation_embeddings_client(llm_profile: dict[str, str]) -> OpenAIEmbeddings:
    """为连接校验构建 Embedding 客户端。"""
    try:
        return OpenAIEmbeddings(
            model=llm_profile["embedding_model"],
            api_key=llm_profile["api_key"],
            base_url=llm_profile["base_url"],
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )
    except TypeError:
        return OpenAIEmbeddings(
            model=llm_profile["embedding_model"],
            openai_api_key=llm_profile["api_key"],
            openai_api_base=llm_profile["base_url"],
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )


async def _validate_llm_profile_connectivity(
    llm_profile: dict[str, str],
    *,
    validate_chat: bool,
    validate_embedding: bool,
) -> tuple[bool, bool, list[str]]:
    """对配置执行轻量连通性验证。"""
    chat_ok = True
    embedding_ok = True
    errors: list[str] = []

    if validate_chat:
        try:
            chat_client = _build_validation_chat_client(llm_profile)
            await chat_client.ainvoke([HumanMessage(content="ping")])
        except Exception as exc:  # noqa: BLE001
            chat_ok = False
            errors.append(f"chat 校验失败: {_safe_error_text(exc)}")

    if validate_embedding:
        if not llm_profile.get("embedding_model", "").strip():
            embedding_ok = True
            errors.append("embedding 校验已跳过：当前会话未配置 embedding 模型。")
        else:
            try:
                embedding_client = _build_validation_embeddings_client(llm_profile)
                await asyncio.to_thread(embedding_client.embed_query, "ping")
            except Exception as exc:  # noqa: BLE001
                embedding_ok = False
                errors.append(f"embedding 校验失败: {_safe_error_text(exc)}")

    return chat_ok, embedding_ok, errors


async def _resolve_llm_profile(session_id: str | None, *, owner_id: str) -> dict[str, str]:
    """解析会话级 LLM profile（优先 session，其次默认环境）。"""
    normalized_session_id = (session_id or "").strip()
    if normalized_session_id:
        if not _session_store_ready:
            raise HTTPException(status_code=503, detail="LLM 会话存储未就绪，请稍后重试")
        profile = await _session_store.get_session(normalized_session_id, owner_id=owner_id)
        if profile is None:
            raise HTTPException(status_code=401, detail="LLM 会话不存在、已过期或不属于当前用户")
        if "provider" not in profile:
            profile["provider"] = _infer_provider_from_base_url(profile["base_url"])
        return profile

    if REQUIRE_LLM_SESSION:
        raise HTTPException(status_code=401, detail="请先创建 LLM 会话并携带 session_id")

    default_profile = _default_llm_profile()
    if default_profile is not None:
        return default_profile

    raise HTTPException(status_code=401, detail="未配置默认模型，请先创建 LLM 会话并携带 session_id")


def _try_auto_save_memory(
    user_id: str,
    query: str,
    llm_profile: dict[str, str],
) -> str | None:
    """尝试将用户自述背景自动写入长期记忆。"""
    if not _should_auto_save_memory(query):
        return None
    if not llm_profile.get("embedding_model", "").strip():
        logger.info("自动记忆写入跳过 | user_id=%s | reason=embedding_disabled", user_id)
        return None

    try:
        manager = _get_memory_manager()
        memory_text = f"用户自述背景：{query.strip()}"
        memory_id = manager.save_preference(
            user_id=user_id,
            preference_text=memory_text,
            embedding_profile=llm_profile,
        )
        if memory_id:
            logger.info("自动记忆写入成功 | user_id=%s | memory_id=%s", user_id, memory_id)
        return memory_id
    except MemoryProviderError as exc:
        logger.warning(
            "自动记忆写入失败（模型侧） | user_id=%s | reason=%s | error=%s",
            user_id,
            exc.reason,
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("自动记忆写入失败（不中断主流程） | user_id=%s | error=%s", user_id, exc)
        return None


def _to_sse(payload: dict[str, Any]) -> str:
    """将字典消息包装为 SSE data 帧。"""
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _extract_token_content(event: dict[str, Any]) -> str:
    """从 on_chat_model_stream 事件中提取 token 文本。"""
    data = event.get("data", {}) if isinstance(event, dict) else {}
    chunk = data.get("chunk")
    if chunk is None:
        return ""

    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts)

    if isinstance(chunk, dict):
        dict_content = chunk.get("content")
        if isinstance(dict_content, str):
            return dict_content

    return ""


def _sanitize_stream_token_text(token_text: str) -> str:
    """清理流式 token 中的控制标记，避免污染前端展示。"""
    if not token_text:
        return token_text
    cleaned = STREAM_CONTROL_EMAIL_ARG_PATTERN.sub("", token_text)
    cleaned = STREAM_CONTROL_CONTENT_ARG_OPEN_PATTERN.sub("", cleaned)
    cleaned = STREAM_CONTROL_FRAGMENT_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"</?\s*>", "", cleaned)
    lowered = cleaned.lower()
    if any(keyword in lowered for keyword in STREAM_CONTROL_KEYWORDS):
        if ("<" in cleaned) or (">" in cleaned) or ("string=" in lowered):
            return ""
    return cleaned


def _extract_bearer_token(authorization: str) -> str:
    """从 Authorization 头提取 Bearer Token。"""
    auth_value = authorization.strip()
    if not auth_value:
        return ""

    parts = auth_value.split(" ", 1)
    if len(parts) != 2:
        return ""
    if parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()


def _looks_like_jwt(token: str) -> bool:
    """判断字符串是否形如 JWT（header.payload.signature）。"""
    return token.count(".") == 2


def _is_allowed_api_key(candidate: str) -> bool:
    """常量时序比较，避免简单字符串比较泄漏时间特征。"""
    if not candidate:
        return False
    return any(compare_digest(candidate, allowed_key) for allowed_key in ALLOWED_API_KEYS)


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    """校验并解码 JWT，返回 claims。"""
    decode_options: dict[str, Any] = {
        "require": ["sub"],
        "verify_aud": bool(JWT_AUDIENCE),
        "verify_iss": bool(JWT_ISSUER),
    }
    decode_kwargs: dict[str, Any] = {
        "algorithms": JWT_ALGORITHMS or ["RS256"],
        "options": decode_options,
        "leeway": JWT_LEEWAY_SECONDS,
    }
    if JWT_AUDIENCE:
        decode_kwargs["audience"] = JWT_AUDIENCE
    if JWT_ISSUER:
        decode_kwargs["issuer"] = JWT_ISSUER

    try:
        if _jwt_jwk_client is not None:
            signing_key = _jwt_jwk_client.get_signing_key_from_jwt(token)
            key_material = signing_key.key
            return jwt.decode(token, key=key_material, **decode_kwargs)

        if JWT_HS256_SECRET:
            return jwt.decode(token, key=JWT_HS256_SECRET, **decode_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="JWT 校验失败，请重新登录") from exc

    raise HTTPException(
        status_code=503,
        detail="JWT 校验未完成配置（缺少 JWT_JWKS_URL 或 JWT_HS256_SECRET）",
    )


def _subject_from_claims(claims: dict[str, Any]) -> str:
    """从 JWT claims 提取 subject。"""
    subject = str(claims.get("sub", "")).strip()
    if not subject:
        raise HTTPException(status_code=401, detail="JWT 缺少 sub 字段")
    return subject


def _resolve_effective_user_id(*, token_subject: str, client_user_id: str, source: str) -> str:
    """统一以 token_subject 作为真实用户 ID，忽略客户端传入 user_id。"""
    normalized_client_user_id = client_user_id.strip()
    if normalized_client_user_id and normalized_client_user_id != token_subject:
        if DEBUG_MODE:
            logger.warning(
                "检测到 user_id 与 JWT sub 不一致，已忽略客户端 user_id | source=%s | sub=%s | client_user_id=%s",
                source,
                token_subject,
                normalized_client_user_id,
            )
        else:
            logger.warning(
                "检测到 user_id 与 JWT sub 不一致，已忽略客户端 user_id | source=%s | sub=%s",
                source,
                token_subject,
            )
    return token_subject


async def _require_api_auth_context(request: Request) -> AuthContext:
    """统一接口鉴权（JWT 优先，兼容 API Key）。"""
    if not REQUIRE_API_AUTH:
        return {"auth_type": "disabled", "subject": None}

    bearer_token = _extract_bearer_token(request.headers.get("Authorization", ""))
    api_key_header = (request.headers.get("X-API-Key") or "").strip()

    if bearer_token and _looks_like_jwt(bearer_token):
        claims = _decode_jwt_claims(bearer_token)
        return {"auth_type": "jwt", "subject": _subject_from_claims(claims)}

    if not ALLOWED_API_KEYS:
        logger.error("鉴权已启用，但未配置 AGENT_API_KEYS")
        raise HTTPException(status_code=503, detail="服务端鉴权未完成配置，请联系管理员")

    candidate = bearer_token or api_key_header

    if _is_allowed_api_key(candidate):
        if AUTH_REQUIRE_USER_SUB and not AUTH_ALLOW_API_KEY_FALLBACK:
            raise HTTPException(status_code=401, detail="当前接口仅接受带 sub 的 JWT 令牌")
        fallback_subject = (request.headers.get("X-User-Sub") or "").strip() if AUTH_ALLOW_API_KEY_FALLBACK else ""
        return {"auth_type": "api_key", "subject": fallback_subject or None}

    # 打印认证失败详情
    # logger.warning("认证失败 - 候选Token: '%s', 允许的Token: %s", candidate, ALLOWED_API_KEYS)
    raise HTTPException(status_code=401, detail="未授权：请提供有效的 API Token")


async def _require_user_context(
    auth_context: AuthContext = Depends(_require_api_auth_context),
) -> AuthContext:
    """要求调用方必须携带可识别用户身份（JWT sub）。"""
    if auth_context.get("subject"):
        return auth_context
    if AUTH_REQUIRE_USER_SUB:
        raise HTTPException(status_code=401, detail="当前操作需要带 sub 的 JWT 身份令牌")
    return auth_context


async def _require_api_auth(request: Request) -> None:
    """兼容旧依赖签名：仅执行鉴权，不关心用户身份。"""
    _ = await _require_api_auth_context(request)


def _require_subject(auth_context: AuthContext) -> str:
    """从认证上下文取出 subject。"""
    subject = str(auth_context.get("subject") or "").strip()
    
    # 如果不要求JWT sub且subject为空，返回默认subject
    if not subject and not AUTH_REQUIRE_USER_SUB:
        logger.info("不要求JWT sub，使用默认subject")
        return "default_user"
    
    if not subject:
        logger.error("未从认证上下文获取 subject")
        raise HTTPException(status_code=401, detail="未授权：请提供有效的 API Token")
    
    return subject


def _streaming_headers() -> dict[str, str]:
    """统一返回 SSE 头，避免代理缓冲。"""
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


def _graph_config(user_id: str, llm_profile: dict[str, str]) -> dict[str, Any]:
    """构造 LangGraph 配置（线程 + 会话级模型配置）。"""
    return {
        "configurable": {"thread_id": user_id},
        "metadata": {"llm_profile": llm_profile},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }


def _is_waiting_for_tools_node(state: Any) -> bool:
    """判断图是否被中断在 tools_node 前。"""
    next_nodes = getattr(state, "next", ()) or ()

    if isinstance(next_nodes, str):
        return next_nodes == "tools_node"

    try:
        return "tools_node" in set(next_nodes)
    except TypeError:
        return False


def _extract_pending_tool_calls(state: Any) -> list[dict[str, Any]]:
    """从中断状态中提取待审批 tool_calls。"""
    values = getattr(state, "values", {}) or {}
    if not isinstance(values, dict):
        return []

    messages = values.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return []

    last_message = messages[-1]

    if isinstance(last_message, AIMessage):
        tool_calls = last_message.tool_calls or []
        return [call for call in tool_calls if isinstance(call, dict)]

    maybe_tool_calls = getattr(last_message, "tool_calls", None)
    if isinstance(maybe_tool_calls, list):
        return [call for call in maybe_tool_calls if isinstance(call, dict)]

    return []


_SUPERVISOR_ROUTE_WORDS = {"DATASCIENTIST", "REPORTER", "ASSISTANT", "FINISH"}


def _message_content_to_text(content: Any) -> str:
    """将消息 content 尽力转换为纯文本。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts).strip()
    return str(content).strip()


def _extract_final_ai_answer_text(state: Any) -> str:
    """当流式 token 为空时，从最终状态提取可展示的 AI 文本。"""
    values = getattr(state, "values", {}) or {}
    if not isinstance(values, dict):
        return ""

    messages = values.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return ""

    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        text = _message_content_to_text(getattr(message, "content", ""))
        if not text:
            continue
        normalized = text.replace('"', "").replace("'", "").strip().upper()
        if normalized in _SUPERVISOR_ROUTE_WORDS:
            continue
        return text
    return ""


def _build_reject_tool_messages(tool_calls: list[dict[str, Any]]) -> list[ToolMessage]:
    """构造“用户拒绝执行工具”的 ToolMessage 列表。"""
    tool_messages: list[ToolMessage] = []

    for call in tool_calls:
        tool_call_id = str(call.get("id", "")).strip()
        if not tool_call_id:
            continue
        tool_messages.append(
            ToolMessage(
                content="用户拒绝了此操作",
                tool_call_id=tool_call_id,
            )
        )

    return tool_messages


def _is_email_delivery_query(query: str) -> bool:
    """轻量判断当前问题是否明确要求“发邮件”。"""
    normalized = query.strip()
    if not normalized:
        return False
    lower_text = normalized.lower()
    has_send_action = any(keyword in normalized for keyword in ("发送", "发到", "发给", "寄给", "转发"))
    has_mail_target = any(keyword in normalized for keyword in ("邮件", "邮箱", "电邮")) or ("email" in lower_text)
    has_email_address = bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", normalized))
    return (has_send_action and has_mail_target) or (has_send_action and has_email_address)


def _all_pending_tool_calls_are_send_report(tool_calls: list[dict[str, Any]]) -> bool:
    """判断待审批工具是否全部为 send_report。"""
    if not tool_calls:
        return False
    for call in tool_calls:
        if not isinstance(call, dict):
            return False
        name = str(call.get("name", "")).strip()
        if name not in {"tool_send_report", "send_report"}:
            return False
    return True


async def _stream_pending_interrupt_only(
    *,
    user_id: str,
    pending_tool_calls: list[dict[str, Any]],
) -> AsyncIterator[str]:
    """仅回放待审批中断事件，帮助前端在漏事件后恢复审批状态。"""
    logger.info(
        "回放 HITL 待审批事件 | user_id=%s | pending_tools=%d",
        user_id,
        len(pending_tool_calls),
    )
    yield _to_sse(
        {
            "type": "interrupt",
            "tool_calls": _sanitize_tool_calls_for_client(pending_tool_calls),
        }
    )
    yield _to_sse(
        {
            "type": "error",
            "message": "检测到上一轮存在待审批工具调用，请先审批后再继续聊天。",
        }
    )
    yield "data: [DONE]\n\n"


async def _stream_resume_no_pending(*, user_id: str) -> AsyncIterator[str]:
    """resume 幂等兜底：无待审批时返回可读提示而非报错。"""
    logger.info("resume 幂等返回 | user_id=%s | reason=no_pending_interrupt", user_id)
    yield _to_sse({"type": "token", "content": "审批状态已更新：当前没有待审批工具调用。"})
    yield "data: [DONE]\n\n"


async def _stream_graph_events(
    initial_input: Any,
    *,
    config: dict[str, Any],
    user_id: str,
    emit_interrupt: bool,
) -> AsyncIterator[str]:
    """复用的图事件流 -> SSE 转换器。"""
    active_node = ""
    emitted_worker_token = False
    worker_nodes = {"data_scientist_node", "reporter_node", "assistant_node"}
    broadcast_nodes = {"supervisor_node", "data_scientist_node", "reporter_node", "assistant_node"}

    try:
        async for event in graph_runtime.get_app_graph().astream_events(initial_input, config=config, version="v1"):
            event_name = event.get("event", "")
            node_name = str(event.get("name", "")).strip()

            if event_name == "on_chain_start" and node_name in broadcast_nodes:
                active_node = node_name
                yield _to_sse({"type": "agent_switch", "agent": node_name})

            if event_name == "on_chat_model_stream":
                # 仅将 Worker 的 token 流推送给前端，避免暴露主管路由词。
                if active_node not in worker_nodes:
                    continue
                token_text = _extract_token_content(event)
                token_text = _sanitize_stream_token_text(token_text)
                if token_text:
                    emitted_worker_token = True
                    yield _to_sse({"type": "token", "content": token_text})

            elif event_name == "on_tool_start":
                data = event.get("data", {}) if isinstance(event, dict) else {}
                tool_name = event.get("name", "unknown_tool")
                raw_tool_input = data.get("input")
                tool_input = _safe_tool_payload(raw_tool_input)
                logger.info("流式事件 | tool_start | user_id=%s | tool=%s", user_id, tool_name)
                payload: dict[str, Any] = {"type": "tool_start", "tool": tool_name}
                if DEBUG_MODE:
                    payload["input"] = tool_input
                else:
                    payload["input"] = _tool_payload_summary(tool_input)
                    payload["redacted"] = True
                yield _to_sse(payload)

            elif event_name == "on_tool_end":
                data = event.get("data", {}) if isinstance(event, dict) else {}
                tool_name = event.get("name", "unknown_tool")
                raw_tool_output = data.get("output")
                tool_output = _safe_tool_payload(raw_tool_output)
                logger.info("流式事件 | tool_end | user_id=%s | tool=%s", user_id, tool_name)
                payload = {"type": "tool_end", "tool": tool_name}
                if DEBUG_MODE:
                    payload["output"] = tool_output
                else:
                    payload["output"] = _tool_payload_summary(tool_output)
                    payload["redacted"] = True
                yield _to_sse(payload)

        state: Any | None = None
        if emit_interrupt or not emitted_worker_token:
            state = await graph_runtime.get_app_graph().aget_state(config)

        if emit_interrupt and state is not None:
            if _is_waiting_for_tools_node(state):
                pending_tool_calls = _extract_pending_tool_calls(state)
                if pending_tool_calls:
                    logger.info(
                        "命中 HITL 中断 | user_id=%s | pending_tools=%d",
                        user_id,
                        len(pending_tool_calls),
                    )
                    yield _to_sse(
                        {
                            "type": "interrupt",
                            "tool_calls": _sanitize_tool_calls_for_client(pending_tool_calls),
                        }
                    )
            elif not emitted_worker_token:
                fallback_answer = _extract_final_ai_answer_text(state)
                if fallback_answer:
                    yield _to_sse({"type": "token", "content": fallback_answer})
        elif not emitted_worker_token and state is not None:
            fallback_answer = _extract_final_ai_answer_text(state)
            if fallback_answer:
                yield _to_sse({"type": "token", "content": fallback_answer})

        yield "data: [DONE]\n\n"
    except Exception as exc:  # noqa: BLE001
        if DEBUG_MODE:
            logger.exception("流式聊天执行失败 | user_id=%s | error=%s", user_id, exc)
            message = _safe_error_text(exc, max_len=300)
        else:
            logger.exception("流式聊天执行失败 | user_id=%s", user_id)
            message = "流式处理出现异常，请稍后重试。"
        yield _to_sse({"type": "error", "message": message})
        yield "data: [DONE]\n\n"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """初始化并释放会话级基础设施。"""
    global _session_store_ready
    logger.info(
        "安全配置 | env=%s | require_api_auth=%s | require_user_sub=%s | cors_origins=%s | cors_credentials=%s | require_llm_session=%s | llm_allowlist_count=%d",
        ENVIRONMENT,
        REQUIRE_API_AUTH,
        AUTH_REQUIRE_USER_SUB,
        ",".join(CORS_ALLOW_ORIGINS) if CORS_ALLOW_ORIGINS else "(none)",
        CORS_ALLOW_CREDENTIALS,
        REQUIRE_LLM_SESSION,
        len(ALLOWED_LLM_BASE_URLS),
    )
    
    # 打印LLM配置信息
    logger.info("LLM配置 - API_KEY: %s, BASE_URL: %s, MODEL: %s", 
                "*" * 8 + os.getenv("OPENAI_API_KEY", "")[-4:] if os.getenv("OPENAI_API_KEY") else "未设置",
                os.getenv("OPENAI_BASE_URL", "未设置"),
                os.getenv("OPENAI_MODEL", "未设置"))
    if _is_production_environment() and not ALLOWED_LLM_BASE_URLS:
        raise RuntimeError("生产环境必须配置 ALLOWED_LLM_BASE_URLS，当前为空，服务已拒绝启动。")
    if not _is_production_environment() and not ALLOWED_LLM_BASE_URLS:
        logger.warning("开发环境未配置 ALLOWED_LLM_BASE_URLS：将放行任意模型网关，仅建议本地调试使用。")
    if REQUIRE_API_AUTH and AUTH_REQUIRE_USER_SUB and not (JWT_JWKS_URL or JWT_HS256_SECRET):
        logger.warning("已启用用户身份校验，但未配置 JWT_JWKS_URL/JWT_HS256_SECRET，用户请求将无法通过。")
    if not LLM_SESSION_MASTER_KEY_RAW:
        logger.warning("未配置 LLM_SESSION_MASTER_KEY，当前回退使用 JWT_HS256_SECRET 或 AGENT_API_TOKEN。建议生产环境使用独立主密钥。")

    await graph_runtime.init_graph_runtime()

    try:
        await _session_store.startup()
        _session_store_ready = True
    except Exception as exc:  # noqa: BLE001
        _session_store_ready = False
        logger.warning("LLM Session Store 启动失败，将仅支持默认环境变量模式 | error=%s", exc)

    try:
        yield
    finally:
        try:
            await _session_store.shutdown()
        finally:
            _session_store_ready = False
        await graph_runtime.shutdown_graph_runtime()


app = FastAPI(title="NanoAgent Service", version="1.0.0", lifespan=lifespan)

# 生产环境建议使用精确白名单，避免跨域滥用。
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """全局兜底异常处理器。"""
    logger.exception("应用发生未处理异常：%s", exc)
    return JSONResponse(status_code=500, content={"detail": "服务器内部错误"})


@app.get("/health")
async def health() -> dict[str, str]:
    """服务存活探针。"""
    return {"status": "ok", "service": "agent_service"}


@app.get("/health/mcp")
async def mcp_health() -> dict[str, Any]:
    """检查上游 MCP 服务可达性与健康状态。"""
    mcp_health_url = f"{MCP_BASE_URL}/health"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(mcp_health_url)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001
        if DEBUG_MODE:
            logger.warning("MCP 健康检查失败 | url=%s | error=%s", mcp_health_url, exc)
            return {
                "status": "degraded",
                "mcp_url": mcp_health_url,
                "error": "上游 MCP 服务暂不可用",
                "debug_error": _safe_error_text(exc),
            }
        logger.warning("MCP 健康检查失败 | url=%s", mcp_health_url)
        return {"status": "degraded", "mcp_url": mcp_health_url, "error": "上游 MCP 服务暂不可用"}

    return {"status": "ok", "mcp_url": mcp_health_url, "mcp_response": payload}


@app.get(
    "/api/v1/session/llm/providers",
    response_model=LLMProviderListResponse,
    dependencies=[Depends(_require_api_auth)],
)
async def list_llm_providers() -> LLMProviderListResponse:
    """返回后端支持的 LLM Provider 列表（供前端配置页使用）。"""
    items: list[LLMProviderItem] = []
    for provider in SUPPORTED_PROVIDERS:
        preset = PROVIDER_PRESETS[provider]
        default_base_url = str(preset.get("base_url", "")).strip() or None
        requires_base_url = bool(preset.get("requires_base_url", provider == "other"))
        if "embedding_model" in preset:
            default_embedding_model = str(preset.get("embedding_model", "")).strip()
        else:
            default_embedding_model = DEFAULT_EMBEDDING_MODEL
        items.append(
            LLMProviderItem(
                provider=provider,
                label=str(preset.get("label", provider)).strip() or provider,
                requires_base_url=requires_base_url,
                default_base_url=default_base_url,
                default_embedding_model=default_embedding_model,
            )
        )

    return LLMProviderListResponse(items=items)


@app.post(
    "/api/v1/session/llm/validate",
    response_model=LLMSessionValidateResponse,
    dependencies=[Depends(_require_api_auth)],
)
async def validate_llm_session_config(request: LLMSessionValidateRequest) -> LLMSessionValidateResponse:
    """校验用户输入的 LLM 配置是否可用于聊天与长期记忆。"""
    llm_profile = _build_llm_profile(request)
    chat_ok, embedding_ok, errors = await _validate_llm_profile_connectivity(
        llm_profile,
        validate_chat=request.validate_chat,
        validate_embedding=request.validate_embedding,
    )

    return LLMSessionValidateResponse(
        provider=llm_profile.get("provider", "other"),
        model=llm_profile["model"],
        base_url=llm_profile["base_url"],
        embedding_model=llm_profile["embedding_model"],
        chat_ok=chat_ok,
        embedding_ok=embedding_ok,
        errors=errors,
    )


@app.post(
    "/api/v1/session/llm",
    response_model=LLMSessionCreateResponse,
)
async def create_llm_session(
    request: LLMSessionCreateRequest,
    auth_context: AuthContext = Depends(_require_user_context),
) -> LLMSessionCreateResponse:
    """创建会话级 LLM 凭据（BYOK）。"""
    if not _session_store_ready:
        raise HTTPException(status_code=503, detail="LLM 会话存储未就绪，请稍后重试")

    owner_id = _require_subject(auth_context)
    llm_profile = _build_llm_profile(request)

    try:
        session_id, effective_ttl = await _session_store.create_session(
            llm_profile,
            owner_id=owner_id,
            ttl_seconds=request.ttl_seconds,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info(
        "创建 LLM 会话成功 | owner_id=%s | session_id=%s | provider=%s | model=%s | ttl=%d",
        owner_id,
        session_id,
        llm_profile.get("provider", "other"),
        llm_profile["model"],
        effective_ttl,
    )
    return LLMSessionCreateResponse(
        session_id=session_id,
        provider=llm_profile.get("provider", "other"),
        expires_in=effective_ttl,
        model=llm_profile["model"],
        base_url=llm_profile["base_url"],
        embedding_model=llm_profile["embedding_model"],
    )


@app.delete(
    "/api/v1/session/llm/{session_id}",
    response_model=LLMSessionDeleteResponse,
)
async def delete_llm_session(
    session_id: str = Path(..., min_length=8),
    auth_context: AuthContext = Depends(_require_user_context),
) -> LLMSessionDeleteResponse:
    """主动销毁会话级 LLM 凭据。"""
    if not _session_store_ready:
        raise HTTPException(status_code=503, detail="LLM 会话存储未就绪，请稍后重试")

    owner_id = _require_subject(auth_context)
    normalized_session_id = session_id.strip()
    try:
        deleted = await _session_store.delete_session(normalized_session_id, owner_id=owner_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在、已过期或不属于当前用户")

    logger.info("删除 LLM 会话成功 | owner_id=%s | session_id=%s", owner_id, normalized_session_id)
    return LLMSessionDeleteResponse(
        session_id=normalized_session_id,
        status="success",
        message="LLM 会话已删除。",
    )


@app.post("/api/v1/chat")
async def chat(
    request: ChatRequest,
    auth_context: AuthContext = Depends(_require_user_context),
) -> StreamingResponse:
    """以 SSE 流式方式执行 NanoAgent 图并实时返回 token。"""
    token_subject = _require_subject(auth_context)
    user_id = _resolve_effective_user_id(
        token_subject=token_subject,
        client_user_id=request.user_id,
        source="/api/v1/chat",
    )
    query = request.query.strip()
    llm_profile = await _resolve_llm_profile(request.session_id, owner_id=user_id)

    logger.info("收到流式聊天请求 | user_id=%s | query_len=%d", user_id, len(query))

    config = _graph_config(user_id, llm_profile)

    # 若上一轮工具调用仍待审批，优先回放 interrupt，避免前端“无提示卡住”。
    try:
        pending_state = await graph_runtime.get_app_graph().aget_state(config)
        if _is_waiting_for_tools_node(pending_state):
            pending_tool_calls = _extract_pending_tool_calls(pending_state)
            if pending_tool_calls:
                # 新问题并未要求发邮件时，自动清理“旧的 send_report 待审批”，避免误锁对话。
                if _all_pending_tool_calls_are_send_report(pending_tool_calls) and not _is_email_delivery_query(query):
                    reject_messages = _build_reject_tool_messages(pending_tool_calls)
                    if reject_messages:
                        await graph_runtime.get_app_graph().aupdate_state(
                            config,
                            {"messages": reject_messages},
                            as_node="tools_node",
                        )
                        logger.info(
                            "自动清理待审批 send_report | user_id=%s | pending_tools=%d | reason=new_non_email_query",
                            user_id,
                            len(reject_messages),
                        )
                    else:
                        logger.warning(
                            "待审批 send_report 缺少 tool_call_id，无法自动清理 | user_id=%s",
                            user_id,
                        )
                        return StreamingResponse(
                            _stream_pending_interrupt_only(
                                user_id=user_id,
                                pending_tool_calls=pending_tool_calls,
                            ),
                            media_type="text/event-stream",
                            headers=_streaming_headers(),
                        )
                else:
                    return StreamingResponse(
                        _stream_pending_interrupt_only(
                            user_id=user_id,
                            pending_tool_calls=pending_tool_calls,
                        ),
                        media_type="text/event-stream",
                        headers=_streaming_headers(),
                    )
    except Exception as exc:  # noqa: BLE001
        logger.warning("预检查待审批状态失败 | user_id=%s | error=%s", user_id, exc)

    # 尝试自动写入用户自述背景（失败不影响主流程）。
    auto_memory_id = _try_auto_save_memory(user_id, query, llm_profile)
    if auto_memory_id:
        logger.info("本轮对话已自动记录背景信息 | user_id=%s | memory_id=%s", user_id, auto_memory_id)

    initial_state: dict[str, Any] = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id,
        "memory_context": "",
        "sender": "",
    }

    return StreamingResponse(
        _stream_graph_events(
            initial_state,
            config=config,
            user_id=user_id,
            emit_interrupt=True,
        ),
        media_type="text/event-stream",
        headers=_streaming_headers(),
    )


@app.post("/api/v1/chat/resume")
async def chat_resume(
    request: ChatResumeRequest,
    auth_context: AuthContext = Depends(_require_user_context),
) -> StreamingResponse:
    """对已中断的工具调用进行审批并恢复执行。"""
    token_subject = _require_subject(auth_context)
    user_id = _resolve_effective_user_id(
        token_subject=token_subject,
        client_user_id=request.user_id,
        source="/api/v1/chat/resume",
    )
    action = request.action
    llm_profile = await _resolve_llm_profile(request.session_id, owner_id=user_id)
    config = _graph_config(user_id, llm_profile)

    logger.info("收到审批续跑请求 | user_id=%s | action=%s", user_id, action)

    try:
        state = await graph_runtime.get_app_graph().aget_state(config)
        if not _is_waiting_for_tools_node(state):
            return StreamingResponse(
                _stream_resume_no_pending(user_id=user_id),
                media_type="text/event-stream",
                headers=_streaming_headers(),
            )

        pending_tool_calls = _extract_pending_tool_calls(state)
        if not pending_tool_calls:
            return StreamingResponse(
                _stream_resume_no_pending(user_id=user_id),
                media_type="text/event-stream",
                headers=_streaming_headers(),
            )

        if action == "reject":
            tool_messages = _build_reject_tool_messages(pending_tool_calls)
            if not tool_messages:
                raise HTTPException(status_code=400, detail="待审批工具调用缺少 tool_call_id，无法拒绝")

            await graph_runtime.get_app_graph().aupdate_state(
                config,
                {"messages": tool_messages},
                as_node="tools_node",
            )
            logger.info(
                "审批结果 | user_id=%s | action=reject | tool_calls=%d",
                user_id,
                len(tool_messages),
            )
        else:
            logger.info(
                "审批结果 | user_id=%s | action=approve | tool_calls=%d",
                user_id,
                len(pending_tool_calls),
            )

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("审批续跑失败 | user_id=%s | action=%s | error=%s", user_id, action, exc)
        raise HTTPException(status_code=500, detail="服务器内部错误") from exc

    return StreamingResponse(
        _stream_graph_events(
            None,
            config=config,
            user_id=user_id,
            emit_interrupt=True,
        ),
        media_type="text/event-stream",
        headers=_streaming_headers(),
    )


@app.post("/api/v1/memory", response_model=MemoryResponse)
async def save_memory(
    request: MemoryRequest,
    auth_context: AuthContext = Depends(_require_user_context),
) -> MemoryResponse:
    """将用户偏好写入长期记忆。"""
    token_subject = _require_subject(auth_context)
    user_id = _resolve_effective_user_id(
        token_subject=token_subject,
        client_user_id=request.user_id,
        source="/api/v1/memory",
    )
    preference_text = request.preference_text.strip()

    logger.info(
        "收到记忆写入请求 | user_id=%s | text_len=%d",
        user_id,
        len(preference_text),
    )

    try:
        llm_profile = await _resolve_llm_profile(request.session_id, owner_id=user_id)
        if not llm_profile.get("embedding_model", "").strip():
            raise HTTPException(
                status_code=400,
                detail="当前会话未配置 embedding 模型，无法写入长期记忆。请切换支持 embedding 的提供商或填写可用 embedding 模型。",
            )
        manager = _get_memory_manager()
        memory_id = manager.save_preference(
            user_id=user_id,
            preference_text=preference_text,
            embedding_profile=llm_profile,
        )
        if not memory_id:
            raise HTTPException(status_code=400, detail="长期记忆写入失败，请检查输入内容后重试")
        return MemoryResponse(
            user_id=user_id,
            status="success",
            message="偏好已成功写入长期记忆。",
            memory_id=memory_id,
        )
    except MemoryProviderError as exc:
        status_code_by_reason = {
            "auth": 401,
            "rate_limit": 429,
            "timeout": 504,
            "bad_request": 400,
            "unknown": 502,
        }
        status_code = status_code_by_reason.get(exc.reason, 502)
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("记忆写入失败 | user_id=%s | error=%s", user_id, exc)
        raise HTTPException(status_code=500, detail="服务器内部错误") from exc


@app.get("/api/v1/memory/{user_id}", response_model=MemoryListResponse)
async def list_memory(
    user_id: str = Path(..., min_length=1),
    limit: int = Query(default=50, ge=1, le=200),
    auth_context: AuthContext = Depends(_require_user_context),
) -> MemoryListResponse:
    """查看某个用户的长期记忆列表。"""
    # 在不要求JWT sub的情况下，使用客户端提供的user_id
    if AUTH_REQUIRE_USER_SUB:
        token_subject = _require_subject(auth_context)
        normalized_user_id = _resolve_effective_user_id(
            token_subject=token_subject,
            client_user_id=user_id,
            source="/api/v1/memory/{user_id}",
        )
    else:
        # 不要求JWT sub时，直接使用客户端提供的user_id
        normalized_user_id = user_id.strip()
        if not normalized_user_id:
            raise HTTPException(status_code=422, detail="user_id 不能为空")

    try:
        manager = _get_memory_manager()
        items = manager.list_memories(user_id=normalized_user_id, limit=limit)
        return MemoryListResponse(
            user_id=normalized_user_id,
            items=[MemoryItem(**item) for item in items],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("记忆列表查询失败 | user_id=%s | error=%s", normalized_user_id, exc)
        raise HTTPException(status_code=500, detail="服务器内部错误") from exc


@app.delete(
    "/api/v1/memory/{user_id}/{memory_id}",
    response_model=MemoryDeleteResponse,
)
async def delete_memory(
    user_id: str = Path(..., min_length=1),
    memory_id: str = Path(..., min_length=1),
    auth_context: AuthContext = Depends(_require_user_context),
) -> MemoryDeleteResponse:
    """删除某个用户的一条长期记忆。"""
    token_subject = _require_subject(auth_context)
    normalized_user_id = _resolve_effective_user_id(
        token_subject=token_subject,
        client_user_id=user_id,
        source="/api/v1/memory/{user_id}/{memory_id}",
    )
    normalized_memory_id = memory_id.strip()

    try:
        manager = _get_memory_manager()
        deleted = manager.delete_memory(user_id=normalized_user_id, memory_id=normalized_memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="未找到可删除的记忆记录")

        return MemoryDeleteResponse(
            user_id=normalized_user_id,
            memory_id=normalized_memory_id,
            status="success",
            message="记忆已删除。",
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "记忆删除失败 | user_id=%s | memory_id=%s | error=%s",
            normalized_user_id,
            normalized_memory_id,
            exc,
        )
        raise HTTPException(status_code=500, detail="服务器内部错误") from exc