from __future__ import annotations

import sys
import asyncio

# Windows 兼容性修复
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from graph import init as graph_runtime
from session_store import LLMSessionStore

from config import (
    ALLOWED_LLM_BASE_URLS,
    AUTH_ALLOW_API_KEY_FALLBACK,
    AUTH_REQUIRE_USER_SUB,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_ORIGINS,
    ENVIRONMENT,
    JWT_HS256_SECRET,
    JWT_JWKS_URL,
    LLM_SESSION_MASTER_KEY_RAW,
    LLM_SESSION_MAX_TTL_SECONDS,
    LLM_SESSION_MASTER_KEY,
    LLM_SESSION_TTL_SECONDS,
    REDIS_URL,
    REQUIRE_API_AUTH,
)
from utils import _is_production_environment
from routes import register_routes

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger("nanoagent.agent_service.main")

_session_store = LLMSessionStore(
    REDIS_URL,
    master_key=LLM_SESSION_MASTER_KEY,
    default_ttl_seconds=LLM_SESSION_TTL_SECONDS,
    max_ttl_seconds=LLM_SESSION_MAX_TTL_SECONDS,
)
_session_store_ready = False


@asynccontextmanager
async def lifespan(_: FastAPI):
    """初始化并释放会话级基础设施。"""
    global _session_store_ready
    logger.info(
        "安全配置 | env=%s | require_api_auth=%s | require_user_sub=%s | cors_origins=%s | cors_credentials=%s | require_llm_session=%s | llm_allowlist_count=%d",
        ENVIRONMENT,
        REQUIRE_API_AUTH,
        AUTH_REQUIRE_USER_SUB,
        ",".join(CORS_ALLOW_ORIGINS) if CORS_ALLOW_ORIGINS else "(none)",
        CORS_ALLOW_CREDENTIALS,
        os.getenv("REQUIRE_LLM_SESSION", "true"),
        len(ALLOWED_LLM_BASE_URLS),
    )
    
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_routes(app, _session_store, _session_store_ready)