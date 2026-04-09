"""API 路由模块。"""

import logging
from typing import Any

import httpx
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.requests import Request
from auth import _resolve_effective_user_id
from graph.workflow import get_app_graph
from auth import _resolve_effective_user_id
import graph.workflow as get_app_graph

from auth import (
    AuthContext,
    _require_api_auth,
    _require_subject,
    _require_user_context,
)
from config import (
    DEFAULT_EMBEDDING_MODEL,
    PROVIDER_PRESETS,
    SUPPORTED_PROVIDERS,
)
from utils import (
    _all_pending_tool_calls_are_send_report,
    _build_reject_tool_messages,
    _extract_pending_tool_calls,
    _get_memory_manager,
    _graph_config,
    _is_email_delivery_query,
    _is_waiting_for_tools_node,
    _normalize_provider,
    _resolve_base_url_for_provider,
    _resolve_embedding_model_for_provider,
    _resolve_llm_profile,
    _safe_error_text,
    _stream_graph_events,
    _stream_pending_interrupt_only,
    _stream_resume_no_pending,
    _try_auto_save_memory,
    _validate_llm_profile_connectivity,
)
from memory import MemoryProviderError

logger = logging.getLogger("nanoagent.agent_service.routes")


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


def _streaming_headers() -> dict[str, str]:
    """SSE 响应头。"""
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


def register_routes(app: FastAPI, session_store: Any, session_store_ready: bool) -> None:
    """注册所有路由到 FastAPI 应用。"""

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
        from config import MCP_BASE_URL

        mcp_health_url = f"{MCP_BASE_URL}/health"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(mcp_health_url)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001
            from config import DEBUG_MODE

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
        if not session_store_ready:
            raise HTTPException(status_code=503, detail="LLM 会话存储未就绪，请稍后重试")

        owner_id = _require_subject(auth_context)
        llm_profile = _build_llm_profile(request)

        try:
            session_id, effective_ttl = await session_store.create_session(
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
        if not session_store_ready:
            raise HTTPException(status_code=503, detail="LLM 会话存储未就绪，请稍后重试")

        owner_id = _require_subject(auth_context)
        normalized_session_id = session_id.strip()
        try:
            deleted = await session_store.delete_session(normalized_session_id, owner_id=owner_id)
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
        llm_profile = await _resolve_llm_profile(request.session_id, owner_id=user_id, session_store=session_store, session_store_ready=session_store_ready)

        logger.info("收到流式聊天请求 | user_id=%s | query_len=%d", user_id, len(query))

        config = _graph_config(user_id, llm_profile)

        try:
            pending_state = await get_app_graph().aget_state(config)
            if _is_waiting_for_tools_node(pending_state):
                pending_tool_calls = _extract_pending_tool_calls(pending_state)
                if pending_tool_calls:
                    if _all_pending_tool_calls_are_send_report(pending_tool_calls) and not _is_email_delivery_query(query):
                        reject_messages = _build_reject_tool_messages(pending_tool_calls)
                        if reject_messages:
                            await get_app_graph().aupdate_state(
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

        auto_memory_id = _try_auto_save_memory(user_id, query, llm_profile)
        if auto_memory_id:
            logger.info("本轮对话已自动记录背景信息 | user_id=%s | memory_id=%s", user_id, auto_memory_id)

        from langchain_core.messages import HumanMessage

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
        llm_profile = await _resolve_llm_profile(request.session_id, owner_id=user_id, session_store=session_store, session_store_ready=session_store_ready)
        config = _graph_config(user_id, llm_profile)

        logger.info("收到审批续跑请求 | user_id=%s | action=%s", user_id, action)

        try:
            state = await get_app_graph().aget_state(config)
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

                await get_app_graph().aupdate_state(
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
        from auth import _resolve_effective_user_id

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
            llm_profile = await _resolve_llm_profile(request.session_id, owner_id=user_id, session_store=session_store, session_store_ready=session_store_ready)
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
        from auth import _resolve_effective_user_id

        token_subject = _require_subject(auth_context)
        normalized_user_id = _resolve_effective_user_id(
            token_subject=token_subject,
            client_user_id=user_id,
            source="/api/v1/memory/{user_id}",
        )

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
        from auth import _resolve_effective_user_id

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