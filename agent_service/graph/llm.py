"""NanoAgent LLM管理模块。

处理LLM客户端的创建、缓存和配置。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from .tools import tool_query_database, tool_get_current_time, tool_search, tool_upsert_user_setting

from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    _graph_runtime_globals,
)

if TYPE_CHECKING:
    from tools import tool_get_current_time, tool_query_database, tool_search, tool_upsert_user_setting

_llm_cache = _graph_runtime_globals["_llm_cache"]
_non_stream_llm_cache = _graph_runtime_globals["_non_stream_llm_cache"]
_bound_llm_cache = _graph_runtime_globals["_bound_llm_cache"]


def _default_llm_profile() -> dict[str, str] | None:
    """读取环境变量默认 LLM 配置（兼容旧模式）。"""
    if not DEFAULT_OPENAI_API_KEY:
        return None
    return {
        "api_key": DEFAULT_OPENAI_API_KEY,
        "base_url": DEFAULT_OPENAI_BASE_URL,
        "model": DEFAULT_OPENAI_MODEL,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
    }


def _normalize_llm_profile(profile: Any) -> dict[str, str] | None:
    """标准化会话级 LLM 配置。"""
    if not isinstance(profile, dict):
        return None

    api_key = str(profile.get("api_key", "")).strip()
    base_url = str(profile.get("base_url", "")).strip().rstrip("/")
    model = str(profile.get("model", "")).strip()
    embedding_model = str(profile.get("embedding_model", "")).strip()
    has_embedding_model_field = "embedding_model" in profile

    if not api_key or not base_url or not model:
        return None

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "embedding_model": embedding_model if has_embedding_model_field else (embedding_model or DEFAULT_EMBEDDING_MODEL),
    }


def _llm_profile_from_config(config: RunnableConfig | None) -> dict[str, str] | None:
    """优先从请求级 metadata 读取 llm_profile，缺失时回退到默认环境变量。"""
    if isinstance(config, dict):
        metadata = config.get("metadata", {})
        if isinstance(metadata, dict):
            profile = _normalize_llm_profile(metadata.get("llm_profile"))
            if profile is not None:
                return profile

    return _default_llm_profile()


def _chat_llm_from_profile(profile: dict[str, str]) -> ChatOpenAI:
    """按 profile 获取可复用 ChatOpenAI 客户端。"""
    cache_key = (profile["model"], profile["base_url"], profile["api_key"])
    cached = _llm_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        llm_client = ChatOpenAI(
            model=profile["model"],
            api_key=profile["api_key"],
            base_url=profile["base_url"],
            streaming=True,
            temperature=0,
            timeout=60,
            max_retries=2,
        )
    except TypeError:
        llm_client = ChatOpenAI(
            model=profile["model"],
            openai_api_key=profile["api_key"],
            openai_api_base=profile["base_url"],
            streaming=True,
            temperature=0,
            timeout=60,
            max_retries=2,
        )

    _llm_cache[cache_key] = llm_client
    return llm_client


def _non_stream_chat_llm_from_profile(profile: dict[str, str]) -> ChatOpenAI:
    """按 profile 获取非流式 ChatOpenAI 客户端（用于路由/判定器）。"""
    cache_key = (profile["model"], profile["base_url"], profile["api_key"])
    cached = _non_stream_llm_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        llm_client = ChatOpenAI(
            model=profile["model"],
            api_key=profile["api_key"],
            base_url=profile["base_url"],
            streaming=False,
            temperature=0,
            timeout=30,
            max_retries=1,
        )
    except TypeError:
        llm_client = ChatOpenAI(
            model=profile["model"],
            openai_api_key=profile["api_key"],
            openai_api_base=profile["base_url"],
            streaming=False,
            temperature=0,
            timeout=30,
            max_retries=1,
        )

    _non_stream_llm_cache[cache_key] = llm_client
    return llm_client


def _get_chat_llm(config: RunnableConfig | None) -> ChatOpenAI:
    """获取当前请求对应的基础 LLM 客户端。"""
    profile = _llm_profile_from_config(config)
    if profile is None:
        raise RuntimeError("未提供可用的 LLM 配置，请先创建会话或设置默认 OPENAI_API_KEY。")
    return _chat_llm_from_profile(profile)


def _get_non_stream_chat_llm(config: RunnableConfig | None) -> ChatOpenAI:
    """获取当前请求对应的非流式 LLM 客户端。"""
    profile = _llm_profile_from_config(config)
    if profile is None:
        raise RuntimeError("未提供可用的 LLM 配置，请先创建会话或设置默认 OPENAI_API_KEY。")
    return _non_stream_chat_llm_from_profile(profile)


def _get_bound_llm(
    config: RunnableConfig | None,
    worker: Literal["knowledge_worker", "reporter", "travel_planner"],
) -> Any:
    """获取绑定工具后的 Worker LLM。"""
    profile = _llm_profile_from_config(config)
    if profile is None:
        raise RuntimeError("未提供可用的 LLM 配置，请先创建会话或设置默认 OPENAI_API_KEY。")

    base_llm = _chat_llm_from_profile(profile)
    cache_key = (profile["model"], profile["base_url"], profile["api_key"], worker)
    cached = _bound_llm_cache.get(cache_key)
    if cached is not None:
        return cached

    if worker == "knowledge_worker":
        bound = base_llm.bind_tools([tool_query_database, tool_get_current_time, tool_search])
    elif worker == "travel_planner":
        bound = base_llm.bind_tools([tool_get_current_time])
    else:
        bound = base_llm.bind_tools([tool_upsert_user_setting])

    _bound_llm_cache[cache_key] = bound
    return bound