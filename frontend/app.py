"""NanoAgent 的 Streamlit 前端。"""

from __future__ import annotations

import base64
import binascii
import json
import os
import queue
import re
import threading
import time
from typing import Any, Iterator
from urllib.parse import quote

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _jwt_subject_from_token(token: str) -> str:
    """从 JWT 文本中提取 sub（不做签名校验，仅用于前端展示与提示）。"""
    normalized = token.strip()
    if normalized.count(".") != 2:
        return ""
    try:
        payload_segment = normalized.split(".")[1]
        padding = "=" * (-len(payload_segment) % 4)
        payload_raw = base64.urlsafe_b64decode(payload_segment + padding).decode("utf-8")
        payload = json.loads(payload_raw)
    except (ValueError, TypeError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError):
        return ""
    subject = payload.get("sub")
    return subject.strip() if isinstance(subject, str) else ""


AGENT_API_BASE_URL = os.getenv("AGENT_API_BASE_URL", "http://localhost:8080").rstrip("/")
AGENT_API_TOKEN = os.getenv("AGENT_API_TOKEN", "").strip()
AGENT_TOKEN_SUB = _jwt_subject_from_token(AGENT_API_TOKEN)
APPROVAL_CLICK_COOLDOWN_SECONDS = float(os.getenv("APPROVAL_CLICK_COOLDOWN_SECONDS", "1.5"))
CHAT_ENDPOINT = f"{AGENT_API_BASE_URL}/api/v1/chat"
CHAT_RESUME_ENDPOINT = f"{AGENT_API_BASE_URL}/api/v1/chat/resume"
MEMORY_ENDPOINT = f"{AGENT_API_BASE_URL}/api/v1/memory"
MEMORY_LIST_ENDPOINT_TEMPLATE = f"{AGENT_API_BASE_URL}/api/v1/memory/{{user_id}}"
MEMORY_DELETE_ENDPOINT_TEMPLATE = f"{AGENT_API_BASE_URL}/api/v1/memory/{{user_id}}/{{memory_id}}"
LLM_PROVIDER_ENDPOINT = f"{AGENT_API_BASE_URL}/api/v1/session/llm/providers"
LLM_VALIDATE_ENDPOINT = f"{AGENT_API_BASE_URL}/api/v1/session/llm/validate"
LLM_SESSION_CREATE_ENDPOINT = f"{AGENT_API_BASE_URL}/api/v1/session/llm"
LLM_SESSION_DELETE_ENDPOINT_TEMPLATE = f"{AGENT_API_BASE_URL}/api/v1/session/llm/{{session_id}}"

MODEL_PRESETS: dict[str, list[str]] = {
    "qwen": ["qwen3.5-plus", "qwen-plus-latest", "qwen-max-latest"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "other": ["自定义输入"],
}

FALLBACK_PROVIDER_ITEMS: list[dict[str, Any]] = [
    {
        "provider": "qwen",
        "label": "Tongyi Qwen",
        "requires_base_url": False,
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_embedding_model": "",
    },
    {
        "provider": "openai",
        "label": "OpenAI",
        "requires_base_url": False,
        "default_base_url": "https://api.openai.com/v1",
        "default_embedding_model": "",
    },
    {
        "provider": "deepseek",
        "label": "DeepSeek",
        "requires_base_url": False,
        "default_base_url": "https://api.deepseek.com/v1",
        # DeepSeek 会话默认不启用 embedding；可手动填写第三方可用 embedding 模型。
        "default_embedding_model": "",
    },
    {
        "provider": "groq",
        "label": "Groq",
        "requires_base_url": False,
        "default_base_url": "https://api.groq.com/openai/v1",
        "default_embedding_model": "",
    },
    {
        "provider": "other",
        "label": "Other",
        "requires_base_url": True,
        "default_base_url": None,
        "default_embedding_model": "",
    },
]

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
    "string=\"true\"",
)
# Add these constants to the existing constants section (around line 40)
USER_THREADS_ENDPOINT_TEMPLATE = f"{AGENT_API_BASE_URL}/api/v1/user_threads/{{user_id}}"
CHAT_HISTORY_ENDPOINT_TEMPLATE = f"{AGENT_API_BASE_URL}/api/v1/chat/history/{{thread_id}}?user_id={{user_id}}"

# Add these functions after the existing API functions like stream_chat_api (around line 300)
def get_user_threads_api(user_id: str) -> list[str]:
    """获取用户的所有thread_ids"""
    endpoint = USER_THREADS_ENDPOINT_TEMPLATE.format(user_id=_quote_path(user_id))
    try:
        response = requests.get(endpoint, headers=_request_headers(), timeout=30)
        response.raise_for_status()
        payload = response.json()
        thread_ids = payload.get("thread_ids", [])
        return [tid for tid in thread_ids if isinstance(tid, str)]
    except requests.exceptions.RequestException as e:
        st.error(f"获取用户对话线程失败: {e}")
        return []

def get_chat_history_api(thread_id: str, user_id: str) -> list[dict[str, Any]]:
    """获取特定thread_id的聊天历史"""
    endpoint = CHAT_HISTORY_ENDPOINT_TEMPLATE.format(thread_id=_quote_path(thread_id), user_id=_quote_path(user_id))
    try:
        response = requests.get(endpoint, headers=_request_headers(), timeout=30)
        response.raise_for_status()
        payload = response.json()
        messages = payload.get("messages", [])
        # Convert the messages to the format expected by the frontend
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.get("role", "assistant"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", ""),
                "tool_calls": msg.get("tool_calls", None)  # Add tool_calls if available
            })
        return formatted_messages
    except requests.exceptions.RequestException as e:
        st.error(f"获取聊天历史失败: {e}")
        return []

def _quote_path(value: str) -> str:
    return quote(value, safe="")


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _request_headers(*, sse: bool = False) -> dict[str, str]:
    headers: dict[str, str] = {}
    if sse:
        headers["Accept"] = "text/event-stream"
    if AGENT_API_TOKEN:
        headers["Authorization"] = f"Bearer {AGENT_API_TOKEN}"
    return headers


def _request_error_detail(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    try:
        payload = response.json()
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    except ValueError:
        pass
    text = (response.text or "").strip()
    return text or str(exc)


def _raise_for_status_with_detail(response: requests.Response) -> None:
    """在流式请求场景下保留后端 detail，避免只显示笼统的 HTTPError。"""
    if response.status_code < 400:
        return

    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            maybe_detail = payload.get("detail")
            if isinstance(maybe_detail, str):
                detail = maybe_detail.strip()
    except ValueError:
        pass

    if not detail:
        detail = (response.text or "").strip()

    message = detail or f"HTTP {response.status_code} {response.reason}"
    raise requests.HTTPError(message, response=response, request=response.request)


def _is_invalid_llm_session_error(detail: str) -> bool:
    """判断是否为后端返回的 LLM 会话失效错误。"""
    normalized = detail.strip()
    return "LLM 会话不存在、已过期或不属于当前用户" in normalized


def _is_no_pending_interrupt_error(detail: str) -> bool:
    """判断是否为“当前没有待审批工具调用”这类可恢复状态。"""
    normalized = detail.strip()
    return "当前没有待审批的工具调用" in normalized or "未找到待审批的工具调用明细" in normalized


def _sanitize_stream_token(token: str) -> str:
    """清理模型流式 token 中的 DSML/函数调用控制标记。"""
    normalized = token.replace("\r", "")
    cleaned = DSML_CONTROL_TOKEN_PATTERN.sub("", normalized)
    cleaned = CONTROL_EMAIL_ARG_PATTERN.sub("", cleaned)
    cleaned = CONTROL_CONTENT_ARG_OPEN_PATTERN.sub("", cleaned)
    cleaned = CONTROL_MARKUP_FRAGMENT_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"</?\s*>", "", cleaned)
    lowered = cleaned.lower()
    if any(keyword in lowered for keyword in CONTROL_MARKUP_KEYWORDS):
        if ("<" in cleaned) or (">" in cleaned) or ("string=" in lowered):
            return ""
    return cleaned


def _sanitize_answer_text(answer: str) -> str:
    """对完整回答再次做控制标记清理。"""
    if not answer:
        return answer
    cleaned = DSML_CONTROL_TOKEN_PATTERN.sub("", answer)
    cleaned = CONTROL_EMAIL_ARG_PATTERN.sub("", cleaned)
    cleaned = CONTROL_CONTENT_ARG_OPEN_PATTERN.sub("", cleaned)
    cleaned = CONTROL_MARKUP_FRAGMENT_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"</?\s*>", "", cleaned)
    cleaned_lines: list[str] = []
    for line in cleaned.splitlines():
        normalized_line = line.strip()
        lowered = normalized_line.lower()
        if normalized_line and any(keyword in lowered for keyword in CONTROL_MARKUP_KEYWORDS):
            if ("<" in normalized_line) or (">" in normalized_line) or ("string=" in lowered):
                continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _iter_sse_events(response: requests.Response) -> Iterator[dict[str, Any]]:
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:") :].strip()
        if not data_str:
            continue
        if data_str == "[DONE]":
            yield {"type": "done"}
            break
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            yield {"type": "error", "message": f"无法解析 SSE 数据：{data_str[:200]}"}
            continue
        yield event if isinstance(event, dict) else {"type": "error", "message": f"收到非字典事件：{event}"}


def _active_session_id() -> str | None:
    value = st.session_state.get("llm_session_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def stream_chat_api(user_id: str, query: str, session_id: str | None = None,thread_id: str | None = None) -> Iterator[dict[str, Any]]:
    payload: dict[str, Any] = {"user_id": user_id, "query": query}
    if session_id and session_id.strip():
        payload["session_id"] = session_id.strip()
    if thread_id and thread_id.strip():
        payload["thread_id"] = thread_id.strip()
    with requests.post(
        CHAT_ENDPOINT,
        json=payload,
        headers=_request_headers(sse=True),
        timeout=(10, 600),
        stream=True,
    ) as response:
        _raise_for_status_with_detail(response)
        yield from _iter_sse_events(response)


def stream_resume_api(user_id: str, action: str, session_id: str | None = None, thread_id: str | None = None) -> Iterator[dict[str, Any]]:
    payload: dict[str, Any] = {"user_id": user_id, "action": action}
    if session_id and session_id.strip():
        payload["session_id"] = session_id.strip()
    if thread_id and thread_id.strip():
        payload["thread_id"] = thread_id.strip()
    with requests.post(
        CHAT_RESUME_ENDPOINT,
        json=payload,
        headers=_request_headers(sse=True),
        timeout=(10, 600),
        stream=True,
    ) as response:
        _raise_for_status_with_detail(response)
        yield from _iter_sse_events(response)


def save_memory_api(user_id: str, preference_text: str, session_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"user_id": user_id, "preference_text": preference_text}
    if session_id and session_id.strip():
        payload["session_id"] = session_id.strip()
    response = requests.post(MEMORY_ENDPOINT, json=payload, headers=_request_headers(), timeout=30)
    response.raise_for_status()
    return response.json()


def list_memory_api(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    endpoint = MEMORY_LIST_ENDPOINT_TEMPLATE.format(user_id=_quote_path(user_id))
    response = requests.get(endpoint, params={"limit": limit}, headers=_request_headers(), timeout=30)
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items")
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def delete_memory_api(user_id: str, memory_id: str) -> dict[str, Any]:
    endpoint = MEMORY_DELETE_ENDPOINT_TEMPLATE.format(user_id=_quote_path(user_id), memory_id=_quote_path(memory_id))
    response = requests.delete(endpoint, headers=_request_headers(), timeout=30)
    response.raise_for_status()
    return response.json()


def list_llm_providers_api() -> list[dict[str, Any]]:
    response = requests.get(LLM_PROVIDER_ENDPOINT, headers=_request_headers(), timeout=20)
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items")
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def validate_llm_session_api(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(LLM_VALIDATE_ENDPOINT, json=payload, headers=_request_headers(), timeout=90)
    response.raise_for_status()
    return response.json()


def create_llm_session_api(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(LLM_SESSION_CREATE_ENDPOINT, json=payload, headers=_request_headers(), timeout=30)
    response.raise_for_status()
    return response.json()


def delete_llm_session_api(session_id: str) -> dict[str, Any]:
    endpoint = LLM_SESSION_DELETE_ENDPOINT_TEMPLATE.format(session_id=_quote_path(session_id))
    response = requests.delete(endpoint, headers=_request_headers(), timeout=30)
    response.raise_for_status()
    return response.json()

def init_session_state() -> None:
    default_user_id = AGENT_TOKEN_SUB or "user_001"
    import uuid
    default_conversation_id = f"{default_user_id}_conv_{str(uuid.uuid4())[:8]}"
    if "user_id" not in st.session_state:
        st.session_state.user_id = default_user_id
    elif AGENT_TOKEN_SUB:
        st.session_state.user_id = AGENT_TOKEN_SUB
    if "conversations" not in st.session_state:
        # Initialize conversations dictionary
        st.session_state.conversations = {f"{default_conversation_id}": []}
        # Only initialize with thread IDs, not full content yet
        st.session_state.thread_ids = []
        # Load thread IDs for the user (but not the content yet)
        try:
            thread_ids = get_user_threads_api(st.session_state.user_id)
            if thread_ids:
                st.session_state.thread_ids = thread_ids
                for thread_id in thread_ids:
                    st.session_state.conversations[thread_id] = []
                st.success(f"发现 {len(thread_ids)} 个历史对话，点击对话ID加载内容")
        except Exception as e:
            st.warning(f"加载历史对话ID时出现错误: {e}")
            st.session_state.thread_ids = []
    if "current_conversation_id" not in st.session_state:
        # 当前激活的对话ID
        st.session_state.current_conversation_id = default_conversation_id
    if "pending_interrupt" not in st.session_state:
        st.session_state.pending_interrupt = False
    if "pending_tool_calls" not in st.session_state:
        st.session_state.pending_tool_calls = []
    if "approval_in_progress" not in st.session_state:
        st.session_state.approval_in_progress = False
    if "deferred_approval_action" not in st.session_state:
        st.session_state.deferred_approval_action = ""
    if "approval_last_click_ts" not in st.session_state:
        st.session_state.approval_last_click_ts = 0.0
    if "chat_input_text" not in st.session_state:
        st.session_state.chat_input_text = ""
    if "clear_chat_input_next_run" not in st.session_state:
        st.session_state.clear_chat_input_next_run = False
    if "chat_generating" not in st.session_state:
        st.session_state.chat_generating = False
    if "chat_stop_requested" not in st.session_state:
        st.session_state.chat_stop_requested = False
    if "chat_stream_answer" not in st.session_state:
        st.session_state.chat_stream_answer = ""
    if "chat_stream_tool_events" not in st.session_state:
        st.session_state.chat_stream_tool_events = []
    if "chat_stream_interrupted" not in st.session_state:
        st.session_state.chat_stream_interrupted = False
    if "chat_stream_saw_error" not in st.session_state:
        st.session_state.chat_stream_saw_error = False
    if "chat_stream_agent_tip" not in st.session_state:
        st.session_state.chat_stream_agent_tip = ""
    if "chat_stream_user_query" not in st.session_state:
        st.session_state.chat_stream_user_query = ""
    if "chat_stream_worker_done" not in st.session_state:
        st.session_state.chat_stream_worker_done = False
    if "chat_event_queue" not in st.session_state:
        st.session_state.chat_event_queue = None
    if "chat_stop_event" not in st.session_state:
        st.session_state.chat_stop_event = None
    if "chat_worker_thread" not in st.session_state:
        st.session_state.chat_worker_thread = None
    if "memory_items_cache" not in st.session_state:
        st.session_state.memory_items_cache = []
    if "memory_cache_user_id" not in st.session_state:
        st.session_state.memory_cache_user_id = ""
    if "memory_cache_error" not in st.session_state:
        st.session_state.memory_cache_error = ""
    if "memory_cache_loaded" not in st.session_state:
        st.session_state.memory_cache_loaded = False
    if "memory_refresh_needed" not in st.session_state:
        st.session_state.memory_refresh_needed = True

    if "llm_provider_catalog" not in st.session_state:
        st.session_state.llm_provider_catalog = [dict(item) for item in FALLBACK_PROVIDER_ITEMS]
    if "llm_provider_catalog_error" not in st.session_state:
        st.session_state.llm_provider_catalog_error = ""
    if "llm_provider_catalog_loaded" not in st.session_state:
        st.session_state.llm_provider_catalog_loaded = False

    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "qwen"
    if "llm_model_pick" not in st.session_state:
        st.session_state.llm_model_pick = "qwen3.5-plus"
    if "llm_model_custom" not in st.session_state:
        st.session_state.llm_model_custom = ""
    if "llm_api_key_input" not in st.session_state:
        st.session_state.llm_api_key_input = ""
    if "llm_base_url_input" not in st.session_state:
        st.session_state.llm_base_url_input = ""
    if "llm_embedding_model_input" not in st.session_state:
        st.session_state.llm_embedding_model_input = ""
    if "llm_session_id" not in st.session_state:
        st.session_state.llm_session_id = None
    if "llm_session_meta" not in st.session_state:
        st.session_state.llm_session_meta = {}
    if "llm_validation_result" not in st.session_state:
        st.session_state.llm_validation_result = None
    if "clear_llm_api_key_next_run" not in st.session_state:
        st.session_state.clear_llm_api_key_next_run = False
    if "loaded_conversations" not in st.session_state:
        # 跟踪哪些对话已被加载
        st.session_state.loaded_conversations = set()

def load_conversation_content(conversation_id: str) -> None:
    """按需加载对话内容"""
    if conversation_id in st.session_state.loaded_conversations:
        # Already loaded, nothing to do
        return
    
    # Check if this is a historical thread that needs loading
    if conversation_id in st.session_state.thread_ids:
        # This is a historical thread, load its content
        history = get_chat_history_api(conversation_id, st.session_state.user_id)
        if history:
            st.session_state.conversations[conversation_id] = history
            st.session_state.loaded_conversations.add(conversation_id)
            st.success(f"已加载对话内容: {conversation_id[:12]}...")
    else:
        # This is a new conversation, initialize with empty list
        st.session_state.conversations[conversation_id] = []
        st.session_state.loaded_conversations.add(conversation_id)

def switch_conversation(conversation_id: str) -> None:
    """切换到指定对话"""
    # Load content if not already loaded
    load_conversation_content(conversation_id)
    
    st.session_state.current_conversation_id = conversation_id
    st.session_state.messages = st.session_state.conversations[conversation_id]


def new_conversation() -> None:
    """创建新对话"""
    import uuid
    conversation_id = f"conv_{str(uuid.uuid4())[:8]}"
    st.session_state.conversations[conversation_id] = []
    switch_conversation(conversation_id)


def delete_conversation(conversation_id: str) -> None:
    """删除指定对话"""
    if conversation_id in st.session_state.conversations:
        del st.session_state.conversations[conversation_id]
        # 如果删除的是当前对话，则切换到第一个可用对话
        if st.session_state.current_conversation_id == conversation_id:
            available_convs = list(st.session_state.conversations.keys())
            if available_convs:
                switch_conversation(available_convs[0])
            else:
                new_conversation()


def _normalize_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    return [item for item in raw_tool_calls if isinstance(item, dict)] if isinstance(raw_tool_calls, list) else []


def _set_pending_interrupt(tool_calls: list[dict[str, Any]]) -> None:
    st.session_state.pending_interrupt = True
    st.session_state.pending_tool_calls = tool_calls


def _clear_pending_interrupt() -> None:
    st.session_state.pending_interrupt = False
    st.session_state.pending_tool_calls = []


def _approval_cooldown_remaining() -> float:
    """返回审批按钮剩余冷却秒数。"""
    try:
        last_click = float(st.session_state.get("approval_last_click_ts", 0.0) or 0.0)
    except (TypeError, ValueError):
        last_click = 0.0
    remaining = APPROVAL_CLICK_COOLDOWN_SECONDS - (time.monotonic() - last_click)
    return remaining if remaining > 0 else 0.0


def _clear_active_session() -> None:
    st.session_state.llm_session_id = None
    st.session_state.llm_session_meta = {}


def _mark_memory_refresh_needed() -> None:
    st.session_state.memory_refresh_needed = True


def _load_memory_items(user_id: str, *, force_refresh: bool = False) -> list[dict[str, Any]]:
    """按需加载记忆列表，避免流式轮询期间高频请求后端。"""
    normalized_user_id = user_id.strip()
    cache_user_id = str(st.session_state.get("memory_cache_user_id", "")).strip()
    cache_loaded = bool(st.session_state.get("memory_cache_loaded", False))
    refresh_needed = bool(st.session_state.get("memory_refresh_needed", False))

    should_refresh = (
        force_refresh
        or refresh_needed
        or (not cache_loaded)
        or (cache_user_id != normalized_user_id)
    )

    if not should_refresh:
        cached_items = st.session_state.get("memory_items_cache", [])
        return [item for item in cached_items if isinstance(item, dict)] if isinstance(cached_items, list) else []

    try:
        items = list_memory_api(normalized_user_id, limit=80)
        st.session_state.memory_items_cache = items
        st.session_state.memory_cache_user_id = normalized_user_id
        st.session_state.memory_cache_error = ""
        st.session_state.memory_cache_loaded = True
        st.session_state.memory_refresh_needed = False
        return items
    except requests.RequestException as exc:
        st.session_state.memory_cache_error = _request_error_detail(exc)
    except ValueError as exc:
        st.session_state.memory_cache_error = str(exc)

    st.session_state.memory_cache_loaded = True
    if cache_user_id != normalized_user_id:
        st.session_state.memory_items_cache = []
        st.session_state.memory_cache_user_id = normalized_user_id

    cached_items = st.session_state.get("memory_items_cache", [])
    return [item for item in cached_items if isinstance(item, dict)] if isinstance(cached_items, list) else []


def _reset_chat_stream_runtime(*, keep_input: bool = True) -> None:
    """重置前端流式生成运行态。"""
    st.session_state.chat_generating = False
    st.session_state.chat_stop_requested = False
    st.session_state.chat_stream_answer = ""
    st.session_state.chat_stream_tool_events = []
    st.session_state.chat_stream_interrupted = False
    st.session_state.chat_stream_saw_error = False
    st.session_state.chat_stream_agent_tip = ""
    st.session_state.chat_stream_worker_done = False
    st.session_state.chat_event_queue = None
    st.session_state.chat_stop_event = None
    st.session_state.chat_worker_thread = None
    if not keep_input:
        st.session_state.clear_chat_input_next_run = True


def _chat_stream_worker(
    user_id: str,
    query: str,
    session_id: str | None,
    thread_id: str | None,
    event_queue: "queue.Queue[dict[str, Any]]",
    stop_event: threading.Event,
) -> None:
    """后台线程：消费后端 SSE 并投递到队列。"""
    payload: dict[str, Any] = {"user_id": user_id, "query": query}
    if session_id and session_id.strip():
        payload["session_id"] = session_id.strip()
    if thread_id and thread_id.strip():
        payload["thread_id"] = thread_id.strip()

    stopped = False
    try:
        with requests.post(
            CHAT_ENDPOINT,
            json=payload,
            headers=_request_headers(sse=True),
            timeout=(10, 600),
            stream=True,
        ) as response:
            _raise_for_status_with_detail(response)
            for event in _iter_sse_events(response):
                if stop_event.is_set():
                    stopped = True
                    break
                event_queue.put(event)
    except requests.RequestException as exc:
        event_queue.put({"type": "error", "message": _request_error_detail(exc)})
    except Exception as exc:  # noqa: BLE001
        event_queue.put({"type": "error", "message": f"流式处理出现异常：{exc}"})
    finally:
        if stop_event.is_set() or stopped:
            event_queue.put({"type": "stopped"})
        event_queue.put({"type": "worker_done"})


def _start_chat_stream(user_id: str, query: str) -> None:
    """启动后台流式生成线程。"""
    import uuid
    thread_id = st.session_state.current_conversation_id   
    event_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_chat_stream_worker,
        args=(user_id, query, _active_session_id(), thread_id,event_queue, stop_event),
        daemon=True,
        name="nanoagent_chat_stream",
    )

    st.session_state.chat_generating = True
    st.session_state.chat_stop_requested = False
    st.session_state.chat_stream_answer = ""
    st.session_state.chat_stream_tool_events = []
    st.session_state.chat_stream_interrupted = False
    st.session_state.chat_stream_saw_error = False
    st.session_state.chat_stream_agent_tip = ""
    st.session_state.chat_stream_worker_done = False
    st.session_state.chat_stream_user_query = query
    st.session_state.chat_event_queue = event_queue
    st.session_state.chat_stop_event = stop_event
    st.session_state.chat_worker_thread = worker

    worker.start()


def _finalize_chat_stream() -> None:
    """将后台流式结果收敛为一条 assistant 消息。"""
    full_answer = _sanitize_answer_text(str(st.session_state.get("chat_stream_answer", "")))
    tool_events = _normalize_tool_calls(st.session_state.get("chat_stream_tool_events"))
    interrupted = bool(st.session_state.get("chat_stream_interrupted", False))
    stopped = bool(st.session_state.get("chat_stop_requested", False))
    saw_error = bool(st.session_state.get("chat_stream_saw_error", False))

    if not full_answer:
        if interrupted:
            full_answer = "操作已暂停，等待人工审批。"
        elif stopped:
            full_answer = "已暂停生成。"
        elif saw_error:
            full_answer = "请求失败，请稍后重试。"
        else:
            full_answer = "后端未返回内容。"

    # 将助手回复添加到当前对话
    current_conv_id = st.session_state.current_conversation_id
    if current_conv_id not in st.session_state.conversations:
        st.session_state.conversations[current_conv_id] = []
    st.session_state.conversations[current_conv_id].append(
        {"role": "assistant", "content": full_answer, "tool_calls": tool_events or None}
    )
    
    # 同步到messages变量
    st.session_state.messages = st.session_state.conversations[current_conv_id]

    if not interrupted:
        _clear_pending_interrupt()

    _reset_chat_stream_runtime(keep_input=True)


def _process_chat_stream_events() -> None:
    """处理后台线程投递的 SSE 事件。"""
    if not bool(st.session_state.get("chat_generating", False)):
        return

    event_queue = st.session_state.get("chat_event_queue")
    if not isinstance(event_queue, queue.Queue):
        st.session_state.chat_stream_saw_error = True
        st.session_state.chat_stream_answer = "流式队列异常，请重试。"
        st.session_state.chat_stream_worker_done = True
        _finalize_chat_stream()
        return

    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            break

        event_type = str(event.get("type", "")).strip()
        if event_type == "agent_switch":
            tip = _agent_tip(str(event.get("agent", "")))
            st.session_state.chat_stream_agent_tip = tip
            events = _normalize_tool_calls(st.session_state.get("chat_stream_tool_events"))
            events.append(event)
            st.session_state.chat_stream_tool_events = events
        elif event_type == "token":
            token = _sanitize_stream_token(str(event.get("content", "")))
            if token:
                st.session_state.chat_stream_answer = str(st.session_state.get("chat_stream_answer", "")) + token
        elif event_type in {"tool_start", "tool_end"}:
            events = _normalize_tool_calls(st.session_state.get("chat_stream_tool_events"))
            events.append(event)
            st.session_state.chat_stream_tool_events = events
        elif event_type == "interrupt":
            calls = _normalize_tool_calls(event.get("tool_calls"))
            _set_pending_interrupt(calls)
            events = _normalize_tool_calls(st.session_state.get("chat_stream_tool_events"))
            events.append({"type": "interrupt", "tool_calls": calls})
            st.session_state.chat_stream_tool_events = events
            st.session_state.chat_stream_interrupted = True
            if not str(st.session_state.get("chat_stream_answer", "")).strip():
                st.session_state.chat_stream_answer = "操作已暂停，等待人工审批。"
        elif event_type == "error":
            st.session_state.chat_stream_saw_error = True
            if not str(st.session_state.get("chat_stream_answer", "")).strip():
                error_message = str(event.get("message", "流式处理出现错误")).strip() or "流式处理出现错误"
                if _is_invalid_llm_session_error(error_message):
                    _clear_active_session()
                st.session_state.chat_stream_answer = f"请求失败：{error_message}"
        elif event_type in {"done", "stopped", "worker_done"}:
            st.session_state.chat_stream_worker_done = True

    worker = st.session_state.get("chat_worker_thread")
    if isinstance(worker, threading.Thread) and not worker.is_alive():
        st.session_state.chat_stream_worker_done = True

    if bool(st.session_state.get("chat_stream_worker_done", False)):
        _finalize_chat_stream()
        # 关键修复：本轮 render_chat_history 已执行过，finalize 后需要立刻 rerun
        # 才能把刚追加的 assistant 消息显示出来，避免“必须再点一次发送才看到”。
        rerun_app()
        return


def _render_inflight_assistant_message() -> None:
    """渲染当前正在生成中的 assistant 临时消息。"""
    if not bool(st.session_state.get("chat_generating", False)):
        return

    tip = str(st.session_state.get("chat_stream_agent_tip", "")).strip()
    answer = _sanitize_answer_text(str(st.session_state.get("chat_stream_answer", "")))
    tool_events = _normalize_tool_calls(st.session_state.get("chat_stream_tool_events"))

    with st.chat_message("assistant"):
        if tip:
            st.info(tip)
        st.markdown(answer or "正在生成中...")
        if tool_events:
            tool_placeholder = st.empty()
            _render_tool_events(tool_placeholder, tool_events)


def _provider_map(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for item in items:
        provider = str(item.get("provider", "")).strip()
        if provider:
            mapped[provider] = item
    return mapped


def _load_provider_catalog(force_refresh: bool = False) -> list[dict[str, Any]]:
    if st.session_state.llm_provider_catalog_loaded and not force_refresh:
        items = st.session_state.llm_provider_catalog
        if isinstance(items, list) and items:
            return items

    try:
        items = list_llm_providers_api()
        if items:
            st.session_state.llm_provider_catalog = items
            st.session_state.llm_provider_catalog_error = ""
            st.session_state.llm_provider_catalog_loaded = True
            return items
    except requests.RequestException as exc:
        st.session_state.llm_provider_catalog_error = _request_error_detail(exc)
    except ValueError as exc:
        st.session_state.llm_provider_catalog_error = str(exc)

    fallback = st.session_state.llm_provider_catalog
    if not isinstance(fallback, list) or not fallback:
        fallback = [dict(item) for item in FALLBACK_PROVIDER_ITEMS]
        st.session_state.llm_provider_catalog = fallback
    st.session_state.llm_provider_catalog_loaded = True
    return fallback


def _default_model_options_for_provider(provider: str) -> list[str]:
    options = list(MODEL_PRESETS.get(provider, []))
    if "自定义输入" not in options:
        options.append("自定义输入")
    return options


def _resolve_selected_model_name(provider: str) -> str:
    model_pick = str(st.session_state.get("llm_model_pick", "")).strip()
    if model_pick and model_pick != "自定义输入":
        return model_pick
    custom = str(st.session_state.get("llm_model_custom", "")).strip()
    if custom:
        return custom
    defaults = MODEL_PRESETS.get(provider, [])
    return defaults[0] if defaults else ""


def _build_llm_payload(
    *,
    provider: str,
    provider_info: dict[str, Any],
    model_name: str,
    api_key: str,
    embedding_model: str,
    ttl_seconds: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": provider,
        "api_key": api_key,
        "model": model_name,
        "ttl_seconds": ttl_seconds,
        # 统一策略：留空即禁用 Embedding，不再按 provider 自动补默认值。
        "embedding_model": embedding_model.strip(),
    }
    requires_base_url = bool(provider_info.get("requires_base_url", provider == "other"))
    if requires_base_url:
        payload["base_url"] = str(st.session_state.get("llm_base_url_input", "")).strip()
    return payload


def _render_validation_result(result: dict[str, Any]) -> None:
    chat_ok = bool(result.get("chat_ok", False))
    embedding_ok = bool(result.get("embedding_ok", False))
    errors = result.get("errors") if isinstance(result.get("errors"), list) else []
    embedding_enabled = bool(str(result.get("embedding_model", "")).strip())

    if chat_ok and (embedding_ok or not embedding_enabled):
        if embedding_enabled:
            st.success("连接校验通过。")
        else:
            st.success("连接校验通过（Embedding 未启用）。")
    else:
        st.warning("连接校验未通过，请检查配置。")

    if errors:
        st.caption("；".join(str(item) for item in errors[:2]))

def render_ai_config_panel() -> None:
    # Streamlit 不允许在同一轮中修改已实例化组件的 session key。
    # 因此在新一轮渲染开始时，先处理上轮设置的清空标记。
    if bool(st.session_state.get("clear_llm_api_key_next_run", False)):
        st.session_state.llm_api_key_input = ""
        st.session_state.clear_llm_api_key_next_run = False

    refresh_col, _ = st.columns([1, 1])
    force_refresh = refresh_col.button("刷新模型列表", use_container_width=True)
    provider_items = _load_provider_catalog(force_refresh=force_refresh)

    provider_error = str(st.session_state.get("llm_provider_catalog_error", "")).strip()
    if provider_error:
        st.caption(f"提示：模型列表刷新失败，已回退本地默认配置。原因：{provider_error}")

    provider_map = _provider_map(provider_items)
    provider_options = list(provider_map.keys())
    if not provider_options:
        provider_options = ["qwen"]
        provider_map = _provider_map([dict(item) for item in FALLBACK_PROVIDER_ITEMS])

    current_provider = str(st.session_state.get("llm_provider", "qwen")).strip()
    if current_provider not in provider_options:
        st.session_state.llm_provider = provider_options[0]

    st.selectbox(
        "模型提供商",
        options=provider_options,
        key="llm_provider",
        format_func=lambda p: f"{provider_map.get(p, {}).get('label', p)} ({p})",
    )

    selected_provider = str(st.session_state.get("llm_provider", provider_options[0]))
    provider_info = provider_map.get(selected_provider, {})

    model_options = _default_model_options_for_provider(selected_provider)
    if str(st.session_state.get("llm_model_pick", "")).strip() not in model_options:
        st.session_state.llm_model_pick = model_options[0]

    st.selectbox(
        "模型名称",
        options=model_options,
        key="llm_model_pick",
        help="可直接选择常用模型；若选择“自定义输入”，可手填模型名。",
    )

    if st.session_state.llm_model_pick == "自定义输入":
        st.text_input("自定义模型名称", key="llm_model_custom", placeholder="例如：qwen3.5-plus")

    st.text_input("API Key", key="llm_api_key_input", type="password", placeholder="请输入你自己的 API Key")

    requires_base_url = bool(provider_info.get("requires_base_url", selected_provider == "other"))
    default_base_url = provider_info.get("default_base_url")
    if requires_base_url:
        st.text_input(
            "Base URL",
            key="llm_base_url_input",
            placeholder="例如：https://your-provider.example.com/v1",
        )

    embedding_placeholder = str(provider_info.get("default_embedding_model", "")).strip()
    st.text_input(
        "Embedding 模型（可选）",
        key="llm_embedding_model_input",
        placeholder=embedding_placeholder or "留空表示不启用 Embedding",
    )

    validate_col, activate_col = st.columns(2)
    validate_clicked = validate_col.button("验证连接", use_container_width=True)
    activate_clicked = activate_col.button("保存并启用", use_container_width=True)

    model_name = _resolve_selected_model_name(selected_provider)
    api_key = str(st.session_state.get("llm_api_key_input", "")).strip()
    embedding_model = str(st.session_state.get("llm_embedding_model_input", "")).strip()

    if validate_clicked or activate_clicked:
        if not model_name:
            st.error("请先填写模型名称。")
            return
        if not api_key:
            st.error("请先填写 API Key。")
            return
        if requires_base_url and not str(st.session_state.get("llm_base_url_input", "")).strip():
            st.error("当前 provider 需要填写 Base URL。")
            return

    if validate_clicked:
        try:
            payload = _build_llm_payload(
                provider=selected_provider,
                provider_info=provider_info,
                model_name=model_name,
                api_key=api_key,
                embedding_model=embedding_model,
                ttl_seconds=3600,
            )
            payload.pop("ttl_seconds", None)
            payload["validate_chat"] = True
            payload["validate_embedding"] = bool(embedding_model.strip())
            result = validate_llm_session_api(payload)
            st.session_state.llm_validation_result = result
            _render_validation_result(result)
        except requests.RequestException as exc:
            st.error(f"调用 /api/v1/session/llm/validate 失败：{_request_error_detail(exc)}")
        except ValueError as exc:
            st.error(f"解析校验结果失败：{exc}")

    if activate_clicked:
        try:
            payload = _build_llm_payload(
                provider=selected_provider,
                provider_info=provider_info,
                model_name=model_name,
                api_key=api_key,
                embedding_model=embedding_model,
                ttl_seconds=3600,
            )
            result = create_llm_session_api(payload)
            old_session_id = _active_session_id()
            new_session_id = str(result.get("session_id", "")).strip()
            if not new_session_id:
                st.error("创建会话失败：后端未返回 session_id")
                return

            st.session_state.llm_session_id = new_session_id
            st.session_state.llm_session_meta = {
                "provider": result.get("provider", selected_provider),
                "model": result.get("model", model_name),
                "base_url": result.get("base_url", default_base_url),
                "embedding_model": result.get("embedding_model", embedding_placeholder),
                "expires_in": result.get("expires_in"),
            }
            # 会话创建后下一轮再清空输入框中的敏感信息，避免组件 key 冲突。
            st.session_state.clear_llm_api_key_next_run = True

            if old_session_id and old_session_id != new_session_id:
                try:
                    delete_llm_session_api(old_session_id)
                except requests.RequestException:
                    pass

            st.success("AI 会话已启用，后续聊天与记忆写入将使用该配置。")
            rerun_app()
        except requests.RequestException as exc:
            st.error(f"调用 /api/v1/session/llm 失败：{_request_error_detail(exc)}")
        except ValueError as exc:
            st.error(f"解析创建会话结果失败：{exc}")

    validation_result = st.session_state.get("llm_validation_result")
    if isinstance(validation_result, dict) and not validate_clicked:
        _render_validation_result(validation_result)

    session_id = _active_session_id()
    if session_id:
        meta = st.session_state.get("llm_session_meta", {})
        provider = meta.get("provider", "unknown")
        model = meta.get("model", "unknown")
        expires_in = meta.get("expires_in")
        st.success(f"当前会话已启用：`{session_id}`")
        st.caption(f"Provider={provider} | Model={model} | TTL={expires_in}")

        if st.button("停用当前会话", use_container_width=True):
            try:
                delete_llm_session_api(session_id)
            except requests.RequestException as exc:
                st.warning(f"后端删除会话失败（将本地清理）：{_request_error_detail(exc)}")
            _clear_active_session()
            rerun_app()
    else:
        st.caption("当前未启用会话：将使用后端默认模型配置（若后端已设置）。")

def render_sidebar() -> str:
    with st.sidebar:
        st.title("NanoAgent 控制台")
        if AGENT_TOKEN_SUB:
            st.text_input("用户 ID（由登录令牌决定）", value=AGENT_TOKEN_SUB, disabled=True)
            st.session_state.user_id = AGENT_TOKEN_SUB
            st.caption("后端会以 JWT sub 作为真实用户身份。")
        else:
            user_id = st.text_input("用户 ID", value=st.session_state.user_id)
            st.session_state.user_id = user_id.strip() or "user_001"
            st.caption("当前为手动用户 ID 模式（仅用于本地调试）。")

        st.markdown("---")
        with st.expander("对话管理", expanded=True):
            # 对话管理UI
            st.write("**对话管理**")
            
            # 创建新对话按钮
            if st.button("新建对话", use_container_width=True):
                new_conversation()
                st.rerun()
            
            # 显示对话列表
            conversation_ids = list(st.session_state.conversations.keys())
            if len(conversation_ids) > 1:
                # 如果有多个对话，提供切换选项
                conversation_names = []
                for cid in conversation_ids:
                    # 为每个对话生成简单名称
                    if cid == "default":
                        name = "默认对话"
                    else:
                        name = f"对话 {cid.split('_')[-1][-4:]}"
                    conversation_names.append(name)
                
                # 获取当前对话索引
                current_idx = conversation_ids.index(st.session_state.current_conversation_id)
                
                selected_idx = st.selectbox(
                    "选择对话", 
                    options=range(len(conversation_ids)),
                    format_func=lambda x: conversation_names[x],
                    index=current_idx
                )
                
                if selected_idx != current_idx:
                    switch_conversation(conversation_ids[selected_idx])
                    st.rerun()
            else:
                st.info(f"当前对话: 对话 {'default' if st.session_state.current_conversation_id == 'default' else st.session_state.current_conversation_id.split('_')[-1][-4:]}")
            
            # 删除当前对话按钮
            if len(conversation_ids) > 1:
                if st.button("删除当前对话", use_container_width=True, type="secondary"):
                    delete_conversation(st.session_state.current_conversation_id)
                    st.rerun()

        st.markdown("---")
        with st.expander("AI 配置", expanded=False):
            render_ai_config_panel()

        st.markdown("---")
        with st.expander("写入长期记忆", expanded=False):
            with st.form("memory_form", clear_on_submit=True):
                preference_text = st.text_area("偏好内容", placeholder="例如：我是一名计算机专业学生。", height=120)
                submitted = st.form_submit_button("保存偏好")
                if submitted:
                    text = preference_text.strip()
                    if not text:
                        st.warning("提交前请先输入偏好内容。")
                    else:
                        try:
                            result = save_memory_api(st.session_state.user_id, text, session_id=_active_session_id())
                            memory_id = result.get("memory_id")
                            st.success(f"已保存，memory_id: {memory_id}" if memory_id else result.get("message", "已保存。"))
                            _mark_memory_refresh_needed()
                            rerun_app()
                        except requests.RequestException as exc:
                            st.error(f"调用 /api/v1/memory 失败：{_request_error_detail(exc)}")
                        except ValueError as exc:
                            st.error(f"解析后端响应失败：{exc}")

        st.markdown("---")
        with st.expander("长期记忆管理", expanded=False):
            refresh_clicked = st.button("刷新记忆列表", use_container_width=True)
            if refresh_clicked:
                _mark_memory_refresh_needed()

            memory_items = _load_memory_items(st.session_state.user_id, force_refresh=refresh_clicked)
            memory_error = str(st.session_state.get("memory_cache_error", "")).strip()
            if memory_error:
                st.error(f"获取记忆列表失败：{memory_error}")
            if not memory_items:
                st.caption("当前暂无长期记忆。")
            else:
                for index, item in enumerate(memory_items, start=1):
                    memory_id = str(item.get("memory_id", "")).strip()
                    text = str(item.get("preference_text", "")).strip()
                    display_text = text or "(空内容)"
                    st.markdown(display_text)

                    delete_key = f"delete_{memory_id}" if memory_id else f"delete_fallback_{index}"
                    if memory_id and st.button("删除这条记忆", key=delete_key):
                        try:
                            result = delete_memory_api(st.session_state.user_id, memory_id)
                            st.success(result.get("message", "删除成功。"))
                            _mark_memory_refresh_needed()
                            rerun_app()
                        except requests.RequestException as exc:
                            st.error(f"删除失败：{_request_error_detail(exc)}")
                        except ValueError as exc:
                            st.error(f"解析删除结果失败：{exc}")
                    st.divider()

    return st.session_state.user_id


def render_chat_history() -> None:
    # 确保当前对话的消息被同步到messages变量
    if "conversations" not in st.session_state:
        # 如果对话系统未初始化，则初始化
        st.session_state.conversations = {"default": []}
        st.session_state.current_conversation_id = "default"
    
    if st.session_state.current_conversation_id in st.session_state.conversations:
        st.session_state.messages = st.session_state.conversations[st.session_state.current_conversation_id]
    else:
        st.session_state.conversations[st.session_state.current_conversation_id] = []
        st.session_state.messages = []
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        tool_calls = message.get("tool_calls")
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant" and tool_calls:
                with st.expander("Agent 思维链 / 工具调用"):
                    st.json(tool_calls)


def _render_tool_events(tool_placeholder: Any, tool_events: list[dict[str, Any]]) -> None:
    if not tool_events:
        return
    with tool_placeholder.container():
        with st.expander("Agent 思维链 / 工具调用", expanded=True):
            for event in tool_events:
                event_type = event.get("type")
                if event_type == "tool_start":
                    tool_name = event.get("tool", "unknown_tool")
                    st.markdown(f"- 正在调用工具：`{tool_name}`")
                    if event.get("input") is not None:
                        st.code(json.dumps(event.get("input"), ensure_ascii=False, indent=2, default=str), language="json")
                elif event_type == "tool_end":
                    tool_name = event.get("tool", "unknown_tool")
                    st.markdown(f"- 工具调用完成：`{tool_name}`")
                    if event.get("output") is not None:
                        st.code(json.dumps(event.get("output"), ensure_ascii=False, indent=2, default=str), language="json")
                elif event_type == "agent_switch":
                    st.markdown(f"- 节点切换：`{event.get('agent', 'unknown_node')}`")
                elif event_type == "interrupt":
                    st.markdown("- 已触发人工审批拦截")
                    calls = _normalize_tool_calls(event.get("tool_calls"))
                    if calls:
                        st.json(calls)
                elif event_type == "error":
                    st.error(str(event.get("message", "流式处理出现错误")))


def _agent_tip(agent: str) -> str:
    if agent == "supervisor_node":
        return "主管节点正在思考..."
    if agent == "knowledge_worker_node":
        return "数据科学家正在处理..."
    if agent == "reporter_node":
        return "报告专家正在处理..."
    if agent == "assistant_node":
        return "Assistant 正在处理一般对话..."
    return f"节点切换：{agent}"


def _consume_stream_events(
    event_iter: Iterator[dict[str, Any]],
    *,
    answer_placeholder: Any,
    tool_placeholder: Any,
    agent_status_placeholder: Any,
) -> tuple[str, list[dict[str, Any]], bool]:
    full_answer = ""
    tool_events: list[dict[str, Any]] = []
    interrupted = False
    saw_error = False

    for event in event_iter:
        event_type = event.get("type")
        if event_type == "agent_switch":
            tip = _agent_tip(str(event.get("agent", "")))
            agent_status_placeholder.info(tip)
            if hasattr(st, "toast"):
                st.toast(tip)
            tool_events.append(event)
            _render_tool_events(tool_placeholder, tool_events)
        elif event_type == "token":
            token = _sanitize_stream_token(str(event.get("content", "")))
            if token:
                full_answer += token
                answer_placeholder.markdown(full_answer)
        elif event_type in {"tool_start", "tool_end"}:
            tool_events.append(event)
            _render_tool_events(tool_placeholder, tool_events)
        elif event_type == "error":
            tool_events.append(event)
            _render_tool_events(tool_placeholder, tool_events)
            saw_error = True
            error_message = str(event.get("message", "流式处理出现错误")).strip() or "流式处理出现错误"
            if not full_answer:
                full_answer = f"请求失败：{error_message}"
                answer_placeholder.markdown(full_answer)
        elif event_type == "interrupt":
            calls = _normalize_tool_calls(event.get("tool_calls"))
            _set_pending_interrupt(calls)
            tool_events.append({"type": "interrupt", "tool_calls": calls})
            _render_tool_events(tool_placeholder, tool_events)
            if not full_answer:
                full_answer = "操作已暂停，等待人工审批。"
            answer_placeholder.markdown(full_answer)
            interrupted = True
            break
        elif event_type == "done":
            break

    full_answer = _sanitize_answer_text(full_answer)

    if not full_answer and not interrupted and not saw_error:
        full_answer = "后端未返回内容。"
        answer_placeholder.markdown(full_answer)

    if tool_events:
        _render_tool_events(tool_placeholder, tool_events)

    # 本轮事件消费完成后清理顶部“正在处理”提示，避免用户误以为仍在运行。
    if hasattr(agent_status_placeholder, "empty"):
        agent_status_placeholder.empty()

    return full_answer, tool_events, interrupted

def _resume_after_approval(user_id: str, action: str, session_id: str | None) -> None:
    if bool(st.session_state.get("approval_in_progress", False)):
        return
    st.session_state.approval_last_click_ts = time.monotonic()
    st.session_state.approval_in_progress = True
    # 立刻收起审批面板，避免用户在网络抖动下重复点击触发竞态。
    _clear_pending_interrupt()
    full_answer = ""
    tool_events: list[dict[str, Any]] = []
    interrupted = False

    with st.chat_message("assistant"):
        agent_status_placeholder = st.empty()
        answer_placeholder = st.empty()
        tool_placeholder = st.empty()

        try:
            full_answer, tool_events, interrupted = _consume_stream_events(
                stream_resume_api(user_id, action, session_id=session_id, thread_id=st.session_state.current_conversation_id),
                answer_placeholder=answer_placeholder,
                tool_placeholder=tool_placeholder,
                agent_status_placeholder=agent_status_placeholder,
            )
        except requests.RequestException as exc:
            detail = _request_error_detail(exc)
            if _is_no_pending_interrupt_error(detail):
                full_answer = "审批状态已更新：当前没有待审批工具调用（可能已执行完成，或正在继续处理）。请查看上方最新回复。"
                _clear_pending_interrupt()
            else:
                full_answer = f"调用 /api/v1/chat/resume 失败：{detail}"
            answer_placeholder.markdown(full_answer)
        except ValueError as exc:
            full_answer = f"解析续跑流式响应失败：{exc}"
            answer_placeholder.markdown(full_answer)
        finally:
            st.session_state.approval_in_progress = False

    current_conv_id = st.session_state.current_conversation_id
    if current_conv_id not in st.session_state.conversations:
        st.session_state.conversations[current_conv_id] = []
    st.session_state.conversations[current_conv_id].append({"role": "assistant", "content": full_answer, "tool_calls": tool_events or None})

    # 同步到messages变量
    st.session_state.messages = st.session_state.conversations[current_conv_id]

    if not interrupted:
        _clear_pending_interrupt()

    rerun_app()


def _run_deferred_approval_if_any(user_id: str) -> None:
    """执行延迟审批动作：用于先隐藏审批框，再发起后端 resume 请求。"""
    action = str(st.session_state.get("deferred_approval_action", "")).strip().lower()
    if action not in {"approve", "reject"}:
        return
    st.session_state.deferred_approval_action = ""
    _resume_after_approval(user_id=user_id, action=action, session_id=_active_session_id())


def _clear_backend_pending_interrupt(user_id: str) -> None:
    """强制清理后端待审批状态（按 reject 执行）。"""
    session_id = _active_session_id()
    try:
        for _ in stream_resume_api(user_id, "reject", session_id=session_id, thread_id=st.session_state.current_conversation_id):
            pass
    except requests.RequestException as exc:
        detail = _request_error_detail(exc)
        if not _is_no_pending_interrupt_error(detail):
            st.warning(f"清理后端待审批失败：{detail}")
    except ValueError as exc:
        st.warning(f"解析清理响应失败：{exc}")

    _clear_pending_interrupt()
    st.session_state.approval_in_progress = False
    _reset_chat_stream_runtime(keep_input=True)
    rerun_app()


def render_interrupt_panel(user_id: str) -> None:
    if not st.session_state.pending_interrupt:
        return

    st.warning("Agent 申请执行以下高危工具，请审批：")

    pending_calls = _normalize_tool_calls(st.session_state.pending_tool_calls)
    if not pending_calls:
        st.caption("未找到待审批工具明细。")
    else:
        for index, call in enumerate(pending_calls, start=1):
            tool_name = str(call.get("name", "unknown_tool"))
            tool_args = call.get("args", call.get("arguments", {}))
            st.markdown(f"{index}. 待执行工具：`{tool_name}`")
            st.code(json.dumps(tool_args, ensure_ascii=False, indent=2, default=str), language="json")

    approve_col, reject_col = st.columns(2)
    in_progress = bool(st.session_state.get("approval_in_progress", False))
    generating = bool(st.session_state.get("chat_generating", False))
    cooldown_remaining = _approval_cooldown_remaining()
    if in_progress:
        st.info("审批请求处理中，已锁定审批按钮，请勿重复点击。")
        return
    if generating:
        st.info("系统正在继续处理，本轮暂不允许再次审批。")
        return
    if cooldown_remaining > 0:
        st.caption(f"防重复点击冷却中：{cooldown_remaining:.1f}s")

    if approve_col.button(
        "✅ 允许执行",
        use_container_width=True,
        key="approve_tool_execution",
        disabled=cooldown_remaining > 0,
    ):
        st.session_state.approval_last_click_ts = time.monotonic()
        _clear_pending_interrupt()
        st.session_state.deferred_approval_action = "approve"
        rerun_app()
    if reject_col.button(
        "❌ 拒绝执行",
        use_container_width=True,
        key="reject_tool_execution",
        disabled=cooldown_remaining > 0,
    ):
        st.session_state.approval_last_click_ts = time.monotonic()
        _clear_pending_interrupt()
        st.session_state.deferred_approval_action = "reject"
        rerun_app()


def handle_user_input(user_id: str) -> None:
    _process_chat_stream_events()
    _render_inflight_assistant_message()

    if bool(st.session_state.get("clear_chat_input_next_run", False)):
        st.session_state.chat_input_text = ""
        st.session_state.clear_chat_input_next_run = False

    generating = bool(st.session_state.get("chat_generating", False))
    pending_interrupt = bool(st.session_state.get("pending_interrupt", False))
    in_progress_approval = bool(st.session_state.get("approval_in_progress", False))

    if (pending_interrupt or in_progress_approval) and not generating:
        with st.expander("输入被锁定？", expanded=False):
            st.caption("如果审批面板未正常显示，可先解除前端锁定，再重新发送请求以回放待审批状态。")
            unlock_col, clear_col = st.columns(2)
            if unlock_col.button("解除前端锁定", key="unlock_frontend_state", use_container_width=True):
                _clear_pending_interrupt()
                st.session_state.approval_in_progress = False
                _reset_chat_stream_runtime(keep_input=True)
                rerun_app()
            if clear_col.button("清理后端待审批", key="clear_backend_interrupt", use_container_width=True):
                _clear_backend_pending_interrupt(user_id)

    input_col, upload_col, action_col = st.columns([8, 1, 1])
    with input_col:
        st.text_input(
            "请输入你的问题...",
            key="chat_input_text",
            label_visibility="collapsed",
            placeholder="请输入你的问题...",
            disabled=generating or pending_interrupt or in_progress_approval,
        )
    with action_col:
        action_label = "暂停" if generating else "发送"
        action_clicked = st.button(
            action_label,
            key="chat_send_or_pause_button",
            use_container_width=True,
            disabled=pending_interrupt and not generating,
        )
    uploaded_file = st.file_uploader(
            "上传文件",
            key="file_uploader",
            label_visibility="collapsed",
            accept_multiple_files=False,
            type=["txt", "pdf", "docx", "xlsx", "csv", "jpg", "jpeg", "png"],
            disabled=generating or pending_interrupt or in_progress_approval,
        )
    
    # 处理文件上传
    if uploaded_file is not None and action_clicked:
        import requests
        import io
        
        # 获取当前用户ID
        current_user_id = user_id
        
        # 准备上传数据
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        params = {'user_id': current_user_id}
        
        try:
            # 调用后端上传API - 使用正确的API URL和认证头
            headers = _request_headers()
            upload_url = f"{AGENT_API_BASE_URL}/api/v1/upload"
            response = requests.post(upload_url, files=files, params=params, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"文件上传成功: {uploaded_file.name}")
                
                # 将文件信息添加到对话中
                # file_info = f"用户上传了文件: {uploaded_file.name} ({uploaded_file.type})，文件已可供AI分析。"
                # st.session_state.messages.append({"role": "user", "content": file_info})
                
                # # 不直接修改widget状态，而是触发页面重载来重置上传器
                # # st.rerun()
            else:
                st.error(f"文件上传失败: {response.json().get('detail', '未知错误')}")
        except Exception as e:
            st.error(f"上传过程中出错: {str(e)}")
    
    if action_clicked and generating:
        stop_event = st.session_state.get("chat_stop_event")
        if isinstance(stop_event, threading.Event):
            stop_event.set()
            st.session_state.chat_stop_requested = True
            st.caption("已请求暂停当前生成，等待后端结束本轮流式响应...")
        else:
            st.warning("当前没有可暂停的生成任务。")

    if action_clicked and not generating:
        user_query = str(st.session_state.get("chat_input_text", "")).strip()
        if not user_query:
            return

        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.clear_chat_input_next_run = True
        _start_chat_stream(user_id, user_query)
        rerun_app()
        return

    # 自动轮询后台流式状态，保证 token 能持续刷新。
    if bool(st.session_state.get("chat_generating", False)):
        time.sleep(0.2)
        rerun_app()
def switch_conversation(conversation_id: str) -> None:
    """切换到指定对话"""
    # 如果需要，加载内容
    load_conversation_content(conversation_id)
    
    st.session_state.current_conversation_id = conversation_id
    # 确保对话存在
    if conversation_id not in st.session_state.conversations:
        st.session_state.conversations[conversation_id] = []
    st.session_state.messages = st.session_state.conversations[conversation_id]

def new_conversation() -> None:
    """创建新对话"""
    import uuid
    conversation_id = f"conv_{str(uuid.uuid4())[:8]}"
    st.session_state.conversations[conversation_id] = []
    switch_conversation(conversation_id)

def delete_conversation(conversation_id: str) -> None:
    """删除指定对话"""
    if conversation_id in st.session_state.conversations:
        del st.session_state.conversations[conversation_id]
        # 如果删除的是当前对话，则切换到第一个可用对话
        if st.session_state.current_conversation_id == conversation_id:
            available_convs = list(st.session_state.conversations.keys())
            if available_convs:
                switch_conversation(available_convs[0])
            else:
                new_conversation()

def main() -> None:
    st.set_page_config(page_title="NanoAgent 前端", page_icon=":robot_face:", layout="wide")
    init_session_state()
    # 在 init_session_state() 函数中添加
    if "conversations" not in st.session_state:
        st.session_state.conversations = {"default": []}
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = "default"
    # 确保 messages 也初始化
    if "messages" not in st.session_state:
        current_conv_id = st.session_state.current_conversation_id
        st.session_state.messages = st.session_state.conversations[current_conv_id]
    user_id = render_sidebar()

    st.title("NanoAgent")
    st.caption(f"当前用户：`{user_id}`")

    session_id = _active_session_id()
    if session_id:
        meta = st.session_state.get("llm_session_meta", {})
        st.caption(f"当前 AI 会话：`{meta.get('provider', 'unknown')}` / `{meta.get('model', 'unknown')}`")
    else:
        st.caption("当前 AI 会话：未启用（将使用后端默认配置）")

    render_chat_history()
    _run_deferred_approval_if_any(user_id)
    handle_user_input(user_id)
    render_interrupt_panel(user_id)


if __name__ == "__main__":
    main()