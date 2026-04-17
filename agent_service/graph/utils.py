from __future__ import annotations

import hashlib
import json
import re
from uuid import uuid4
from typing import Any
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from .config import (
    CONTROL_CONTENT_ARG_OPEN_PATTERN,
    CONTROL_EMAIL_ARG_PATTERN,
    CONTROL_MARKUP_FRAGMENT_PATTERN,
    CONTROL_MARKUP_KEYWORDS,
    DATABASE_HELP_KEYWORDS,
    DSML_CONTROL_TOKEN_PATTERN,
    EMAIL_PATTERN,
    MAX_MODEL_HISTORY_MESSAGES,
    REPORT_CONTENT_SOFT_LIMIT,
    SQL_SNIPPET_PATTERN,
    TOOL_LOG_MAX_TEXT_LENGTH,
    logger,
)


def _truncate_for_log(value: str, *, max_len: int = TOOL_LOG_MAX_TEXT_LENGTH) -> str:
    """截断工具日志文本，避免记录过长输入。"""
    text = value.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...(truncated)"


def _mask_email_for_log(value: str) -> str:
    """邮箱脱敏，仅用于日志。"""
    email = value.strip()
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    if not local:
        return f"***@{domain}"
    keep = 1 if len(local) < 4 else 2
    return f"{local[:keep]}***@{domain}"


def _short_text_digest(value: str) -> str:
    """生成固定长度摘要，便于在日志中定位请求而不暴露原文。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _message_to_text(message: BaseMessage) -> str:
    """尽力将 LangChain 消息内容转换为纯文本。"""
    content = message.content
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
        return " ".join(parts).strip()
    return str(content)


def _latest_user_query(messages: list[BaseMessage]) -> str:
    """从消息列表中提取最新一条用户问题。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _message_to_text(msg).strip()
    return ""


def _strip_dsml_control_tokens(text: str) -> str:
    """移除部分模型返回的 DSML 控制标记，避免污染前端可读输出。"""
    if not text:
        return text
    cleaned = DSML_CONTROL_TOKEN_PATTERN.sub("", text)
    cleaned = CONTROL_EMAIL_ARG_PATTERN.sub("", cleaned)
    cleaned = CONTROL_CONTENT_ARG_OPEN_PATTERN.sub("", cleaned)
    cleaned = CONTROL_MARKUP_FRAGMENT_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"</?\s*>", "", cleaned)
    cleaned_lines: list[str] = []
    for line in cleaned.splitlines():
        normalized_line = line.strip()
        lower_line = normalized_line.lower()
        if normalized_line and any(keyword in lower_line for keyword in CONTROL_MARKUP_KEYWORDS):
            if ("<" in normalized_line) or (">" in normalized_line) or ("string=" in lower_line):
                continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _sanitize_ai_message_text(message: BaseMessage) -> BaseMessage:
    """清理 AIMessage 中的控制标记文本。"""
    if not isinstance(message, AIMessage):
        return message

    original_text = _message_to_text(message)
    cleaned_text = _strip_dsml_control_tokens(original_text)
    if cleaned_text == original_text:
        return message

    try:
        return message.model_copy(update={"content": cleaned_text})
    except AttributeError:
        return message.copy(update={"content": cleaned_text})


def _sanitize_history_for_model(
    messages: list[BaseMessage],
    *,
    max_messages: int = MAX_MODEL_HISTORY_MESSAGES,
    config: RunnableConfig | None = None,
) -> list[BaseMessage]:
    """清理对模型不友好的历史片段（未闭环 tool_calls / 孤立 ToolMessage）。"""
    if not messages:
        return messages

    sanitized: list[BaseMessage] = []
    dropped_count = 0
    index = 0
    total = len(messages)

    while index < total:
        message = messages[index]

        if isinstance(message, AIMessage) and message.tool_calls:
            tool_call_ids = [
                str(call.get("id", "")).strip()
                for call in message.tool_calls
                if isinstance(call, dict) and str(call.get("id", "")).strip()
            ]

            next_index = index + 1
            following_tools: list[ToolMessage] = []
            while next_index < total and isinstance(messages[next_index], ToolMessage):
                following_tools.append(messages[next_index])  # type: ignore[arg-type]
                next_index += 1

            if not tool_call_ids:
                dropped_count += 1
                index = next_index
                continue

            required_ids = set(tool_call_ids)
            found_ids = {
                str(getattr(tool_message, "tool_call_id", "")).strip()
                for tool_message in following_tools
                if str(getattr(tool_message, "tool_call_id", "")).strip()
            }

            if required_ids.issubset(found_ids):
                sanitized.append(message)
                for tool_message in following_tools:
                    tool_call_id = str(getattr(tool_message, "tool_call_id", "")).strip()
                    if tool_call_id in required_ids:
                        sanitized.append(tool_message)
                    else:
                        dropped_count += 1
                index = next_index
                continue

            dropped_count += 1 + len(following_tools)
            index = next_index
            continue

        if isinstance(message, ToolMessage):
            dropped_count += 1
            index += 1
            continue

        sanitized.append(message)
        index += 1
    summary_msg = None
    if max_messages > 0 and len(sanitized) > max_messages:
        logger.info("历史消息清理 | max_messages=%d", max_messages)
        tail = sanitized[-max_messages:]
        sanitized = _sanitize_history_for_model(tail, max_messages=0)
    if dropped_count > 0:
        logger.info("历史消息清理完成 | dropped=%d | kept=%d", dropped_count, len(sanitized))
    return sanitized

def summaries_messages(messages: list[BaseMessage],config: RunnableConfig,sum_messages:list[BaseMessage]=None) -> list[BaseMessage]:
    # 将早期消息转换为文本格式，用于摘要生成
    messages=_sanitize_history_for_model(messages,config=config,max_messages=0)
    summary_text = "\n".join([
        f"{msg.type}: {msg.content}" for msg in messages
    ])
    # logger.info(f"原始历史消息: {summary_text}")
    if sum_messages:
        summary_text = f"历史摘要：{sum_messages[0].content}\n{summary_text}"
    # 创建摘要提示
    summary_prompt = f"""
    请总结以下对话历史，提取关键信息和上下文，限制在100字以内：
    {summary_text}
    """
    try:
        # 使用大模型生成摘要
        from .llm import _get_non_stream_chat_llm
        summary_result = _get_non_stream_chat_llm(config).invoke(summary_prompt)
        # 创建摘要消息对象
        from langchain_core.messages import SystemMessage
        summary_msg = SystemMessage(content=f"更久以前的对话历史摘要: {summary_result.content}")
        # 返回摘要消息加上最近5条消息
        logger.info(f"Summary message成功: {summary_msg}")
        # 存储到 Redis
    except Exception:
        logger.error("摘要生成失败")
        summary_msg = None
    return [summary_msg]

def _latest_human_index(messages: list[BaseMessage]) -> int:
    """返回最近一条 HumanMessage 的索引，不存在则返回 -1。"""
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            return idx
    return -1


def _has_recent_send_report_tool_result(messages: list[BaseMessage]) -> bool:
    """判断当前轮次是否已出现 send_report 工具结果。"""
    if not messages:
        return False

    last_human_idx = _latest_human_index(messages)
    start_idx = last_human_idx + 1 if last_human_idx >= 0 else 0

    for message in messages[start_idx:]:
        if not isinstance(message, ToolMessage):
            continue

        text = _message_to_text(message).lower()
        if not text:
            continue
        message_name = str(getattr(message, "name", "")).strip().lower()
        if message_name in {"send_report", "tool_send_report"}:
            return True

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                tool_name = str(payload.get("tool", "")).strip().lower()
                if tool_name in {"send_report", "tool_send_report"}:
                    return True
        except Exception:  # noqa: BLE001
            pass

        if "send_report" in text or "tool_send_report" in text:
            return True

    return False


def _has_database_intent(query: str) -> bool:
    """判断用户问题是否与数据库查询有关。"""
    normalized = query.strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in DATABASE_HELP_KEYWORDS)


def _has_sql_snippet(query: str) -> bool:
    """判断输入中是否包含可执行 SQL 片段。"""
    normalized = query.strip()
    if not normalized:
        return False
    return bool(SQL_SNIPPET_PATTERN.search(normalized))


def _build_database_help_answer() -> str:
    """生成数据库查询引导文案（面向非 SQL 用户）。"""
    return (
        "我可以帮你查数据库，但需要你给出更具体的查询目标或 SQL。下面是最快可用的提问方式：\n\n"
        "1. 你可以查什么\n"
        "- 当前有哪些数据库\n"
        "- 当前库有哪些表\n"
        "- 某张表有哪些字段\n"
        "- 简单统计（总数、分组计数、最近 N 条记录）\n\n"
        "2. 推荐提问模板（直接复制）\n"
        "- 请调用数据库工具执行 SQL：SELECT datname FROM pg_database;\n"
        "- 请调用数据库工具执行 SQL：SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;\n"
        "- 请调用数据库工具执行 SQL：SELECT column_name, data_type FROM information_schema.columns WHERE table_name='你的表名' ORDER BY ordinal_position;\n"
        "- 请调用数据库工具执行 SQL：SELECT COUNT(*) AS total FROM 你的表名;\n\n"
        "3. 常见失败原因（以及怎么改）\n"
        "- 只允许只读查询：仅支持 SELECT/CTE，不能 INSERT/UPDATE/DELETE/DDL\n"
        "- 不支持多语句：一条请求里只能有一条 SQL（不要写分号后第二条）\n"
        "- 表名/字段名不存在：先查 information_schema 确认结构\n"
        "- 查询超时或结果过大：加 WHERE / LIMIT，缩小范围\n\n"
        "你现在只要告诉我'要查哪张表、查什么字段、时间范围'，我就能帮你拼出可执行 SQL。"
    )


def _extract_first_email(text: str) -> str:
    """从文本中提取第一个邮箱地址。"""
    match = EMAIL_PATTERN.search(text)
    return match.group(0).strip() if match else ""


def _fallback_report_content_from_query(query: str) -> str:
    """当模型未给出可用正文时，从用户问题生成可发送的兜底正文。"""
    normalized = query.strip()
    if not normalized:
        return "请查收本次自动生成的报告。"

    without_email = EMAIL_PATTERN.sub("", normalized)
    without_email = re.sub(
        r"(请|帮我|麻烦)?\s*(发送|发到|发给|寄给|转发)\s*(到)?\s*(我的)?\s*(邮件|邮箱|电邮)",
        "",
        without_email,
        flags=re.IGNORECASE,
    )
    without_email = re.sub(r"(上述|上面|以上|这段|这份)\s*内容", "", without_email)
    without_email = re.sub(r"[，,。；;:：\\s]+", " ", without_email).strip()
    if not without_email:
        return "请查收本次自动生成的报告。"
    return without_email[:REPORT_CONTENT_SOFT_LIMIT]


def _latest_assistant_answer_before_last_user(messages: list[BaseMessage]) -> str:
    """提取最近一轮用户输入之前的最后一条 assistant 可读回答。"""
    last_human_idx = _latest_human_index(messages)
    if last_human_idx <= 0:
        return ""

    for idx in range(last_human_idx - 1, -1, -1):
        message = messages[idx]
        if not isinstance(message, AIMessage):
            continue

        text = _strip_dsml_control_tokens(_message_to_text(message)).strip()
        if not text:
            continue

        upper_text = text.upper().replace('"', "").replace("'", "").strip()
        if upper_text in {"KNOWLEDGEWORKER", "REPORTER", "ASSISTANT", "FINISH"}:
            continue

        return text

    return ""


def _derive_report_content(
    *,
    latest_query: str,
    model_response_text: str,
    history: list[BaseMessage],
) -> str:
    """合成稳定可发送的邮件正文，优先模型正文，其次复用上一条助手回答。"""
    cleaned_model_text = _strip_dsml_control_tokens(model_response_text).strip()
    if cleaned_model_text and len(cleaned_model_text) >= 12:
        return cleaned_model_text[:REPORT_CONTENT_SOFT_LIMIT]

    if any(keyword in latest_query for keyword in ("上述内容", "上面内容", "以上内容", "刚才内容")):
        previous_answer = _latest_assistant_answer_before_last_user(history)
        if previous_answer:
            return previous_answer[:REPORT_CONTENT_SOFT_LIMIT]

    return _fallback_report_content_from_query(latest_query)


def _extract_report_content_from_query(query: str) -> str:
    """从用户输入中尽量提取"邮件正文"字段。"""
    normalized = query.strip()
    if not normalized:
        return ""

    patterns = [
        r"(?:内容|正文)\s*[：:]\s*(.+)$",
        r"(?:发送|发给|发到).{0,30}(?:内容|正文)\s*(?:是|为)\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        extracted = _strip_dsml_control_tokens(match.group(1)).strip()
        if extracted:
            return extracted[:REPORT_CONTENT_SOFT_LIMIT]

    return ""


def _build_reporter_success_message(tool_payload: str) -> str:
    """基于 send_report 工具结果生成稳定、可读的确认文案。"""
    parsed_payload: Any = None
    try:
        parsed_payload = json.loads(tool_payload)
    except Exception:  # noqa: BLE001
        parsed_payload = None

    if isinstance(parsed_payload, dict):
        status = str(parsed_payload.get("status", "")).strip().lower()
        message = str(parsed_payload.get("message", "")).strip()
        if status == "success":
            return message or "邮件已发送成功。"
        if status == "error":
            return message or "邮件发送失败，请稍后重试。"

    cleaned = _strip_dsml_control_tokens(tool_payload).strip()
    if cleaned:
        return cleaned[:300]
    return "邮件操作已执行完成。"


def _build_tool_call_message(tool_name: str, args: dict[str, Any]) -> AIMessage:
    """构造可被 ToolNode 执行的 AIMessage(tool_calls)。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": f"call_{uuid4().hex[:12]}",
                "type": "tool_call",
                "name": tool_name,
                "args": args,
            }
        ],
    )


def _normalize_send_report_args(
    raw_args: Any,
    *,
    latest_query: str,
    history: list[BaseMessage],
) -> dict[str, Any] | None:
    """标准化 send_report 参数；无法构造有效参数时返回 None。"""
    args = raw_args if isinstance(raw_args, dict) else {}
    email = _extract_first_email(str(args.get("email", "")).strip()) or _extract_first_email(latest_query)
    if not email:
        return None

    raw_content = str(args.get("content", "")).strip()
    content = _derive_report_content(
        latest_query=latest_query,
        model_response_text=raw_content,
        history=history,
    )
    if not content:
        content = "请查收本次自动生成的报告。"

    return {"email": email, "content": content}

def store_messages_info_to_redis(message_len: int, config: RunnableConfig, sum_messages: list[BaseMessage]) -> None:
    """
    将消息长度和摘要消息存储到Redis，覆盖原有数据
    
    Args:
        message_len: 消息长度
        config: 配置对象
        sum_messages: 摘要消息列表
    """
    try:
        import redis
        import json
        from .config import REDIS_URL
        
        # 连接到Redis
        redis_client = redis.from_url(REDIS_URL)
        
        # 生成统一的键，基于会话ID，这样会覆盖原有的数据
        thread_id = f"{config.get('configurable', {}).get('thread_id', 'default_thread')}"
        redis_key = f"messages_info:{thread_id}:latest"
        
        # 准备要存储的数据 - 仅存储message_len和sum_messages
        messages_info = {
            'message_len': message_len,
            'sum_messages_count': len(sum_messages),
            'sum_messages_content': [
                {
                    'type': str(getattr(msg, 'type', 'unknown')),
                    'content': _message_to_text(msg) if msg else '',
                    'additional_kwargs': getattr(msg, 'additional_kwargs', {}) if msg else {}
                } for msg in sum_messages if msg
            ],
            'timestamp': __import__('time').time(),
            'thread_id': thread_id
        }
        
        # 存储到Redis，设置24小时过期
        redis_client.setex(redis_key, 86400, json.dumps(messages_info, ensure_ascii=False))
        
        logger.info(f"消息信息已存储到Redis: {redis_key}, 消息长度: {message_len}, 摘要消息数: {len(sum_messages)}")
        
    except ImportError:
        logger.warning("Redis 模块未安装，跳过存储到Redis")
    except Exception as e:
        logger.error(f"存储消息信息到Redis失败: {e}")


def get_messages_info_from_redis(config):
    """
    从Redis获取最新的消息信息
    
    Args:
        thread_id: 会话ID
    
    Returns:
        包含message_len和sum_messages的字典，如果没有找到则返回None
    """
    try:
        import redis
        import json
        from .config import REDIS_URL
        from langchain_core.messages import SystemMessage

        # 连接到Redis
        redis_client = redis.from_url(REDIS_URL)

        # 直接获取该会话的最新消息信息键
        thread_id = f"{config.get('configurable', {}).get('thread_id', 'default_thread')}"
        redis_key = f"messages_info:{thread_id}:latest"
        data = redis_client.get(redis_key)

        if data:
            messages_info = json.loads(data.decode('utf-8'))

            # 重构摘要消息对象
            sum_messages = []
            for msg_data in messages_info.get('sum_messages_content', []):
                if msg_data.get('type') == 'system':
                    msg = SystemMessage(
                        content=msg_data.get('content', ''),
                        additional_kwargs=msg_data.get('additional_kwargs', {})
                    )
                    sum_messages.append(msg)

            # 返回重构的消息信息
            result = {
                'message_len': messages_info.get('message_len'),
                'sum_messages': sum_messages,
                'sum_messages_count': messages_info.get('sum_messages_count'),
                'timestamp': messages_info.get('timestamp'),
                'thread_id': messages_info.get('thread_id'),
                'key': redis_key
            }

            logger.info(f"从Redis获取最新摘要消息信息成功: {result['sum_messages']}")
            return result
        else:
            logger.info(f"Redis中没有找到键: {redis_key}")
            return None

    except ImportError:
        logger.warning("Redis 模块未安装")
        return None
    except Exception as e:
        logger.error(f"从Redis获取消息信息失败: {e}")
        return None
    

def get_summary_text(config: RunnableConfig) -> str:
    sum_messages = None
    summary_text = "（无）"
    message_info = get_messages_info_from_redis(config)
    if message_info:
        sum_messages = message_info['sum_messages']
        # 提取摘要消息的内容部分，而不是整个消息对象
        if sum_messages:
            summary_texts = []
            for msg in sum_messages:
                if hasattr(msg, 'content'):
                    summary_texts.append(str(msg.content))
            summary_text = "\n".join(summary_texts) if summary_texts else "（无）"
    return summary_text