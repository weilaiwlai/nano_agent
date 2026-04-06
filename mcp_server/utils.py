"""MCP 服务通用工具函数模块。"""

import json
import re
from datetime import datetime, timezone
from typing import Any

from config import MAX_LOG_TEXT_LENGTH


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
    from config import REPORT_ALLOWED_EMAIL_DOMAINS
    
    if not REPORT_ALLOWED_EMAIL_DOMAINS:
        return True
    return _email_domain(value) in REPORT_ALLOWED_EMAIL_DOMAINS


def _build_report_attachment_filename(timestamp: str) -> str:
    """构造稳定、可读的报告附件名。"""
    from config import REPORT_ATTACHMENT_PREFIX
    
    # 例：2026-03-27T18:30:25.123456+00:00 -> 20260327_183025
    compact = re.sub(r"[^0-9]", "", timestamp)
    date_part = compact[:8] if len(compact) >= 8 else datetime.now(timezone.utc).strftime("%Y%m%d")
    time_part = compact[8:14] if len(compact) >= 14 else datetime.now(timezone.utc).strftime("%H%M%S")
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]", "_", REPORT_ATTACHMENT_PREFIX) or "nanoagent_report"
    return f"{safe_prefix}_{date_part}_{time_part}.txt"


def _prepare_report_email_payload(content: str, *, timestamp: str) -> tuple[str, bytes | None, str | None, bool]:
    """对超长报告做优雅降级：正文摘要 + 附件保留全文。"""
    from config import REPORT_SOFT_BODY_CHARS, REPORT_SUMMARY_PREVIEW_CHARS, REPORT_ATTACH_OVERFLOW
    
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