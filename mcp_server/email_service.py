"""MCP 服务邮件发送功能模块。"""

import asyncio
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Optional, Tuple

from config import (
    REPORT_PROVIDER, REPORT_SUBJECT, SMTP_HOST, SMTP_PORT, SMTP_USERNAME, 
    SMTP_PASSWORD, SMTP_FROM, SMTP_USE_TLS, SMTP_USE_SSL, SMTP_TIMEOUT_SECONDS,
    REPORT_MAX_CONTENT_CHARS, DEBUG_MODE, logger
)
from utils import _json_response, _truncate_text, _mask_email, _is_valid_email, _is_report_email_allowed, _prepare_report_email_payload


async def send_report_tool(email: str, content: str) -> str:
    """发送报告邮件工具实现。"""
    timestamp = datetime.now(timezone.utc).isoformat()
    normalized_email = email.strip()
    normalized_content = content.strip()
    masked_email = _mask_email(normalized_email)
    content_preview = _truncate_text(normalized_content)

    if not _is_valid_email(normalized_email):
        return _json_response(
            {
                "status": "error",
                "message": "邮箱格式无效，请提供合法邮箱地址。",
            }
        )

    if not normalized_content:
        return _json_response(
            {
                "status": "error",
                "message": "content 不能为空。",
            }
        )

    if len(normalized_content) > REPORT_MAX_CONTENT_CHARS:
        return _json_response(
            {
                "status": "error",
                "message": f"content 过长（>{REPORT_MAX_CONTENT_CHARS} 字符），请缩短后重试。",
            }
        )

    mail_body, attachment_bytes, attachment_filename, downgraded = _prepare_report_email_payload(
        normalized_content,
        timestamp=timestamp,
    )

    if not _is_report_email_allowed(normalized_email):
        return _json_response(
            {
                "status": "error",
                "message": "目标邮箱域名不在白名单内，已拒绝发送。",
            }
        )

    if REPORT_PROVIDER not in {"mock", "smtp"}:
        logger.warning("REPORT_PROVIDER=%s 不受支持，回退为 mock。", REPORT_PROVIDER)

    provider = REPORT_PROVIDER if REPORT_PROVIDER in {"mock", "smtp"} else "mock"

    if provider == "smtp":
        return await _send_smtp_email(
            normalized_email, masked_email, mail_body, attachment_bytes, 
            attachment_filename, downgraded, timestamp, normalized_content
        )
    else:
        return _send_mock_email(
            normalized_email, masked_email, mail_body, attachment_bytes,
            attachment_filename, downgraded, timestamp, normalized_content, content_preview
        )


async def _send_smtp_email(
    email: str, masked_email: str, mail_body: str, attachment_bytes: Optional[bytes],
    attachment_filename: Optional[str], downgraded: bool, timestamp: str, original_content: str
) -> str:
    """通过SMTP发送邮件。"""
    missing_fields: list[str] = []
    if not SMTP_HOST:
        missing_fields.append("SMTP_HOST")
    if not SMTP_FROM:
        missing_fields.append("SMTP_FROM")
    if SMTP_USERNAME and not SMTP_PASSWORD:
        missing_fields.append("SMTP_PASSWORD")
    if missing_fields:
        return _json_response(
            {
                "status": "error",
                "message": f"SMTP 配置不完整，缺少: {', '.join(missing_fields)}",
            }
        )

    try:
        await asyncio.to_thread(
            _smtp_send_report_sync,
            to_email=email,
            subject=REPORT_SUBJECT,
            content=mail_body,
            attachment_bytes=attachment_bytes,
            attachment_filename=attachment_filename,
        )
        logger.info(
            "SMTP 发送报告成功 | email=%s | timestamp=%s | content_length=%d | downgraded=%s | attachment=%s",
            masked_email,
            timestamp,
            len(original_content),
            downgraded,
            bool(attachment_bytes),
        )
        return _json_response(
            {
                "status": "success",
                "provider": "smtp",
                "delivery": "sent",
                "email": masked_email,
                "timestamp": timestamp,
                "downgraded": downgraded,
                "attachment_filename": attachment_filename,
                "message": (
                    "报告发送成功。正文已摘要，完整内容作为附件发送。"
                    if downgraded and attachment_bytes
                    else "报告发送成功。"
                ),
            }
        )
    except (smtplib.SMTPException, OSError, ssl.SSLError) as exc:
        if DEBUG_MODE:
            logger.exception("SMTP 发送失败 | email=%s | error=%s", masked_email, exc)
            return _json_response(
                {
                    "status": "error",
                    "message": "SMTP 发送失败，请检查邮件服务配置或网络连通性。",
                    "details": _truncate_text(str(exc)),
                }
            )
        logger.error("SMTP 发送失败 | email=%s", masked_email)
        return _json_response(
            {
                "status": "error",
                "message": "SMTP 发送失败，请检查邮件服务配置或网络连通性。",
            }
        )
    except Exception as exc:  # noqa: BLE001
        if DEBUG_MODE:
            logger.exception("send_report 出现未知错误 | email=%s | error=%s", masked_email, exc)
            return _json_response(
                {
                    "status": "error",
                    "message": "发送报告时发生未知错误。",
                    "details": _truncate_text(str(exc)),
                }
            )
        logger.error("send_report 出现未知错误 | email=%s", masked_email)
        return _json_response(
            {
                "status": "error",
                "message": "发送报告时发生未知错误。",
            }
        )


def _send_mock_email(
    email: str, masked_email: str, mail_body: str, attachment_bytes: Optional[bytes],
    attachment_filename: Optional[str], downgraded: bool, timestamp: str, 
    original_content: str, content_preview: str
) -> str:
    """模拟发送邮件（用于开发和测试）。"""
    if DEBUG_MODE:
        logger.info(
            "模拟发送报告 | email=%s | timestamp=%s | content_length=%d | downgraded=%s | attachment=%s | content_preview=%s",
            masked_email,
            timestamp,
            len(original_content),
            downgraded,
            bool(attachment_bytes),
            content_preview,
        )
    else:
        logger.info(
            "模拟发送报告 | email=%s | timestamp=%s | content_length=%d | downgraded=%s | attachment=%s",
            masked_email,
            timestamp,
            len(original_content),
            downgraded,
            bool(attachment_bytes),
        )
    return _json_response(
        {
            "status": "success",
            "provider": "mock",
            "delivery": "simulated",
            "email": masked_email,
            "timestamp": timestamp,
            "downgraded": downgraded,
            "attachment_filename": attachment_filename,
            "message": (
                "报告发送成功（模拟）。正文已摘要，完整内容作为附件发送（模拟）。"
                if downgraded and attachment_bytes
                else "报告发送成功（模拟）。"
            ),
        }
    )


def _smtp_send_report_sync(
    *,
    to_email: str,
    subject: str,
    content: str,
    attachment_bytes: Optional[bytes] = None,
    attachment_filename: Optional[str] = None,
) -> None:
    """同步 SMTP 发送实现（由 asyncio.to_thread 调用）。"""
    message = EmailMessage()
    message["From"] = SMTP_FROM
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(content)
    if attachment_bytes:
        filename = (attachment_filename or "report.txt").strip() or "report.txt"
        message.add_attachment(
            attachment_bytes,
            maintype="text",
            subtype="plain",
            filename=filename,
        )

    if SMTP_USE_SSL:
        with smtplib.SMTP_SSL(
            host=SMTP_HOST,
            port=SMTP_PORT,
            timeout=SMTP_TIMEOUT_SECONDS,
            context=ssl.create_default_context(),
        ) as client:
            if SMTP_USERNAME:
                client.login(SMTP_USERNAME, SMTP_PASSWORD)
            client.send_message(message)
        return

    with smtplib.SMTP(host=SMTP_HOST, port=SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS) as client:
        client.ehlo()
        if SMTP_USE_TLS:
            client.starttls(context=ssl.create_default_context())
            client.ehlo()
        if SMTP_USERNAME:
            client.login(SMTP_USERNAME, SMTP_PASSWORD)
        client.send_message(message)