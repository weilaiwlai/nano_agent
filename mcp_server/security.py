"""MCP 服务安全认证和校验模块。"""

import re
from hmac import compare_digest
from typing import Optional

from fastapi import HTTPException
from starlette.requests import Request

from config import MCP_SERVICE_TOKEN, FORBIDDEN_SQL_FUNCTIONS, MCP_SQL_MAX_LENGTH


def _extract_service_token(request: Request) -> str:
    """从请求头提取服务间鉴权令牌。"""
    header_token = (request.headers.get("X-Service-Token") or "").strip()
    if header_token:
        return header_token

    authorization = (request.headers.get("Authorization") or "").strip()
    if not authorization:
        return ""
    parts = authorization.split(" ", 1)
    if len(parts) != 2:
        return ""
    if parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()


def _is_authorized_service_request(request: Request) -> bool:
    """校验服务间调用令牌。"""
    token = _extract_service_token(request)
    if not token:
        return False
    return compare_digest(token, MCP_SERVICE_TOKEN)


def _service_user_id_from_header(request: Request) -> str:
    """读取服务侧透传的用户标识。"""
    return (request.headers.get("X-NanoAgent-User-Id") or "").strip()


def _resolve_effective_setting_user_id(request: Request, payload_user_id: str) -> str:
    """解析受控写工具的最终 user_id，并做服务头与请求体一致性校验。"""
    header_user_id = _service_user_id_from_header(request)
    normalized_payload_user_id = payload_user_id.strip()

    if not header_user_id:
        raise HTTPException(status_code=422, detail="upsert_user_setting 缺少服务侧用户上下文")

    if header_user_id and normalized_payload_user_id and header_user_id != normalized_payload_user_id:
        from config import logger
        logger.warning(
            "拒绝跨租户写入 | header_user_id=%s | payload_user_id=%s",
            header_user_id,
            normalized_payload_user_id,
        )
        raise HTTPException(status_code=403, detail="用户标识不一致，拒绝写入。")

    return header_user_id


def _is_protected_path(path: str) -> bool:
    """判断当前路径是否需要服务间鉴权。"""
    normalized_path = path.rstrip("/") or "/"
    return normalized_path.startswith("/tools/") or normalized_path.startswith("/mcp")


def _normalize_sql_for_check(sql: str) -> str:
    """标准化 SQL 文本，便于白名单校验。"""
    return " ".join(sql.strip().split()).lower()


def _sql_safety_error(sql: str) -> Optional[str]:
    """返回 SQL 安全检查错误，安全时返回 None。"""
    normalized_sql = _normalize_sql_for_check(sql)
    if not normalized_sql:
        return "sql 不能为空。"

    if len(normalized_sql) > MCP_SQL_MAX_LENGTH:
        return f"sql 过长（>{MCP_SQL_MAX_LENGTH} 字符），请缩短后重试。"

    # 禁止多语句执行（简单且有效）。
    if ";" in normalized_sql:
        return "仅允许单条只读 SELECT 查询，不支持多语句。"

    if not (normalized_sql.startswith("select ") or normalized_sql.startswith("with ")):
        return "仅允许只读 SELECT/CTE 查询。"

    forbidden_tokens = (
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " alter ",
        " create ",
        " truncate ",
        " grant ",
        " revoke ",
        " merge ",
        " call ",
        " execute ",
        " do ",
        " copy ",
    )
    padded = f" {normalized_sql} "
    for token in forbidden_tokens:
        if token in padded:
            return "仅允许只读 SELECT 查询；增删改和 DDL 已被禁用。"

    for func_name in FORBIDDEN_SQL_FUNCTIONS:
        pattern = rf"\b{re.escape(func_name)}\s*\("
        if re.search(pattern, normalized_sql):
            return f"SQL 包含高风险函数 `{func_name}`，已拒绝执行。"

    return None


def _build_limited_select_sql(sql: str, *, row_limit: int) -> str:
    """将任意只读查询包裹为外层 LIMIT，控制结果集规模。"""
    return f"SELECT * FROM ({sql}) AS nanoagent_safe_query LIMIT {row_limit}"