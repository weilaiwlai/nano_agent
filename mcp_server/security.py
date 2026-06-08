"""MCP 服务安全认证和校验模块。

SQL 安全校验基于 sqlglot AST 解析，而非字符串匹配：
- 解析 SQL 为抽象语法树，精确识别语句类型（SELECT/INSERT/DELETE/...）
- 遍历 AST 子节点，拦截 CTE 或子查询中夹带的写操作
- 彻底消除字符串匹配的误杀问题（如 SELECT * FROM insert_log 被错误拦截）
"""

from hmac import compare_digest
from typing import Optional

import sqlglot
from sqlglot import exp
from fastapi import HTTPException
from starlette.requests import Request

from config import MCP_SERVICE_TOKEN, FORBIDDEN_SQL_FUNCTIONS, MCP_SQL_MAX_LENGTH, logger

# AST 遍历时要拦截的 DML/DDL 操作节点类型
# 这些对应 INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/GRANT/REVOKE/MERGE
# 以及 Command（CALL/EXECUTE/DO）和 Copy
_FORBIDDEN_EXPR_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.TruncateTable,
    exp.Alter,
    exp.Create,
    exp.Grant,
    exp.Revoke,
    exp.Merge,
    exp.Command,
    exp.Copy,
)


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


def _sql_safety_error(sql: str) -> Optional[str]:
    """基于 sqlglot AST 解析校验 SQL 安全性。

    校验策略（由浅入深）：
    1. 空值 / 长度上限检查
    2. 分号检查（sqlglot 默认也分句，这里作为快速路径和纵深防御）
    3. AST 解析 → 仅允许单条 SELECT / CTE（WITH ... SELECT）
    4. 遍历 AST 所有节点 → 拦截嵌套的 DML/DDL 操作
    5. 遍历 AST 所有节点 → 拦截高风险函数调用

    返回：安全时 None，不安全时返回错误描述字符串。
    """
    normalized_sql = sql.strip()
    if not normalized_sql:
        return "sql 不能为空。"

    if len(normalized_sql) > MCP_SQL_MAX_LENGTH:
        return f"sql 过长（>{MCP_SQL_MAX_LENGTH} 字符），请缩短后重试。"

    # 纵深防御：直接禁止分号，防止解析器逃逸或解析差异
    if ";" in normalized_sql:
        return "仅允许单条只读 SELECT 查询，不支持多语句。"

    # AST 解析
    try:
        statements = sqlglot.parse(normalized_sql, read="postgres")
    except sqlglot.errors.ParseError:
        logger.warning("SQL 解析失败: %s", _truncate_debug_sql(normalized_sql))
        return "SQL 语法解析失败，请检查语句是否合法。"

    if not statements:
        return "SQL 解析结果为空。"

    if len(statements) > 1:
        return "仅允许单条只读 SELECT 查询，不支持多语句。"

    stmt = statements[0]

    # 顶层必须为 SELECT（含 WITH ... SELECT，sqlglot 将其解析为 Select，CTE 作为内部节点）
    if not isinstance(stmt, exp.Select):
        return "仅允许只读 SELECT/CTE 查询。"

    # 遍历 AST 所有节点，拦截嵌套的 DML/DDL
    for node in stmt.walk():
        # 拦截增删改和 DDL 操作（即使在 CTE 或子查询中）
        if isinstance(node, _FORBIDDEN_EXPR_TYPES):
            return "仅允许只读 SELECT 查询；增删改和 DDL 已被禁用。"

        # 拦截高风险函数调用
        if isinstance(node, exp.Anonymous):
            # Anonymous 节点的 .name 是实际函数名（已小写）
            func_name = node.name
            if func_name in FORBIDDEN_SQL_FUNCTIONS:
                return f"SQL 包含高风险函数 `{func_name}`，已拒绝执行。"
        elif isinstance(node, exp.Func):
            # 标准函数子类（Count, Sum 等），sql_name() 返回大写 SQL 名
            func_name = node.sql_name().lower()
            if func_name in FORBIDDEN_SQL_FUNCTIONS:
                return f"SQL 包含高风险函数 `{func_name}`，已拒绝执行。"

    return None


def _truncate_debug_sql(sql: str, max_len: int = 80) -> str:
    """截断 SQL 用于日志输出。"""
    return sql if len(sql) <= max_len else sql[:max_len] + "…"


def _build_limited_select_sql(sql: str, *, row_limit: int) -> str:
    """将任意只读查询包裹为外层 LIMIT，控制结果集规模。"""
    return f"SELECT * FROM ({sql}) AS nanoagent_safe_query LIMIT {row_limit}"