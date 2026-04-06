"""认证模块。"""

from hmac import compare_digest
from typing import Any, Literal, TypedDict

import jwt
from fastapi import Depends, HTTPException
from starlette.requests import Request

from config import (
    ALLOWED_API_KEYS,
    AUTH_ALLOW_API_KEY_FALLBACK,
    AUTH_REQUIRE_USER_SUB,
    JWT_ALGORITHMS,
    JWT_AUDIENCE,
    JWT_HS256_SECRET,
    JWT_ISSUER,
    JWT_LEEWAY_SECONDS,
    REQUIRE_API_AUTH,
    _jwt_jwk_client,
)


class AuthContext(TypedDict):
    """请求认证上下文。"""

    auth_type: Literal["jwt", "api_key", "disabled"]
    subject: str | None


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
    subject = claims.get("sub", "")
    if not isinstance(subject, str) or not subject.strip():
        raise HTTPException(status_code=401, detail="JWT claims 缺少有效的 sub 字段")
    return subject.strip()


def _resolve_effective_user_id(*, token_subject: str, client_user_id: str, source: str) -> str:
    """根据配置决定最终使用的 user_id。"""
    if not AUTH_REQUIRE_USER_SUB:
        return client_user_id.strip()

    if token_subject == client_user_id.strip():
        return client_user_id.strip()

    raise HTTPException(
        status_code=403,
        detail=f"用户身份不匹配：JWT subject ({token_subject}) 与请求 user_id ({client_user_id}) 不一致",
    )


async def _require_api_auth_context(request: Request) -> AuthContext:
    """认证依赖：解析并返回认证上下文。"""
    if not REQUIRE_API_AUTH:
        return {"auth_type": "disabled", "subject": None}

    authorization = request.headers.get("authorization", "")
    bearer_token = _extract_bearer_token(authorization)

    if not bearer_token:
        if AUTH_ALLOW_API_KEY_FALLBACK and ALLOWED_API_KEYS:
            api_key = request.headers.get("x-api-key", "")
            if _is_allowed_api_key(api_key):
                return {"auth_type": "api_key", "subject": None}
        raise HTTPException(status_code=401, detail="缺少认证信息")

    if _looks_like_jwt(bearer_token):
        claims = _decode_jwt_claims(bearer_token)
        subject = _subject_from_claims(claims)
        return {"auth_type": "jwt", "subject": subject}

    if AUTH_ALLOW_API_KEY_FALLBACK and _is_allowed_api_key(bearer_token):
        return {"auth_type": "api_key", "subject": None}

    raise HTTPException(status_code=401, detail="认证信息格式错误")


async def _require_user_context(request: Request) -> AuthContext:
    """认证依赖：要求用户身份上下文。"""
    auth_context = await _require_api_auth_context(request)
    if AUTH_REQUIRE_USER_SUB and auth_context["auth_type"] != "jwt":
        raise HTTPException(status_code=403, detail="当前接口要求用户身份认证（JWT）")
    return auth_context


async def _require_api_auth(request: Request) -> None:
    """认证依赖：仅验证 API 认证，不要求用户身份。"""
    await _require_api_auth_context(request)


def _require_subject(auth_context: AuthContext) -> str:
    """从认证上下文中提取 subject，若不存在则抛出异常。"""
    subject = auth_context.get("subject")
    if not subject:
        raise HTTPException(status_code=403, detail="当前接口要求用户身份认证（JWT）")
    return subject