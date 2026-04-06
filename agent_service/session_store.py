"""LLM session store backed by Redis with TTL and at-rest API key encryption."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from typing import Any
from uuid import uuid4

from cryptography.fernet import Fernet, InvalidToken
from redis.asyncio import Redis
from redis.exceptions import RedisError

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger("nanoagent.agent_service.session_store")


class LLMSessionStore:
    """Store short-lived BYOK LLM profiles in Redis.

    Security model:
    - `api_key` is encrypted before writing to Redis (data-at-rest protection).
    - On read, encrypted payload is decrypted in memory and never logged.
    - Legacy plaintext records remain readable for backward compatibility.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        master_key: str,
        key_prefix: str = "nanoagent:llm_session:",
        default_ttl_seconds: int = 3600,
        max_ttl_seconds: int = 86400,
    ) -> None:
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl_seconds = max(60, default_ttl_seconds)
        self.max_ttl_seconds = max(self.default_ttl_seconds, max_ttl_seconds)
        self._redis: Redis | None = None

        normalized_master_key = master_key.strip()
        if not normalized_master_key:
            raise RuntimeError("LLM_SESSION_MASTER_KEY is required for secure session storage.")
        self._fernet = Fernet(self._build_fernet_key(normalized_master_key))

    @staticmethod
    def _build_fernet_key(master_key: str) -> bytes:
        """Derive a Fernet-compatible key from an arbitrary master secret string."""
        digest = hashlib.sha256(master_key.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest)

    async def startup(self) -> None:
        """Initialize Redis client and verify connectivity."""
        self._redis = Redis.from_url(self.redis_url, decode_responses=True)
        await self._redis.ping()
        logger.info("LLM Session Store connected to Redis.")

    async def shutdown(self) -> None:
        """Release Redis connection."""
        if self._redis is None:
            return
        await self._redis.aclose()
        self._redis = None
        logger.info("LLM Session Store Redis connection closed.")

    def _redis_client(self) -> Redis:
        if self._redis is None:
            raise RuntimeError("LLM Session Store is not initialized.")
        return self._redis

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}{session_id}"

    def _ttl(self, ttl_seconds: int | None) -> int:
        if ttl_seconds is None:
            return self.default_ttl_seconds
        return max(60, min(int(ttl_seconds), self.max_ttl_seconds))

    def _encrypt_api_key(self, plain_api_key: str) -> str:
        token = self._fernet.encrypt(plain_api_key.encode("utf-8"))
        return token.decode("utf-8")

    def _decrypt_api_key(self, encrypted_api_key: str) -> str | None:
        try:
            decrypted = self._fernet.decrypt(encrypted_api_key.encode("utf-8"))
        except InvalidToken:
            return None
        return decrypted.decode("utf-8").strip()

    async def _load_payload(self, session_id: str) -> dict[str, Any] | None:
        """Load raw session payload from Redis."""
        try:
            redis_client = self._redis_client()
            raw = await redis_client.get(self._key(session_id))
            if not raw:
                return None
            data: Any = json.loads(raw)
        except RedisError as exc:
            logger.exception("Failed to read LLM session from Redis.")
            raise RuntimeError("Failed to read LLM session, please retry later.") from exc
        except json.JSONDecodeError:
            logger.warning("LLM session payload is corrupted, session_id=%s", session_id)
            return None

        if not isinstance(data, dict):
            return None
        return data

    async def create_session(
        self,
        llm_profile: dict[str, str],
        *,
        owner_id: str,
        ttl_seconds: int | None = None,
    ) -> tuple[str, int]:
        """Create a session and return `(session_id, effective_ttl)`."""
        normalized_owner_id = owner_id.strip()
        if not normalized_owner_id:
            raise RuntimeError("Failed to create LLM session: owner_id is required.")

        raw_api_key = str(llm_profile.get("api_key", "")).strip()
        if not raw_api_key:
            raise RuntimeError("Failed to create LLM session: api_key is required.")

        session_id = uuid4().hex
        effective_ttl = self._ttl(ttl_seconds)

        stored_profile = {
            "base_url": str(llm_profile.get("base_url", "")).strip(),
            "model": str(llm_profile.get("model", "")).strip(),
            "embedding_model": str(llm_profile.get("embedding_model", "")).strip(),
            "api_key_encrypted": self._encrypt_api_key(raw_api_key),
            "encryption": "fernet-v1",
        }
        provider = llm_profile.get("provider")
        if isinstance(provider, str) and provider.strip():
            stored_profile["provider"] = provider.strip().lower()

        payload = {"llm_profile": stored_profile, "owner_id": normalized_owner_id}

        try:
            redis_client = self._redis_client()
            await redis_client.set(
                self._key(session_id),
                json.dumps(payload, ensure_ascii=False),
                ex=effective_ttl,
            )
            return session_id, effective_ttl
        except RedisError as exc:
            logger.exception("Failed to create LLM session in Redis.")
            raise RuntimeError("Failed to create LLM session, please retry later.") from exc

    async def get_session(self, session_id: str, *, owner_id: str) -> dict[str, str] | None:
        """Fetch session profile. Returns `None` when not found/expired/invalid."""
        normalized_id = session_id.strip()
        normalized_owner_id = owner_id.strip()
        if not normalized_id or not normalized_owner_id:
            return None

        data = await self._load_payload(normalized_id)
        if data is None:
            return None

        payload_owner_id = str(data.get("owner_id", "")).strip()
        if not payload_owner_id or payload_owner_id != normalized_owner_id:
            return None

        profile = data.get("llm_profile")
        if not isinstance(profile, dict):
            return None

        # Backward compatibility:
        # - Legacy payload: plaintext `api_key`
        # - New payload: encrypted `api_key_encrypted`
        api_key_plain = str(profile.get("api_key", "")).strip()
        if not api_key_plain:
            encrypted = str(profile.get("api_key_encrypted", "")).strip()
            if not encrypted:
                return None
            api_key_plain = self._decrypt_api_key(encrypted) or ""
            if not api_key_plain:
                logger.warning("Cannot decrypt LLM session api_key, session_id=%s", normalized_id)
                return None

        model_value = profile.get("model")
        if not isinstance(model_value, str) or not model_value.strip():
            return None

        base_url_value = profile.get("base_url", "")
        if base_url_value is None:
            normalized_base_url = ""
        elif isinstance(base_url_value, str):
            normalized_base_url = base_url_value.strip()
        else:
            return None

        embedding_value = profile.get("embedding_model", "")
        if embedding_value is None:
            normalized_embedding_model = ""
        elif isinstance(embedding_value, str):
            normalized_embedding_model = embedding_value.strip()
        else:
            return None

        normalized_profile: dict[str, str] = {
            "api_key": api_key_plain,
            "model": model_value.strip(),
            "base_url": normalized_base_url,
            "embedding_model": normalized_embedding_model,
        }

        provider = profile.get("provider")
        if isinstance(provider, str) and provider.strip():
            normalized_profile["provider"] = provider.strip().lower()

        return normalized_profile

    async def delete_session(self, session_id: str, *, owner_id: str) -> bool:
        """Delete a session by id. Returns `True` only when a key was removed."""
        normalized_id = session_id.strip()
        normalized_owner_id = owner_id.strip()
        if not normalized_id or not normalized_owner_id:
            return False

        data = await self._load_payload(normalized_id)
        if data is None:
            return False

        payload_owner_id = str(data.get("owner_id", "")).strip()
        if not payload_owner_id or payload_owner_id != normalized_owner_id:
            return False

        try:
            redis_client = self._redis_client()
            deleted = await redis_client.delete(self._key(normalized_id))
            return bool(deleted)
        except RedisError as exc:
            logger.exception("Failed to delete LLM session from Redis.")
            raise RuntimeError("Failed to delete LLM session, please retry later.") from exc
