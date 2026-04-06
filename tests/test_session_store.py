"""Unit tests for LLMSessionStore security and compatibility behavior."""

from __future__ import annotations

import json
import unittest
from typing import Any

try:
    from src.session_store import LLMSessionStore
    _SESSION_STORE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    try:
        from agent_service.src.session_store import LLMSessionStore
        _SESSION_STORE_IMPORT_ERROR = None
    except Exception as nested_exc:  # noqa: BLE001
        LLMSessionStore = None  # type: ignore[assignment]
        _SESSION_STORE_IMPORT_ERROR = nested_exc


class _FakeRedis:
    """Minimal async Redis stub for unit tests."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        _ = ex
        self.data[key] = value
        return True

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def delete(self, key: str) -> int:
        existed = key in self.data
        self.data.pop(key, None)
        return 1 if existed else 0


@unittest.skipUnless(
    LLMSessionStore is not None,
    f"session_store dependencies not available: {_SESSION_STORE_IMPORT_ERROR}",
)
class LLMSessionStoreTests(unittest.IsolatedAsyncioTestCase):
    """Verify encryption-at-rest and backward compatibility."""

    async def asyncSetUp(self) -> None:
        self.store = LLMSessionStore(
            "redis://unit-test",
            master_key="unit-test-master-key",
            default_ttl_seconds=3600,
            max_ttl_seconds=86400,
        )
        self.fake_redis = _FakeRedis()
        # Inject fake redis client to avoid external dependencies.
        self.store._redis = self.fake_redis  # type: ignore[attr-defined]

    async def test_create_session_encrypts_api_key_before_persist(self) -> None:
        profile = {
            "provider": "qwen",
            "api_key": "test-key-unit-plaintext",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3.5-plus",
            "embedding_model": "text-embedding-v3",
        }

        session_id, effective_ttl = await self.store.create_session(
            profile,
            owner_id="user_001",
        )

        self.assertTrue(session_id)
        self.assertGreaterEqual(effective_ttl, 60)

        raw = await self.fake_redis.get(self.store._key(session_id))
        self.assertIsNotNone(raw)
        payload: dict[str, Any] = json.loads(str(raw))
        persisted_profile = payload["llm_profile"]

        self.assertEqual(payload["owner_id"], "user_001")
        self.assertNotIn("api_key", persisted_profile)
        self.assertIn("api_key_encrypted", persisted_profile)
        self.assertEqual(persisted_profile.get("encryption"), "fernet-v1")
        self.assertNotEqual(persisted_profile["api_key_encrypted"], profile["api_key"])

    async def test_get_session_decrypts_encrypted_profile(self) -> None:
        profile = {
            "provider": "qwen",
            "api_key": "test-key-unit-encrypted",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3.5-plus",
            "embedding_model": "text-embedding-v3",
        }
        session_id, _ = await self.store.create_session(profile, owner_id="user_001")

        loaded = await self.store.get_session(session_id, owner_id="user_001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["api_key"], "test-key-unit-encrypted")
        self.assertEqual(loaded["model"], "qwen3.5-plus")
        self.assertEqual(loaded["provider"], "qwen")

    async def test_get_session_supports_legacy_plaintext_payload(self) -> None:
        legacy_id = "legacy_plaintext_session"
        legacy_payload = {
            "owner_id": "user_001",
            "llm_profile": {
                "provider": "qwen",
                "api_key": "legacy-plaintext-key",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen3.5-plus",
                "embedding_model": "text-embedding-v3",
            },
        }
        await self.fake_redis.set(
            self.store._key(legacy_id),
            json.dumps(legacy_payload, ensure_ascii=False),
            ex=300,
        )

        loaded = await self.store.get_session(legacy_id, owner_id="user_001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["api_key"], "legacy-plaintext-key")
        self.assertEqual(loaded["provider"], "qwen")

    async def test_get_session_allows_empty_embedding_model(self) -> None:
        profile = {
            "provider": "deepseek",
            "api_key": "test-key-no-embedding",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "embedding_model": "",
        }
        session_id, _ = await self.store.create_session(profile, owner_id="user_001")

        loaded = await self.store.get_session(session_id, owner_id="user_001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["embedding_model"], "")
        self.assertEqual(loaded["provider"], "deepseek")

    async def test_get_session_enforces_owner_binding(self) -> None:
        profile = {
            "provider": "qwen",
            "api_key": "test-key-owner-bound",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3.5-plus",
            "embedding_model": "text-embedding-v3",
        }
        session_id, _ = await self.store.create_session(profile, owner_id="user_001")

        loaded = await self.store.get_session(session_id, owner_id="user_002")
        self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
