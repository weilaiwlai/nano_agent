"""NanoAgent 长期记忆模块。

该模块使用 ChromaDB 保存用户偏好向量，并按语义检索相关记忆片段，
用于后续注入到模型提示词中。
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import chromadb
from langchain_openai import OpenAIEmbeddings

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        PermissionDeniedError,
        RateLimitError,
    )
except Exception:  # noqa: BLE001
    class APITimeoutError(Exception):
        """当 openai 异常类不可用时的兜底超时异常。"""

    class APIConnectionError(Exception):
        """当 openai 异常类不可用时的兜底连接异常。"""

    class AuthenticationError(Exception):
        """当 openai 异常类不可用时的兜底鉴权异常。"""

    class BadRequestError(Exception):
        """当 openai 异常类不可用时的兜底请求异常。"""

    class PermissionDeniedError(Exception):
        """当 openai 异常类不可用时的兜底权限异常。"""

    class RateLimitError(Exception):
        """当 openai 异常类不可用时的兜底限流异常。"""


if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger("nanoagent.agent_service.memory")


class MemoryProviderError(RuntimeError):
    """表示外部模型服务导致的可诊断异常。"""

    def __init__(self, message: str, *, reason: str, retriable: bool = False) -> None:
        super().__init__(message)
        self.reason = reason
        self.retriable = retriable


class UserMemoryManager:
    """管理用户长期偏好记忆的向量存储、检索、查看与删除。"""

    def __init__(
        self,
        persist_path: str = None,
        collection_name: str = "user_preferences",
    ) -> None:
        """初始化 Embedding 模型与 Chroma 持久化集合。"""
        # 使用环境变量或默认的Windows路径
        self.persist_path = persist_path or os.getenv("CHROMA_PERSIST_PATH", "./data/chroma")
        self.collection_name = collection_name

        os.makedirs(self.persist_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_path)
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
        openai_api_key = os.getenv("OPENAI_API_KEY", os.getenv("QWEN_API_KEY"))
        openai_base_url = os.getenv(
            "OPENAI_BASE_URL",
            os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.default_embedding_model = embedding_model
        self.default_openai_base_url = openai_base_url

        if not openai_api_key:
            logger.warning("未配置 OPENAI_API_KEY/QWEN_API_KEY，Embedding 调用将失败。")

        try:
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                api_key=openai_api_key,
                base_url=openai_base_url,
                # DashScope/Qwen compatible endpoint expects raw text input,
                # not token-id arrays from tiktoken preprocessing.
                check_embedding_ctx_length=False,
                tiktoken_enabled=False,
            )
        except TypeError:
            # 兼容较老版本 langchain-openai 的参数命名。
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=openai_api_key,
                openai_api_base=openai_base_url,
                check_embedding_ctx_length=False,
                tiktoken_enabled=False,
            )
        self._embedding_clients: dict[tuple[str, str, str], OpenAIEmbeddings] = {}

        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        logger.info(
            "UserMemoryManager 初始化完成 | persist_path=%s | collection=%s | embedding_model=%s",
            self.persist_path,
            self.collection_name,
            embedding_model,
        )

    @staticmethod
    def _text_hash(text: str) -> str:
        """生成文本指纹，用于避免重复写入。"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        """安全解析时间戳，失败时返回最小时间。"""
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return datetime.min.replace(tzinfo=timezone.utc)

    def _find_existing_memory_id(self, user_id: str, text_hash: str) -> str | None:
        """按 user_id + text_hash 查找已存在的记忆记录。"""
        try:
            where_filter = {"$and": [{"user_id": user_id}, {"text_hash": text_hash}]}
            try:
                result = self.collection.get(
                    where=where_filter,
                    include=["metadatas"],
                    limit=1,
                )
            except TypeError:
                result = self.collection.get(
                    where=where_filter,
                    include=["metadatas"],
                )

            ids = result.get("ids") or []
            return ids[0] if ids else None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "查询重复记忆失败，继续正常写入 | user_id=%s | error=%s",
                user_id,
                exc,
            )
            return None

    def _normalize_embedding_profile(
        self,
        embedding_profile: dict[str, str] | None,
    ) -> dict[str, str] | None:
        """标准化会话级 embedding 配置。"""
        if not embedding_profile:
            return None

        api_key = str(embedding_profile.get("api_key", "")).strip()
        if not api_key:
            return None

        base_url = str(
            embedding_profile.get("base_url")
            or self.default_openai_base_url
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip()

        # 显式传入空 embedding_model 视为“本会话禁用 embedding”。
        if "embedding_model" in embedding_profile:
            model = str(embedding_profile.get("embedding_model", "")).strip()
            if not model:
                return None
        else:
            model = str(self.default_embedding_model or "text-embedding-v3").strip()

        if not base_url or not model:
            return None

        return {"api_key": api_key, "base_url": base_url, "embedding_model": model}

    def _get_embeddings_client(
        self,
        embedding_profile: dict[str, str] | None = None,
    ) -> OpenAIEmbeddings:
        """按会话配置返回 Embeddings 客户端（带缓存）。"""
        if embedding_profile is not None and "embedding_model" in embedding_profile:
            explicit_model = str(embedding_profile.get("embedding_model", "")).strip()
            if not explicit_model:
                raise MemoryProviderError(
                    "当前会话未配置 embedding 模型，无法执行语义向量操作。",
                    reason="bad_request",
                )

        normalized = self._normalize_embedding_profile(embedding_profile)
        if normalized is None:
            return self.embeddings

        cache_key = (
            normalized["api_key"],
            normalized["base_url"],
            normalized["embedding_model"],
        )
        cached = self._embedding_clients.get(cache_key)
        if cached is not None:
            return cached

        try:
            client = OpenAIEmbeddings(
                model=normalized["embedding_model"],
                api_key=normalized["api_key"],
                base_url=normalized["base_url"],
                check_embedding_ctx_length=False,
                tiktoken_enabled=False,
            )
        except TypeError:
            client = OpenAIEmbeddings(
                model=normalized["embedding_model"],
                openai_api_key=normalized["api_key"],
                openai_api_base=normalized["base_url"],
                check_embedding_ctx_length=False,
                tiktoken_enabled=False,
            )

        self._embedding_clients[cache_key] = client
        return client

    def save_preference(
        self,
        user_id: str,
        preference_text: str,
        *,
        embedding_profile: dict[str, str] | None = None,
    ) -> str | None:
        """保存用户偏好到向量库，并附带元数据。

        返回：新建或已存在的 memory_id；输入不合法时返回 None。
        外部模型服务异常时抛出 MemoryProviderError。
        """
        normalized_user_id = user_id.strip()
        normalized_text = preference_text.strip()
        if not normalized_user_id or not normalized_text:
            logger.warning("save_preference 跳过：user_id 或 preference_text 为空。")
            return None

        text_hash = self._text_hash(normalized_text)
        existing_id = self._find_existing_memory_id(normalized_user_id, text_hash)
        if existing_id:
            logger.info(
                "命中重复记忆，跳过写入 | user_id=%s | memory_id=%s",
                normalized_user_id,
                existing_id,
            )
            return existing_id

        timestamp = datetime.now(timezone.utc).isoformat()
        record_id = f"{normalized_user_id}:{uuid4().hex}"
        metadata: dict[str, Any] = {
            "user_id": normalized_user_id,
            "timestamp": timestamp,
            "text_hash": text_hash,
        }

        try:
            vector = self._get_embeddings_client(embedding_profile).embed_query(normalized_text)
            self.collection.add(
                ids=[record_id],
                documents=[normalized_text],
                embeddings=[vector],
                metadatas=[metadata],
            )
            logger.info(
                "偏好保存成功 | user_id=%s | record_id=%s",
                normalized_user_id,
                record_id,
            )
            return record_id
        except MemoryProviderError:
            raise
        except APITimeoutError as exc:
            logger.warning(
                "save_preference Embedding 超时 | user_id=%s | error=%s",
                normalized_user_id,
                exc,
            )
            raise MemoryProviderError(
                "Embedding 请求超时，请稍后重试。",
                reason="timeout",
                retriable=True,
            ) from exc
        except TimeoutError as exc:
            logger.warning(
                "save_preference 超时 | user_id=%s | error=%s",
                normalized_user_id,
                exc,
            )
            raise MemoryProviderError(
                "Embedding 请求超时，请稍后重试。",
                reason="timeout",
                retriable=True,
            ) from exc
        except (AuthenticationError, PermissionDeniedError) as exc:
            logger.warning(
                "save_preference 鉴权失败 | user_id=%s | error=%s",
                normalized_user_id,
                exc,
            )
            raise MemoryProviderError(
                "模型凭据校验失败，请检查 API Key、Base URL 与模型名称。",
                reason="auth",
            ) from exc
        except RateLimitError as exc:
            logger.warning(
                "save_preference 触发限流 | user_id=%s | error=%s",
                normalized_user_id,
                exc,
            )
            raise MemoryProviderError(
                "当前请求触发限流，请稍后重试。",
                reason="rate_limit",
                retriable=True,
            ) from exc
        except (BadRequestError, APIConnectionError) as exc:
            logger.warning(
                "save_preference 请求失败 | user_id=%s | error=%s",
                normalized_user_id,
                exc,
            )
            raise MemoryProviderError(
                "模型请求参数或网络连接异常，请检查配置后重试。",
                reason="bad_request",
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "save_preference 出现未知错误 | user_id=%s | error=%s",
                normalized_user_id,
                exc,
            )
            raise MemoryProviderError(
                "写入长期记忆失败，请稍后重试。",
                reason="unknown",
                retriable=True,
            ) from exc

    def list_memories(self, user_id: str, limit: int = 50) -> list[dict[str, str]]:
        """列出某个用户的长期记忆（按时间倒序）。"""
        normalized_user_id = user_id.strip()
        if not normalized_user_id:
            return []

        safe_limit = max(1, min(limit, 200))

        try:
            try:
                result = self.collection.get(
                    where={"user_id": normalized_user_id},
                    include=["documents", "metadatas"],
                    limit=safe_limit,
                )
            except TypeError:
                result = self.collection.get(
                    where={"user_id": normalized_user_id},
                    include=["documents", "metadatas"],
                )

            ids = result.get("ids") or []
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []

            items: list[dict[str, str]] = []
            for idx, memory_id in enumerate(ids):
                memory_text = str(documents[idx]) if idx < len(documents) else ""
                metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
                timestamp = str(metadata.get("timestamp", ""))
                items.append(
                    {
                        "memory_id": str(memory_id),
                        "preference_text": memory_text,
                        "timestamp": timestamp,
                    }
                )

            items.sort(
                key=lambda item: self._parse_timestamp(item.get("timestamp", "")),
                reverse=True,
            )
            return items[:safe_limit]
        except Exception as exc:  # noqa: BLE001
            logger.exception("list_memories 失败 | user_id=%s | error=%s", normalized_user_id, exc)
            return []

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """删除指定用户的一条记忆。"""
        normalized_user_id = user_id.strip()
        normalized_memory_id = memory_id.strip()
        if not normalized_user_id or not normalized_memory_id:
            return False

        try:
            result = self.collection.get(ids=[normalized_memory_id], include=["metadatas"])
            ids = result.get("ids") or []
            if not ids:
                logger.warning(
                    "delete_memory 未找到目标记录 | user_id=%s | memory_id=%s",
                    normalized_user_id,
                    normalized_memory_id,
                )
                return False

            metadatas = result.get("metadatas") or []
            metadata = metadatas[0] if metadatas and metadatas[0] else {}
            owner_id = str(metadata.get("user_id", "")).strip()

            if owner_id and owner_id != normalized_user_id:
                logger.warning(
                    "delete_memory 拒绝跨用户删除 | requester=%s | owner=%s | memory_id=%s",
                    normalized_user_id,
                    owner_id,
                    normalized_memory_id,
                )
                return False

            if not owner_id and not normalized_memory_id.startswith(f"{normalized_user_id}:"):
                logger.warning(
                    "delete_memory 无法确认归属，拒绝删除 | user_id=%s | memory_id=%s",
                    normalized_user_id,
                    normalized_memory_id,
                )
                return False

            self.collection.delete(ids=[normalized_memory_id])
            logger.info(
                "记忆删除成功 | user_id=%s | memory_id=%s",
                normalized_user_id,
                normalized_memory_id,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "delete_memory 失败 | user_id=%s | memory_id=%s | error=%s",
                normalized_user_id,
                normalized_memory_id,
                exc,
            )
            return False

    def retrieve_context(
        self,
        user_id: str,
        current_query: str,
        k: int = 3,
        *,
        embedding_profile: dict[str, str] | None = None,
    ) -> str:
        """按 user_id 过滤并检索 Top-k 语义相关记忆，附加最近记忆兜底。"""
        normalized_user_id = user_id.strip()
        normalized_query = current_query.strip()
        if not normalized_user_id or not normalized_query or k <= 0:
            return ""

        semantic_items: list[dict[str, str]] = []

        embedding_disabled = (
            embedding_profile is not None
            and "embedding_model" in embedding_profile
            and not str(embedding_profile.get("embedding_model", "")).strip()
        )
        if embedding_disabled:
            logger.info("retrieve_context 跳过语义检索 | user_id=%s | reason=embedding_disabled", normalized_user_id)

        if not embedding_disabled:
            try:
                query_vector = self._get_embeddings_client(embedding_profile).embed_query(normalized_query)
                result = self.collection.query(
                    query_embeddings=[query_vector],
                    n_results=max(k, 3),
                    where={"user_id": normalized_user_id},
                    include=["documents", "metadatas", "distances"],
                )

                ids_nested = result.get("ids") or []
                documents_nested = result.get("documents") or []
                metadatas_nested = result.get("metadatas") or []

                ids = ids_nested[0] if ids_nested else []
                documents = documents_nested[0] if documents_nested else []
                metadatas = metadatas_nested[0] if metadatas_nested else []

                for idx, memory_id in enumerate(ids):
                    memory_text = str(documents[idx]) if idx < len(documents) else ""
                    metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
                    timestamp = str(metadata.get("timestamp", ""))
                    semantic_items.append(
                        {
                            "memory_id": str(memory_id),
                            "preference_text": memory_text,
                            "timestamp": timestamp,
                            "source": "semantic",
                        }
                    )
            except APITimeoutError as exc:
                logger.warning(
                    "retrieve_context Embedding 超时 | user_id=%s | error=%s",
                    normalized_user_id,
                    exc,
                )
            except TimeoutError as exc:
                logger.warning(
                    "retrieve_context 超时 | user_id=%s | error=%s",
                    normalized_user_id,
                    exc,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "retrieve_context 语义检索失败 | user_id=%s | error=%s",
                    normalized_user_id,
                    exc,
                )

        recent_items = self.list_memories(normalized_user_id, limit=max(k * 2, 6))
        for item in recent_items:
            item["source"] = "recent"

        merged: list[dict[str, str]] = []
        seen_ids: set[str] = set()

        for item in semantic_items + recent_items:
            memory_id = item.get("memory_id", "")
            if not memory_id or memory_id in seen_ids:
                continue
            if not item.get("preference_text", "").strip():
                continue
            merged.append(item)
            seen_ids.add(memory_id)
            if len(merged) >= max(k, 5):
                break

        if not merged:
            logger.info("未命中记忆 | user_id=%s", normalized_user_id)
            return ""

        lines: list[str] = []
        for item in merged:
            ts = item.get("timestamp", "") or "unknown_time"
            src = "语义相关" if item.get("source") == "semantic" else "近期记录"
            lines.append(f"- [{src}][{ts}] {item.get('preference_text', '')}")

        context_block = "用户长期记忆（请优先参考）：\n" + "\n".join(lines)
        logger.info(
            "记忆检索完成 | user_id=%s | semantic_hits=%d | total_used=%d",
            normalized_user_id,
            len(semantic_items),
            len(merged),
        )
        return context_block