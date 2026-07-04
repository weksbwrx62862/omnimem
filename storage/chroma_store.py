"""ChromaDB VectorStore 适配器（基于 storage/base.py 抽象接口）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omnimem.storage.base import VectorStore

logger = logging.getLogger(__name__)


def _suppress_telemetry_noise() -> None:
    """抑制 ChromaDB telemetry 噪音日志。"""
    for name in ("chromadb.telemetry.product.posthog", "chromadb.telemetry"):
        log = logging.getLogger(name)
        log.setLevel(logging.WARNING)
        log.addFilter(logging.Filter())


_suppress_telemetry_noise()


class ChromaVectorStore(VectorStore):
    """基于 ChromaDB 的向量存储，调用方提供 embeddings。"""

    def __init__(
        self,
        collection_name: str = "omnimem",
        persist_dir: str | Path = "/tmp/omnimem/storage/chroma",
        embedding_dimension: int | None = None,
    ):
        self._collection_name = collection_name
        self._persist_dir = Path(persist_dir)
        self._embedding_dimension = embedding_dimension
        self._client: Any = None
        self._collection: Any = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self._persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            logger.warning("chromadb not installed — ChromaVectorStore unavailable")
        except Exception as e:
            logger.warning("ChromaVectorStore init failed: %s", e)
        self._initialized = True

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """使用预计算 embeddings 写入 ChromaDB。"""
        self._ensure_initialized()
        if self._collection is None:
            return
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            self._persist_client()
        except Exception as e:
            logger.warning("ChromaVectorStore add failed: %s", e)
            raise RuntimeError(f"ChromaVectorStore add failed: {e}") from e

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """使用查询向量检索相似结果。"""
        self._ensure_initialized()
        if self._collection is None:
            return []
        try:
            count = self._collection.count()
            if count == 0:
                return []
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": min(top_k, count),
                "include": ["metadatas", "distances"],
            }
            if filters is not None:
                kwargs["where"] = filters
            results = self._collection.query(**kwargs)
            output: list[dict[str, Any]] = []
            metas = results.get("metadatas", [[]])[0] if results else []
            dists = results.get("distances", [[]])[0] if results else []
            ids = results.get("ids", [[]])[0] if results else []
            for doc_id, meta, dist in zip(ids, metas, dists, strict=False):
                entry = dict(meta) if meta else {}
                entry["id"] = doc_id
                entry["score"] = 1.0 - dist
                output.append(entry)
            return output
        except Exception as e:
            logger.warning("ChromaVectorStore search failed: %s", e)
            return []

    def delete(self, ids: list[str]) -> None:
        """删除指定 ID 的向量。"""
        self._ensure_initialized()
        if self._collection is None or not ids:
            return
        try:
            self._collection.delete(ids=ids)
            self._persist_client()
        except Exception as e:
            logger.warning("ChromaVectorStore delete failed: %s", e)

    def count(self) -> int:
        """返回文档数量。"""
        self._ensure_initialized()
        if self._collection is None:
            return 0
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    def reset(self) -> None:
        """重置集合。"""
        self._ensure_initialized()
        if self._client is None:
            return
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.warning("ChromaVectorStore reset failed: %s", e)

    def _persist_client(self) -> None:
        try:
            if self._client and hasattr(self._client, "persist"):
                self._client.persist()
        except Exception as e:
            logger.warning("ChromaVectorStore persist failed: %s", e)

    def close(self) -> None:
        """释放客户端。"""
        try:
            self._persist_client()
            self._collection = None
            self._client = None
            self._initialized = False
        except Exception as e:
            logger.warning("ChromaVectorStore close failed: %s", e)
