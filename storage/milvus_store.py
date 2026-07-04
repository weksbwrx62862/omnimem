"""Milvus VectorStore 适配器（可选依赖）。

pymilvus 未安装时，初始化阶段即抛出明确错误，便于工厂层优雅降级。
"""

from __future__ import annotations

import logging
from typing import Any

from omnimem.storage.base import VectorStore

logger = logging.getLogger(__name__)


class MilvusVectorStore(VectorStore):
    """基于 Milvus 的向量存储，调用方提供 embeddings。

    默认使用 "omnimem" collection，主键字段为 "id"，向量字段为 "embedding"。
    可通过配置调整 collection / uri / token / dim 等参数。
    """

    def __init__(
        self,
        collection_name: str = "omnimem",
        uri: str = "http://localhost:19530",
        token: str = "",
        embedding_dimension: int = 384,
        metric_type: str = "COSINE",
        consistency_level: str = "Bounded",
    ):
        self._collection_name = collection_name
        self._uri = uri
        self._token = token
        self._embedding_dimension = embedding_dimension
        self._metric_type = metric_type
        self._consistency_level = consistency_level
        self._client: Any = None
        self._initialized = False

    def _ensure_runtime(self) -> Any:
        """确保 pymilvus 已安装，返回模块。"""
        try:
            from pymilvus import MilvusClient
        except ImportError as e:
            raise RuntimeError(
                "pymilvus 未安装，无法使用 MilvusVectorStore。"
                "请执行: pip install pymilvus"
            ) from e
        return MilvusClient

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        milvus_client_cls = self._ensure_runtime()
        kwargs: dict[str, Any] = {"uri": self._uri}
        if self._token:
            kwargs["token"] = self._token
        self._client = milvus_client_cls(**kwargs)
        if not self._client.has_collection(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                dimension=self._embedding_dimension,
                primary_field_name="id",
                vector_field_name="embedding",
                metric_type=self._metric_type,
                consistency_level=self._consistency_level,
                auto_id=False,
            )
        self._initialized = True

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """使用预计算 embeddings 写入 Milvus。"""
        self._ensure_initialized()
        if not ids:
            return
        try:
            data: list[dict[str, Any]] = []
            for doc_id, emb, meta in zip(ids, embeddings, metadatas, strict=False):
                record = dict(meta)
                record["id"] = doc_id
                record["embedding"] = emb
                data.append(record)
            self._client.upsert(collection_name=self._collection_name, data=data)
        except Exception as e:
            logger.warning("MilvusVectorStore add failed: %s", e)
            raise RuntimeError(f"MilvusVectorStore add failed: {e}") from e

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """使用查询向量检索相似结果。"""
        self._ensure_initialized()
        try:
            count = self.count()
            if count == 0:
                return []
            kwargs: dict[str, Any] = {
                "collection_name": self._collection_name,
                "data": [query_embedding],
                "limit": min(top_k, max(1, count)),
                "output_fields": ["*"],
            }
            if filters is not None:
                kwargs["filter"] = self._build_filter(filters)
            results = self._client.search(**kwargs)
            output: list[dict[str, Any]] = []
            # MilvusClient.search 返回 [[{...}, ...]] 结构
            for group in results:
                for item in group:
                    entity = dict(item.get("entity", {}))
                    entity["id"] = item.get("id", entity.get("id", ""))
                    entity["score"] = float(item.get("distance", 0.0))
                    output.append(entity)
            return output
        except Exception as e:
            logger.warning("MilvusVectorStore search failed: %s", e)
            return []

    def delete(self, ids: list[str]) -> None:
        """删除指定 ID 的向量。"""
        self._ensure_initialized()
        if not ids:
            return
        try:
            self._client.delete(
                collection_name=self._collection_name,
                ids=ids,
            )
        except Exception as e:
            logger.warning("MilvusVectorStore delete failed: %s", e)

    def count(self) -> int:
        """返回文档数量。"""
        self._ensure_initialized()
        try:
            stats = self._client.get_collection_stats(self._collection_name)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    def reset(self) -> None:
        """清空 collection。"""
        self._ensure_initialized()
        try:
            if self._client.has_collection(self._collection_name):
                self._client.drop_collection(self._collection_name)
            self._client.create_collection(
                collection_name=self._collection_name,
                dimension=self._embedding_dimension,
                primary_field_name="id",
                vector_field_name="embedding",
                metric_type=self._metric_type,
                consistency_level=self._consistency_level,
                auto_id=False,
            )
        except Exception as e:
            logger.warning("MilvusVectorStore reset failed: %s", e)

    def close(self) -> None:
        """释放客户端。"""
        try:
            if self._client is not None:
                self._client.close()
                self._client = None
                self._initialized = False
        except Exception as e:
            logger.warning("MilvusVectorStore close failed: %s", e)

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> str:
        """将 dict 过滤条件转换为 Milvus filter 表达式。

        当前仅支持简单的等值过滤，复杂表达式由调用方自行构造后传入。
        """
        conditions: list[str] = []
        for key, value in filters.items():
            if isinstance(value, str):
                conditions.append(f'{key} == "{value}"')
            elif isinstance(value, bool):
                conditions.append(f"{key} == {str(value).lower()}")
            elif isinstance(value, (int, float)):
                conditions.append(f"{key} == {value}")
        return " and ".join(conditions) if conditions else ""
