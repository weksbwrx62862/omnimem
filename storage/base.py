"""向量存储统一抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """向量数据库抽象基类，调用方负责计算 embeddings。"""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """添加向量与元数据。"""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """使用查询向量检索最相似的 top_k 条结果。"""
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """删除指定 id 的向量。"""
        ...

    @abstractmethod
    def count(self) -> int:
        """返回向量库中的文档数量。"""
        ...
