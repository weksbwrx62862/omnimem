"""检索通道统一抽象接口。"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    """检索结果统一数据结构。

    Attributes:
        results: 检索命中的文档列表，每个文档为 dict。
        scores: 与 results 一一对应的相似度/相关度分数。
        channel: 检索通道名称，例如 "vector"、"bm25"。
    """

    results: list[dict[str, Any]]
    scores: list[float]
    channel: str


class BaseRetriever(ABC):
    """所有检索通道的抽象基类。

    注：为保持与现有同步调用代码的兼容，``search`` 为同步方法；
    异步入口使用 ``asearch``，默认会在线程池中执行 ``search``。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """检索通道名称。"""
        ...

    @abstractmethod
    def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        """同步检索。"""
        ...

    async def asearch(self, query: str, **kwargs: Any) -> RetrievalResult:
        """异步检索，默认在线程池中执行同步 ``search``。"""
        return await asyncio.to_thread(self.search, query, **kwargs)
