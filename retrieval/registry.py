"""检索通道插件化注册表。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omnimem.retrieval.base import BaseRetriever, RetrievalResult
from omnimem.retrieval.bm25 import BM25Retriever
from omnimem.retrieval.vector import VectorRetriever

logger = logging.getLogger(__name__)


class _GraphRetriever(BaseRetriever):
    """图谱检索占位实现（Phase 3 扩展）。"""

    def __init__(self, data_dir: Path | None = None, config: Any | None = None, **kwargs: Any) -> None:
        pass

    @property
    def name(self) -> str:
        return "graph"

    def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        return RetrievalResult(results=[], scores=[], channel=self.name)


class _TemporalRetriever(BaseRetriever):
    """时间检索占位实现（Phase 3 扩展）。"""

    def __init__(self, data_dir: Path | None = None, config: Any | None = None, **kwargs: Any) -> None:
        pass

    @property
    def name(self) -> str:
        return "temporal"

    def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        return RetrievalResult(results=[], scores=[], channel=self.name)


class RetrieverRegistry:
    """检索通道注册表，支持按名称注册与获取。"""

    def __init__(self) -> None:
        self._registry: dict[str, type[Any]] = {}

    def register(self, name: str, retriever_class: type[Any]) -> None:
        """注册检索通道类。"""
        self._registry[name] = retriever_class
        logger.debug("Registered retriever channel: %s", name)

    def get(self, name: str) -> type[Any] | None:
        """获取已注册的检索通道类。"""
        return self._registry.get(name)

    def list_channels(self) -> list[str]:
        """返回所有已注册通道名称。"""
        return list(self._registry.keys())

    def unregister(self, name: str) -> None:
        """注销指定通道。"""
        self._registry.pop(name, None)


def _build_default_registry() -> RetrieverRegistry:
    """构建默认注册表。"""
    registry = RetrieverRegistry()
    registry.register("vector", VectorRetriever)
    registry.register("bm25", BM25Retriever)
    registry.register("graph", _GraphRetriever)
    registry.register("temporal", _TemporalRetriever)
    return registry


# 全局默认注册表实例
DEFAULT_REGISTRY = _build_default_registry()

__all__ = [
    "DEFAULT_REGISTRY",
    "RetrieverRegistry",
    "_GraphRetriever",
    "_TemporalRetriever",
]
