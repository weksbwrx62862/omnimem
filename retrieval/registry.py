"""检索通道插件化注册表。"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import Any

from omnimem.governance.temporal_kg import TemporalKnowledgeGraph
from omnimem.retrieval.base import BaseRetriever, RetrievalResult
from omnimem.retrieval.bm25 import BM25Retriever
from omnimem.retrieval.vector import VectorRetriever

logger = logging.getLogger(__name__)


class _GraphRetriever(BaseRetriever):
    """基于 TemporalKnowledgeGraph 的图谱检索通道。

    延迟初始化 TKG 以避免启动时的 SQLite 开销，
    仅在首次时序查询时加载图谱实例。
    """

    # 实体提取正则：2-4字中文词组 + 英文大写词
    _ENTITY_PATTERNS = [
        re.compile(r'[\u4e00-\u9fff]{2,4}'),
        re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),
    ]

    def __init__(self, data_dir: Path | None = None, config: Any | None = None, **kwargs: Any) -> None:
        self._data_dir = data_dir
        self._config = config
        self._tkg: TemporalKnowledgeGraph | None = None

    @property
    def name(self) -> str:
        return "graph"

    def _ensure_tkg(self) -> TemporalKnowledgeGraph | None:
        """延迟初始化 TemporalKnowledgeGraph，失败时返回 None。"""
        if self._tkg is not None:
            return self._tkg
        if self._data_dir is None:
            logger.debug("图谱检索：未提供 data_dir，跳过初始化")
            return None
        try:
            self._tkg = TemporalKnowledgeGraph(self._data_dir, config=self._config)
            return self._tkg
        except Exception as e:
            logger.warning("TemporalKnowledgeGraph 初始化失败: %s", e)
            return None

    @classmethod
    def _extract_entities(cls, query: str) -> list[str]:
        """从查询中提取实体词。"""
        entities: list[str] = []
        for pattern in cls._ENTITY_PATTERNS:
            entities.extend(pattern.findall(query))
        return entities

    def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        """执行图谱检索：仅对时序查询生效。"""
        # 非时序查询直接返回空结果
        if not _TemporalRetriever.is_temporal_query(query):
            return RetrievalResult(results=[], scores=[], channel=self.name)

        # 延迟初始化 TKG
        tkg = self._ensure_tkg()
        if tkg is None:
            return RetrievalResult(results=[], scores=[], channel=self.name)

        # 从查询中提取实体词并检索
        entities = self._extract_entities(query)
        result_entries: list[dict[str, Any]] = []
        for entity in entities:
            try:
                triples = tkg.query_current(subject=entity, predicate="")
                for triple in triples:
                    result_entries.append({
                        "content": f"{triple.subject} {triple.predicate} {triple.object}",
                        "memory_id": triple.source_memory_id or triple.id,
                        "score": 0.5,
                        "source": "graph_temporal",
                        "metadata": {"valid_at": triple.valid_at, "confidence": triple.confidence},
                    })
            except Exception as e:
                logger.warning("图谱检索实体 '%s' 失败: %s", entity, e)

        scores = [0.5] * len(result_entries)
        return RetrievalResult(results=result_entries, scores=scores, channel=self.name)


class _TemporalRetriever(BaseRetriever):
    """时间检索通道：不独立检索，仅提供时序关键词检测和加权排序工具方法。

    实际的时序重排序由 HybridOrchestrator 在 RRF 融合后调用
    ``apply_temporal_rerank`` 完成，而非通过 search() 独立检索。
    """

    # 时序关键词（正则模式，忽略大小写）
    _TEMPORAL_PATTERN = re.compile(
        r'(?:'
        # 中文时序关键词
        r'最近|上次|什么时候|第一次|最后|之前|之后|刚才|以前|后来'
        r'|'
        # 英文时序关键词
        r'when\b|last\b|first\b|recently\b|before\b|after\b'
        r'|earlier\b|later\b|previous\b|latest\b'
        r')',
        re.IGNORECASE,
    )

    def __init__(self, data_dir: Path | None = None, config: Any | None = None, **kwargs: Any) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "temporal"

    def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        """TemporalRetriever 不独立检索，始终返回空结果。

        时序重排序逻辑由 HybridOrchestrator._apply_temporal_rerank() 执行。
        """
        return RetrievalResult(results=[], scores=[], channel=self.name)

    # ── 工具方法（供 HybridOrchestrator 调用）──

    @classmethod
    def is_temporal_query(cls, query: str) -> bool:
        """检测查询是否包含时序关键词。"""
        return bool(cls._TEMPORAL_PATTERN.search(query))

    @classmethod
    def compute_temporal_weight(
        cls,
        result: dict[str, Any],
        now: datetime | None = None,
        decay_lambda: float = 0.1,
    ) -> float:
        """计算单条检索结果的时间衰减权重。

        Args:
            result: 检索结果条目，需包含 metadata 中的时间戳
            now: 当前时间，默认为 UTC now
            decay_lambda: 衰减系数，默认 0.1（约 7 天半衰期）

        Returns:
            时间衰减权重，范围 [0, 1]，越近越大
        """
        if now is None:
            now = datetime.now(tz=timezone.utc)

        # 从 metadata 或顶层提取时间戳
        ts_str = ""
        metadata = result.get("metadata", {})
        if isinstance(metadata, dict):
            ts_str = metadata.get("created_at", "") or metadata.get("timestamp", "")
        if not ts_str:
            ts_str = result.get("created_at", "") or result.get("timestamp", "")

        if not ts_str:
            return 0.0

        try:
            ts = _parse_datetime(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            days_ago = (now - ts).total_seconds() / 86400.0
            if days_ago < 0:
                days_ago = 0.0
            return exp(-decay_lambda * days_ago)
        except Exception:
            logger.debug("时间戳解析失败: %s", ts_str)
            return 0.0

    @classmethod
    def apply_temporal_rerank(
        cls,
        query: str,
        results: list[dict[str, Any]],
        alpha: float = 0.5,
        decay_lambda: float = 0.1,
    ) -> list[dict[str, Any]]:
        """对检索结果应用时序重排序。

        仅当查询包含时序关键词时才生效，否则原样返回。

        Args:
            query: 用户查询
            results: RRF 融合后的检索结果
            alpha: 时间因素融合系数，默认 0.5
                   final_score = original_score * (1 + alpha * time_weight)
            decay_lambda: 时间衰减系数，默认 0.1

        Returns:
            重排序后的结果列表
        """
        if not results or not cls.is_temporal_query(query):
            return results

        now = datetime.now(tz=timezone.utc)
        for r in results:
            time_weight = cls.compute_temporal_weight(r, now=now, decay_lambda=decay_lambda)
            original_score = r.get("score", r.get("rrf_score", 0.0))
            final_score = original_score * (1.0 + alpha * time_weight)
            r["score"] = round(final_score, 6)
            r["_temporal_weight"] = round(time_weight, 4)
            r["_temporal_reranked"] = True

        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results


def _parse_datetime(ts_str: str) -> datetime:
    """尝试多种格式解析时间戳字符串。"""
    # ISO 8601 格式
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            pass
        try:
            return datetime.strptime(ts_str, fmt)
        except (ValueError, TypeError):
            pass
    raise ValueError(f"无法解析时间戳: {ts_str}")


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
