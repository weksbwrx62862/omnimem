"""多通道检索编排与 RRF 融合。

负责通道并行调度、RRF/additive 融合、类型加权、缓存管理、索引重建等核心检索逻辑。
共享线程池已拆分到 executor.py，融合逻辑拆分到 fusion.py，缓存管理拆分到 cache.py。
"""

from __future__ import annotations

import asyncio as _asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from omnimem.retrieval.base import BaseRetriever
from omnimem.retrieval.cache import QueryCacheMixin
from omnimem.retrieval.executor import (  # noqa: F401 - compat re-export for tests
    _shared_executor,
    _shared_executor_lock,
    _shared_executor_refs,
)
from omnimem.retrieval.executor import (
    acquire_shared_executor as _acquire_shared_executor,
)
from omnimem.retrieval.executor import (
    release_shared_executor as _release_shared_executor,
)
from omnimem.retrieval.fusion import FusionMixin
from omnimem.retrieval.index_admin import IndexAdminMixin
from omnimem.retrieval.query_quality import is_garbage_query
from omnimem.retrieval.synonym_expander import SynonymExpander
from omnimem.retrieval.vector_store import _emit
from omnimem.utils.cache import CacheKeyBuilder

logger = logging.getLogger(__name__)


class HybridOrchestrator(FusionMixin, QueryCacheMixin, IndexAdminMixin):
    """混合检索编排器：承载多通道检索、融合与缓存管理逻辑。"""


    # ★ Task 2: 知识更新标记的默认分数提升
    _DEFAULT_UPDATED_BOOST: float = 0.3

    def __init__(self, facade: Any) -> None:
        self._facade = facade
        self._synonym_expander = SynonymExpander(facade._synonym_map)
        self._executor = self._create_executor()
        # ★ Task 2: 从 config 读取 updated_boost，默认 0.3
        config = getattr(facade, "_config", None)
        if config is not None:
            self._updated_boost = float(config.get("updated_boost", self._DEFAULT_UPDATED_BOOST))
        else:
            self._updated_boost = self._DEFAULT_UPDATED_BOOST
        # ★ Task 3: 查询增强配置
        self._query_expansion_enabled = bool(config.get("query_expansion_enabled", True)) if config else True
        self._entity_boost_weight = float(config.get("entity_boost_weight", 1.5)) if config else 1.5
        # ★ 偏好查询改写：对包含偏好信号词的查询追加偏好同义词
        self._preference_rewrite_enabled = bool(config.get("preference_rewrite_enabled", True)) if config else True

    def _create_executor(self) -> ThreadPoolExecutor:
        """获取检索线程池（★ P2: 全进程共享，max_workers 可通过配置调整）。"""
        config = getattr(self._facade, "_config", None)
        default_workers = min(32, (os.cpu_count() or 1) + 4)
        if config is not None:
            try:
                max_workers = int(config.get("retrieval_max_workers", default_workers))
            except Exception:
                max_workers = default_workers
        else:
            max_workers = default_workers
        max_workers = max(1, max_workers)
        return _acquire_shared_executor(max_workers)

    def shutdown(self) -> None:
        """释放共享线程池引用（最后一个实例释放时真正关闭）。"""
        if self._executor is not None:
            _release_shared_executor(wait=True)
            self._executor = None

    # ── 通道级检索 ──

    def vector_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """向量检索通道。"""
        return self._facade._vector.search(query, top_k=top_k)

    def bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """BM25 检索通道（含同义词扩展）。"""
        return self._synonym_expander.search(self._facade._bm25, query, top_k)

    @staticmethod
    def _search_base_retriever(
        retriever: BaseRetriever, query: str, top_k: int
    ) -> list[dict[str, Any]]:
        """统一调用 BaseRetriever 并将 RetrievalResult 转为文档列表。"""
        return retriever.search(query, top_k=top_k).results

    def catalog_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """目录递归检索通道（内化 OpenViking find()）。"""
        if not self._facade._catalog:
            return []
        try:
            return self._facade._catalog.search(query, top_k=top_k)
        except Exception as e:
            logger.warning("Catalog search failed (non-fatal): %s", e)
            return []

    def dispatch_channels(
        self,
        query: str,
        top_k: int,
        allowed_channels: set[str] | None,
        trace: Any,
        bm25_query: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """执行多通道并行检索，按 recall_strategy 分流 + 超时降级。

        Args:
            query: 原始查询（用于向量/目录等通道）
            top_k: 每通道返回数量
            allowed_channels: 限制检索通道集合
            trace: 追踪对象
            bm25_query: BM25 通道增强查询（含同义扩展词），为 None 时使用原始 query
        """
        # BM25 使用增强后的查询，其他通道使用原始查询
        effective_bm25_query = bm25_query if bm25_query is not None else query
        channel_results: dict[str, list[dict[str, Any]]] = {}
        facade = self._facade

        if facade._recall_strategy == "keyword":
            if "bm25" in facade._channels and (not allowed_channels or "bm25" in allowed_channels):
                channel_results["bm25"] = self.bm25_search(effective_bm25_query, top_k)
        elif facade._recall_strategy == "embedding":
            if "vector" in facade._channels and (not allowed_channels or "vector" in allowed_channels):
                if facade._vector_breaker.should_skip():
                    logger.warning("CircuitBreaker OPEN: skipping vector search, no results")
                    _emit("[OmniMem] ⚠ 向量检索不可用，已降级到关键词模式")
                else:
                    try:
                        channel_results["vector"] = self.vector_search(query, top_k)
                        facade._vector_breaker.record_success()
                    except Exception:
                        facade._vector_breaker.record_failure()
        else:
            timeout_sec = facade._recall_timeout_ms / 1000.0
            futures: dict[str, Any] = {}

            for name, (retriever, _weight) in facade._channels.items():
                if allowed_channels and name not in allowed_channels:
                    continue
                if name == "vector" and facade._vector_breaker.should_skip():
                    logger.warning(
                        "CircuitBreaker OPEN: skipping vector search, degrading to BM25+Catalog only"
                    )
                    _emit("[OmniMem] ⚠ 向量检索不可用，已降级到关键词模式")
                    continue
                if name == "bm25":
                    futures[name] = self._executor.submit(self.bm25_search, effective_bm25_query, top_k)
                elif isinstance(retriever, BaseRetriever):
                    futures[name] = self._executor.submit(
                        self._search_base_retriever, retriever, query, top_k
                    )
                else:
                    futures[name] = self._executor.submit(retriever.search, query, top_k=top_k)

            if facade._catalog and (not allowed_channels or "catalog" in allowed_channels):
                futures["catalog"] = self._executor.submit(self.catalog_search, query, top_k)

            for name, future in futures.items():
                try:
                    channel_results[name] = future.result(timeout=timeout_sec)
                    if name == "vector":
                        facade._vector_breaker.record_success()
                except (TimeoutError, Exception) as e:
                    channel_results[name] = []
                    future.cancel()
                    if name == "vector":
                        facade._vector_breaker.record_failure()
                    logger.warning("%s search timeout/degraded (%dms): %s",
                                   name, facade._recall_timeout_ms, e)

            if trace:
                for ch_name, ch_results in channel_results.items():
                    trace.add_step("channel_search", channel=ch_name,
                                   output_count=len(ch_results))

            active = [n for n, r in channel_results.items() if r]
            if len(active) == 1 and active[0] != "vector":
                logger.info("Recall degraded: only %s channel has results", active[0])
                _emit("[OmniMem] ⚠ 向量检索不可用，已降级到关键词模式")

        return channel_results

    # ── 主检索入口 ──

    def search(
        self,
        query: str,
        max_tokens: int = 1500,
        mode: str = "rag",
        top_k: int = 40,
        allowed_channels: set[str] | None = None,
        enable_trace: bool = False,
    ) -> list[dict[str, Any]]:
        """混合检索：向量 + BM25 + RRF 融合。"""
        from omnimem.retrieval.trace import SearchTrace
        trace = SearchTrace(query) if enable_trace else None

        is_garbage = is_garbage_query(query)
        try:
            doc_count = self._facade._vector.count()
        except Exception:
            doc_count = 0

        if is_garbage:
            top_k = min(top_k, 2)
            max_tokens = min(max_tokens, 200)
            if doc_count >= 100:
                top_k = 1
            elif doc_count >= 30:
                top_k = min(top_k, 1)

        if mode == "llm" and not is_garbage:
            top_k = max(top_k, 20)
            max_tokens = max(max_tokens, 3000)

        # ★ COUNT 查询（how many/how much）KMAG优化：提升 BM25 权重 + 扩大 top_k
        # ★ 修复 C7：原实现临时修改共享 _source_weights，并发检索会读到污染权重。
        #   改为 try/finally 保证恢复，并加锁保护读取-修改-恢复的原子性。
        import re as _re_count
        _is_count = bool(_re_count.match(r'^(how many|how much)\b', query.strip().lower()))
        _orig_count_weights = {}
        _count_lock = getattr(self._facade, "_source_weights_lock", None)
        if _is_count:
            top_k = max(top_k, 60)
            if self._facade._source_weights is not None:
                if _count_lock is not None:
                    with _count_lock:
                        _orig_count_weights = dict(self._facade._source_weights)
                        self._facade._source_weights["bm25"] = 5.0
                else:
                    _orig_count_weights = dict(self._facade._source_weights)
                    self._facade._source_weights["bm25"] = 5.0

        cache_key = CacheKeyBuilder.build_recall_key(query, max_tokens, mode, top_k)
        cached = self.check_cache(cache_key)
        if cached is not None:
            return cached

        # ★ Task 3.1: 查询同义扩展 — 仅影响 BM25 通道
        bm25_query = query
        if self._query_expansion_enabled:
            expanded_terms = self._synonym_expander.expand(query)
            if expanded_terms:
                bm25_query = query + " " + " ".join(expanded_terms)

        # ★ 偏好查询改写：对偏好类查询追加偏好同义词，缩小语义鸿沟
        if self._preference_rewrite_enabled:
            pref_signals = {"recommend", "suggest", "prefer", "favorite", "favourite",
                             "best", "ideal", "suitable", "looking for"}
            query_lower = query.lower()
            if any(sig in query_lower for sig in pref_signals):
                bm25_query = bm25_query + " prefer like enjoy"

        channel_results = self.dispatch_channels(query, top_k, allowed_channels, trace, bm25_query=bm25_query)
        results = self.fuse_and_filter(
            query, channel_results,
            is_garbage=is_garbage, doc_count=doc_count,
            top_k=top_k, max_tokens=max_tokens, trace=trace,
        )
        self.set_cache(cache_key, results)

        # ★ COUNT 查询：恢复原始权重，防止污染后续检索（修复 C7：加锁恢复）
        if _is_count and self._facade._source_weights is not None:
            if _count_lock is not None:
                with _count_lock:
                    self._facade._source_weights.clear()
                    self._facade._source_weights.update(_orig_count_weights)
            else:
                self._facade._source_weights.clear()
                self._facade._source_weights.update(_orig_count_weights)

        if trace and results:
            results[-1]["_trace"] = trace.to_dict()

        return results

    async def async_search(
        self,
        query: str,
        max_tokens: int = 1500,
        mode: str = "rag",
        top_k: int = 40,
        allowed_channels: set[str] | None = None,
        enable_trace: bool = False,
    ) -> list[dict[str, Any]]:
        """异步混合检索：使用 asyncio 并行执行各通道检索。"""
        from omnimem.retrieval.trace import SearchTrace
        trace = SearchTrace(query) if enable_trace else None
        facade = self._facade

        is_garbage = is_garbage_query(query)
        try:
            doc_count = facade._vector.count()
        except Exception:
            doc_count = 0

        if is_garbage:
            top_k = min(top_k, 2)
            max_tokens = min(max_tokens, 200)
            if doc_count >= 100:
                top_k = 1
            elif doc_count >= 30:
                top_k = min(top_k, 1)

        if mode == "llm" and not is_garbage:
            top_k = max(top_k, 20)
            max_tokens = max(max_tokens, 3000)

        cache_key = CacheKeyBuilder.build_recall_key(query, max_tokens, mode, top_k)
        cached = self.check_cache(cache_key)
        if cached is not None:
            logger.debug("HybridRetriever async query cache hit: %s", query[:50])
            return cached

        # ★ Task 3.1: 查询同义扩展 — 仅影响 BM25 通道
        bm25_query = query
        if self._query_expansion_enabled:
            expanded_terms = self._synonym_expander.expand(query)
            if expanded_terms:
                bm25_query = query + " " + " ".join(expanded_terms)

        # ★ 偏好查询改写：对偏好类查询追加偏好同义词，缩小语义鸿沟
        if self._preference_rewrite_enabled:
            pref_signals = {"recommend", "suggest", "prefer", "favorite", "favourite",
                             "best", "ideal", "suitable", "looking for"}
            query_lower = query.lower()
            if any(sig in query_lower for sig in pref_signals):
                bm25_query = bm25_query + " prefer like enjoy"

        channel_results: dict[str, list[dict[str, Any]]] = {}

        if facade._recall_strategy == "keyword":
            if "bm25" in facade._channels and (not allowed_channels or "bm25" in allowed_channels):
                channel_results["bm25"] = await _asyncio.to_thread(self.bm25_search, bm25_query, top_k)
        elif facade._recall_strategy == "embedding":
            if "vector" in facade._channels and (not allowed_channels or "vector" in allowed_channels):
                if facade._vector_breaker.should_skip():
                    logger.warning("CircuitBreaker OPEN: skipping async vector search")
                    _emit("[OmniMem] ⚠ 向量检索不可用，已降级到关键词模式")
                else:
                    try:
                        channel_results["vector"] = await _asyncio.to_thread(
                            self.vector_search, query, top_k
                        )
                        facade._vector_breaker.record_success()
                    except Exception:
                        facade._vector_breaker.record_failure()
        else:
            async_tasks: dict[str, Any] = {}

            for name, (retriever, _weight) in facade._channels.items():
                if allowed_channels and name not in allowed_channels:
                    continue
                if name == "vector" and facade._vector_breaker.should_skip():
                    logger.warning(
                        "CircuitBreaker OPEN: skipping async vector search, degrading to BM25+Catalog only"
                    )
                    _emit("[OmniMem] ⚠ 向量检索不可用，已降级到关键词模式")
                    continue
                if name == "bm25":
                    async_tasks[name] = _asyncio.to_thread(self.bm25_search, bm25_query, top_k)
                elif isinstance(retriever, BaseRetriever):
                    async_tasks[name] = _asyncio.to_thread(
                        self._search_base_retriever, retriever, query, top_k
                    )
                else:
                    async_tasks[name] = _asyncio.to_thread(retriever.search, query, top_k=top_k)

            if facade._catalog and (not allowed_channels or "catalog" in allowed_channels):
                async_tasks["catalog"] = _asyncio.to_thread(self.catalog_search, query, top_k)

            if async_tasks:
                task_names = list(async_tasks.keys())
                task_coros = list(async_tasks.values())
                results_list = await _asyncio.gather(*task_coros, return_exceptions=True)
                for name, result in zip(task_names, results_list):
                    if isinstance(result, Exception):
                        channel_results[name] = []
                        if name == "vector":
                            facade._vector_breaker.record_failure()
                        logger.warning("async %s search failed: %s", name, result)
                    else:
                        channel_results[name] = result
                        if name == "vector":
                            facade._vector_breaker.record_success()

            if trace:
                for ch_name, ch_results in channel_results.items():
                    trace.add_step("channel_search", channel=ch_name,
                                   output_count=len(ch_results))

            active = [n for n, r in channel_results.items() if r]
            if len(active) == 1 and active[0] != "vector":
                logger.info("Async recall degraded: only %s channel has results", active[0])
                _emit("[OmniMem] ⚠ 向量检索不可用，已降级到关键词模式")

        results = self.rrf_fuse(
            query,
            channel_results,
            is_garbage=is_garbage,
            doc_count=doc_count,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        if trace:
            trace.add_step("rrf_fuse",
                           input_count=sum(len(r) for r in channel_results.values()),
                           output_count=len(results))

        results = [r for r in results if r.get("source") != "sync_turn"]
        # ★ ADD-only 策略：过滤已被 superseded 的旧记忆，只保留最新版本
        results = [r for r in results if not (r.get("is_superseded") or r.get("metadata", {}).get("is_superseded"))]
        results = self.supplement_low_recall_types(query, results, top_k)
        results = self.apply_type_boost(results, updated_boost=self._updated_boost,
                                        query=query, entity_boost_weight=self._entity_boost_weight)
        results = self._apply_temporal_rerank(query, results)
        self.set_cache(cache_key, results)

        if trace and results:
            results[-1]["_trace"] = trace.to_dict()

        return results
