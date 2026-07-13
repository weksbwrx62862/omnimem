"""多通道检索编排与 RRF 融合。

负责通道并行调度、RRF/additive 融合、类型加权、缓存管理、索引重建等核心检索逻辑。
"""

from __future__ import annotations

import asyncio as _asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from omnimem.retrieval.base import BaseRetriever
from omnimem.retrieval.query_quality import is_garbage_query, trim_to_budget
from omnimem.retrieval.synonym_expander import SynonymExpander
from omnimem.retrieval.vector_store import _emit
from omnimem.utils.cache import CacheKeyBuilder

logger = logging.getLogger(__name__)


class HybridOrchestrator:
    """混合检索编排器：承载多通道检索、融合与缓存管理逻辑。"""

    # reasoning/action 类型的记忆包含高价值信息但关键词密度低，
    # 需要提高权重避免被 fact/preference 等高频词类型淹没。
    _TYPE_BOOST: dict[str, float] = {
        "reasoning": 1.3,
        "action": 1.3,
        "correction": 1.1,
    }

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

    def _create_executor(self) -> ThreadPoolExecutor:
        """创建实例级检索线程池，max_workers 可通过配置调整。"""
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
        return ThreadPoolExecutor(max_workers=max_workers)

    def shutdown(self) -> None:
        """关闭检索线程池，等待已提交任务完成。"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
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

    # ── 融合与过滤 ──

    def fuse_and_filter(
        self,
        query: str,
        channel_results: dict[str, list[dict[str, Any]]],
        *,
        is_garbage: bool,
        doc_count: int,
        top_k: int,
        max_tokens: int,
        trace: Any = None,
        fusion_mode: str = "rrf",
    ) -> list[dict[str, Any]]:
        """融合 + 类型补充 + 时序重排序 + 过滤。"""
        if fusion_mode == "additive":
            results = self.additive_fuse(
                query, channel_results,
                is_garbage=is_garbage,
                top_k=top_k, max_tokens=max_tokens,
            )
            if trace:
                trace.add_step("additive_fuse",
                               input_count=sum(len(r) for r in channel_results.values()),
                               output_count=len(results))
        else:
            results = self.rrf_fuse(
                query, channel_results,
                is_garbage=is_garbage, doc_count=doc_count,
                top_k=top_k, max_tokens=max_tokens,
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
        # 时序重排序：当查询包含时序关键词时，对融合结果按时间衰减重新排序
        results = self._apply_temporal_rerank(query, results)
        return results

    def rrf_fuse(
        self,
        query: str,
        channel_results: dict[str, list[dict[str, Any]]],
        *,
        is_garbage: bool,
        doc_count: int,
        top_k: int,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """RRF 融合 + 数据量自适应阈值 + 垃圾查询二次验证 + Rerank + Token 裁剪。"""
        adaptive_min_rrf = 0.035
        if doc_count >= 200 or doc_count >= 100:
            adaptive_min_rrf = 0.04
        elif doc_count >= 50 or doc_count >= 20:
            adaptive_min_rrf = 0.035
        elif doc_count < 10:
            adaptive_min_rrf = 0.01

        base_weights: list[float] = []
        result_lists: list[list[dict[str, Any]]] = []
        active_names: list[str] = []
        for name, results in channel_results.items():
            if not results:
                continue
            if name in self._facade._channels:
                weight = self._facade._channels[name][1]
            elif name == "catalog":
                weight = 2.0
            else:
                weight = 1.0
            base_weights.append(weight)
            result_lists.append(results)
            active_names.append(name)

        active_channels = len(result_lists)
        if active_channels <= 1:
            adaptive_min_rrf = min(adaptive_min_rrf, 0.01)
            if active_channels == 1:
                logger.warning("RRF degraded: only %s channel has results, weight=%.1f",
                             active_names[0], base_weights[0] if base_weights else 0)

        if self._facade._source_weights:
            for i, name in enumerate(active_names):
                base_weights[i] *= self._facade._source_weights.get(name, 1.0)

        if not result_lists:
            return []

        fused = self._facade._rrf.merge(
            result_lists,
            min_rrf=adaptive_min_rrf,
            weights=base_weights,
        )

        # 相关性过滤：移除分数远低于最高分的结果
        if fused:
            max_score = fused[0].get("score", 0)
            if max_score > 0:
                threshold = max_score * 0.1
                fused = [r for r in fused if r.get("score", 0) >= threshold]

        if is_garbage and fused:
            fused = []

        if self._facade._reranker and len(fused) > 3:
            fused = self._facade._reranker.rerank(query, fused, top_k=top_k)

        return trim_to_budget(fused, max_tokens)

    def additive_fuse(
        self,
        query: str,
        channel_results: dict[str, list[dict[str, Any]]],
        *,
        is_garbage: bool,
        top_k: int,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """Additive fusion with entity boost (inspired by mem0 three-signal)."""
        from omnimem.retrieval.entity_extractor import EntityExtractor
        extractor = EntityExtractor()

        query_entities = extractor.extract(query, max_entities=8)
        doc_scores: dict[str, dict[str, Any]] = {}
        channel_weights = {"vector": 3.0, "bm25": 1.0, "catalog": 2.0}

        for name, results in channel_results.items():
            if not results:
                continue
            weight = channel_weights.get(name, 1.0)
            for doc in results:
                doc_id = doc.get("memory_id", "") or f"hash-{hash(doc.get('content', ''))}"
                score = doc.get("score", 0.0)
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"scores": {}, "entry": doc}
                doc_scores[doc_id]["scores"][name] = score * weight

        if not doc_scores:
            return []

        results = []
        for doc_id, data in doc_scores.items():
            scores = data["scores"]
            total_weight = sum(channel_weights.get(n, 1.0) for n in scores)
            base_score = sum(scores.values()) / total_weight if total_weight > 0 else 0.0

            doc_entities = data["entry"].get("metadata", {}).get("entities", [])
            entity_boost = extractor.compute_entity_overlap(query_entities, doc_entities)

            if entity_boost > 0:
                fused_score = (base_score * total_weight + entity_boost) / (total_weight + 1.0)
            else:
                fused_score = base_score

            entry = dict(data["entry"])
            entry["additive_score"] = round(fused_score, 6)
            entry["score"] = round(fused_score, 6)
            entry["_channels"] = list(scores.keys())
            if entity_boost > 0:
                entry["_entity_boost"] = round(entity_boost, 6)
            results.append(entry)

        results.sort(key=lambda x: x["score"], reverse=True)

        if is_garbage and results:
            results = []

        results = [r for r in results if r["score"] >= 0.05]

        if self._facade._reranker and len(results) > 3:
            results = self._facade._reranker.rerank(query, results, top_k=top_k)

        return trim_to_budget(results[:top_k], max_tokens)

    @classmethod
    def apply_type_boost(
        cls,
        results: list[dict[str, Any]],
        updated_boost: float = 0.3,
        query: str = "",
        entity_boost_weight: float = 1.0,
    ) -> list[dict[str, Any]]:
        """对 reasoning/action/correction 类型应用分数加权，并对 is_updated 记忆提升排序。

        ★ Task 3.2: 当传入 query 和 entity_boost_weight > 1.0 时，
        对包含查询关键实体的结果额外加权，提升实体匹配度高的记忆排序。
        """
        # ★ Task 3.2: 查询实体提取与加权
        query_entities: set[str] = set()
        if query and entity_boost_weight > 1.0:
            from omnimem.retrieval.entity_extractor import EntityExtractor
            extractor = EntityExtractor()
            extracted = extractor.extract(query, max_entities=5)
            query_entities = {e.lower() for e in extracted}

        for r in results:
            mem_type = r.get("type", "")
            boost = cls._TYPE_BOOST.get(mem_type, 1.0)
            if boost > 1.0:
                current_score = r.get("score", r.get("rrf_score", 0))
                r["score"] = round(current_score * boost, 5)
                r["type_boost"] = boost
            # ★ Task 2: is_updated 记忆获得分数提升
            metadata = r.get("metadata", {})
            if metadata.get("is_updated") or r.get("is_updated"):
                current_score = r.get("score", r.get("rrf_score", 0))
                r["score"] = round(current_score * (1 + updated_boost), 5)
                r["updated_boost"] = updated_boost
            # ★ Task 3.2: 关键实体 BM25 加权
            if query_entities:
                doc_entities = metadata.get("entities", [])
                doc_entity_set = {e.lower() for e in doc_entities}
                overlap = query_entities & doc_entity_set
                if overlap:
                    current_score = r.get("score", r.get("rrf_score", 0))
                    r["score"] = round(current_score * entity_boost_weight, 5)
                    r["entity_boost"] = entity_boost_weight
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    def supplement_low_recall_types(
        self, query: str, results: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """对 reasoning/action 类型做扩展查询，弥补关键词密度低导致的召回不足。"""
        existing_ids = {r.get("memory_id", "") for r in results}
        type_counts = {}
        for r in results:
            t = r.get("type", "")
            type_counts[t] = type_counts.get(t, 0) + 1

        need_reasoning = type_counts.get("reasoning", 0) < 2
        need_action = type_counts.get("action", 0) < 2

        if not need_reasoning and not need_action:
            return results

        extra_queries = []
        if need_reasoning:
            extra_queries.append(f"[教训/经验/踩坑] {query}")
        if need_action:
            extra_queries.append(f"[Agent行为/工具调用] {query}")

        for eq in extra_queries:
            extra_results = self._facade._bm25.search(eq, top_k=5)
            for r in extra_results:
                mid = r.get("memory_id", "")
                if mid in existing_ids:
                    continue
                mem_type = r.get("type", "")
                if mem_type not in ("reasoning", "action"):
                    continue
                r["_source"] = "type_supplement"
                r["score"] = r.get("score", 0) * 0.8
                results.append(r)
                existing_ids.add(mid)

        return results

    # ── 缓存管理 ──

    def _apply_temporal_rerank(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """对融合后的结果应用时序重排序。

        仅当查询包含时序关键词时生效，否则原样返回。
        委托给 _TemporalRetriever.apply_temporal_rerank() 执行。
        """
        from omnimem.retrieval.registry import _TemporalRetriever

        facade = self._facade
        alpha = getattr(facade, "_temporal_rerank_alpha", 0.5)
        decay_lambda = getattr(facade, "_temporal_decay_lambda", 0.1)

        return _TemporalRetriever.apply_temporal_rerank(
            query, results, alpha=alpha, decay_lambda=decay_lambda,
        )

    def check_cache(self, cache_key: str) -> list[dict[str, Any]] | None:
        """查询缓存检查（需在读锁内调用）。"""
        facade = self._facade
        if facade._ml_cache is not None:
            try:
                cached = facade._ml_cache.get(cache_key)
                if cached is not None:
                    logger.debug("HybridRetriever ML cache hit: %s", cache_key[:50])
                    return cached
                return None
            except Exception as e:
                logger.debug("ML cache get failed, falling back to dict: %s", e)
        now = time.time()
        if cache_key in facade._query_cache:
            cached_results, cached_time = facade._query_cache[cache_key]
            if now - cached_time < facade._query_cache_ttl:
                logger.debug("HybridRetriever query cache hit")
                return cached_results
        return None

    def set_cache(self, cache_key: str, results: list[dict[str, Any]]) -> None:
        """写入查询缓存（需在读锁内调用）。"""
        facade = self._facade
        tags: set[str] = set()
        for r in results:
            mid = r.get("memory_id", "")
            if mid:
                tags.add(f"memory:{mid}")
        if facade._ml_cache is not None:
            try:
                facade._ml_cache.set(cache_key, results, ttl=facade._query_cache_ttl, tags=tags)
                return
            except Exception as e:
                logger.debug("ML cache set failed, falling back to dict: %s", e)
        facade._query_cache[cache_key] = (results, time.time())

    def invalidate_cache_by_memory(self, memory_id: str) -> None:
        """按 memory_id 精准失效相关缓存（用于 add 时调用）。"""
        facade = self._facade
        if facade._ml_cache is not None:
            try:
                facade._ml_cache.invalidate_memory(memory_id)
                return
            except Exception as e:
                logger.debug("ML cache invalidate_memory failed: %s", e)
        facade._query_cache.clear()

    def clear_all_cache(self) -> None:
        """清空所有查询缓存（用于 delete/update/rebuild 等全量失效场景）。"""
        facade = self._facade
        if facade._ml_cache is not None:
            try:
                facade._ml_cache.clear()
            except Exception as e:
                logger.debug("ML cache clear failed: %s", e)
        facade._query_cache.clear()

    def cleanup_query_cache(self) -> None:
        """主动清理过期的查询缓存条目。"""
        facade = self._facade
        now = time.time()
        expired_keys = [
            key for key, (_, cached_time) in facade._query_cache.items()
            if now - cached_time >= facade._query_cache_ttl
        ]
        for key in expired_keys:
            del facade._query_cache[key]

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
        import re as _re_count
        _is_count = bool(_re_count.match(r'^(how many|how much)\b', query.strip().lower()))
        _orig_count_weights = {}
        if _is_count:
            top_k = max(top_k, 60)
            if self._facade._source_weights is not None:
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

        channel_results = self.dispatch_channels(query, top_k, allowed_channels, trace, bm25_query=bm25_query)
        results = self.fuse_and_filter(
            query, channel_results,
            is_garbage=is_garbage, doc_count=doc_count,
            top_k=top_k, max_tokens=max_tokens, trace=trace,
        )
        self.set_cache(cache_key, results)

        # ★ COUNT 查询：恢复原始权重，防止污染后续检索
        if _is_count and self._facade._source_weights is not None:
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

    # ── 索引维护 ──

    def enrich_for_rebuild(self, content: str, mem_type: str, room: str = "") -> str:
        """重建索引时为各类型附加可搜索描述。"""
        type_prefixes = {
            "secret": "[加密信息/密钥/凭证]",
            "skill": "[技能/步骤/教程]",
            "procedural": "[流程/操作/指南]",
            "reasoning": "[教训/经验/踩坑]",
            "action": "[Agent行为/工具调用]",
        }
        prefix = type_prefixes.get(mem_type, "")
        if prefix:
            return f"{prefix} {room} {content}"
        return content

    def rebuild_bm25_from_entries(self, entries: list[dict[str, Any]]) -> int:
        """从索引条目重建 BM25 检索通道（跨会话持久化恢复）。

        使用 BM25Retriever 的增量更新能力，避免全量重建。
        """
        bm25 = self._facade._bm25
        enriched_entries = []
        for entry in entries:
            memory_id = entry.get("memory_id", "")
            content = entry.get("content", "") or entry.get("summary", "")
            if not memory_id or not content:
                continue
            mem_type = entry.get("type", "fact")
            room = entry.get("room", "")
            enriched = self.enrich_for_rebuild(content, mem_type, room)
            enriched_entries.append({**entry, "content": enriched})
        result = bm25.update_from_entries(enriched_entries)
        total = result.get("added", 0) + result.get("updated", 0)
        logger.info(
            "BM25 rebuild (incremental): added=%d, updated=%d, deleted=%d",
            result.get("added", 0),
            result.get("updated", 0),
            result.get("deleted", 0),
        )
        return total

    def rebuild_all_from_entries(self, entries: list[dict[str, Any]]) -> dict[str, int]:
        """全量重建向量+BM25检索索引（解决历史向量退化问题）。

        向量索引使用分批并行 embedding 计算 + 批量写入；
        BM25 使用增量更新，避免全量重建。
        """
        facade = self._facade
        self.clear_all_cache()

        # 读取重建参数配置（默认 batch_size=32, max_workers=4）
        config = getattr(facade, "_config", None)
        batch_size = 32
        max_workers = 4
        if config is not None:
            try:
                batch_size = int(config.get("rebuild_batch_size", 32))
                max_workers = int(config.get("rebuild_max_workers", 4))
            except Exception:
                batch_size, max_workers = 32, 4
        batch_size = max(1, batch_size)
        max_workers = max(1, max_workers)

        # 1. 清空现有向量索引
        try:
            if hasattr(facade._vector, "reset"):
                facade._vector.reset()
            else:
                for entry in entries:
                    mid = entry.get("memory_id", "")
                    if mid:
                        facade._vector.delete(mid)
        except Exception as e:
            logger.warning("Vector reset/delete failed in rebuild: %s", e)

        # 2. 并行重建向量索引
        vec_count = 0
        try:
            vec_count = facade._vector.rebuild_vectors_parallel(
                entries, batch_size=batch_size, max_workers=max_workers
            )
        except Exception as e:
            logger.warning("rebuild_vectors_parallel failed: %s", e)

        # 3. BM25 增量更新
        bm25_count = 0
        try:
            bm25_entries = []
            for entry in entries:
                mid = entry.get("memory_id", "")
                content = entry.get("content", "")
                if not mid or not content:
                    continue
                mem_type = entry.get("type", "fact")
                room = entry.get("room", "")
                enriched = self.enrich_for_rebuild(content, mem_type, room)
                bm25_entries.append({**entry, "content": enriched})
            result = facade._bm25.update_from_entries(bm25_entries)
            bm25_count = result.get("added", 0) + result.get("updated", 0)
        except Exception as e:
            logger.warning("BM25 incremental rebuild failed: %s", e)

        facade._vector.flush()
        logger.info(
            "HybridRetriever rebuild: vector=%d, bm25=%d from %d entries",
            vec_count, bm25_count, len(entries),
        )
        return {"vector": vec_count, "bm25": bm25_count}

    def cleanup_sync_turn_entries(self) -> None:
        """清理旧的 sync_turn 条目，防止索引膨胀。"""
        facade = self._facade
        max_entries = facade._max_sync_turn_entries
        while len(facade._sync_turn_ids) > max_entries:
            old_id = facade._sync_turn_ids.popleft()
            try:
                facade._bm25.delete(old_id)
            except Exception as e:
                logger.warning("BM25 delete sync_turn %s failed: %s", old_id, e)
            try:
                facade._vector.delete(old_id)
            except Exception as e:
                logger.warning("Vector delete sync_turn %s failed: %s", old_id, e)

    def index_update(self, user_content: str, _assistant_content: str) -> None:
        """后台异步索引更新（从 sync_turn 调用）。"""
        import re
        import uuid

        clean_user = re.sub(
            r"### Relevant Memories(?:\s*\(prefetched\))?\s*\n.*"
            r"(?=\n(?!- )|\Z)",
            "",
            user_content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        clean_user = re.sub(r"^- \[cached\].*$", "", clean_user, flags=re.MULTILINE).strip()

        content = clean_user[:200].strip() if clean_user else ""
        if not content or len(content) < 5:
            return

        idx_id = f"sync-{uuid.uuid4().hex[:8]}"
        facade = self._facade
        self.clear_all_cache()
        facade._vector.add(content, memory_id=idx_id, metadata={"source": "sync_turn"})
        facade._bm25.add(content, memory_id=idx_id, metadata={"source": "sync_turn"})
        facade._sync_turn_ids.append(idx_id)
        self.cleanup_sync_turn_entries()
