"""检索结果融合与过滤（★ M6-9：从 hybrid_orchestrator.py 拆分）。"""

from __future__ import annotations

import logging
from typing import Any

from omnimem.retrieval.query_quality import trim_to_budget

logger = logging.getLogger(__name__)


class FusionMixin:
    """RRF/additive 融合 + 类型加权 + 低召回补充 + 时序重排。"""

    # reasoning/action 类型的记忆包含高价值信息但关键词密度低，
    # 需要提高权重避免被 fact/preference 等高频词类型淹没。
    _TYPE_BOOST: dict[str, float] = {
        "reasoning": 1.3,
        "action": 1.3,
        "correction": 1.1,
    }

    # ★ 缺陷3: 偏好意图信号词 — 查询本身关于偏好时不过滤偏好记忆
    _PREF_INTENT_WORDS: tuple[str, ...] = (
        "偏好", "喜欢", "习惯", "倾向", "偏爱", "爱好", "口味", "常用",
        "prefer", "like", "favorite", "favourite", "enjoy", "habit", "taste",
    )

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

        # ★ 缺陷2修复: 绝对语义相关性地板 —— 向量最高余弦低于下限且无 BM25 词法命中时,
        #   判定为"无相关结果"直接返回空集, 避免对完全无关查询硬塞最近邻噪声。
        min_rel = getattr(self, "_min_relevance_score", 0.0)
        vec_hits = channel_results.get("vector", [])
        # 仅当向量通道确实参与且有结果时才应用语义地板;
        # 纯 BM25/自定义通道无法判定语义相似度, 不门控以免误清空。
        if min_rel > 0.0 and vec_hits:
            max_vec = max((float(r.get("score", 0.0)) for r in vec_hits), default=0.0)
            has_lexical = self._has_meaningful_lexical_hit(
                query, channel_results.get("bm25") or []
            )
            if max_vec < min_rel and not has_lexical:
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

        # ★ 缺陷3修复: 偏好记忆查询相关性门控
        if getattr(self, "_preference_gate_enabled", True):
            fused = self._gate_preferences(query, fused)

        if self._facade._reranker and len(fused) > 3:
            # ★ 先截断到 rerank 候选数，避免对 100+ 条做 Cross-Encoder 推理
            _rerank_candidates = min(30, len(fused))
            fused = self._facade._reranker.rerank(query, fused[:_rerank_candidates], top_k=top_k)

        return trim_to_budget(fused, max_tokens)

    @staticmethod
    def _has_meaningful_lexical_hit(
        query: str, bm25_results: list[dict[str, Any]]
    ) -> bool:
        """BM25 命中仅在与查询存在非噪声实词重叠时才算词法证据。

        避免"相关/使用"这类噪声词的弱命中豁免语义地板 —— 例如查询
        "完全不相关的查询xyz"被切出"相关"后与任意含"相关"的记忆
        弱匹配(score≈0.01), 不应视为有效词法命中。
        """
        if not bm25_results:
            return False
        try:
            from omnimem.retrieval.bm25 import (
                _MINIMAL_ZH_STOPWORDS,
                _NOISE_WORDS,
                _tokenize,
            )
        except Exception:
            return True  # 无法分词时保守放行, 维持原行为
        stop = _NOISE_WORDS | _MINIMAL_ZH_STOPWORDS
        q_tokens = {t for t in _tokenize(query) if len(t) >= 2 and t not in stop}
        if not q_tokens:
            return False
        for r in bm25_results[:5]:
            c_tokens = set(_tokenize(r.get("content", "") or ""))
            if q_tokens & c_tokens:
                return True
        return False

    def _gate_preferences(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """偏好类记忆易泛化匹配任意查询, 仅在查询确与其相关时保留。

        保留条件(满足其一):
          1. 查询本身是偏好意图(含"偏好/喜欢/prefer"等信号词);
          2. 偏好内容与查询存在实词(非停用词)重叠。
        否则过滤该泛化偏好, 避免污染无关查询结果(零结果/召回噪声)。
        """
        if not results:
            return results
        if not any(r.get("type") in ("preference", "preferences") for r in results):
            return results

        ql = query.lower()
        if any(sig in query or sig in ql for sig in self._PREF_INTENT_WORDS):
            return results

        try:
            from omnimem.retrieval.bm25 import (
                _MINIMAL_ZH_STOPWORDS,
                _NOISE_WORDS,
                _tokenize,
            )
        except Exception:
            return results  # 无法分词则保持原行为
        stop = _NOISE_WORDS | _MINIMAL_ZH_STOPWORDS
        q_tokens = {t for t in _tokenize(query) if len(t) >= 2 and t not in stop}

        gated: list[dict[str, Any]] = []
        for r in results:
            if r.get("type") in ("preference", "preferences"):
                c_tokens = set(_tokenize(r.get("content", "") or ""))
                if q_tokens & c_tokens:
                    gated.append(r)
                # 否则: 丢弃与查询无实词重叠的泛化偏好
            else:
                gated.append(r)
        return gated

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
            # ★ 先截断到 rerank 候选数
            _rerank_candidates = min(30, len(results))
            results = self._facade._reranker.rerank(query, results[:_rerank_candidates], top_k=top_k)

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
