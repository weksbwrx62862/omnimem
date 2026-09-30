"""retrieval/fusion.py — FusionMixin 单元测试。

覆盖: fuse_and_filter 分派 / rrf_fuse 阈值与语义地板 / additive_fuse 加权
     apply_type_boost 分差护栏 / supplement_low_recall_types / _gate_preferences
     _has_meaningful_lexical_hit / _apply_temporal_rerank
"""

from __future__ import annotations

from typing import Any

import pytest
from omnimem.retrieval.fusion import FusionMixin

# ─── 脚手架 ────────────────────────────────────────────────


class _FakeRRF:
    """记录参数的 RRF 桩：直接拼接 result_lists，按 score 排序返回。"""

    def __init__(self, merged: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._merged = merged

    def merge(
        self,
        lists: list[list[dict[str, Any]]],
        *,
        min_rrf: float,
        weights: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append({"min_rrf": min_rrf, "weights": list(weights or [])})
        if self._merged is not None:
            return list(self._merged)
        flat: list[dict[str, Any]] = []
        for group in lists:
            for r in group:
                flat.append(dict(r))
        flat.sort(key=lambda x: x.get("score", 0), reverse=True)
        return flat


class _FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query: str, results: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
        self.calls.append((query, len(results)))
        return [{"memory_id": r.get("memory_id"), "score": 1.0 - i * 0.01} for i, r in enumerate(results)]


class _FakeBM25:
    def __init__(self, results_by_query: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self._results = results_by_query or {}
        self.queries: list[str] = []

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        self.queries.append(query)
        for q, rs in self._results.items():
            if q in query:
                return [dict(r) for r in rs]
        return []


class _FakeFacade:
    def __init__(self, **kw: Any) -> None:
        self._channels: dict[str, tuple] = kw.get("_channels", {})
        self._source_weights: dict[str, float] = kw.get("_source_weights", {})
        self._rrf: _FakeRRF = kw.get("_rrf", _FakeRRF())
        self._reranker: Any = kw.get("_reranker")
        self._bm25: Any = kw.get("_bm25", _FakeBM25())


class _Engine(FusionMixin):
    def __init__(self, facade: _FakeFacade | None = None, **overrides: Any) -> None:
        self._facade = facade or _FakeFacade()
        self._updated_boost = overrides.get("updated_boost", 0.3)
        self._entity_boost_weight = overrides.get("entity_boost_weight", 1.0)
        self._min_relevance_score = overrides.get("min_relevance_score", 0.0)
        self._preference_gate_enabled = overrides.get("preference_gate_enabled", True)


# ─── _has_meaningful_lexical_hit ───────────────────────────


class TestHasMeaningfulLexicalHit:
    def test_empty_bm25_returns_false(self):
        assert FusionMixin._has_meaningful_lexical_hit("python", []) is False

    def test_overlap_returns_true(self):
        hits = [{"content": "python programming tutorial"}]
        assert FusionMixin._has_meaningful_lexical_hit("python tutorial", hits) is True

    def test_no_overlap_returns_false(self):
        hits = [{"content": "java enterprise beans"}]
        assert FusionMixin._has_meaningful_lexical_hit("python snake", hits) is False

    def test_query_all_noise_tokens_returns_false(self):
        # 查询切出的 token 全部为停用/短词 → q_tokens 为空
        hits = [{"content": "some text"}]
        assert FusionMixin._has_meaningful_lexical_hit("的了是在", hits) is False

    def test_only_checks_first_five(self):
        # 第 6 条起才出现真实重叠, 前 5 条无匹配 → False
        hits = [{"content": f"unrelated {i}"} for i in range(5)]
        hits.append({"content": "python"})
        assert FusionMixin._has_meaningful_lexical_hit("python", hits) is False

    def test_chinese_overlap(self):
        hits = [{"content": "深度学习框架对比"}]
        assert FusionMixin._has_meaningful_lexical_hit("深度学习优化", hits) is True


# ─── _gate_preferences ─────────────────────────────────────


class TestGatePreferences:
    def test_empty_results(self):
        eng = _Engine()
        assert eng._gate_preferences("q", []) == []

    def test_no_preference_types_passthrough(self):
        eng = _Engine()
        rs = [{"type": "fact", "content": "x"}]
        assert eng._gate_preferences("q", rs) is rs

    def test_pref_intent_query_passthrough_all(self):
        eng = _Engine()
        rs = [
            {"type": "preference", "content": "unrelated stuff"},
            {"type": "fact", "content": "y"},
        ]
        # "偏好" 属于 _PREF_INTENT_WORDS
        out = eng._gate_preferences("我的偏好是什么", rs)
        assert out == rs

    def test_english_pref_intent_passthrough(self):
        eng = _Engine()
        rs = [{"type": "preference", "content": "no overlap"}]
        out = eng._gate_preferences("what do I prefer", rs)
        assert len(out) == 1

    def test_preference_with_overlap_kept(self):
        eng = _Engine()
        rs = [{"type": "preference", "content": "python code style"}]
        out = eng._gate_preferences("python 缩进", rs)
        assert len(out) == 1

    def test_preference_without_overlap_dropped(self):
        eng = _Engine()
        rs = [{"type": "preference", "content": "chocolate cake"}]
        out = eng._gate_preferences("database index", rs)
        assert out == []

    def test_non_preference_always_kept(self):
        eng = _Engine()
        rs = [
            {"type": "preference", "content": "chocolate cake"},
            {"type": "fact", "content": "sqlite index"},
        ]
        out = eng._gate_preferences("database query", rs)
        assert len(out) == 1
        assert out[0]["type"] == "fact"


# ─── apply_type_boost ──────────────────────────────────────


class TestApplyTypeBoost:
    def test_reasoning_boosted_1_3x(self):
        rs = [{"memory_id": "r", "type": "reasoning", "score": 0.10}]
        out = FusionMixin.apply_type_boost(rs)
        assert out[0]["score"] == pytest.approx(0.13)
        assert out[0]["type_boost"] == 1.3

    def test_action_boosted(self):
        rs = [{"memory_id": "a", "type": "action", "score": 0.10}]
        out = FusionMixin.apply_type_boost(rs)
        assert out[0]["score"] == pytest.approx(0.13)

    def test_fact_no_boost(self):
        rs = [{"memory_id": "f", "type": "fact", "score": 0.10}]
        out = FusionMixin.apply_type_boost(rs)
        assert out[0]["score"] == pytest.approx(0.10)
        assert "type_boost" not in out[0]

    def test_is_updated_metadata(self):
        rs = [{"memory_id": "u", "type": "fact", "score": 0.10,
               "metadata": {"is_updated": True}}]
        out = FusionMixin.apply_type_boost(rs, updated_boost=0.5)
        assert out[0]["score"] == pytest.approx(0.15)
        assert out[0]["updated_boost"] == 0.5

    def test_is_updated_top_level(self):
        rs = [{"memory_id": "u", "type": "fact", "score": 0.10, "is_updated": True}]
        out = FusionMixin.apply_type_boost(rs, updated_boost=0.3)
        assert out[0]["score"] == pytest.approx(0.13)

    def test_pref_intent_query_skips_type_boost(self):
        rs = [{"memory_id": "r", "type": "reasoning", "score": 0.10}]
        out = FusionMixin.apply_type_boost(rs, query="我的偏好是什么")
        # _pref_intent 命中 → boost 被强制 1.0
        assert out[0]["score"] == pytest.approx(0.10)
        assert "type_boost" not in out[0]

    def test_entity_boost_overlap(self):
        rs = [
            {"memory_id": "hit", "type": "fact", "score": 0.10,
             "metadata": {"entities": ["Python"]}},
        ]
        out = FusionMixin.apply_type_boost(
            rs, query="Python tips", entity_boost_weight=1.2,
        )
        # Python 大小写不敏感命中 → 0.10 * 1.2 = 0.12
        assert out[0]["score"] == pytest.approx(0.12)
        assert out[0]["entity_boost"] == 1.2

    def test_entity_weight_no_query(self):
        rs = [{"memory_id": "x", "type": "fact", "score": 0.10,
               "metadata": {"entities": ["Python"]}}]
        out = FusionMixin.apply_type_boost(
            rs, query="", entity_boost_weight=1.2,
        )
        # query 空 → 不提取 query_entities → 不加权
        assert "entity_boost" not in out[0]

    def test_gap_guard_caps_flip(self):
        # 强 fact 0.20 vs 弱 reasoning 0.10: 预分差 2.0 > 1.10
        # 若无护栏: 0.10 * 1.3 = 0.13 仍 < 0.20, 不翻转, 不触发 cap
        # 用 0.15 vs 0.10 更强触发翻转: 0.10*1.3=0.13 仍 < 0.15, 不翻转
        # 需要更强 boost 才能翻: entity_boost_weight 1.6 → 0.10*1.6=0.16 > 0.15
        rs = [
            {"memory_id": "strong", "type": "fact", "score": 0.15,
             "metadata": {"entities": []}},
            {"memory_id": "weak", "type": "fact", "score": 0.10,
             "metadata": {"entities": ["Python"]}},
        ]
        out = FusionMixin.apply_type_boost(
            rs, query="Python tips", entity_boost_weight=1.6,
        )
        # 护栏: 弱项被 cap 到 0.15*0.999 ≈ 0.14985
        weak = next(r for r in out if r["memory_id"] == "weak")
        assert weak.get("boost_capped") is True
        assert weak["score"] < 0.15

    def test_gap_guard_allows_close_flip(self):
        # 预分差 <= 1.10 时允许翻转
        rs = [
            {"memory_id": "strong", "type": "fact", "score": 0.10,
             "metadata": {"entities": []}},
            {"memory_id": "weak", "type": "reasoning", "score": 0.099,
             "metadata": {"entities": []}},
        ]
        out = FusionMixin.apply_type_boost(rs)
        # 0.099 * 1.3 = 0.1287 > 0.10 → 翻转, 且预分差 0.10/0.099 ≈ 1.01 <= 1.10
        # 护栏跳过 → 无 boost_capped
        assert out[0]["memory_id"] == "weak"
        assert "boost_capped" not in out[0]

    def test_returns_sorted_desc(self):
        rs = [
            {"memory_id": "low", "type": "fact", "score": 0.05},
            {"memory_id": "high", "type": "reasoning", "score": 0.20},
        ]
        out = FusionMixin.apply_type_boost(rs)
        assert [r["memory_id"] for r in out] == ["high", "low"]

    def test_empty_results(self):
        assert FusionMixin.apply_type_boost([]) == []


# ─── rrf_fuse ──────────────────────────────────────────────


class TestRrfFuse:
    def test_doc_count_ge_100_threshold_004(self):
        eng = _Engine()
        eng.rrf_fuse(
            "q",
            {
                "vector": [{"memory_id": "a", "score": 0.5}],
                "bm25": [{"memory_id": "b", "score": 0.4}],
            },
            is_garbage=False, doc_count=150, top_k=5, max_tokens=1000,
        )
        assert eng._facade._rrf.calls[0]["min_rrf"] == 0.04

    def test_doc_count_lt_10_threshold_clamped_001(self):
        # doc_count<10 → adaptive_min_rrf=0.01, 且单通道 → min(0.01,0.01)=0.01
        eng = _Engine()
        eng.rrf_fuse(
            "q", {"vector": [{"memory_id": "a", "score": 0.5}]},
            is_garbage=False, doc_count=5, top_k=5, max_tokens=1000,
        )
        assert eng._facade._rrf.calls[0]["min_rrf"] == 0.01

    def test_doc_count_mid_uses_default_0035(self):
        # doc_count=30 (>=20) → 0.035; 多通道不降
        eng = _Engine()
        eng.rrf_fuse(
            "q",
            {
                "vector": [{"memory_id": "a", "score": 0.5}],
                "bm25": [{"memory_id": "b", "score": 0.3}],
            },
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        assert eng._facade._rrf.calls[0]["min_rrf"] == 0.035

    def test_channel_weights_lookup(self):
        # vector 通道权重来自 _channels["vector"][1]
        facade = _FakeFacade(_channels={"vector": (None, 3.0), "bm25": (None, 1.0)})
        eng = _Engine(facade)
        eng.rrf_fuse(
            "q",
            {
                "vector": [{"memory_id": "a", "score": 0.5}],
                "bm25": [{"memory_id": "b", "score": 0.3}],
            },
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        assert eng._facade._rrf.calls[0]["weights"] == [3.0, 1.0]

    def test_catalog_default_weight_2(self):
        # catalog 不在 _channels → 默认 2.0
        eng = _Engine()
        eng.rrf_fuse(
            "q", {"catalog": [{"memory_id": "c"}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        assert eng._facade._rrf.calls[0]["weights"] == [2.0]

    def test_source_weights_multiplicative(self):
        facade = _FakeFacade(
            _channels={"vector": (None, 3.0)},
            _source_weights={"vector": 0.5},
        )
        eng = _Engine(facade)
        eng.rrf_fuse(
            "q", {"vector": [{"memory_id": "a", "score": 0.5}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        assert eng._facade._rrf.calls[0]["weights"] == pytest.approx([1.5])

    def test_planner_weights_stacked(self):
        facade = _FakeFacade(
            _channels={"vector": (None, 3.0)},
            _source_weights={"vector": 2.0},
        )
        eng = _Engine(facade)
        eng.rrf_fuse(
            "q", {"vector": [{"memory_id": "a", "score": 0.5}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
            planner_weights={"vector": 0.25},
        )
        # 3.0 * 2.0 * 0.25 = 1.5
        assert eng._facade._rrf.calls[0]["weights"] == pytest.approx([1.5])

    def test_empty_channel_results_returns_empty(self):
        eng = _Engine()
        out = eng.rrf_fuse(
            "q", {}, is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        assert out == []

    def test_min_relevance_floor_blocks_when_no_lexical(self):
        # 向量最高分 0.3 < 阈值 0.5, bm25 无重叠 → 语义地板拒绝
        eng = _Engine(_FakeFacade(), min_relevance_score=0.5)
        out = eng.rrf_fuse(
            "quantum field",
            {
                "vector": [{"memory_id": "v", "score": 0.3}],
                "bm25": [{"memory_id": "b", "content": "python basics", "score": 0.1}],
            },
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        assert out == []

    def test_min_relevance_floor_bypassed_via_lexical(self):
        # 向量低但 bm25 有重叠 → 地板放行
        eng = _Engine(_FakeFacade(), min_relevance_score=0.5)
        out = eng.rrf_fuse(
            "quantum field",
            {
                "vector": [{"memory_id": "v", "score": 0.3}],
                "bm25": [{"memory_id": "b", "content": "quantum physics", "score": 0.4}],
            },
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        assert len(out) >= 1

    def test_is_garbage_clears_results(self):
        merged = [{"memory_id": "a", "score": 0.5}]
        facade = _FakeFacade(_rrf=_FakeRRF(merged=merged))
        eng = _Engine(facade)
        out = eng.rrf_fuse(
            "q", {"vector": [{"memory_id": "a", "score": 0.5}]},
            is_garbage=True, doc_count=30, top_k=5, max_tokens=1000,
        )
        assert out == []

    def test_rerank_invoked_when_len_gt_3(self):
        merged = [{"memory_id": f"m{i}", "score": 0.5 - i * 0.01} for i in range(5)]
        facade = _FakeFacade(
            _rrf=_FakeRRF(merged=merged),
            _reranker=_FakeReranker(),
        )
        eng = _Engine(facade)
        out = eng.rrf_fuse(
            "q", {"vector": [{"memory_id": "x", "score": 0.5}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        assert len(facade._reranker.calls) == 1
        assert len(out) > 0

    def test_rerank_skipped_when_short(self):
        merged = [{"memory_id": "a", "score": 0.5}]
        rr = _FakeReranker()
        facade = _FakeFacade(_rrf=_FakeRRF(merged=merged), _reranker=rr)
        eng = _Engine(facade)
        eng.rrf_fuse(
            "q", {"vector": [{"memory_id": "x", "score": 0.5}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        assert rr.calls == []

    def test_relative_relevance_threshold_filters_weak(self):
        # top score=0.10 → 过滤 < 0.01 的
        merged = [
            {"memory_id": "top", "score": 0.10},
            {"memory_id": "mid", "score": 0.05},
            {"memory_id": "bot", "score": 0.005},
        ]
        facade = _FakeFacade(_rrf=_FakeRRF(merged=merged))
        eng = _Engine(facade)
        out = eng.rrf_fuse(
            "q", {"vector": [{"memory_id": "x", "score": 0.5}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        ids = {r["memory_id"] for r in out}
        assert "bot" not in ids
        assert "top" in ids

    def test_trim_to_budget_applied(self):
        # 每条 20 字符 → 每条约 5 tokens; max_tokens=6 → 只能保留 1 条
        merged = [{"memory_id": f"m{i}", "content": "x" * 20, "score": 0.5 - i * 0.01}
                  for i in range(5)]
        facade = _FakeFacade(_rrf=_FakeRRF(merged=merged))
        eng = _Engine(facade)
        out = eng.rrf_fuse(
            "q", {"vector": [{"memory_id": "x", "score": 0.5}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=6,
        )
        assert len(out) <= 1


# ─── additive_fuse ─────────────────────────────────────────


class TestAdditiveFuse:
    def test_empty_returns_empty(self):
        eng = _Engine()
        out = eng.additive_fuse(
            "q", {}, is_garbage=False, top_k=5, max_tokens=1000,
        )
        assert out == []

    def test_channel_weight_vector_3x_bm25_1x(self):
        eng = _Engine()
        out = eng.additive_fuse(
            "unrelated",
            {
                "vector": [{"memory_id": "v", "content": "aa", "score": 0.4}],
                "bm25": [{"memory_id": "b", "content": "bb", "score": 0.4}],
            },
            is_garbage=False, top_k=5, max_tokens=1000,
        )
        by_id = {r["memory_id"]: r for r in out}
        # vector: score 0.4*3 / 3 = 0.4
        # bm25:   score 0.4*1 / 1 = 0.4
        # 两者得分相同, 但 vector 权重高时归一化后应等值
        assert "v" in by_id and "b" in by_id

    def test_score_floor_005_filters(self):
        eng = _Engine()
        out = eng.additive_fuse(
            "q",
            {"vector": [{"memory_id": "low", "content": "x", "score": 0.01}]},
            is_garbage=False, top_k=5, max_tokens=1000,
        )
        assert out == []

    def test_is_garbage_clears(self):
        eng = _Engine()
        out = eng.additive_fuse(
            "q",
            {"vector": [{"memory_id": "v", "content": "python", "score": 0.9}]},
            is_garbage=True, top_k=5, max_tokens=1000,
        )
        assert out == []

    def test_planner_weights_stacked(self):
        # planner_weights 影响 weight → 最终 base_score 变化
        eng = _Engine()
        out1 = eng.additive_fuse(
            "q",
            {"bm25": [{"memory_id": "b", "content": "python", "score": 0.5}]},
            is_garbage=False, top_k=5, max_tokens=1000,
        )
        out2 = eng.additive_fuse(
            "q",
            {"bm25": [{"memory_id": "b", "content": "python", "score": 0.5}]},
            is_garbage=False, top_k=5, max_tokens=1000,
            planner_weights={"bm25": 2.0},
        )
        # planner 加权后 score * 2.0 归一化除以 total_weight=1 → 0.5→1.0
        assert out2[0]["score"] > out1[0]["score"]

    def test_entity_boost_added_when_overlap(self):
        eng = _Engine()
        out = eng.additive_fuse(
            "Python programming",
            {
                "vector": [{
                    "memory_id": "v",
                    "content": "Python tips and tricks",
                    "score": 0.6,
                    "metadata": {"entities": ["Python", "programming"]},
                }],
            },
            is_garbage=False, top_k=5, max_tokens=1000,
        )
        assert len(out) == 1
        # entity_boost 只在 overlap>0 时打标
        assert out[0].get("_entity_boost", 0) > 0

    def test_sorted_desc(self):
        eng = _Engine()
        out = eng.additive_fuse(
            "q",
            {
                "vector": [
                    {"memory_id": "hi", "content": "aaa", "score": 0.9},
                    {"memory_id": "lo", "content": "bbb", "score": 0.1},
                ],
            },
            is_garbage=False, top_k=5, max_tokens=1000,
        )
        scores = [r["score"] for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_invoked_when_len_gt_3(self):
        rr = _FakeReranker()
        facade = _FakeFacade(_reranker=rr)
        eng = _Engine(facade)
        docs = [{"memory_id": f"m{i}", "content": "python code", "score": 0.8 - i * 0.01}
                for i in range(5)]
        eng.additive_fuse(
            "q", {"vector": docs},
            is_garbage=False, top_k=5, max_tokens=1000,
        )
        assert len(rr.calls) == 1


# ─── supplement_low_recall_types ───────────────────────────


class TestSupplementLowRecallTypes:
    def test_already_two_each_no_supplement(self):
        bm25 = _FakeBM25({"教训": [{"memory_id": "x", "type": "reasoning", "score": 0.5}]})
        facade = _FakeFacade(_bm25=bm25)
        eng = _Engine(facade)
        rs = [
            {"memory_id": "r1", "type": "reasoning", "score": 0.5},
            {"memory_id": "r2", "type": "reasoning", "score": 0.4},
            {"memory_id": "a1", "type": "action", "score": 0.5},
            {"memory_id": "a2", "type": "action", "score": 0.4},
        ]
        out = eng.supplement_low_recall_types("q", rs, top_k=10)
        assert out == rs
        assert bm25.queries == []

    def test_missing_reasoning_triggers_query(self):
        bm25 = _FakeBM25({
            "教训": [{"memory_id": "new_r", "type": "reasoning", "score": 0.5}],
        })
        eng = _Engine(_FakeFacade(_bm25=bm25))
        rs = [
            {"memory_id": "a1", "type": "action", "score": 0.5},
            {"memory_id": "a2", "type": "action", "score": 0.4},
            {"memory_id": "r1", "type": "reasoning", "score": 0.6},
        ]
        out = eng.supplement_low_recall_types("q", rs, top_k=10)
        assert any("教训" in q for q in bm25.queries)
        assert any(r["memory_id"] == "new_r" for r in out)

    def test_missing_action_triggers_agent_query(self):
        bm25 = _FakeBM25({
            "Agent行为": [{"memory_id": "new_a", "type": "action", "score": 0.5}],
        })
        eng = _Engine(_FakeFacade(_bm25=bm25))
        rs = [
            {"memory_id": "r1", "type": "reasoning", "score": 0.5},
            {"memory_id": "r2", "type": "reasoning", "score": 0.4},
        ]
        out = eng.supplement_low_recall_types("q", rs, top_k=10)
        assert any("Agent行为" in q for q in bm25.queries)
        assert any(r["memory_id"] == "new_a" for r in out)

    def test_skips_existing_ids(self):
        bm25 = _FakeBM25({
            "教训": [{"memory_id": "r1", "type": "reasoning", "score": 0.5}],
        })
        eng = _Engine(_FakeFacade(_bm25=bm25))
        rs = [
            {"memory_id": "r1", "type": "reasoning", "score": 0.6},
            {"memory_id": "a1", "type": "action", "score": 0.5},
            {"memory_id": "a2", "type": "action", "score": 0.4},
        ]
        out = eng.supplement_low_recall_types("q", rs, top_k=10)
        # r1 已在集合, 不重复加入
        assert sum(1 for r in out if r["memory_id"] == "r1") == 1

    def test_ignores_non_reasoning_action(self):
        bm25 = _FakeBM25({
            "教训": [{"memory_id": "junk", "type": "fact", "score": 0.5}],
        })
        eng = _Engine(_FakeFacade(_bm25=bm25))
        rs = [{"memory_id": "a1", "type": "action", "score": 0.5},
              {"memory_id": "a2", "type": "action", "score": 0.4}]
        out = eng.supplement_low_recall_types("q", rs, top_k=10)
        assert all(r["memory_id"] != "junk" for r in out)

    def test_score_multiplier_and_source_tag(self):
        bm25 = _FakeBM25({
            "教训": [{"memory_id": "new", "type": "reasoning", "score": 0.5}],
        })
        eng = _Engine(_FakeFacade(_bm25=bm25))
        rs = [
            {"memory_id": "a1", "type": "action", "score": 0.5},
            {"memory_id": "a2", "type": "action", "score": 0.4},
        ]
        out = eng.supplement_low_recall_types("q", rs, top_k=10)
        new = next(r for r in out if r["memory_id"] == "new")
        assert new["score"] == pytest.approx(0.4)
        assert new["_source"] == "type_supplement"


# ─── _apply_temporal_rerank ────────────────────────────────


class TestApplyTemporalRerank:
    def test_non_temporal_query_passthrough(self):
        eng = _Engine()
        rs = [{"memory_id": "a", "score": 0.5}]
        out = eng._apply_temporal_rerank("python basics", rs)
        # 非时序查询 → 原样返回
        assert out == rs

    def test_temporal_query_uses_facade_params(self):
        # 时序关键词命中; 不抛异常即可 (真实衰减计算依赖 timestamp)
        eng = _Engine()
        rs = [{"memory_id": "a", "score": 0.5}]
        out = eng._apply_temporal_rerank("最近发生了什么", rs)
        assert isinstance(out, list)


# ─── fuse_and_filter ───────────────────────────────────────


class TestFuseAndFilter:
    def test_dispatches_to_rrf_by_default(self):
        eng = _Engine()
        orig = eng.rrf_fuse

        called: dict[str, bool] = {}

        def spy(*a, **kw):
            called["rrf"] = True
            return orig(*a, **kw)

        eng.rrf_fuse = spy  # type: ignore[method-assign]
        eng.fuse_and_filter(
            "q", {"vector": [{"memory_id": "v", "score": 0.5}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        assert called.get("rrf")

    def test_dispatches_to_additive(self):
        eng = _Engine()
        orig = eng.additive_fuse
        called: dict[str, bool] = {}

        def spy(*a, **kw):
            called["additive"] = True
            return orig(*a, **kw)

        eng.additive_fuse = spy  # type: ignore[method-assign]
        eng.fuse_and_filter(
            "q", {"vector": [{"memory_id": "v", "content": "python", "score": 0.9}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
            fusion_mode="additive",
        )
        assert called.get("additive")

    def test_trace_records_rrf_step(self):
        eng = _Engine()
        steps: list[str] = []

        class _Trace:
            def add_step(self, name: str, **kw: Any) -> None:
                steps.append(name)

        eng.fuse_and_filter(
            "q", {"vector": [{"memory_id": "v", "score": 0.5}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
            trace=_Trace(),
        )
        assert "rrf_fuse" in steps

    def test_trace_records_additive_step(self):
        eng = _Engine()
        steps: list[str] = []

        class _Trace:
            def add_step(self, name: str, **kw: Any) -> None:
                steps.append(name)

        eng.fuse_and_filter(
            "q", {"vector": [{"memory_id": "v", "content": "python", "score": 0.9}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
            trace=_Trace(), fusion_mode="additive",
        )
        assert "additive_fuse" in steps

    def test_filters_sync_turn_source(self):
        merged = [
            {"memory_id": "a", "score": 0.5, "source": "sync_turn"},
            {"memory_id": "b", "score": 0.4},
        ]
        facade = _FakeFacade(_rrf=_FakeRRF(merged=merged))
        eng = _Engine(facade)
        out = eng.fuse_and_filter(
            "q", {"vector": [{"memory_id": "x", "score": 0.5}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        assert all(r.get("source") != "sync_turn" for r in out)

    def test_filters_top_level_superseded(self):
        merged = [
            {"memory_id": "old", "score": 0.5, "is_superseded": True},
            {"memory_id": "new", "score": 0.4},
        ]
        facade = _FakeFacade(_rrf=_FakeRRF(merged=merged))
        eng = _Engine(facade)
        out = eng.fuse_and_filter(
            "q", {"vector": [{"memory_id": "x", "score": 0.5}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        ids = {r["memory_id"] for r in out}
        assert "old" not in ids
        assert "new" in ids

    def test_filters_metadata_superseded(self):
        merged = [
            {"memory_id": "old", "score": 0.5, "metadata": {"is_superseded": True}},
            {"memory_id": "new", "score": 0.4},
        ]
        facade = _FakeFacade(_rrf=_FakeRRF(merged=merged))
        eng = _Engine(facade)
        out = eng.fuse_and_filter(
            "q", {"vector": [{"memory_id": "x", "score": 0.5}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        ids = {r["memory_id"] for r in out}
        assert "old" not in ids

    def test_pipeline_order_calls_supplement_and_boost(self):
        # 触发 supplement (reasoning/action 计数不足) + type_boost (reasoning 1.3x)
        bm25 = _FakeBM25({
            "教训": [{"memory_id": "sup", "type": "reasoning", "score": 0.10}],
        })
        facade = _FakeFacade(_bm25=bm25)
        eng = _Engine(facade)
        merged = [{"memory_id": "r", "type": "reasoning", "score": 0.20}]
        facade._rrf = _FakeRRF(merged=merged)
        out = eng.fuse_and_filter(
            "q", {"vector": [{"memory_id": "r", "score": 0.5}]},
            is_garbage=False, doc_count=30, top_k=5, max_tokens=1000,
        )
        # 至少触发一次补充查询
        assert bm25.queries
        # 结果包含 reasoning 类型
        assert any(r.get("type") == "reasoning" for r in out)


# ─── 类常量 ────────────────────────────────────────────────


class TestClassConstants:
    def test_type_boost_reasoning_action(self):
        assert FusionMixin._TYPE_BOOST["reasoning"] == 1.3
        assert FusionMixin._TYPE_BOOST["action"] == 1.3
        assert "correction" not in FusionMixin._TYPE_BOOST

    def test_pref_intent_words(self):
        assert "偏好" in FusionMixin._PREF_INTENT_WORDS
        assert "prefer" in FusionMixin._PREF_INTENT_WORDS
