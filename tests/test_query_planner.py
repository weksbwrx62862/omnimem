"""HMS Planner（规则版）单元与集成测试。

覆盖：
  - 意图检测：temporal / entity / count / preference / conversational / general / mixed
  - QueryPlan 接口：weight_for / is_default
  - 集成：dispatch_channels 通道裁剪、rrf_fuse 权重叠加、search 全链路传递
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from omnimem.retrieval.planner import QueryPlan, QueryPlanner

# ── 单元：意图检测 ──

class TestQueryPlannerIntents:
    """各类查询意图的检测正确性。"""

    def setup_method(self) -> None:
        self.planner = QueryPlanner()

    def test_temporal_chinese(self) -> None:
        """中文时间词 → temporal 意图。"""
        plan = self.planner.plan("上个月处理了几个问题")
        assert plan.intent in ("temporal", "mixed")
        assert "temporal" in plan.rationale
        assert plan.channel_weights["temporal"] >= 2.0
        assert not plan.is_default

    def test_temporal_english(self) -> None:
        """英文时间词 → temporal 意图。"""
        plan = self.planner.plan("what did we discuss last week")
        assert "temporal" in plan.rationale

    def test_temporal_absolute_date(self) -> None:
        """绝对日期（2024年5月）→ temporal 意图。"""
        plan = self.planner.plan("2024年5月发生了什么")
        assert "temporal" in plan.rationale

    def test_entity_quoted(self) -> None:
        """引号专名 → entity 意图。"""
        plan = self.planner.plan("关于\"OmniMem\"的检索优化")
        assert "entity" in plan.rationale
        assert plan.channel_weights["graph"] >= 2.0
        assert plan.channel_weights["entity"] >= 1.5

    def test_entity_zh_suffix(self) -> None:
        """中文称谓（张总/李经理）→ entity 意图。"""
        assert "entity" in self.planner.plan("张总上次说什么了").rationale
        assert "entity" in self.planner.plan("李经理的项目进展").rationale

    def test_entity_org(self) -> None:
        """组织后缀（公司/研究院）→ entity 意图。"""
        assert "entity" in self.planner.plan("阿里云科技最近动态").rationale

    def test_entity_camel(self) -> None:
        """CamelCase 专名 → entity 意图。"""
        assert "entity" in self.planner.plan("OpenAI 的 GPT-5 进展").rationale

    def test_count(self) -> None:
        """计数类查询 → count 意图，BM25 加权。"""
        plan = self.planner.plan("how many projects did we complete")
        assert "count" in plan.rationale
        assert plan.channel_weights["bm25"] >= 2.0

    def test_count_chinese(self) -> None:
        """中文计数（几个/多少）→ count 意图。"""
        assert "count" in self.planner.plan("我们一共几个客户").rationale

    def test_preference(self) -> None:
        """偏好查询 → preference 意图。"""
        plan = self.planner.plan("推荐哪个模型")
        assert "preference" in plan.rationale
        assert plan.channel_weights["vector"] > 1.0

    def test_conversational(self) -> None:
        """寒暄查询 → conversational 意图，削减召回规模。"""
        plan = self.planner.plan("你好")
        assert plan.intent == "conversational"
        assert plan.top_k_scale < 1.0
        assert plan.channel_weights["vector"] < 1.0

    def test_general_default(self) -> None:
        """普通查询 → general，is_default=True（调用方零开销跳过）。"""
        plan = self.planner.plan("解释一下记忆系统的架构")
        assert plan.is_default is True
        assert plan.intent == "general"
        assert plan.channel_weights == {}

    def test_empty_query(self) -> None:
        """空查询 → general 默认。"""
        assert self.planner.plan("").is_default is True
        assert self.planner.plan("   ").is_default is True

    def test_mixed_temporal_entity(self) -> None:
        """时间+实体组合 → mixed，两套权重同时生效。"""
        plan = self.planner.plan("上个月张总说过什么")
        assert plan.intent == "mixed"
        assert "temporal" in plan.rationale and "entity" in plan.rationale
        assert plan.channel_weights["temporal"] >= 2.0
        assert plan.channel_weights["graph"] >= 2.0


# ── 单元：QueryPlan 接口 ──

class TestQueryPlanInterface:
    """QueryPlan 数据类行为。"""

    def test_weight_for_default(self) -> None:
        """未配置的通道返回 1.0。"""
        plan = QueryPlan()
        assert plan.weight_for("vector") == 1.0
        assert plan.weight_for("nonexistent") == 1.0

    def test_weight_for_configured(self) -> None:
        """已配置的通道返回实际权重。"""
        plan = QueryPlan(channel_weights={"bm25": 2.5})
        assert plan.weight_for("bm25") == 2.5


# ── 集成：dispatch_channels 通道裁剪 ──

def _make_facade(**config_overrides) -> MagicMock:
    """构造最小 facade mock（与 test_query_enhancement 同风格）。"""
    facade = MagicMock()
    facade._synonym_map = {}
    facade._config = {
        "query_expansion_enabled": False,
        "entity_boost_weight": 1.5,
        **config_overrides,
    }
    facade._channels = {
        "bm25": (MagicMock(), 1.0),
        "vector": (MagicMock(), 3.0),
    }
    facade._recall_strategy = "hybrid"
    facade._recall_timeout_ms = 5000
    facade._source_weights = {}
    facade._vector_breaker = MagicMock()
    facade._vector_breaker.should_skip.return_value = False
    facade._catalog = None
    facade._reranker = None
    facade._rrf = MagicMock()
    facade._vector = MagicMock()
    facade._vector.count.return_value = 10
    facade._bm25 = MagicMock()
    facade._query_cache = {}
    facade._query_cache_ttl = 60
    facade._ml_cache = None
    facade._max_sync_turn_entries = 5
    facade._sync_turn_ids = []
    return facade


class TestDispatchChannelSkipping:
    """Planner 低权重通道应在 dispatch 阶段被裁剪。"""

    def test_low_weight_channel_skipped(self) -> None:
        """权重低于阈值的通道不执行检索。"""
        facade = _make_facade()
        orchestrator = _make_orchestrator_with_mock_searches(facade)
        skipped = {"bm25"}

        orchestrator.dispatch_channels(
            "测试查询", 10, None, None, planner_weights={"bm25": 0.1}
        )
        # bm25 被裁剪 → 其 search 不应被调用
        assert skipped & set(orchestrator.called_channels) == set()
        assert "vector" in orchestrator.called_channels

    def test_normal_weight_channel_runs(self) -> None:
        """权重正常的通道照常执行。"""
        facade = _make_facade()
        orchestrator = _make_orchestrator_with_mock_searches(facade)

        orchestrator.dispatch_channels(
            "测试查询", 10, None, None, planner_weights={"bm25": 2.0}
        )
        assert "bm25" in orchestrator.called_channels
        assert "vector" in orchestrator.called_channels

    def test_no_planner_weights_no_change(self) -> None:
        """不传 planner_weights 时行为不变（全通道执行）。"""
        facade = _make_facade()
        orchestrator = _make_orchestrator_with_mock_searches(facade)

        orchestrator.dispatch_channels("测试查询", 10, None, None)
        assert "bm25" in orchestrator.called_channels
        assert "vector" in orchestrator.called_channels

    def test_catalog_skipped_with_low_weight(self) -> None:
        """catalog 通道同样受 planner 裁剪。"""
        facade = _make_facade()
        facade._catalog = MagicMock()
        facade._catalog.search.return_value = [{"memory_id": "c1", "content": "cat"}]
        orchestrator = _make_orchestrator_with_mock_searches(facade)

        orchestrator.dispatch_channels(
            "测试查询", 10, None, None, planner_weights={"catalog": 0.1}
        )
        assert "catalog" not in orchestrator.called_channels


def _make_orchestrator_with_mock_searches(facade) -> MagicMock:
    """构造 orchestrator，mock 各通道 search 记录调用。"""
    from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator

    orchestrator = HybridOrchestrator(facade)
    orchestrator.called_channels: list[str] = []

    def _fake_search(name: str):
        def _search(query: str, top_k: int) -> list[dict]:
            orchestrator.called_channels.append(name)
            return [{"memory_id": f"{name}1", "content": f"{name}-result", "score": 0.5}]
        return _search

    for ch_name in ("bm25", "vector"):
        facade._channels[ch_name][0].search.side_effect = _fake_search(ch_name)
    # bm25_search 走 synonym_expander.search 封装，需单独 mock
    original_bm25 = orchestrator.bm25_search

    def _bm25_wrapper(query: str, top_k: int):
        orchestrator.called_channels.append("bm25")
        return [{"memory_id": "bm251", "content": "bm25-result", "score": 0.5}]
    orchestrator.bm25_search = _bm25_wrapper
    orchestrator._original_bm25 = original_bm25
    return orchestrator


# ── 集成：rrf_fuse 权重叠加 ──

class TestRrfFusePlannerWeights:
    """Planner 局部权重应乘入 RRF base_weights。"""

    def _facade_with_rrf_capture(self) -> tuple[MagicMock, list]:
        facade = _make_facade()
        captured: list = []

        class _FakeRRF:
            def merge(self, result_lists, min_rrf=0.0, weights=None):
                captured.append(weights)
                fused = []
                for lst in result_lists:
                    fused.extend(lst)
                return fused

        facade._rrf = _FakeRRF()
        return facade, captured

    def test_planner_weights_multiply_base(self) -> None:
        """vector 3.0 × planner 1.2 → base_weights 中 vector 权重被放大。"""
        from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator

        facade, captured = self._facade_with_rrf_capture()
        orchestrator = HybridOrchestrator(facade)
        channel_results = {
            "vector": [{"memory_id": "v1", "content": "a", "score": 0.9}],
            "bm25": [{"memory_id": "b1", "content": "b", "score": 0.7}],
        }
        orchestrator.rrf_fuse(
            "测试", channel_results,
            is_garbage=False, doc_count=10, top_k=10, max_tokens=1500,
            planner_weights={"vector": 1.2, "bm25": 0.5},
        )
        assert captured, "RRF.merge 应被调用"
        weights = captured[0]
        # 通道注册权重: vector=3.0, bm25=1.0 → planner 叠加后 vector=3.6, bm25=0.5
        by_name = dict(zip(("vector", "bm25"), weights))
        assert by_name["vector"] == pytest.approx(3.0 * 1.2)
        assert by_name["bm25"] == pytest.approx(1.0 * 0.5)

    def test_no_planner_weights_unchanged(self) -> None:
        """不传 planner_weights 时权重不变。"""
        from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator

        facade, captured = self._facade_with_rrf_capture()
        orchestrator = HybridOrchestrator(facade)
        channel_results = {
            "vector": [{"memory_id": "v1", "content": "a", "score": 0.9}],
            "bm25": [{"memory_id": "b1", "content": "b", "score": 0.7}],
        }
        orchestrator.rrf_fuse(
            "测试", channel_results,
            is_garbage=False, doc_count=10, top_k=10, max_tokens=1500,
        )
        weights = captured[0]
        by_name = dict(zip(("vector", "bm25"), weights))
        assert by_name["vector"] == pytest.approx(3.0)
        assert by_name["bm25"] == pytest.approx(1.0)


# ── 集成：search 全链路 ──

class TestSearchPipeline:
    """search() 主入口应把 planner 权重传递到 dispatch 与 fuse。"""

    def test_search_passes_planner_weights(self) -> None:
        """实体查询：dispatch 与 rrf_fuse 都收到 planner_weights。"""
        from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator

        facade = _make_facade()
        orchestrator = HybridOrchestrator(facade)
        captured_fuse_weights: list = []

        # mock dispatch_channels 与 rrf_fuse 捕获参数
        original_dispatch = orchestrator.dispatch_channels

        def _fake_dispatch(query, top_k, allowed, trace, bm25_query=None, planner_weights=None):
            return {"vector": [{"memory_id": "v1", "content": "x", "score": 0.8}]}
        orchestrator.dispatch_channels = _fake_dispatch

        original_fuse = orchestrator.fuse_and_filter

        def _fake_fuse(query, channel_results, *, is_garbage, doc_count, top_k, max_tokens, trace=None, fusion_mode="rrf", planner_weights=None):
            captured_fuse_weights.append(planner_weights)
            return [{"memory_id": "v1", "content": "x", "score": 0.8}]
        orchestrator.fuse_and_filter = _fake_fuse

        orchestrator.search("张总的项目进展", mode="rag", top_k=10)

        assert captured_fuse_weights and captured_fuse_weights[0] is not None
        assert captured_fuse_weights[0]["graph"] >= 2.0

    def test_search_general_no_planner_overhead(self) -> None:
        """general 查询：planner_weights 为 None，零额外开销。"""
        from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator

        facade = _make_facade()
        orchestrator = HybridOrchestrator(facade)
        captured_fuse_weights: list = []

        def _fake_dispatch(query, top_k, allowed, trace, bm25_query=None, planner_weights=None):
            return {"vector": [{"memory_id": "v1", "content": "x", "score": 0.8}]}
        orchestrator.dispatch_channels = _fake_dispatch

        def _fake_fuse(query, channel_results, *, is_garbage, doc_count, top_k, max_tokens, trace=None, fusion_mode="rrf", planner_weights=None):
            captured_fuse_weights.append(planner_weights)
            return [{"memory_id": "v1", "content": "x", "score": 0.8}]
        orchestrator.fuse_and_filter = _fake_fuse

        orchestrator.search("解释一下记忆系统的架构", mode="rag", top_k=10)
        assert captured_fuse_weights == [None]

    def test_planner_can_be_disabled_by_config(self) -> None:
        """planner_enabled=False 时跳过 planner。"""
        facade = _make_facade(planner_enabled=False)
        from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator

        orchestrator = HybridOrchestrator(facade)
        assert orchestrator._planner_enabled is False

    @pytest.mark.asyncio
    async def test_async_search_planner(self) -> None:
        """async_search 同样传递 planner 权重。"""
        from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator

        facade = _make_facade()
        orchestrator = HybridOrchestrator(facade)
        captured: list = []

        def _fake_fuse(query, channel_results, *, is_garbage, doc_count, top_k, max_tokens, planner_weights=None):
            captured.append(planner_weights)
            return [{"memory_id": "v1", "content": "x", "score": 0.8}]
        orchestrator.rrf_fuse = _fake_fuse

        # 阻断真实通道检索：直接 mock bm25/vector
        facade._channels["bm25"][0].search.return_value = []
        facade._channels["vector"][0].search.return_value = []
        await orchestrator.async_search("上周讨论了什么", mode="rag", top_k=10)
        assert captured and captured[0] is not None
        assert captured[0]["temporal"] >= 2.0
