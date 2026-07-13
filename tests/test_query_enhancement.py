"""查询增强（Task 3）单元测试：同义扩展、实体加权、配置项。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator
from omnimem.retrieval.synonym_expander import SynonymExpander


# ── 辅助：构造最小 facade ──

def _make_facade(**config_overrides) -> MagicMock:
    """构造最小 facade mock，支持查询增强配置。"""
    facade = MagicMock()
    facade._synonym_map = {
        "编程": ["代码", "开发", "程序"],
        "宠物": ["猫咪", "狗狗"],
    }
    facade._config = {
        "query_expansion_enabled": True,
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
    facade._vector_breaker.should_skip.return_value = True  # 跳过向量检索以简化测试
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


# ── 测试 1: SynonymExpander.expand() 扩展查询 ──

class TestSynonymExpanderExpand:
    """测试 SynonymExpander.expand() 方法。"""

    def test_expand_returns_synonyms_for_matched_key(self) -> None:
        """查询包含映射 key 时，应返回对应的同义词列表。"""
        expander = SynonymExpander(synonym_map={"编程": ["代码", "开发"]})
        result = expander.expand("我想学编程")
        assert "代码" in result
        assert "开发" in result

    def test_expand_excludes_terms_already_in_query(self) -> None:
        """已在查询中出现同义词不应重复返回。"""
        expander = SynonymExpander(synonym_map={"编程": ["代码", "开发"]})
        result = expander.expand("编程代码")
        # "代码" 已在查询中，不应返回
        assert "代码" not in result
        assert "开发" in result

    def test_expand_returns_empty_when_no_match(self) -> None:
        """查询不包含任何映射 key 时，返回空列表。"""
        expander = SynonymExpander(synonym_map={"编程": ["代码"]})
        result = expander.expand("今天天气不错")
        assert result == []

    def test_expand_deduplicates(self) -> None:
        """多个 key 映射到同一同义词时，应去重。"""
        expander = SynonymExpander(synonym_map={
            "编程": ["代码", "开发"],
            "代码": ["编程", "源码"],
        })
        result = expander.expand("编程代码")
        # 不应有重复项
        assert len(result) == len(set(result))


# ── 测试 2: 查询增强后 BM25 检索结果更多 ──

class TestQueryExpansionBM25:
    """测试查询增强后 BM25 通道接收增强查询。"""

    def test_bm25_receives_expanded_query_via_dispatch(self) -> None:
        """dispatch_channels 传入 bm25_query 时，BM25 应使用增强查询。"""
        facade = _make_facade()
        orchestrator = HybridOrchestrator(facade)
        assert orchestrator._query_expansion_enabled is True

        # mock bm25_search 以验证传入的查询
        captured_queries: list[str] = []
        def _capture_bm25(query: str, top_k: int) -> list[dict]:
            captured_queries.append(query)
            return [{"memory_id": "m1", "content": "test", "score": 0.5}]
        orchestrator.bm25_search = _capture_bm25

        # 构造增强查询，手动传入 dispatch_channels
        expanded_terms = orchestrator._synonym_expander.expand("编程")
        bm25_query = "编程 " + " ".join(expanded_terms)
        orchestrator.dispatch_channels("编程", top_k=10, allowed_channels=None, trace=None, bm25_query=bm25_query)
        # BM25 应收到扩展后的查询
        assert len(captured_queries) == 1
        assert "代码" in captured_queries[0] or "开发" in captured_queries[0]

    def test_search_method_expands_query_for_bm25(self) -> None:
        """search() 方法应在调用 dispatch_channels 前自动扩展查询。"""
        facade = _make_facade()
        orchestrator = HybridOrchestrator(facade)

        # 捕获 dispatch_channels 的 bm25_query 参数
        captured_bm25_queries: list[str | None] = []
        original_dispatch = orchestrator.dispatch_channels

        def _capture_dispatch(query, top_k, allowed_channels, trace, bm25_query=None):
            captured_bm25_queries.append(bm25_query)
            # 返回空结果以简化流程
            return {"bm25": [{"memory_id": "m1", "content": "test", "score": 0.5}]}

        orchestrator.dispatch_channels = _capture_dispatch
        # mock fuse_and_filter 返回空列表
        orchestrator.fuse_and_filter = lambda *a, **kw: []
        # mock 缓存检查返回 None
        orchestrator.check_cache = lambda k: None
        orchestrator.set_cache = lambda k, v: None

        orchestrator.search("编程")
        # bm25_query 应不为 None 且包含同义词
        assert captured_bm25_queries[0] is not None
        assert "代码" in captured_bm25_queries[0] or "开发" in captured_bm25_queries[0]

    def test_vector_receives_original_query(self) -> None:
        """向量检索应使用原始查询，不受同义扩展影响。"""
        facade = _make_facade()
        facade._vector_breaker.should_skip.return_value = False

        # 捕获向量检索通道的查询
        vector_retriever = MagicMock()
        captured_vector_queries: list[str] = []
        def _capture_vector_search(query, top_k=10):
            captured_vector_queries.append(query)
            return MagicMock(results=[{"memory_id": "v1", "content": "test", "score": 0.5}])
        vector_retriever.search.side_effect = _capture_vector_search
        # 更新 facade 的 channels，使向量通道使用 mock retriever
        facade._channels = {
            "bm25": (MagicMock(), 1.0),
            "vector": (vector_retriever, 3.0),
        }

        orchestrator = HybridOrchestrator(facade)

        # mock bm25_search
        orchestrator.bm25_search = lambda q, k: []

        # 传入 bm25_query 以验证向量通道不受影响
        orchestrator.dispatch_channels(
            "编程", top_k=10, allowed_channels=None, trace=None,
            bm25_query="编程 代码 开发",
        )
        # 向量检索应收到原始查询（不含同义扩展词）
        assert captured_vector_queries == ["编程"]


# ── 测试 3: entity_boost_weight 配置可覆盖 ──

class TestEntityBoostConfig:
    """测试实体加权配置项。"""

    def test_default_entity_boost_weight(self) -> None:
        """默认 entity_boost_weight 应为 1.5。"""
        facade = _make_facade()
        orchestrator = HybridOrchestrator(facade)
        assert orchestrator._entity_boost_weight == 1.5

    def test_custom_entity_boost_weight(self) -> None:
        """自定义 entity_boost_weight 应正确覆盖默认值。"""
        facade = _make_facade(entity_boost_weight=2.0)
        orchestrator = HybridOrchestrator(facade)
        assert orchestrator._entity_boost_weight == 2.0

    def test_entity_boost_applied_in_apply_type_boost(self) -> None:
        """apply_type_boost 应对实体匹配的结果加权。"""
        results = [
            {
                "memory_id": "m1",
                "score": 0.5,
                "type": "fact",
                "metadata": {"entities": ["Python", "OmniMem"]},
            },
            {
                "memory_id": "m2",
                "score": 0.5,
                "type": "fact",
                "metadata": {"entities": ["Rust"]},
            },
        ]
        # 查询包含 "Python" 实体
        boosted = HybridOrchestrator.apply_type_boost(
            results, updated_boost=0.3, query="Python编程", entity_boost_weight=1.5
        )
        # m1 包含 "Python" 实体，应被加权
        m1 = next(r for r in boosted if r["memory_id"] == "m1")
        m2 = next(r for r in boosted if r["memory_id"] == "m2")
        assert m1["score"] > m2["score"]
        assert m1.get("entity_boost") == 1.5

    def test_entity_boost_not_applied_when_weight_is_one(self) -> None:
        """entity_boost_weight=1.0 时不应加权。"""
        results = [
            {
                "memory_id": "m1",
                "score": 0.5,
                "type": "fact",
                "metadata": {"entities": ["Python"]},
            },
        ]
        boosted = HybridOrchestrator.apply_type_boost(
            results, updated_boost=0.3, query="Python编程", entity_boost_weight=1.0
        )
        # entity_boost_weight=1.0 不触发实体加权
        assert "entity_boost" not in boosted[0]


# ── 测试 4: query_expansion_enabled=False 时跳过扩展 ──

class TestQueryExpansionDisabled:
    """测试 query_expansion_enabled=False 时跳过扩展。"""

    def test_expansion_disabled_no_bm25_enhancement(self) -> None:
        """query_expansion_enabled=False 时 BM25 不应接收扩展查询。"""
        facade = _make_facade(query_expansion_enabled=False)
        orchestrator = HybridOrchestrator(facade)
        assert orchestrator._query_expansion_enabled is False

        captured_queries: list[str] = []
        def _capture_bm25(query: str, top_k: int) -> list[dict]:
            captured_queries.append(query)
            return [{"memory_id": "m1", "content": "test", "score": 0.5}]
        orchestrator.bm25_search = _capture_bm25

        orchestrator.dispatch_channels("编程", top_k=10, allowed_channels=None, trace=None)
        # BM25 应收到原始查询（无扩展）
        assert captured_queries == ["编程"]

    def test_expansion_disabled_config(self) -> None:
        """配置 query_expansion_enabled=False 应正确设置。"""
        facade = _make_facade(query_expansion_enabled=False)
        orchestrator = HybridOrchestrator(facade)
        assert orchestrator._query_expansion_enabled is False

    def test_expansion_enabled_by_default(self) -> None:
        """无 config 时 query_expansion_enabled 默认为 True。"""
        facade = MagicMock()
        facade._synonym_map = {}
        facade._config = None
        facade._channels = {}
        orchestrator = HybridOrchestrator(facade)
        assert orchestrator._query_expansion_enabled is True
        assert orchestrator._entity_boost_weight == 1.5
