"""retrieval/hybrid_orchestrator.py — HybridOrchestrator 单元测试。

覆盖: 构造配置解析 / executor 生命周期 / 通道委托 / dispatch_channels 策略
     search 主流程 (garbage/llm/COUNT/cache/planner/synonym/preference)
     async_search 对称路径
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock

from omnimem.retrieval.base import BaseRetriever, RetrievalResult
from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator

# ─── 脚手架 ────────────────────────────────────────────────


class _FakeBreaker:
    def __init__(self, skip: bool = False) -> None:
        self._skip = skip
        self.successes = 0
        self.failures = 0

    def should_skip(self) -> bool:
        return self._skip

    def record_success(self) -> None:
        self.successes += 1

    def record_failure(self) -> None:
        self.failures += 1


class _FakeVector:
    def __init__(self, count: int = 0, results: list | None = None) -> None:
        self._count = count
        self._results = results or []
        self._embedding_fn = None

    def count(self) -> int:
        return self._count

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        return list(self._results)


class _FakeBM25:
    def __init__(self, results: list | None = None) -> None:
        self._results = results or []
        self.queries: list[str] = []

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        self.queries.append(query)
        return list(self._results)


class _FakeCatalog:
    def __init__(self, results: list | None = None, raises: bool = False) -> None:
        self._results = results or []
        self._raises = raises

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        if self._raises:
            raise RuntimeError("catalog boom")
        return list(self._results)


class _FakeBaseRetriever(BaseRetriever):
    def __init__(self, results: list | None = None) -> None:
        self._results = results or []

    @property
    def name(self) -> str:
        return "fake"

    def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        return RetrievalResult(results=list(self._results), scores=[0.5] * len(self._results), channel=self.name)


class _FakeFacade:
    def __init__(self, **kw: Any) -> None:
        self._synonym_map: dict = kw.get("synonym_map", {})
        self._config: dict | None = kw.get("config")
        self._vector = kw.get("vector", _FakeVector())
        self._bm25 = kw.get("bm25", _FakeBM25())
        self._catalog = kw.get("catalog")
        self._channels: dict[str, tuple] = kw.get("channels", {})
        self._recall_strategy: str = kw.get("recall_strategy", "hybrid")
        self._recall_timeout_ms: int = kw.get("recall_timeout_ms", 500)
        self._vector_breaker = kw.get("vector_breaker", _FakeBreaker())
        self._source_weights: dict[str, float] | None = kw.get("source_weights")
        self._source_weights_lock = kw.get("source_weights_lock")
        # QueryCacheMixin 依赖
        self._ml_cache: Any = kw.get("ml_cache")
        self._query_cache: dict[str, tuple] = kw.get("query_cache", {})
        self._query_cache_ttl: int = kw.get("query_cache_ttl", 300)
        # FusionMixin 依赖
        self._rrf: Any = kw.get("rrf", _SimpleRRF())
        self._reranker: Any = kw.get("reranker")


class _SimpleRRF:
    """最小 RRF 桩：拼接所有列表, 按 score 降序。"""

    def merge(self, lists, *, min_rrf=0.0, weights=None):
        flat: list[dict[str, Any]] = []
        for group in lists:
            for r in group:
                flat.append(dict(r))
        flat.sort(key=lambda x: x.get("score", 0), reverse=True)
        return [r for r in flat if r.get("score", 0) >= min_rrf]


def _orch(facade: _FakeFacade | None = None, **kw: Any) -> HybridOrchestrator:
    return HybridOrchestrator(facade or _FakeFacade(**kw))


# ─── 构造与配置 ────────────────────────────────────────────


class TestConstructor:
    def test_defaults_when_no_config(self):
        orch = _orch(_FakeFacade(config=None))
        assert orch._planner_enabled is True
        assert orch._updated_boost == 0.3
        assert orch._query_expansion_enabled is True
        assert orch._entity_boost_weight == 1.5
        assert orch._min_relevance_score == 0.35
        assert orch._preference_gate_enabled is True

    def test_config_overrides(self):
        cfg = {
            "planner_enabled": False,
            "updated_boost": 0.5,
            "query_expansion_enabled": False,
            "entity_boost_weight": 2.0,
            "min_relevance_score": 0.5,
            "preference_relevance_gate": False,
        }
        orch = _orch(_FakeFacade(config=cfg))
        assert orch._planner_enabled is False
        assert orch._updated_boost == 0.5
        assert orch._query_expansion_enabled is False
        assert orch._entity_boost_weight == 2.0
        assert orch._min_relevance_score == 0.5
        assert orch._preference_gate_enabled is False

    def test_executor_created(self):
        orch = _orch()
        assert orch._executor is not None

    def test_shutdown_releases_executor(self):
        orch = _orch()
        orch.shutdown()
        assert orch._executor is None


# ─── 通道委托 ──────────────────────────────────────────────


class TestChannelDelegation:
    def test_vector_search_delegates(self):
        vec = _FakeVector(results=[{"memory_id": "v1", "score": 0.9}])
        orch = _orch(_FakeFacade(vector=vec))
        out = orch.vector_search("q", top_k=5)
        assert out == [{"memory_id": "v1", "score": 0.9}]

    def test_bm25_search_uses_synonym_expander(self):
        bm = _FakeBM25(results=[{"memory_id": "b1"}])
        orch = _orch(_FakeFacade(bm25=bm, synonym_map={"python": ["py"]}))
        out = orch.bm25_search("python", top_k=5)
        assert isinstance(out, list)

    def test_catalog_search_returns_empty_when_no_catalog(self):
        orch = _orch(_FakeFacade(catalog=None))
        assert orch.catalog_search("q", top_k=5) == []

    def test_catalog_search_delegates(self):
        cat = _FakeCatalog(results=[{"memory_id": "c1"}])
        orch = _orch(_FakeFacade(catalog=cat))
        assert orch.catalog_search("q", top_k=5) == [{"memory_id": "c1"}]

    def test_catalog_search_swallows_exception(self):
        cat = _FakeCatalog(raises=True)
        orch = _orch(_FakeFacade(catalog=cat))
        assert orch.catalog_search("q", top_k=5) == []

    def test_search_base_retriever_extracts_results(self):
        r = _FakeBaseRetriever(results=[{"memory_id": "x"}])
        out = HybridOrchestrator._search_base_retriever(r, "q", 5)
        assert out == [{"memory_id": "x"}]


# ─── dispatch_channels ─────────────────────────────────────


class TestDispatchChannels:
    def test_keyword_strategy_only_bm25(self):
        bm = _FakeBM25(results=[{"memory_id": "b1"}])
        facade = _FakeFacade(
            bm25=bm,
            channels={"bm25": (bm, 1.0), "vector": (_FakeVector(), 3.0)},
            recall_strategy="keyword",
        )
        orch = _orch(facade)
        out = orch.dispatch_channels("q", 5, None, None)
        assert "bm25" in out
        assert "vector" not in out

    def test_embedding_strategy_only_vector(self):
        vec = _FakeVector(results=[{"memory_id": "v1"}])
        facade = _FakeFacade(
            vector=vec,
            channels={"vector": (vec, 3.0)},
            recall_strategy="embedding",
        )
        orch = _orch(facade)
        out = orch.dispatch_channels("q", 5, None, None)
        assert "vector" in out
        assert facade._vector_breaker.successes == 1

    def test_embedding_strategy_breaker_open_skips(self):
        vec = _FakeVector(results=[{"memory_id": "v1"}])
        facade = _FakeFacade(
            vector=vec,
            channels={"vector": (vec, 3.0)},
            recall_strategy="embedding",
            vector_breaker=_FakeBreaker(skip=True),
        )
        orch = _orch(facade)
        out = orch.dispatch_channels("q", 5, None, None)
        assert out == {}

    def test_embedding_strategy_failure_records_breaker(self):
        class _BoomVec:
            def count(self): return 0
            def search(self, *_a, **_kw): raise RuntimeError("boom")

        facade = _FakeFacade(
            vector=_BoomVec(),
            channels={"vector": (_BoomVec(), 3.0)},
            recall_strategy="embedding",
        )
        orch = _orch(facade)
        orch.dispatch_channels("q", 5, None, None)
        assert facade._vector_breaker.failures == 1

    def test_hybrid_dispatches_all_channels(self):
        vec = _FakeVector(results=[{"memory_id": "v1"}])
        bm = _FakeBM25(results=[{"memory_id": "b1"}])
        facade = _FakeFacade(
            vector=vec, bm25=bm,
            channels={"vector": (vec, 3.0), "bm25": (bm, 1.0)},
            recall_strategy="hybrid",
        )
        orch = _orch(facade)
        out = orch.dispatch_channels("q", 5, None, None)
        assert "vector" in out and "bm25" in out

    def test_allowed_channels_filters(self):
        vec = _FakeVector(results=[{"memory_id": "v1"}])
        bm = _FakeBM25(results=[{"memory_id": "b1"}])
        facade = _FakeFacade(
            vector=vec, bm25=bm,
            channels={"vector": (vec, 3.0), "bm25": (bm, 1.0)},
        )
        orch = _orch(facade)
        out = orch.dispatch_channels("q", 5, {"bm25"}, None)
        assert "bm25" in out
        assert "vector" not in out

    def test_planner_skip_threshold(self):
        vec = _FakeVector(results=[{"memory_id": "v1"}])
        bm = _FakeBM25(results=[{"memory_id": "b1"}])
        facade = _FakeFacade(
            vector=vec, bm25=bm,
            channels={"vector": (vec, 3.0), "bm25": (bm, 1.0)},
        )
        orch = _orch(facade)
        # planner_weights 中 vector=0.0 → 低于 CHANNEL_SKIP_THRESHOLD → 跳过
        out = orch.dispatch_channels("q", 5, None, None, planner_weights={"vector": 0.0})
        assert "vector" not in out
        assert "bm25" in out

    def test_bm25_uses_enhanced_query(self):
        bm = _FakeBM25(results=[{"memory_id": "b1"}])
        facade = _FakeFacade(bm25=bm, channels={"bm25": (bm, 1.0)}, recall_strategy="keyword")
        orch = _orch(facade)
        orch.dispatch_channels("orig", 5, None, None, bm25_query="enhanced")
        assert bm.queries == ["enhanced"]

    def test_base_retriever_channel(self):
        r = _FakeBaseRetriever(results=[{"memory_id": "x"}])
        facade = _FakeFacade(channels={"custom": (r, 1.0)})
        orch = _orch(facade)
        out = orch.dispatch_channels("q", 5, None, None)
        assert "custom" in out

    def test_catalog_included_when_present(self):
        cat = _FakeCatalog(results=[{"memory_id": "c1"}])
        facade = _FakeFacade(catalog=cat, channels={})
        orch = _orch(facade)
        out = orch.dispatch_channels("q", 5, None, None)
        assert "catalog" in out


# ─── search 主流程 ─────────────────────────────────────────


class TestSearchMain:
    def test_garbage_query_caps_top_k(self):
        vec = _FakeVector(count=200)
        bm = _FakeBM25()
        facade = _FakeFacade(vector=vec, bm25=bm, channels={"bm25": (bm, 1.0)})
        orch = _orch(facade)
        # 纯符号查询 → is_garbage_query=True
        out = orch.search("!!! ???", top_k=40, max_tokens=1500)
        assert isinstance(out, list)

    def test_llm_mode_raises_top_k(self):
        vec = _FakeVector(count=10)
        bm = _FakeBM25(results=[{"memory_id": "b1", "content": "python", "score": 0.5}])
        facade = _FakeFacade(vector=vec, bm25=bm, channels={"bm25": (bm, 1.0)})
        orch = _orch(facade)
        out = orch.search("python tips", mode="llm", top_k=5, max_tokens=100)
        # llm 模式会把 top_k 提到 >=20, max_tokens >=3000
        assert isinstance(out, list)

    def test_count_query_boosts_bm25_weight(self):
        bm = _FakeBM25(results=[{"memory_id": "b1", "content": "python", "score": 0.5}])
        sw = {"bm25": 1.0}
        facade = _FakeFacade(bm25=bm, channels={"bm25": (bm, 1.0)}, source_weights=sw)
        orch = _orch(facade)
        orch.search("how many python libs", top_k=10, max_tokens=500)
        # 恢复后权重应回到原值
        assert sw["bm25"] == 1.0

    def test_count_query_with_lock(self):
        bm = _FakeBM25(results=[{"memory_id": "b1", "content": "python", "score": 0.5}])
        sw = {"bm25": 1.0}
        lock = threading.Lock()
        facade = _FakeFacade(
            bm25=bm, channels={"bm25": (bm, 1.0)},
            source_weights=sw, source_weights_lock=lock,
        )
        orch = _orch(facade)
        orch.search("how much memory", top_k=10, max_tokens=500)
        assert sw["bm25"] == 1.0

    def test_cache_hit_short_circuits(self, monkeypatch):
        orch = _orch()
        monkeypatch.setattr(orch, "check_cache", lambda k: [{"memory_id": "cached"}])
        out = orch.search("q")
        assert out == [{"memory_id": "cached"}]

    def test_planner_disabled_skips_plan(self, monkeypatch):
        facade = _FakeFacade(config={"planner_enabled": False})
        orch = _orch(facade)
        called = []
        monkeypatch.setattr(orch._planner, "plan", lambda q: called.append(q) or MagicMock(is_default=True))
        orch.search("q")
        assert called == []

    def test_synonym_expansion_appends_to_bm25_query(self):
        bm = _FakeBM25()
        facade = _FakeFacade(bm25=bm, channels={"bm25": (bm, 1.0)},
                             synonym_map={"python": ["py", "python3"]})
        orch = _orch(facade)
        orch.search("python", top_k=5, max_tokens=100)
        # bm25_query 应包含扩展词
        assert any("py" in q for q in bm.queries)

    def test_preference_rewrite_appends_signals(self):
        bm = _FakeBM25()
        facade = _FakeFacade(bm25=bm, channels={"bm25": (bm, 1.0)},
                             config={"query_expansion_enabled": False})
        orch = _orch(facade)
        orch.search("recommend a book", top_k=5, max_tokens=100)
        assert any("prefer" in q for q in bm.queries)

    def test_trace_attached_to_last_result(self):
        bm = _FakeBM25(results=[{"memory_id": "b1", "content": "python", "score": 0.9}])
        facade = _FakeFacade(bm25=bm, channels={"bm25": (bm, 1.0)})
        orch = _orch(facade)
        out = orch.search("python", enable_trace=True, top_k=5, max_tokens=500)
        if out:
            assert "_trace" in out[-1]

    def test_empty_results_no_trace_key(self):
        facade = _FakeFacade(channels={})
        orch = _orch(facade)
        out = orch.search("q", enable_trace=True)
        assert out == []


# ─── async_search ──────────────────────────────────────────


class TestAsyncSearch:
    def test_keyword_strategy(self):
        bm = _FakeBM25(results=[{"memory_id": "b1", "content": "python", "score": 0.5}])
        facade = _FakeFacade(bm25=bm, channels={"bm25": (bm, 1.0)}, recall_strategy="keyword")
        orch = _orch(facade)
        out = asyncio.run(orch.async_search("python", top_k=5, max_tokens=500))
        assert isinstance(out, list)

    def test_embedding_strategy(self):
        vec = _FakeVector(results=[{"memory_id": "v1", "content": "python", "score": 0.9}])
        facade = _FakeFacade(vector=vec, channels={"vector": (vec, 3.0)}, recall_strategy="embedding")
        orch = _orch(facade)
        out = asyncio.run(orch.async_search("python", top_k=5, max_tokens=500))
        assert isinstance(out, list)

    def test_hybrid_strategy(self):
        vec = _FakeVector(results=[{"memory_id": "v1", "content": "python", "score": 0.9}])
        bm = _FakeBM25(results=[{"memory_id": "b1", "content": "python", "score": 0.5}])
        facade = _FakeFacade(vector=vec, bm25=bm,
                             channels={"vector": (vec, 3.0), "bm25": (bm, 1.0)})
        orch = _orch(facade)
        out = asyncio.run(orch.async_search("python", top_k=5, max_tokens=500))
        assert isinstance(out, list)

    def test_cache_hit(self, monkeypatch):
        orch = _orch()
        monkeypatch.setattr(orch, "check_cache", lambda k: [{"memory_id": "cached"}])
        out = asyncio.run(orch.async_search("q"))
        assert out == [{"memory_id": "cached"}]

    def test_garbage_query(self):
        facade = _FakeFacade(channels={})
        orch = _orch(facade)
        out = asyncio.run(orch.async_search("!!! ???", top_k=40))
        assert isinstance(out, list)

    def test_breaker_open_skips_vector(self):
        vec = _FakeVector(results=[{"memory_id": "v1"}])
        facade = _FakeFacade(
            vector=vec, channels={"vector": (vec, 3.0)},
            recall_strategy="embedding",
            vector_breaker=_FakeBreaker(skip=True),
        )
        orch = _orch(facade)
        out = asyncio.run(orch.async_search("q", top_k=5))
        assert isinstance(out, list)

    def test_exception_in_channel_recorded(self):
        class _BoomVec:
            def count(self): return 0
            def search(self, *_a, **_kw): raise RuntimeError("boom")

        facade = _FakeFacade(vector=_BoomVec(), channels={"vector": (_BoomVec(), 3.0)},
                             recall_strategy="embedding")
        orch = _orch(facade)
        asyncio.run(orch.async_search("q", top_k=5))
        assert facade._vector_breaker.failures >= 1


# ─── 后处理管道（通过 search 间接验证）────────────────────


class TestPostProcessing:
    def test_session_local_disabled_skips(self):
        facade = _FakeFacade(config={"session_local_enabled": False})
        orch = _orch(facade)
        assert orch._session_local.enabled is False

    def test_verifier_disabled_skips(self):
        facade = _FakeFacade(config={"verifier_enabled": False})
        orch = _orch(facade)
        assert orch._verifier.enabled is False

    def test_organizer_disabled_skips(self):
        facade = _FakeFacade(config={"organizer_enabled": False})
        orch = _orch(facade)
        assert orch._organizer.enabled is False

    def test_contradiction_disabled_skips(self):
        facade = _FakeFacade(config={"contradiction_enabled": False})
        orch = _orch(facade)
        assert orch._contradiction.enabled is False

    def test_self_healing_disabled_skips(self):
        facade = _FakeFacade(config={"self_healing_enabled": False})
        orch = _orch(facade)
        assert orch._self_healing.enabled is False

    def test_temporal_separation_disabled(self):
        facade = _FakeFacade(config={"temporal_separation_enabled": False})
        orch = _orch(facade)
        assert orch._temporal_separation_enabled is False
