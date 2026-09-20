"""retrieval.index_admin.IndexAdminMixin 离线单元测试。

以最小 Facade 替身挂载 mixin，覆盖：
  - enrich_for_rebuild：5 类前缀 / fact 无前缀
  - rebuild_bm25_from_entries：过滤、enrich、返回 added+updated 计数
  - rebuild_all_from_entries：config 读取、向量+BM25 双路径、失败降级
  - cleanup_sync_turn_entries：deque 弹出、bm25/vector 异常吞掉
  - index_update：preference 头清理、uuid 分配、cache 清空、追加 & 修剪
"""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest
from omnimem.retrieval.index_admin import IndexAdminMixin


class _FakeVector:
    def __init__(self, *, raise_reset: bool = False) -> None:
        self.reset_calls = 0
        self.deleted: list[str] = []
        self.added: list[dict[str, Any]] = []
        self.flush_calls = 0
        self.rebuild_parallel_kwargs: dict[str, Any] | None = None
        self._raise_reset = raise_reset

    def reset(self) -> None:
        if self._raise_reset:
            raise RuntimeError("reset boom")
        self.reset_calls += 1

    def delete(self, mid: str) -> None:
        self.deleted.append(mid)

    def add(self, content: str, memory_id: str = "", metadata: dict | None = None) -> None:
        self.added.append({"content": content, "memory_id": memory_id, "metadata": metadata})

    def flush(self) -> None:
        self.flush_calls += 1

    def rebuild_vectors_parallel(self, entries, *, batch_size: int, max_workers: int) -> int:
        self.rebuild_parallel_kwargs = {
            "n": len(entries),
            "batch_size": batch_size,
            "max_workers": max_workers,
        }
        return len(entries)


class _FakeBM25:
    def __init__(self) -> None:
        self.updated_entries: list[dict[str, Any]] | None = None
        self.added: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    def update_from_entries(self, entries):
        self.updated_entries = list(entries)
        return {"added": len(entries), "updated": 0, "deleted": 0}

    def add(self, content: str, memory_id: str = "", metadata: dict | None = None) -> None:
        self.added.append({"content": content, "memory_id": memory_id, "metadata": metadata})

    def delete(self, mid: str) -> None:
        self.deleted.append(mid)


class _FakeFacade:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._vector = _FakeVector()
        self._bm25 = _FakeBM25()
        self._config = config
        self._max_sync_turn_entries = 3
        self._sync_turn_ids: deque[str] = deque()


class _Engine(IndexAdminMixin):
    def __init__(self, facade: _FakeFacade) -> None:
        self._facade = facade
        self.clear_all_cache_calls = 0

    def clear_all_cache(self) -> None:
        self.clear_all_cache_calls += 1


@pytest.fixture
def facade() -> _FakeFacade:
    return _FakeFacade()


@pytest.fixture
def engine(facade: _FakeFacade) -> _Engine:
    return _Engine(facade)


# ── enrich_for_rebuild ──


def test_enrich_adds_secret_prefix(engine: _Engine) -> None:
    assert engine.enrich_for_rebuild("我的密码", "secret", "工作") == (
        "[加密信息/密钥/凭证] 工作 我的密码"
    )


def test_enrich_adds_skill_prefix(engine: _Engine) -> None:
    assert engine.enrich_for_rebuild("s", "skill").startswith("[技能/步骤/教程]")


def test_enrich_adds_procedural_prefix(engine: _Engine) -> None:
    assert engine.enrich_for_rebuild("s", "procedural").startswith("[流程/操作/指南]")


def test_enrich_adds_reasoning_prefix(engine: _Engine) -> None:
    assert engine.enrich_for_rebuild("s", "reasoning").startswith("[教训/经验/踩坑]")


def test_enrich_adds_action_prefix(engine: _Engine) -> None:
    assert engine.enrich_for_rebuild("s", "action").startswith("[Agent行为/工具调用]")


def test_enrich_no_prefix_for_fact(engine: _Engine) -> None:
    assert engine.enrich_for_rebuild("content", "fact") == "content"


def test_enrich_no_prefix_for_unknown_type(engine: _Engine) -> None:
    assert engine.enrich_for_rebuild("content", "xyz") == "content"


# ── rebuild_bm25_from_entries ──


def test_rebuild_bm25_filters_missing_fields(engine: _Engine) -> None:
    entries = [
        {"memory_id": "m1"},  # 缺 content/summary → skip
        {"content": "c"},  # 缺 memory_id → skip
        {"memory_id": "m2", "content": "c2"},
        {"memory_id": "m3", "summary": "s3"},  # summary 兜底
    ]
    total = engine.rebuild_bm25_from_entries(entries)
    assert total == 2
    passed = engine._facade._bm25.updated_entries
    assert passed is not None
    mids = [e["memory_id"] for e in passed]
    assert mids == ["m2", "m3"]


def test_rebuild_bm25_enriches_type_prefix(engine: _Engine) -> None:
    entries = [{"memory_id": "m1", "content": "我的密码", "type": "secret", "room": "工作"}]
    engine.rebuild_bm25_from_entries(entries)
    enriched = engine._facade._bm25.updated_entries[0]
    assert enriched["content"].startswith("[加密信息/密钥/凭证] 工作 我的密码")


def test_rebuild_bm25_defaults_type_to_fact(engine: _Engine) -> None:
    engine.rebuild_bm25_from_entries([{"memory_id": "m", "content": "abc"}])
    enriched = engine._facade._bm25.updated_entries[0]
    # fact 无 enrich → 原样
    assert enriched["content"] == "abc"


def test_rebuild_bm25_empty_returns_zero(engine: _Engine) -> None:
    assert engine.rebuild_bm25_from_entries([]) == 0


# ── rebuild_all_from_entries ──


def test_rebuild_all_clears_and_uses_default_batching(engine: _Engine) -> None:
    entries = [{"memory_id": "m1", "content": "c"}]
    result = engine.rebuild_all_from_entries(entries)
    assert engine.clear_all_cache_calls == 1
    assert result == {"vector": 1, "bm25": 1}
    kwargs = engine._facade._vector.rebuild_parallel_kwargs
    assert kwargs == {"n": 1, "batch_size": 32, "max_workers": 4}
    assert engine._facade._vector.flush_calls == 1


def test_rebuild_all_honors_config(engine: _Engine) -> None:
    engine._facade._config = {"rebuild_batch_size": 8, "rebuild_max_workers": 2}
    engine.rebuild_all_from_entries([])
    kwargs = engine._facade._vector.rebuild_parallel_kwargs
    assert kwargs["batch_size"] == 8
    assert kwargs["max_workers"] == 2


def test_rebuild_all_floors_batch_and_workers_to_one(engine: _Engine) -> None:
    engine._facade._config = {"rebuild_batch_size": 0, "rebuild_max_workers": -5}
    engine.rebuild_all_from_entries([])
    kwargs = engine._facade._vector.rebuild_parallel_kwargs
    assert kwargs["batch_size"] == 1
    assert kwargs["max_workers"] == 1


def test_rebuild_all_survives_vector_reset_failure(facade: _FakeFacade) -> None:
    facade._vector = _FakeVector(raise_reset=True)
    eng = _Engine(facade)
    out = eng.rebuild_all_from_entries([{"memory_id": "m", "content": "c"}])
    assert out["vector"] == 1  # rebuild_vectors_parallel 仍返回长度
    assert out["bm25"] == 1


def test_rebuild_all_bm25_skips_missing_fields(engine: _Engine) -> None:
    engine.rebuild_all_from_entries(
        [
            {"memory_id": "m1", "content": "c1"},
            {"memory_id": "", "content": "x"},  # skip
            {"memory_id": "m2", "content": ""},  # skip
        ]
    )
    # bm25.update_from_entries 只看到 m1 → added=1
    assert engine._facade._bm25.updated_entries == [
        {"memory_id": "m1", "content": "c1"}
    ]


# ── cleanup_sync_turn_entries ──


def test_cleanup_prunes_to_max(engine: _Engine) -> None:
    facade = engine._facade
    facade._sync_turn_ids.extend(["a", "b", "c", "d", "e"])
    engine.cleanup_sync_turn_entries()
    assert list(facade._sync_turn_ids) == ["c", "d", "e"]
    assert facade._bm25.deleted == ["a", "b"]
    assert facade._vector.deleted == ["a", "b"]


def test_cleanup_noop_under_limit(engine: _Engine) -> None:
    engine._facade._sync_turn_ids.append("x")
    engine.cleanup_sync_turn_entries()
    assert engine._facade._bm25.deleted == []
    assert engine._facade._vector.deleted == []


def test_cleanup_swallows_bm25_delete_exception(engine: _Engine) -> None:
    engine._facade._sync_turn_ids.extend(["a", "b", "c", "d"])

    def boom(_mid: str) -> None:
        raise RuntimeError("nope")

    engine._facade._bm25.delete = boom  # type: ignore[method-assign]
    engine.cleanup_sync_turn_entries()  # 不抛
    assert list(engine._facade._sync_turn_ids) == ["b", "c", "d"]


# ── index_update ──


def test_index_update_skips_too_short_content(engine: _Engine) -> None:
    engine.index_update("abc", "assistant")
    assert engine._facade._bm25.added == []


def test_index_update_strips_prefetched_header(engine: _Engine) -> None:
    raw = (
        "用户原始问题\n\n"
        "### Relevant Memories\n"
        "- [cached] abc\n"
        "- 一条记忆\n"
    )
    engine.index_update(raw, "assistant")
    assert len(engine._facade._bm25.added) == 1
    assert len(engine._facade._vector.added) == 1
    added = engine._facade._bm25.added[0]
    assert added["content"].startswith("用户原始问题")
    assert "Relevant Memories" not in added["content"]
    assert "[cached]" not in added["content"]
    assert added["metadata"] == {"source": "sync_turn"}


def test_index_update_assigns_sync_id_and_appends_deque(engine: _Engine) -> None:
    engine.index_update("这是一段够长的用户消息内容", "assistant")
    assert len(engine._facade._sync_turn_ids) == 1
    assert engine._facade._sync_turn_ids[0].startswith("sync-")


def test_index_update_clears_cache(engine: _Engine) -> None:
    engine.index_update("这是一段够长的用户消息内容", "assistant")
    assert engine.clear_all_cache_calls == 1


def test_index_update_triggers_cleanup(engine: _Engine) -> None:
    engine._facade._max_sync_turn_entries = 2
    for _ in range(3):
        engine.index_update("这是一段够长的用户消息内容", "assistant")
    # 第 3 次触发 cleanup → deque 长度不超过 2
    assert len(engine._facade._sync_turn_ids) <= 2


def test_index_update_truncates_to_200_chars(engine: _Engine) -> None:
    long = "这是一段中文" * 100
    engine.index_update(long, "assistant")
    content = engine._facade._bm25.added[0]["content"]
    assert len(content) <= 200
