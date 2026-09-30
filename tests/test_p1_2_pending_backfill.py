"""P1-2 回归：待回填向量队列有消费者，重建门控按真实缺口触发。

线上实况（2026-09-29 报告）：``vector_index_pending.jsonl`` 积压 6 001 条**没有任何读者**，
而 warmup 的重建触发条件是 ``vec_count == 0``（生产 vec_count=770），所以永远不会重建。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omnimem.core.warmup_manager import WarmupManager
from omnimem.retrieval.vector import VectorRetriever
from omnimem.retrieval.vector_store import read_vector_pending, vector_pending_path


class FakeStore:
    """只实现队列路径 + 记录写入的假向量后端。"""

    def __init__(self, persist_dir: Path, fail_add: bool = False) -> None:
        self._persist_dir = persist_dir
        self.added: list[dict[str, Any]] = []
        self.fail_add = fail_add

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict] | None = None) -> None:
        if self.fail_add:
            raise RuntimeError("simulated chroma failure")
        self.added.extend({"id": i, "doc": d} for i, d in zip(ids, documents, strict=False))

    def pending_path(self) -> Path:
        return vector_pending_path(self._persist_dir)

    def flush(self) -> None:
        return None


def _retriever(store: FakeStore) -> VectorRetriever:
    retriever = object.__new__(VectorRetriever)
    retriever._store = store
    retriever._initialized = True
    retriever._is_new_store = False
    retriever._embedding_provider = None
    retriever._embedding_fn = None
    retriever._chunk_ids = {}
    retriever._data_dir = store._persist_dir.parent
    return retriever


def _write_pending(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record(doc_id: str, text: str = "内容") -> dict[str, Any]:
    return {"ts": 1.0, "id": doc_id, "document": f"{text}{doc_id}", "metadata": {"room": "r"}, "reason": "embed_or_write_failed"}


class TestPendingQueueReader:
    def test_path_is_shared_by_writer_and_reader(self, tmp_path):
        store = FakeStore(tmp_path / "chroma")
        assert store.pending_path() == vector_pending_path(tmp_path / "chroma")
        assert _retriever(store)._pending_path() == store.pending_path()

    def test_read_dedupes_by_id_and_skips_corrupt_lines(self, tmp_path):
        queue = vector_pending_path(tmp_path / "chroma")
        _write_pending(queue, [_record("a", "旧"), _record("a", "新"), {"broken": True}])
        queue.write_text(queue.read_text(encoding="utf-8") + "not-json\n")

        records = read_vector_pending(queue)
        assert [r["id"] for r in records] == ["a"]
        assert records[0]["document"] == "新a"

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert read_vector_pending(vector_pending_path(tmp_path / "nope")) == []


class TestDrainVectorPending:
    def test_drain_replays_and_clears_queue(self, tmp_path):
        store = FakeStore(tmp_path / "chroma")
        retriever = _retriever(store)
        queue = store.pending_path()
        _write_pending(queue, [_record("a"), _record("b")])

        stats = retriever.drain_vector_pending()
        assert stats["pending"] == 2
        assert stats["replayed"] == 2
        assert {added["id"] for added in store.added} == {"a", "b"}
        assert not queue.exists(), "排空后队列不应残留"
        assert not list(tmp_path.glob("*.processing"))

    def test_remainder_beyond_limit_is_queued_back(self, tmp_path):
        store = FakeStore(tmp_path / "chroma")
        retriever = _retriever(store)
        _write_pending(store.pending_path(), [_record(f"id{i}") for i in range(5)])

        stats = retriever.drain_vector_pending(limit=2)
        assert stats["replayed"] == 2
        assert stats["deferred"] == 3
        remaining = read_vector_pending(store.pending_path())
        assert {r["id"] for r in remaining} == {"id2", "id3", "id4"}

    def test_failure_before_reaching_store_is_not_lost(self, tmp_path, monkeypatch):
        """add_batch 在抵达 store 之前抛错时，条目必须退回队列而不是蒸发。"""
        store = FakeStore(tmp_path / "chroma")
        retriever = _retriever(store)
        _write_pending(store.pending_path(), [_record("a")])
        monkeypatch.setattr(retriever, "add_batch", lambda docs: (_ for _ in ()).throw(RuntimeError("boom")))

        stats = retriever.drain_vector_pending()
        assert stats["failed"] == 1
        assert [r["id"] for r in read_vector_pending(store.pending_path())] == ["a"]

    def test_count_pending(self, tmp_path):
        store = FakeStore(tmp_path / "chroma")
        retriever = _retriever(store)
        assert retriever.count_vector_pending() == 0
        _write_pending(store.pending_path(), [_record("a"), _record("b")])
        assert retriever.count_vector_pending() == 2


class TestColdStoreConsumer:
    """★ 消费者必须在「还没人检索过」时也能工作。

    provider 刚起、还没有任何一次向量访问时 ``_store`` 仍是 None。原先
    ``_pending_path()`` 直接返回 None ⇒ count 报 0、drain 空转，warmup 每轮都"看到零积压"，
    线上 6 001 条待回填因此永远排不掉。
    """

    def _cold_retriever(self, store: FakeStore, monkeypatch) -> VectorRetriever:
        retriever = _retriever(store)
        retriever._store = None
        retriever._initialized = False

        def _fake_init():
            retriever._store = store
            retriever._initialized = True

        monkeypatch.setattr(retriever, "_ensure_initialized", _fake_init)
        return retriever

    def test_count_initializes_store(self, tmp_path, monkeypatch):
        store = FakeStore(tmp_path / "chroma")
        _write_pending(store.pending_path(), [_record("a")])
        retriever = self._cold_retriever(store, monkeypatch)

        assert retriever._store is None, "前置：store 尚未懒建"
        assert retriever.count_vector_pending() == 1
        assert retriever._store is store, "count 必须把 store 拉起来，而不是静默返回 0"

    def test_drain_initializes_store(self, tmp_path, monkeypatch):
        store = FakeStore(tmp_path / "chroma")
        _write_pending(store.pending_path(), [_record("a"), _record("b")])
        retriever = self._cold_retriever(store, monkeypatch)

        stats = retriever.drain_vector_pending()
        assert stats["replayed"] == 2
        assert not store.pending_path().exists()


class TestStrandedProcessing:
    """★ 排空被打断后，``.processing`` 残骸必须被回收而不是被下一轮覆盖掉。"""

    def test_orphan_processing_is_recovered(self, tmp_path):
        store = FakeStore(tmp_path / "chroma")
        retriever = _retriever(store)
        queue = store.pending_path()
        stranded = retriever._processing_path(queue)
        _write_pending(stranded, [_record("a"), _record("b")])

        # 排空过程中断（daemon 线程被杀 / 进程退出）时正牌文件不存在
        assert not queue.exists()
        assert retriever.count_vector_pending() == 2, "积压不能因改名而对企业健康检查隐身"

        stats = retriever.drain_vector_pending()
        assert stats["pending"] == 2
        assert stats["replayed"] == 2
        assert {added["id"] for added in store.added} == {"a", "b"}
        assert not stranded.exists(), "回收后不应留下 .processing"

    def test_processing_merges_into_existing_queue(self, tmp_path):
        store = FakeStore(tmp_path / "chroma")
        retriever = _retriever(store)
        queue = store.pending_path()
        _write_pending(queue, [_record("new")])
        _write_pending(retriever._processing_path(queue), [_record("old")])

        stats = retriever.drain_vector_pending()
        assert stats["pending"] == 2
        assert {added["id"] for added in store.added} == {"old", "new"}


class _FakeRetriever:
    def __init__(self, vec_count: int, pending: int, rebuild_result: dict[str, int] | None = None) -> None:
        self._vec_count = vec_count
        self._pending = pending
        self.drained = False
        self.rebuilt = False
        self._rebuild_result = rebuild_result or {"vector": vec_count}

    def _check_vector_health(self) -> dict[str, Any]:
        return {"vector_count": self._vec_count, "breaker_state": "closed"}

    def vector_count(self) -> int:
        return self._rebuild_result["vector"] if self.rebuilt else self._vec_count

    def count_vector_pending(self) -> int:
        return 0 if self.drained else self._pending

    def drain_vector_pending(self, limit: int = 2000) -> dict[str, int]:
        self.drained = True
        return {"pending": self._pending, "replayed": self._pending, "deferred": 0, "failed": 0}

    def rebuild_all_from_entries(self, entries: list[dict[str, Any]]) -> dict[str, int]:
        self.rebuilt = True
        return self._rebuild_result


class _FakeIndex:
    def __init__(self, count: int) -> None:
        self._count = count

    def search_l1(self, limit: int = 5000) -> list[dict[str, Any]]:
        return [{"memory_id": f"m{i}", "content": f"c{i}"} for i in range(min(limit, self._count))]


def _manager(retriever: _FakeRetriever, index: _FakeIndex) -> WarmupManager:
    manager = object.__new__(WarmupManager)
    manager._index = index
    manager._retriever = retriever
    manager._retrieval = type("R", (), {"warmup": lambda self: None})()
    return manager


class TestBackfillGate:
    def test_pending_queue_is_drained_even_when_vectors_are_nonzero(self):
        """生产场景：vec_count=770（≠0）但队列有 6 001 条 —— 原实现什么都不做。"""
        retriever = _FakeRetriever(vec_count=770, pending=6001)
        _manager(retriever, _FakeIndex(712))._backfill_vectors()
        assert retriever.drained, "死信队列没有被消费"

    def test_rebuild_triggers_on_coverage_gap_not_only_on_zero(self):
        retriever = _FakeRetriever(vec_count=100, pending=0, rebuild_result={"vector": 5000})
        _manager(retriever, _FakeIndex(5000))._backfill_vectors()
        assert retriever.rebuilt

    def test_no_rebuild_when_coverage_is_healthy(self):
        """分块会让向量数合法地大于条目数，不能因此反复全量重建（重建会先清空向量库）。"""
        retriever = _FakeRetriever(vec_count=770, pending=0)
        _manager(retriever, _FakeIndex(712))._backfill_vectors()
        assert not retriever.rebuilt

    def test_small_drift_within_tolerance_does_not_rebuild(self):
        retriever = _FakeRetriever(vec_count=700, pending=0)
        _manager(retriever, _FakeIndex(712))._backfill_vectors()
        assert not retriever.rebuilt

    def test_unavailable_vector_channel_skips_backfill(self):
        retriever = _FakeRetriever(vec_count=-1, pending=10)
        _manager(retriever, _FakeIndex(100))._backfill_vectors()
        assert not retriever.drained
        assert not retriever.rebuilt

    def test_backfill_failure_is_not_fatal(self):
        retriever = _FakeRetriever(vec_count=0, pending=0)
        retriever.drain_vector_pending = lambda limit=2000: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]
        manager = _manager(retriever, _FakeIndex(10))
        manager._warmup_retrieval()  # 不抛

    def test_reader_handles_real_retriever_signature(self):
        """drain 的返回键必须与 warmup 读取的一致（防止改名后静默失配）。"""
        retriever = _FakeRetriever(vec_count=1, pending=1)
        stats = retriever.drain_vector_pending()
        for key in ("replayed", "deferred", "failed"):
            assert key in stats


@pytest.mark.parametrize("bad", ["", "   ", "{}\n", "null\n"])
def test_corrupt_and_empty_lines_ignored(tmp_path, bad):
    queue = vector_pending_path(tmp_path / "chroma")
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(bad, encoding="utf-8")
    assert read_vector_pending(queue) == []
