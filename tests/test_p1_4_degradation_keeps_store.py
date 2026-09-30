"""P1-4 回归：降级路径不再丢弃主存储数据。

线上实况（2026-09-29 报告）：
  1. ``_compensate_store`` 在下游步骤（index/retriever）抖动时 ``store.delete(memory_id)``，
     把已经写成功的主抽屉整条删掉 —— 用户记忆凭空消失。
  2. ``drawer_closet`` 的目录在 ``add()`` 入队时创建，真正写盘发生在 ``flush()``，
     中间目录被清掉就 ``[Errno 2]``；而 ``_flush_write_buffer`` 只
     ``logger.warning("Buffered write failed")``，``flush()`` 照旧报告成功。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from omnimem.memory.drawer_closet import DrawerClosetStore
from omnimem.retrieval.vector import VectorRetriever
from omnimem.retrieval.vector_store import read_vector_pending, vector_pending_path


def _store(tmp_path: Path) -> DrawerClosetStore:
    return DrawerClosetStore(tmp_path / "palace", write_buffer_threshold=1000)


class TestDrawerWriteSurvivesMissingDir:
    def test_write_drawer_recreates_missing_parent(self, tmp_path):
        """直接给一个父目录不存在的目标路径：必须先 mkdir 再写。"""
        store = _store(tmp_path)
        target = tmp_path / "palace" / "personal" / "gone" / "drawer" / "abc123.md"
        store._write_drawer(
            target, "内容", "fact", 3, "personal", None, datetime.now(),
        )
        assert target.exists()
        store.close()

    def test_flush_reports_failed_writes_instead_of_claiming_success(self, tmp_path, monkeypatch):
        """写入真失败时 flush() 必须点名返回，不能再让调用方以为成功。"""
        store = _store(tmp_path)
        memory_id = store.add(
            wing="personal", room="people", content="会丢的内容", memory_type="fact"
        )

        def boom(*args: Any, **kwargs: Any) -> None:
            raise OSError(2, "No such file or directory")

        monkeypatch.setattr(store, "_write_drawer", boom)
        failed = store.flush()
        assert memory_id in failed, "写入失败却被 flush() 报告成成功"
        store.close()

    def test_add_then_room_removed_still_flushes_to_disk(self, tmp_path):
        """线上场景：add() 只建目录并入队，随后目录被清 → flush() 仍要落盘。"""
        import shutil

        store = _store(tmp_path)
        memory_id = store.add(
            wing="personal", room="people", content="Alice knows Bob", memory_type="fact"
        )
        for wing_dir in (tmp_path / "palace" / "personal").iterdir():
            if wing_dir.is_dir() and wing_dir.name != ".meta":
                shutil.rmtree(wing_dir)

        failed = store.flush()

        assert failed == [], f"flush 报告了失败: {failed}"
        drawer_files = [p for p in (tmp_path / "palace").rglob(f"{memory_id}.md") if p.parent.name == "drawer"]
        assert drawer_files, "抽屉文件没有在被删掉的目录下重建"
        assert "Alice knows Bob" in drawer_files[0].read_text(encoding="utf-8")
        store.close()

    def test_flushed_content_is_readable_back(self, tmp_path):
        store = _store(tmp_path)
        memory_id = store.add(
            wing="personal", room="people", content="可召回", memory_type="fact"
        )
        store.flush()
        entry = store.get(memory_id)
        assert entry is not None
        assert entry["content"] == "可召回"
        store.close()


class _FakeStore:
    def __init__(self, persist_dir: Path) -> None:
        self._persist_dir = persist_dir
        self.added: list[list[str]] = []

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict] | None = None) -> None:
        self.added.append(list(ids))

    def flush(self) -> None:
        return None


def _retriever(store: Any, data_dir: Path | None = None) -> VectorRetriever:
    retriever = object.__new__(VectorRetriever)
    retriever._store = store
    retriever._initialized = True
    retriever._is_new_store = False
    retriever._embedding_provider = None
    retriever._embedding_fn = None
    retriever._chunk_ids = {}
    persist_dir = data_dir or getattr(store, "_persist_dir", None) or Path("/tmp/omnimem")
    retriever._data_dir = Path(persist_dir).parent
    return retriever


class TestBackfillGapQueueing:
    def test_queue_vector_backfill_writes_a_drainable_record(self, tmp_path):
        """saga 投递的缺口必须能被 P1-2 的排空消费者读回并重放。"""
        store = _FakeStore(tmp_path / "chroma")
        retriever = _retriever(store)

        ok = retriever.queue_vector_backfill(
            "Alice knows Bob", memory_id="m1", metadata={"room": "people"}, reason="saga_compensate"
        )
        assert ok is True

        records = read_vector_pending(vector_pending_path(tmp_path / "chroma"))
        assert len(records) == 1
        assert records[0]["id"] == "m1"
        assert records[0]["reason"] == "saga_compensate"

        stats = retriever.drain_vector_pending()
        assert stats["replayed"] == 1
        assert store.added == [["m1"]]

    def test_empty_inputs_are_rejected(self, tmp_path):
        retriever = _retriever(_FakeStore(tmp_path / "chroma"))
        assert retriever.queue_vector_backfill("", memory_id="m1") is False
        assert retriever.queue_vector_backfill("内容", memory_id="") is False

    def test_backend_without_pending_path_is_not_fatal(self, tmp_path):
        class _NoQueueStore:
            def add(self, *args, **kwargs) -> None:
                return None

        retriever = _retriever(_NoQueueStore(), data_dir=tmp_path / "chroma")
        assert retriever.queue_vector_backfill("内容", memory_id="m1") is False

    def test_recorded_gap_is_plain_jsonl(self, tmp_path):
        retriever = _retriever(_FakeStore(tmp_path / "chroma"))
        retriever.queue_vector_backfill("内容", memory_id="m2", metadata={"type": "fact"})
        line = vector_pending_path(tmp_path / "chroma").read_text(encoding="utf-8").strip()
        record = json.loads(line)
        assert record["document"] == "内容"
        assert record["metadata"] == {"type": "fact"}
        assert isinstance(record["ts"], float)


@pytest.mark.parametrize("reason", ["saga_compensate", "saga_gap", "collection_unavailable"])
def test_reason_is_preserved_end_to_end(tmp_path, reason):
    store = _FakeStore(tmp_path / "chroma")
    retriever = _retriever(store)
    retriever.queue_vector_backfill("内容", memory_id="m9", reason=reason)
    records = read_vector_pending(vector_pending_path(tmp_path / "chroma"))
    assert [r["reason"] for r in records] == [reason]
