"""sync_turn 生命周期回归测试（★ 修复 S1，2026-09-28）

被修缺陷：`HybridRetriever._sync_turn_ids` 只存在于内存，gateway 重启后归零 →
`cleanup_sync_turn_entries()` 的 `while len(...) > max_entries` 永不触发 →
`sync-*` 条目在向量/BM25 索引里只增不减（实测残留 101 条，抢占召回 top-k 坑位）。

本文件验证：队列持久化 / 跨重启恢复 / 启动对账清理遗留 / 淘汰在重启后依然生效
/ 不误删已登记条目。全部用 fake store，不加载 embedding 模型。
"""

import json
from collections import deque
from pathlib import Path

from omnimem.retrieval.engine import HybridRetriever
from omnimem.retrieval.index_admin import IndexAdminMixin


class FakeVector:
    def __init__(self):
        self.docs: dict[str, str] = {}

    def add(self, content, memory_id, metadata=None):
        self.docs[memory_id] = content

    def delete(self, memory_id):
        self.docs.pop(memory_id, None)


class FakeBM25:
    def __init__(self):
        self._documents: list[dict] = []

    def add(self, content, memory_id, metadata=None):
        self._documents.append({"memory_id": memory_id, "content": content})

    def delete(self, memory_id):
        self._documents = [d for d in self._documents if d.get("memory_id") != memory_id]


class StubAdmin(IndexAdminMixin):
    """只借 IndexAdminMixin 的 cleanup，不跑 HybridOrchestrator 的其它初始化。"""

    def __init__(self, facade):
        self._facade = facade

    def clear_all_cache(self):
        pass


def make_bare(tmp_path: Path, max_entries: int = 3) -> HybridRetriever:
    """裸实例：绕过 __init__ 的重组件，只装配本缺陷相关状态。"""
    hr = HybridRetriever.__new__(HybridRetriever)
    hr._sync_turn_ids = deque()
    hr._sync_turn_ids_path = tmp_path / "sync_turn_ids.json"
    hr._bm25 = FakeBM25()
    hr._vector = FakeVector()
    hr._max_sync_turn_entries = max_entries
    return hr


def test_persist_and_restore_roundtrip(tmp_path):
    hr = make_bare(tmp_path)
    hr._sync_turn_ids = deque(["sync-aaa", "sync-bbb"])
    hr.persist_sync_turn_ids()

    assert json.loads((tmp_path / "sync_turn_ids.json").read_text()) == ["sync-aaa", "sync-bbb"]

    hr2 = make_bare(tmp_path)
    hr2._restore_sync_turn_ids()
    assert list(hr2._sync_turn_ids) == ["sync-aaa", "sync-bbb"]


def test_restore_missing_file_is_noop(tmp_path):
    hr = make_bare(tmp_path)
    hr._restore_sync_turn_ids()          # 文件不存在 → 空队列，不抛
    assert list(hr._sync_turn_ids) == []


def test_reconcile_deletes_legacy_and_keeps_registered(tmp_path):
    """启动对账：队列不认识的 sync-* 删掉（两侧），已登记的保留，非 sync 条目不动。"""
    hr = make_bare(tmp_path)
    for mid in ("sync-old1", "sync-old2", "sync-keepme", "real-memory-1"):
        hr._bm25.add(mid, mid)
        hr._vector.add(mid, mid)
    hr._sync_turn_ids = deque(["sync-keepme"])

    hr._reconcile_legacy_sync_entries()

    expected = {"sync-keepme", "real-memory-1"}
    assert {d["memory_id"] for d in hr._bm25._documents} == expected
    assert set(hr._vector.docs) == expected


def test_cleanup_enforces_max_and_persists(tmp_path):
    hr = make_bare(tmp_path, max_entries=3)
    admin = StubAdmin(hr)
    for i in range(5):
        sid = f"sync-{i:04d}"
        hr._vector.add(f"c{i}", sid)
        hr._bm25.add(f"c{i}", sid)
        hr._sync_turn_ids.append(sid)
        admin.cleanup_sync_turn_entries()

    assert list(hr._sync_turn_ids) == ["sync-0002", "sync-0003", "sync-0004"]
    assert {d["memory_id"] for d in hr._bm25._documents} == {"sync-0002", "sync-0003", "sync-0004"}
    assert set(hr._vector.docs) == {"sync-0002", "sync-0003", "sync-0004"}
    assert json.loads((tmp_path / "sync_turn_ids.json").read_text()) == [
        "sync-0002", "sync-0003", "sync-0004",
    ]


def test_cleanup_below_max_still_persists(tmp_path):
    """没触发淘汰也要落盘 —— 否则重启后这批条目就变成了"无人认领的遗留"。"""
    hr = make_bare(tmp_path, max_entries=10)
    admin = StubAdmin(hr)
    hr._sync_turn_ids.append("sync-one")
    hr._vector.add("c", "sync-one")
    hr._bm25.add("c", "sync-one")
    admin.cleanup_sync_turn_entries()

    assert json.loads((tmp_path / "sync_turn_ids.json").read_text()) == ["sync-one"]


def test_restart_keeps_cleanup_working(tmp_path):
    """核心回归：同一 data_dir「重启」后淘汰能力不再失效（修复前这里会退化成 6 条）。"""
    hr = make_bare(tmp_path, max_entries=3)
    admin = StubAdmin(hr)
    for i in range(5):
        sid = f"sync-{i:04d}"
        hr._vector.add(f"c{i}", sid)
        hr._bm25.add(f"c{i}", sid)
        hr._sync_turn_ids.append(sid)
        admin.cleanup_sync_turn_entries()
    assert len(hr._sync_turn_ids) == 3

    # 模拟 gateway 重启：索引已持久化到磁盘，新实例从磁盘恢复队列
    hr2 = make_bare(tmp_path, max_entries=3)
    hr2._bm25._documents = list(hr._bm25._documents)
    hr2._vector.docs = dict(hr._vector.docs)
    hr2._restore_sync_turn_ids()
    hr2._reconcile_legacy_sync_entries()      # 修复前：无此步，遗留无人清

    admin2 = StubAdmin(hr2)
    for i in range(5, 8):
        sid = f"sync-{i:04d}"
        hr2._vector.add(f"c{i}", sid)
        hr2._bm25.add(f"c{i}", sid)
        hr2._sync_turn_ids.append(sid)
        admin2.cleanup_sync_turn_entries()

    assert len(hr2._sync_turn_ids) == 3
    assert len({d["memory_id"] for d in hr2._bm25._documents}) == 3
    assert len(hr2._vector.docs) == 3


def test_control_group_reproduces_old_leak(tmp_path):
    """对照（复现缺陷，非期望行为）：不做 restore/reconcile 时 → 重启即泄漏。

    这正是线上发生的事：重启后队列归零，`while len(ids) > max_entries` 判假，
    淘汰不再触发，索引条目从 3 涨到 6（线上从 47 涨到 101）。
    """
    hr = make_bare(tmp_path, max_entries=3)
    admin = StubAdmin(hr)
    for i in range(3):
        sid = f"sync-{i:04d}"
        hr._vector.add(f"c{i}", sid)
        hr._bm25.add(f"c{i}", sid)
        hr._sync_turn_ids.append(sid)
        admin.cleanup_sync_turn_entries()
    assert len(hr._bm25._documents) == 3

    # "修复前"的重启：不调用 _restore_sync_turn_ids() / _reconcile_legacy_sync_entries()
    hr2 = make_bare(tmp_path, max_entries=3)
    hr2._bm25._documents = list(hr._bm25._documents)
    hr2._vector.docs = dict(hr._vector.docs)
    admin2 = StubAdmin(hr2)
    for i in range(3, 6):
        sid = f"sync-{i:04d}"
        hr2._vector.add(f"c{i}", sid)
        hr2._bm25.add(f"c{i}", sid)
        hr2._sync_turn_ids.append(sid)
        admin2.cleanup_sync_turn_entries()

    assert len(hr2._bm25._documents) == 6        # ← 泄漏未被阻止（缺陷复现）
