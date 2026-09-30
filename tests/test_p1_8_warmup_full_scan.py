"""P1-8 回归：``WarmupManager._warmup_data`` 必须以全量活跃条目驱动 BM25 差集更新。

``BM25Retriever.update_from_entries`` 的语义是"传进来的就是全世界"：
``ids_to_delete = current_ids - new_entries``。原实现喂的是 ``search_l1(limit=2000)``，
副本实测 index 有 3 135 行 ⇒ 每次开机只给最新 2 000 条，其余被当作已删除从 BM25 语料
里剔掉并落盘（日志 ``BM25 rebuild (incremental): added=0, updated=0, deleted=1210``）。
老记忆的关键词通道就这样被一轮一轮削掉 —— 和 P1-1 同一类"召不回"。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omnimem.core.warmup_manager import WarmupManager, _FULL_SCAN_LIMIT


class _FakeIndex:
    def __init__(self, n_active: int = 2500) -> None:
        self.n_active = n_active
        self.requested_limits: list[int] = []
        self.db_path = Path("index.db")

    def search_l1(self, wing: str = "", type: str = "", limit: int = 50) -> list[dict]:
        self.requested_limits.append(limit)
        return [
            {"memory_id": f"m{i:012d}", "content": f"内容 {i}", "type": "fact", "room": "r"}
            for i in range(min(self.n_active, limit))
        ]


class _FakeStore:
    def __init__(self) -> None:
        self.warmed: int = 0
        self.synced_with: list[Path] = []
        self.meta_store = SimpleNamespace(
            sync_from_index=lambda p: self.synced_with.append(p) or (0, 0)
        )

    def warm_up(self, entries) -> int:
        self.warmed = len(entries)
        return self.warmed


class _FakeRetriever:
    def __init__(self) -> None:
        self.bm25_sources: list[int] = []

    def rebuild_bm25_from_entries(self, entries) -> int:
        self.bm25_sources.append(len(entries))
        return len(entries)


def _manager(n_active: int = 2500) -> tuple[WarmupManager, _FakeIndex, _FakeStore, _FakeRetriever]:
    index, store, retriever = _FakeIndex(n_active), _FakeStore(), _FakeRetriever()
    manager = WarmupManager(
        init_reflect_fn=lambda: None,
        init_lora_fn=lambda: None,
        index=index,
        store=store,
        retriever=retriever,
        retrieval=SimpleNamespace(warmup=lambda: None),
        auditor=SimpleNamespace(run_startup_audit=lambda: None),
    )
    return manager, index, store, retriever


def test_bm25_is_driven_by_every_active_entry_not_a_page():
    """★ 核心：2 500 条活跃索引就要给 BM25 2 500 条，一条都不能被 LIMIT 截掉。"""
    manager, index, _store, retriever = _manager(n_active=2500)
    manager._warmup_data()

    assert index.requested_limits == [_FULL_SCAN_LIMIT], "仍在用分页 LIMIT 驱动差集更新"
    assert retriever.bm25_sources == [2500]


def test_scan_limit_is_not_a_page_size():
    """扫描上限必须是全表口径，不能落在任何现实分页尺度上。"""
    assert _FULL_SCAN_LIMIT >= 1_000_000


def test_store_cache_warming_stays_bounded():
    """L1 缓存预热仍然是有界采样（这不是差集更新，截断无害且必须省内存）。"""
    manager, _index, store, _retriever = _manager(n_active=2500)
    manager._warmup_data()

    assert store.warmed == 500


def test_index_sync_still_runs_after_full_scan():
    """P1-7 的同步入口不能被这次改动绕过。"""
    manager, index, store, _retriever = _manager(n_active=10)
    manager._warmup_data()

    assert store.synced_with == [index.db_path]
