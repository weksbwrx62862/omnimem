"""P1-7 回归：``MetaStore.sync_from_index`` 不得把 MetaStore 当 index.db 的删除依据。

线上/副本实测（2026-09-30）：副本磁盘 3 135 个抽屉、``doctor reconcile --apply`` 把
index.db 从 877 行补到 3 308 行；下一次开机 warmup 跑 ``sync_from_index``，
``stale = idx_ids - meta_ids`` 直接把其中 2 431 行（有抽屉、只是 MetaStore 双写缺行）
删回 877，同时把 173 条「meta 有、抽屉已不在磁盘」的幽灵重新写回 index.db。
也就是说：**开机即回滚 P1-1 的对账结果**，而且没有任何 WARNING 可见。

删除判据必须是磁盘。MetaStore 只是并行双写的一方，降级时自己就缺行，不能当基准。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from omnimem.memory.index import ThreeLevelIndex
from omnimem.memory.meta_store import MetaStore


def _index_ids(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {r[0] for r in conn.execute("SELECT memory_id FROM memory_index")}
    finally:
        conn.close()


def _make_meta(tmp_path: Path, memory_ids: list[str], palace_dir: Path | None) -> MetaStore:
    meta = MetaStore(palace_dir / ".meta" if palace_dir else tmp_path / ".meta", palace_dir=palace_dir)
    for mid in memory_ids:
        meta.add(
            mid,
            wing="w",
            type="fact",
            room="r",
            summary=f"摘要 {mid}",
            confidence=3,
            privacy="personal",
            stored_at="2026-09-30T00:00:00",
            content_preview=f"内容 {mid}",
        )
    meta.flush()
    return meta


def _make_palace(tmp_path: Path, memory_ids: list[str]) -> Path:
    palace = tmp_path / "palace"
    for mid in memory_ids:
        drawer_dir = palace / "w" / "h" / "r" / "drawer"
        drawer_dir.mkdir(parents=True, exist_ok=True)
        (drawer_dir / f"{mid}.md").write_text(f"# {mid}\n\n正文\n", encoding="utf-8")
    return palace


def _make_index(tmp_path: Path, memory_ids: list[str]) -> Path:
    index = ThreeLevelIndex(tmp_path / "index")
    for mid in memory_ids:
        index.add(memory_id=mid, wing="w", hall="h", room="r", content=f"内容 {mid}")
    index.flush()
    return index.db_path


def test_index_rows_with_a_drawer_survive_a_sparse_metastore(tmp_path):
    """★ 核心回归：2 431 行「有抽屉、无 meta 行」的合法索引一行都不能被删。"""
    drawer_ids = [f"d{i:012d}" for i in range(30)]
    palace = _make_palace(tmp_path, drawer_ids)
    db_path = _make_index(tmp_path, drawer_ids)
    meta = _make_meta(tmp_path, ["only000000000"], palace)  # MetaStore 只认得 1 条

    before = _index_ids(db_path)
    stale, missing = meta.sync_from_index(db_path)
    after = _index_ids(db_path)

    assert (stale, missing) == (0, 0)
    assert before == after, "MetaStore 缺行被当成幽灵，合法索引行被开机同步删掉了"
    meta.close()


def test_true_ghost_index_rows_are_pruned(tmp_path):
    """既无 meta 行、又无抽屉文件的 index 行，才允许清理。"""
    palace = _make_palace(tmp_path, ["real00000000"])
    db_path = _make_index(tmp_path, ["real00000000", "ghost0000000"])
    meta = _make_meta(tmp_path, [], palace)

    stale, missing = meta.sync_from_index(db_path)

    assert (stale, missing) == (1, 0)
    assert _index_ids(db_path) == {"real00000000"}
    meta.close()


def test_metastore_rows_without_a_drawer_are_not_reinserted(tmp_path):
    """磁盘已无抽屉的记忆，不该被重新写回 index.db 变成幽灵命中。"""
    palace = _make_palace(tmp_path, ["real00000000"])
    db_path = _make_index(tmp_path, ["real00000000"])
    meta = _make_meta(tmp_path, ["gone00000000"], palace)

    stale, missing = meta.sync_from_index(db_path)

    assert (stale, missing) == (0, 0)
    assert "gone00000000" not in _index_ids(db_path)
    meta.close()


def test_missing_index_row_with_a_drawer_is_healed(tmp_path):
    """正向能力要保留：meta 有、磁盘有、index 缺 → 补一行。"""
    palace = _make_palace(tmp_path, ["heal00000000"])
    db_path = _make_index(tmp_path, [])
    meta = _make_meta(tmp_path, ["heal00000000"], palace)

    stale, missing = meta.sync_from_index(db_path)

    assert (stale, missing) == (0, 1)
    assert _index_ids(db_path) == {"heal00000000"}
    meta.close()


def test_unbound_palace_dir_refuses_to_delete(tmp_path):
    """没绑定 palace_dir 就无法核验磁盘 → 整轮同步放弃，一行都不动。"""
    _make_palace(tmp_path, ["a00000000000", "b00000000000"])
    db_path = _make_index(tmp_path, ["a00000000000", "b00000000000"])
    meta = _make_meta(tmp_path, [], None)

    stale, missing = meta.sync_from_index(db_path)

    assert (stale, missing) == (0, 0)
    assert len(_index_ids(db_path)) == 2
    meta.close()


def test_empty_drawer_scan_is_treated_as_unverified(tmp_path):
    """palace 存在却扫不到抽屉（挂载/权限异常）→ fail-closed，不做全量删除。"""
    palace = tmp_path / "palace"
    palace.mkdir()
    db_path = _make_index(tmp_path, ["a00000000000", "b00000000000"])
    meta = _make_meta(tmp_path, [], palace)

    stale, missing = meta.sync_from_index(db_path)

    assert (stale, missing) == (0, 0)
    assert len(_index_ids(db_path)) == 2
    meta.close()
