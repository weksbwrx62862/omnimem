"""P1-10 回归：对账不得把已归档/已遗忘的记忆重新索引。

遗忘曲线归档一条记忆时只改 ``forgetting_state.stage``，抽屉留待后续物理删除，而
``GovernanceAuditor`` 每次开机把它们的 index 行删掉。原对账只看抽屉 ⇒ 把用户忘掉的东西
补回索引，开机审计再删、下次对账再加 —— 永久拉锯，而且违背遗忘语义。
副本实测：某次差额 440 行里 **440 行**都属于已归档/已遗忘。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from omnimem.governance.reconciler import archived_memory_ids, reconcile_index
from omnimem.memory.index import ThreeLevelIndex


def _write_drawer(palace: Path, memory_id: str, content: str = "内容") -> Path:
    path = palace / "personal" / "fact" / "room" / "drawer" / f"{memory_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nmemory_id: {memory_id}\ntype: fact\nconfidence: 3\n"
        f"privacy: personal\nstored_at: '2026-09-01T00:00:00+00:00'\n---\n\n{content}\n",
        encoding="utf-8",
    )
    return path


def _forgetting_db(data_dir: Path, stages: dict[str, str]) -> None:
    """建 governance/forgetting.db 并写入 forgetting_state。"""
    gov = data_dir / "governance"
    gov.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(gov / "forgetting.db")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS forgetting_state ("
        "memory_id TEXT PRIMARY KEY, stage TEXT NOT NULL DEFAULT 'active')"
    )
    conn.executemany(
        "INSERT OR REPLACE INTO forgetting_state (memory_id, stage) VALUES (?, ?)",
        list(stages.items()),
    )
    conn.commit()
    conn.close()


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """磁盘 4 条抽屉、index 一条都没有。"""
    palace = tmp_path / "palace"
    for mid in ("act000000000", "arc000000000", "fgt000000000", "unk000000000"):
        _write_drawer(palace, mid)
    ThreeLevelIndex(tmp_path / "index").flush()
    return tmp_path


def _seed_index(data_dir: Path, ids: list[str]) -> None:
    index = ThreeLevelIndex(data_dir / "index")
    for mid in ids:
        index.add(memory_id=mid, wing="personal", hall="fact", room="room", content="内容")
    index.flush()


def test_archived_and_forgotten_are_not_reindexed(data_dir):
    _forgetting_db(data_dir, {
        "act000000000": "active",
        "arc000000000": "archived",
        "fgt000000000": "forgotten",
        "unk000000000": "consolidating",
    })

    report = reconcile_index(data_dir, apply=False)

    assert sorted(report.missing_in_index) == ["act000000000", "unk000000000"]
    assert sorted(report.skipped_archived) == ["arc000000000", "fgt000000000"]

    applied = reconcile_index(data_dir, apply=True)
    assert applied.repaired == 2, "把已归档/已遗忘的记忆补回索引 = 复活用户忘掉的东西"
    ids = _index_ids(data_dir)
    assert "arc000000000" not in ids and "fgt000000000" not in ids


def test_coverage_excludes_archived_from_the_denominator(data_dir):
    _forgetting_db(data_dir, {
        "act000000000": "active",
        "arc000000000": "archived",
        "fgt000000000": "forgotten",
        "unk000000000": "consolidating",
    })
    _seed_index(data_dir, ["act000000000", "unk000000000"])

    report = reconcile_index(data_dir, apply=False)

    assert report.disk_total == 4
    assert report.recallable_total == 2
    assert report.missing_in_index == []
    assert report.coverage_before == 1.0, "已归档的抽屉不该拉低覆盖率"


def test_missing_forgetting_db_does_not_block_reconcile(data_dir, caplog):
    """没有遗忘库（全新安装）→ 照旧补索引，但要说清楚这层保护没生效。"""
    report = reconcile_index(data_dir, apply=False)

    assert sorted(report.missing_in_index) == [
        "act000000000", "arc000000000", "fgt000000000", "unk000000000",
    ]
    assert report.skipped_archived == []
    assert archived_memory_ids(data_dir) is None


def test_ghost_pruning_is_unaffected(data_dir):
    """幽灵（有 index 无抽屉）与遗忘状态无关，仍应被识别并可清理。"""
    _forgetting_db(data_dir, {})
    _seed_index(data_dir, ["ghost0000000"])

    report = reconcile_index(data_dir, apply=True, prune_ghosts=True)

    assert report.ghosts_in_index == ["ghost0000000"]
    assert report.pruned == 1
    assert "ghost0000000" not in _index_ids(data_dir)


def _index_ids(data_dir: Path) -> set[str]:
    conn = sqlite3.connect(str(data_dir / "index" / "index.db"))
    try:
        return {r[0] for r in conn.execute("SELECT memory_id FROM memory_index")}
    finally:
        conn.close()
