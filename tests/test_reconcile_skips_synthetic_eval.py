"""回归：对账不得把合成评测会话（``provenance.source = api-*``）的抽屉索引进来。

2026-09-30 一次 ``reconcile_index_from_disk --apply`` 以磁盘抽屉为 SSOT 全量补索引，把
LongMemEval 式评测会话的 harness 提示词当作用户记忆灌进 index.db —— 单句
``确认: Please answer yes if the response contains the correct answer`` 重复 603 次，
新增 1 791 条，遗忘曲线覆盖率从 99.9% 掉到 23%，且这批向量占着语义召回的 top-k 坑位。

关键点是**磁盘抽屉不会自己消失**：评测再跑一次就多一批，所以只在事故后清 index.db 不够，
筛除必须落在对账路径上，每次 ``--apply`` 都重新生效。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from omnimem.governance.reconciler import is_synthetic_source, reconcile_index
from omnimem.memory.index import ThreeLevelIndex

EVAL_SOURCE = "api-33f4a8aaeb106e20"
REAL_SOURCE = "20260710_070115_162bedd6"


def _write_drawer(palace: Path, memory_id: str, source: str | None) -> Path:
    provenance = (
        f"provenance:\n  method: reinforcement\n  source: {source}\n"
        if source
        else ""
    )
    path = palace / "personal" / "fact" / "room" / "drawer" / f"{memory_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nmemory_id: {memory_id}\ntype: preference\nconfidence: 3\n"
        f"privacy: personal\nstored_at: '2026-07-10T17:57:50+00:00'\n{provenance}---\n\n"
        "确认: Please answer yes if the response contains the correct answer\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """磁盘 3 条抽屉（评测 / 真实会话 / 无 provenance），index 一条都没有。"""
    palace = tmp_path / "palace"
    _write_drawer(palace, "eval00000000", EVAL_SOURCE)
    _write_drawer(palace, "real00000000", REAL_SOURCE)
    _write_drawer(palace, "noprov000000", None)
    ThreeLevelIndex(tmp_path / "index").flush()
    return tmp_path


def test_eval_drawer_is_skipped_and_never_reindexed(data_dir):
    report = reconcile_index(data_dir, apply=False)

    assert report.skipped_synthetic == ["eval00000000"]
    assert sorted(report.missing_in_index) == ["noprov000000", "real00000000"], (
        "判据是 provenance，不是内容 —— 真实会话里的相同句子仍应正常入索引"
    )

    applied = reconcile_index(data_dir, apply=True)
    assert applied.skipped_synthetic == ["eval00000000"], "筛除要落在 apply 路径上，不止报告"
    assert applied.repaired == 2
    assert "eval00000000" not in _index_ids(data_dir)


def test_coverage_denominator_excludes_eval_drawers(data_dir):
    index = ThreeLevelIndex(data_dir / "index")
    index.add(memory_id="real00000000", wing="personal", hall="fact", room="room", content="x")
    index.add(memory_id="noprov000000", wing="personal", hall="fact", room="room", content="x")
    index.flush()

    report = reconcile_index(data_dir, apply=False)

    assert report.disk_total == 3
    assert report.recallable_total == 2, "评测抽屉不该把覆盖率拉下来"
    assert report.missing_in_index == []
    assert report.coverage_before == 1.0


def test_repeated_reconcile_stays_filtered(data_dir):
    """事故复发路径：清完 index.db 再对账，同一批评测抽屉不能第二次灌回来。"""
    reconcile_index(data_dir, apply=True)
    _purge_index(data_dir)

    again = reconcile_index(data_dir, apply=True)
    assert again.skipped_synthetic == ["eval00000000"]
    assert again.repaired == 2
    assert "eval00000000" not in _index_ids(data_dir)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (EVAL_SOURCE, True),
        ("api-27c9d6ec968e13cb", True),
        (REAL_SOURCE, False),
        ("cron_14228e16f0af_20260916_213014", False),
        ("apix-1", False),
        (None, False),
    ],
)
def test_is_synthetic_source(source, expected):
    provenance = {"source": source} if source is not None else {}
    assert is_synthetic_source({"provenance": provenance}) is expected
    assert is_synthetic_source({"provenance": '{"source": "%s"}' % source}) is expected


def _index_ids(data_dir: Path) -> set[str]:
    conn = sqlite3.connect(str(data_dir / "index" / "index.db"))
    try:
        return {r[0] for r in conn.execute("SELECT memory_id FROM memory_index")}
    finally:
        conn.close()


def _purge_index(data_dir: Path) -> None:
    conn = sqlite3.connect(str(data_dir / "index" / "index.db"))
    conn.execute("DELETE FROM memory_index")
    conn.commit()
    conn.close()


def _purge_index(data_dir: Path) -> None:
    conn = sqlite3.connect(str(data_dir / "index" / "index.db"))
    conn.execute("DELETE FROM memory_index")
    conn.commit()
    conn.close()
