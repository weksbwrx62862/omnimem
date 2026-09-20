"""governance.review_scheduler 离线单元测试。

覆盖：
  - _calculate_due_date：last_accessed / created_at / fallback 明天
  - _estimate_retention：无 last_accessed→0.5、有→曲线递减、坏输入回退
  - _calculate_priority：保持率因子 + 逾期因子
  - _get_due_items：缺 DB → []；正常 sqlite 表读取
  - generate_review_plan：分组、max_items、estimated_time
  - get_today_plan 空/非空
  - get_review_stats 空/非空
  - get_scheduler 单例
  - 便捷 get_today_plan() 字典结构
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from omnimem.governance import review_scheduler as rs
from omnimem.governance.review_scheduler import (
    DailyPlan,
    ReviewItem,
    ReviewScheduler,
    get_scheduler,
)


@pytest.fixture
def scheduler(tmp_path: Path) -> ReviewScheduler:
    return ReviewScheduler(governance_dir=tmp_path)


def _seed_db(path: Path, rows: list[tuple[Any, ...]]) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS forgetting_state ("
        "memory_id TEXT PRIMARY KEY, recall_count INTEGER, "
        "created_at TEXT, last_accessed TEXT)"
    )
    conn.executemany(
        "INSERT OR REPLACE INTO forgetting_state (memory_id, recall_count, created_at, last_accessed) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


# ── _calculate_due_date ──


def test_due_date_uses_last_accessed(scheduler: ReviewScheduler) -> None:
    last = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()
    due = scheduler._calculate_due_date(recall_count=5, created_at=None, last_accessed=last)
    assert due > datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_due_date_falls_back_to_created_at(scheduler: ReviewScheduler) -> None:
    created = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()
    due = scheduler._calculate_due_date(3, created, None)
    assert due > datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_due_date_default_tomorrow_when_no_timestamps(scheduler: ReviewScheduler) -> None:
    now = datetime.now(timezone.utc)
    due = scheduler._calculate_due_date(0, None, None)
    assert (due - now).days >= 0
    assert due > now


def test_due_date_ignores_bad_timestamp(scheduler: ReviewScheduler) -> None:
    # 非法时间戳 → 走默认路径
    due = scheduler._calculate_due_date(2, None, "not-a-date")
    assert isinstance(due, datetime)


# ── _estimate_retention ──


def test_estimate_retention_no_last_accessed_returns_half(scheduler: ReviewScheduler) -> None:
    assert scheduler._estimate_retention(5, None, datetime.now(timezone.utc)) == 0.5


def test_estimate_retention_recent_access_near_one(scheduler: ReviewScheduler) -> None:
    now = datetime.now(timezone.utc)
    last = (now - timedelta(hours=1)).isoformat()
    r = scheduler._estimate_retention(10, last, now)
    assert 0.9 <= r <= 1.0


def test_estimate_retention_decays_with_time(scheduler: ReviewScheduler) -> None:
    now = datetime.now(timezone.utc)
    last = (now - timedelta(days=30)).isoformat()
    r = scheduler._estimate_retention(2, last, now)
    # stability=4, elapsed=30 → (1+30/36)^(-0.5) ≈ 0.738
    assert 0.5 < r < 0.9


def test_estimate_retention_handles_bad_timestamp(scheduler: ReviewScheduler) -> None:
    assert scheduler._estimate_retention(1, "bad", datetime.now(timezone.utc)) == 0.5


# ── _calculate_priority ──


def test_priority_higher_when_low_retention(scheduler: ReviewScheduler) -> None:
    now = datetime.now(timezone.utc)
    due = now + timedelta(days=10)
    hi = scheduler._calculate_priority(0.1, due, now)
    lo = scheduler._calculate_priority(0.9, due, now)
    assert hi > lo


def test_priority_higher_when_overdue(scheduler: ReviewScheduler) -> None:
    now = datetime.now(timezone.utc)
    on_time = scheduler._calculate_priority(0.5, now, now)
    overdue = scheduler._calculate_priority(0.5, now - timedelta(days=7), now)
    assert overdue > on_time


def test_priority_clamped_to_unit(scheduler: ReviewScheduler) -> None:
    now = datetime.now(timezone.utc)
    p = scheduler._calculate_priority(0.0, now - timedelta(days=100), now)
    assert 0.0 <= p <= 1.0


# ── _get_due_items ──


def test_get_due_items_no_db_returns_empty(scheduler: ReviewScheduler) -> None:
    assert scheduler._get_due_items() == []


def test_get_due_items_reads_rows(scheduler: ReviewScheduler, tmp_path: Path) -> None:
    db = tmp_path / "forgetting.db"
    now = datetime.now(timezone.utc)
    _seed_db(
        db,
        [
            ("m1", 5, (now - timedelta(days=10)).isoformat(), (now - timedelta(days=3)).isoformat()),
            ("m2", 0, (now - timedelta(days=1)).isoformat(), None),
        ],
    )
    items = scheduler._get_due_items()
    assert {it.memory_id for it in items} == {"m1", "m2"}
    assert all(isinstance(it, ReviewItem) for it in items)


def test_get_due_items_corrupt_db_returns_empty(scheduler: ReviewScheduler, tmp_path: Path) -> None:
    # 建 DB 但不含 forgetting_state 表
    conn = sqlite3.connect(str(tmp_path / "forgetting.db"))
    conn.execute("CREATE TABLE other (x INTEGER)")
    conn.commit()
    conn.close()
    assert scheduler._get_due_items() == []


# ── generate_review_plan ──


def test_generate_plan_no_items_returns_empty(scheduler: ReviewScheduler) -> None:
    assert scheduler.generate_review_plan(days=3) == []


def test_generate_plan_groups_by_day(scheduler: ReviewScheduler, tmp_path: Path) -> None:
    db = tmp_path / "forgetting.db"
    now = datetime.now(timezone.utc)
    _seed_db(
        db,
        [
            ("m1", 1, (now - timedelta(days=5)).isoformat(), (now - timedelta(days=5)).isoformat()),
            ("m2", 1, (now - timedelta(days=4)).isoformat(), (now - timedelta(days=4)).isoformat()),
        ],
    )
    plans = scheduler.generate_review_plan(days=2, max_items_per_day=50)
    assert len(plans) >= 1
    total = sum(p.total_count for p in plans[:1])
    assert total == 2
    assert plans[0].estimated_time == 2 * len(plans[0].items)


def test_generate_plan_respects_max_per_day(scheduler: ReviewScheduler, tmp_path: Path) -> None:
    db = tmp_path / "forgetting.db"
    now = datetime.now(timezone.utc)
    _seed_db(
        db,
        [
            (f"m{i}", 1, (now - timedelta(days=10)).isoformat(), (now - timedelta(days=10)).isoformat())
            for i in range(5)
        ],
    )
    plans = scheduler.generate_review_plan(days=1, max_items_per_day=2)
    assert plans[0].total_count == 2


def test_generate_plan_sorted_by_priority_desc(scheduler: ReviewScheduler, tmp_path: Path) -> None:
    db = tmp_path / "forgetting.db"
    now = datetime.now(timezone.utc)
    _seed_db(
        db,
        [
            ("low", 0, (now - timedelta(days=1)).isoformat(), (now - timedelta(days=1)).isoformat()),
            ("high", 0, (now - timedelta(days=100)).isoformat(), (now - timedelta(days=100)).isoformat()),
        ],
    )
    plans = scheduler.generate_review_plan(days=1)
    assert plans[0].items[0].priority >= plans[0].items[-1].priority


# ── get_today_plan ──


def test_get_today_plan_empty_returns_stub(scheduler: ReviewScheduler) -> None:
    p = scheduler.get_today_plan()
    assert isinstance(p, DailyPlan)
    assert p.total_count == 0
    assert p.items == []
    assert p.estimated_time == 0


def test_get_today_plan_with_items(scheduler: ReviewScheduler, tmp_path: Path) -> None:
    db = tmp_path / "forgetting.db"
    now = datetime.now(timezone.utc)
    _seed_db(
        db,
        [("m1", 1, (now - timedelta(days=30)).isoformat(), (now - timedelta(days=30)).isoformat())],
    )
    p = scheduler.get_today_plan()
    assert p.total_count >= 1


# ── get_review_stats ──


def test_review_stats_empty_defaults(scheduler: ReviewScheduler) -> None:
    s = scheduler.get_review_stats()
    assert s == {
        "total_due": 0,
        "overdue": 0,
        "avg_retention": 1.0,
        "urgent": 0,
    }


def test_review_stats_counts_overdue_and_urgent(scheduler: ReviewScheduler, tmp_path: Path) -> None:
    db = tmp_path / "forgetting.db"
    now = datetime.now(timezone.utc)
    _seed_db(
        db,
        [
            ("overdue_a", 0, (now - timedelta(days=50)).isoformat(), (now - timedelta(days=50)).isoformat()),
            ("overdue_b", 0, (now - timedelta(days=10)).isoformat(), (now - timedelta(days=10)).isoformat()),
            ("recent", 20, (now - timedelta(days=1)).isoformat(), now.isoformat()),
        ],
    )
    s = scheduler.get_review_stats()
    assert s["total_due"] == 3
    assert s["overdue"] >= 2
    assert 0 <= s["avg_retention"] <= 1


# ── 全局单例 ──


def test_get_scheduler_singleton(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rs, "_scheduler", None)
    a = get_scheduler(tmp_path)
    b = get_scheduler()
    assert a is b


def test_get_today_plan_convenience_returns_dict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rs, "_scheduler", None)
    db = tmp_path / "forgetting.db"
    now = datetime.now(timezone.utc)
    _seed_db(
        db,
        [("m1", 1, (now - timedelta(days=5)).isoformat(), (now - timedelta(days=5)).isoformat())],
    )
    rs.get_scheduler(tmp_path)
    out = rs.get_today_plan()
    assert set(out.keys()) == {"date", "total_count", "estimated_time", "items"}
    if out["items"]:
        item = out["items"][0]
        assert {"memory_id", "due_date", "priority", "retention", "days_overdue"} <= set(item.keys())
