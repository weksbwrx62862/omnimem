"""governance/forgetting_stages.py — _ForgettingStages Mixin 单元测试。

覆盖: 模块级失败计数 / _ensure_pipeline_marker 冷启动 / prune_access_log
     get_stage / get_stage_by_age / archive 状态机 / reactivate / record_access
     set_heat + get_heat / get_recall_count_in_window / get_candidates_by_heat
     mark_upgraded_to_wiki / get_upgrade_candidates
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from omnimem.governance.forgetting_stages import (
    _ACCESS_RETRY_COUNT,
    _bump_access_failures,
    _ForgettingStages,
    _reset_access_failures,
    get_access_failure_streak,
)

# ─── 脚手架 ────────────────────────────────────────────────


def _new_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript("""
        CREATE TABLE forgetting_state (
            memory_id TEXT PRIMARY KEY,
            stage TEXT DEFAULT 'active',
            last_accessed TEXT,
            created_at TEXT,
            recall_count INTEGER DEFAULT 0,
            memory_type TEXT,
            heat TEXT DEFAULT 'neutral',
            heat_updated_at TEXT,
            upgraded_to_wiki INTEGER DEFAULT 0,
            wiki_page_path TEXT
        );
        CREATE TABLE access_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT,
            accessed_at TEXT
        );
        CREATE TABLE pipeline_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    return conn


class _FakeStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.commits = 0

    def commit(self) -> None:
        self._conn.commit()
        self.commits += 1


class _Engine(_ForgettingStages):
    """Mixin 挂载：提供 _conn/_lock/_store/_pending_writes/_stages/_maybe_commit/flush/_set_stage。"""

    _BATCH_THRESHOLD = 10

    def __init__(self) -> None:
        self._conn = _new_conn()
        self._lock = threading.RLock()
        self._store = _FakeStore(self._conn)
        self._pending_writes = 0
        self._stages: dict[str, tuple[int, int | None]] = {
            "active": (0, 7),
            "consolidating": (7, 30),
            "archived": (30, 90),
            "forgotten": (90, None),
        }
        self.maybe_commits = 0

    def _maybe_commit(self) -> None:
        self.maybe_commits += 1
        if self._pending_writes >= self._BATCH_THRESHOLD:
            self._store.commit()
            self._pending_writes = 0

    def flush(self) -> None:
        self._store.commit()
        self._pending_writes = 0

    def _set_stage(self, memory_id: str, stage: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO forgetting_state "
            "(memory_id, stage, last_accessed, created_at) VALUES (?, ?, ?, ?)",
            (memory_id, stage, now, now),
        )
        self._pending_writes += 1
        self._maybe_commit()


def _seed_state(
    eng: _Engine,
    memory_id: str,
    *,
    stage: str = "active",
    heat: str = "neutral",
    recall_count: int = 0,
    upgraded: int = 0,
    memory_type: str = "fact",
    created_at: str | None = None,
    last_accessed: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    eng._conn.execute(
        """INSERT OR REPLACE INTO forgetting_state
           (memory_id, stage, last_accessed, created_at, recall_count, memory_type, heat, upgraded_to_wiki)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (memory_id, stage, last_accessed or now, created_at or now,
         recall_count, memory_type, heat, upgraded),
    )
    eng._conn.commit()


# ─── 模块级失败计数 ────────────────────────────────────────


class TestAccessFailureCounter:
    def test_reset_to_zero(self):
        _reset_access_failures()
        assert get_access_failure_streak() == 0

    def test_bump_increments(self):
        _reset_access_failures()
        assert _bump_access_failures() == 1
        assert _bump_access_failures() == 2
        assert get_access_failure_streak() == 2

    def test_reset_clears_streak(self):
        _bump_access_failures()
        _bump_access_failures()
        _reset_access_failures()
        assert get_access_failure_streak() == 0

    def test_thread_safety(self):
        _reset_access_failures()
        threads = [threading.Thread(target=_bump_access_failures) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert get_access_failure_streak() == 50
        _reset_access_failures()


# ─── _ensure_pipeline_marker ───────────────────────────────


class TestEnsurePipelineMarker:
    def test_first_call_inserts_start_time(self):
        eng = _Engine()
        eng._ensure_pipeline_marker()
        row = eng._conn.execute(
            "SELECT value FROM pipeline_meta WHERE key='start_time'"
        ).fetchone()
        assert row is not None
        # ISO 格式
        assert "T" in row[0]

    def test_second_call_is_idempotent(self):
        eng = _Engine()
        eng._ensure_pipeline_marker()
        first = eng._conn.execute(
            "SELECT value FROM pipeline_meta WHERE key='start_time'"
        ).fetchone()[0]
        eng._ensure_pipeline_marker()
        second = eng._conn.execute(
            "SELECT value FROM pipeline_meta WHERE key='start_time'"
        ).fetchone()[0]
        assert first == second

    def test_get_pipeline_start_time_returns_marker(self):
        eng = _Engine()
        eng._ensure_pipeline_marker()
        assert eng._get_pipeline_start_time() is not None

    def test_get_pipeline_start_time_missing_returns_none(self):
        eng = _Engine()
        # 未调用 _ensure_pipeline_marker, 表存在但无行
        assert eng._get_pipeline_start_time() is None


# ─── prune_access_log ──────────────────────────────────────


class TestPruneAccessLog:
    def test_removes_older_than_cutoff(self):
        eng = _Engine()
        old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        eng._conn.executemany(
            "INSERT INTO access_log (memory_id, accessed_at) VALUES (?, ?)",
            [("m1", old), ("m2", old), ("m3", recent)],
        )
        eng._conn.commit()
        deleted = eng.prune_access_log(days=90)
        assert deleted == 2
        remaining = eng._conn.execute("SELECT COUNT(*) FROM access_log").fetchone()[0]
        assert remaining == 1

    def test_nothing_to_prune(self):
        eng = _Engine()
        recent = datetime.now(timezone.utc).isoformat()
        eng._conn.execute(
            "INSERT INTO access_log (memory_id, accessed_at) VALUES (?, ?)",
            ("m1", recent),
        )
        eng._conn.commit()
        assert eng.prune_access_log(days=90) == 0

    def test_empty_log_returns_zero(self):
        eng = _Engine()
        assert eng.prune_access_log(days=1) == 0


# ─── get_stage / get_stage_by_age ──────────────────────────


class TestStageQueries:
    def test_get_stage_missing_returns_active(self):
        eng = _Engine()
        assert eng.get_stage("nonexistent") == "active"

    def test_get_stage_returns_persisted(self):
        eng = _Engine()
        _seed_state(eng, "m1", stage="archived")
        assert eng.get_stage("m1") == "archived"

    def test_get_stage_by_age_active(self):
        eng = _Engine()
        assert eng.get_stage_by_age(3) == "active"

    def test_get_stage_by_age_consolidating(self):
        eng = _Engine()
        assert eng.get_stage_by_age(15) == "consolidating"

    def test_get_stage_by_age_archived(self):
        eng = _Engine()
        assert eng.get_stage_by_age(60) == "archived"

    def test_get_stage_by_age_forgotten_unbounded(self):
        eng = _Engine()
        assert eng.get_stage_by_age(200) == "forgotten"

    def test_get_stage_by_age_zero_boundary(self):
        eng = _Engine()
        assert eng.get_stage_by_age(0) == "active"
        assert eng.get_stage_by_age(7) == "consolidating"  # 上界排除


# ─── archive / reactivate ──────────────────────────────────


class TestArchiveReactivate:
    def test_archive_from_active(self):
        eng = _Engine()
        _seed_state(eng, "m1", stage="active")
        eng.archive("m1")
        assert eng.get_stage("m1") == "archived"
        assert eng._store.commits >= 1

    def test_archive_from_archived_to_forgotten(self):
        eng = _Engine()
        _seed_state(eng, "m1", stage="archived")
        eng.archive("m1")
        assert eng.get_stage("m1") == "forgotten"

    def test_archive_forgotten_is_noop(self):
        eng = _Engine()
        _seed_state(eng, "m1", stage="forgotten")
        commits_before = eng._store.commits
        eng.archive("m1")
        assert eng.get_stage("m1") == "forgotten"
        assert eng._store.commits == commits_before

    def test_reactivate_resets_to_active(self):
        eng = _Engine()
        _seed_state(eng, "m1", stage="archived")
        eng.reactivate("m1")
        assert eng.get_stage("m1") == "active"
        # last_accessed 被更新
        row = eng._conn.execute(
            "SELECT last_accessed FROM forgetting_state WHERE memory_id='m1'"
        ).fetchone()
        assert row[0] is not None


# ─── record_access ─────────────────────────────────────────


class TestRecordAccess:
    def test_first_access_inserts_row(self):
        eng = _Engine()
        _reset_access_failures()
        eng.record_access("m1", "fact")
        row = eng._conn.execute(
            "SELECT stage, recall_count, memory_type FROM forgetting_state WHERE memory_id='m1'"
        ).fetchone()
        assert row is not None
        assert row[0] == "active"
        assert row[1] == 1
        assert row[2] == "fact"

    def test_second_access_increments_recall(self):
        eng = _Engine()
        _reset_access_failures()
        eng.record_access("m1", "fact")
        eng.record_access("m1", "fact")
        cnt = eng._conn.execute(
            "SELECT recall_count FROM forgetting_state WHERE memory_id='m1'"
        ).fetchone()[0]
        assert cnt == 2

    def test_writes_access_log(self):
        eng = _Engine()
        _reset_access_failures()
        eng.record_access("m1", "fact")
        n = eng._conn.execute(
            "SELECT COUNT(*) FROM access_log WHERE memory_id='m1'"
        ).fetchone()[0]
        assert n == 1

    def test_reactivates_archived_memory(self):
        eng = _Engine()
        _reset_access_failures()
        _seed_state(eng, "m1", stage="archived", recall_count=5)
        eng.record_access("m1", "fact")
        row = eng._conn.execute(
            "SELECT stage, recall_count FROM forgetting_state WHERE memory_id='m1'"
        ).fetchone()
        assert row[0] == "active"
        assert row[1] == 6

    def test_success_resets_failure_counter(self):
        eng = _Engine()
        _bump_access_failures()
        _bump_access_failures()
        eng.record_access("m1", "fact")
        assert get_access_failure_streak() == 0

    def test_persistent_failure_bumps_counter(self, monkeypatch):
        eng = _Engine()
        _reset_access_failures()

        class _BrokenConn:
            def execute(self, *_a, **_kw):
                raise RuntimeError("database is locked")

        eng._conn = _BrokenConn()  # type: ignore[assignment]
        # 缩短退避, 避免测试变慢
        import omnimem.governance.forgetting_stages as mod
        monkeypatch.setattr(mod, "_ACCESS_RETRY_DELAY", 0.0)
        eng.record_access("m1", "fact")
        assert get_access_failure_streak() == 1
        _reset_access_failures()

    def test_retry_constant_is_five(self):
        assert _ACCESS_RETRY_COUNT == 5


# ─── set_heat / get_heat ───────────────────────────────────


class TestHeat:
    def test_get_heat_missing_returns_neutral(self):
        eng = _Engine()
        assert eng.get_heat("nonexistent") == "neutral"

    def test_set_and_get_heat(self):
        eng = _Engine()
        _seed_state(eng, "m1")
        eng.set_heat("m1", "hot")
        assert eng.get_heat("m1") == "hot"

    def test_set_heat_rejects_invalid(self, caplog):
        eng = _Engine()
        _seed_state(eng, "m1")
        eng.set_heat("m1", "blazing")
        # 未写入, 仍为 neutral
        assert eng.get_heat("m1") == "neutral"
        assert any("Invalid heat level" in rec.message for rec in caplog.records)

    def test_set_heat_updates_timestamp(self):
        eng = _Engine()
        _seed_state(eng, "m1")
        eng.set_heat("m1", "cold")
        row = eng._conn.execute(
            "SELECT heat, heat_updated_at FROM forgetting_state WHERE memory_id='m1'"
        ).fetchone()
        assert row[0] == "cold"
        assert row[1] is not None


# ─── get_recall_count_in_window ────────────────────────────


class TestRecallWindow:
    def test_counts_only_recent(self):
        eng = _Engine()
        old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        eng._conn.executemany(
            "INSERT INTO access_log (memory_id, accessed_at) VALUES (?, ?)",
            [("m1", old), ("m1", recent), ("m1", recent)],
        )
        eng._conn.commit()
        assert eng.get_recall_count_in_window("m1", days=7) == 2

    def test_returns_zero_for_unknown_id(self):
        eng = _Engine()
        assert eng.get_recall_count_in_window("nope", days=7) == 0

    def test_empty_log_returns_zero(self):
        eng = _Engine()
        assert eng.get_recall_count_in_window("m1", days=1) == 0


# ─── get_candidates_by_heat ────────────────────────────────


class TestCandidatesByHeat:
    def test_filters_by_heat(self):
        eng = _Engine()
        _seed_state(eng, "m1", heat="hot")
        _seed_state(eng, "m2", heat="cold")
        _seed_state(eng, "m3", heat="hot")
        out = eng.get_candidates_by_heat("hot")
        assert {r["memory_id"] for r in out} == {"m1", "m3"}

    def test_result_shape(self):
        eng = _Engine()
        _seed_state(eng, "m1", heat="warm", recall_count=3, stage="active")
        [r] = eng.get_candidates_by_heat("warm")
        assert r["memory_id"] == "m1"
        assert r["recall_count"] == 3
        assert r["stage"] == "active"
        assert r["heat"] == "warm"
        assert "created_at" in r

    def test_empty_result(self):
        eng = _Engine()
        assert eng.get_candidates_by_heat("cold") == []

    def test_null_recall_count_normalized_to_zero(self):
        eng = _Engine()
        eng._conn.execute(
            "INSERT INTO forgetting_state (memory_id, stage, heat) VALUES (?, ?, ?)",
            ("m1", "active", "hot"),
        )
        eng._conn.commit()
        out = eng.get_candidates_by_heat("hot")
        assert out[0]["recall_count"] == 0


# ─── mark_upgraded_to_wiki ─────────────────────────────────


class TestMarkUpgradedToWiki:
    def test_sets_flag_and_path(self):
        eng = _Engine()
        _seed_state(eng, "m1")
        eng.mark_upgraded_to_wiki("m1", "wiki/python.md")
        row = eng._conn.execute(
            "SELECT upgraded_to_wiki, wiki_page_path FROM forgetting_state WHERE memory_id='m1'"
        ).fetchone()
        assert row[0] == 1
        assert row[1] == "wiki/python.md"

    def test_flush_called(self):
        eng = _Engine()
        _seed_state(eng, "m1")
        commits_before = eng._store.commits
        eng.mark_upgraded_to_wiki("m1", "wiki/x.md")
        assert eng._store.commits > commits_before


# ─── get_upgrade_candidates ────────────────────────────────


class TestGetUpgradeCandidates:
    def test_filters_by_recall_heat_stage(self):
        eng = _Engine()
        _seed_state(eng, "m1", heat="hot", stage="active", recall_count=3)
        _seed_state(eng, "m2", heat="cold", stage="active", recall_count=5)   # heat 不符
        _seed_state(eng, "m3", heat="hot", stage="archived", recall_count=5)  # stage 不符
        _seed_state(eng, "m4", heat="hot", stage="active", recall_count=1)   # recall 不符
        out = eng.get_upgrade_candidates(min_recall=2)
        assert [r["memory_id"] for r in out] == ["m1"]

    def test_excludes_already_upgraded(self):
        eng = _Engine()
        _seed_state(eng, "m1", heat="hot", stage="active", recall_count=3, upgraded=1)
        _seed_state(eng, "m2", heat="hot", stage="active", recall_count=3, upgraded=0)
        out = eng.get_upgrade_candidates(min_recall=2)
        assert [r["memory_id"] for r in out] == ["m2"]

    def test_orders_by_recall_desc(self):
        eng = _Engine()
        _seed_state(eng, "m_low", heat="hot", stage="active", recall_count=2)
        _seed_state(eng, "m_high", heat="hot", stage="active", recall_count=10)
        _seed_state(eng, "m_mid", heat="hot", stage="active", recall_count=5)
        out = eng.get_upgrade_candidates(min_recall=2)
        assert [r["memory_id"] for r in out] == ["m_high", "m_mid", "m_low"]

    def test_custom_min_recall(self):
        eng = _Engine()
        _seed_state(eng, "m1", heat="hot", stage="active", recall_count=5)
        assert eng.get_upgrade_candidates(min_recall=10) == []
        assert len(eng.get_upgrade_candidates(min_recall=5)) == 1

    def test_result_shape(self):
        eng = _Engine()
        _seed_state(eng, "m1", heat="hot", stage="active", recall_count=4)
        [r] = eng.get_upgrade_candidates(min_recall=2)
        assert set(r.keys()) == {"memory_id", "created_at", "recall_count", "heat", "stage"}
        assert r["heat"] == "hot"
        assert r["stage"] == "active"

    def test_empty_table(self):
        eng = _Engine()
        assert eng.get_upgrade_candidates(min_recall=2) == []


# ─── Mixin 属性存在性 ─────────────────────────────────────


class TestMixinSurface:
    def test_all_public_methods_exist(self):
        for name in (
            "_ensure_pipeline_marker",
            "_get_pipeline_start_time",
            "prune_access_log",
            "get_stage",
            "get_stage_by_age",
            "archive",
            "reactivate",
            "record_access",
            "set_heat",
            "get_heat",
            "get_recall_count_in_window",
            "get_candidates_by_heat",
            "mark_upgraded_to_wiki",
            "get_upgrade_candidates",
        ):
            assert hasattr(_ForgettingStages, name), name
