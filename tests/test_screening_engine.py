"""governance/screening_engine.py — ForgettingScreening 单元测试。

覆盖: 三阶段筛选 / warm 降温 / 自动升级 / Wiki 引用计数 / 记忆摘要 / 晋升
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from omnimem.governance.screening_engine import ForgettingScreening

# ─── 脚手架 ────────────────────────────────────────────────


def _new_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE forgetting_state (
            memory_id TEXT PRIMARY KEY,
            stage TEXT DEFAULT 'active',
            created_at TEXT,
            heat_updated_at TEXT,
            recall_count INTEGER DEFAULT 0,
            heat TEXT DEFAULT 'neutral',
            upgraded_to_wiki INTEGER DEFAULT 0
        );
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            content TEXT
        );
    """)
    return conn


class _BrokenConn:
    """execute() 抛异常的假连接, 用于触发内部 try/except 路径。"""

    def execute(self, *_a, **_kw):
        raise RuntimeError("boom")


def _now(days_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


class _State:
    """回调状态收集器。"""

    def __init__(self) -> None:
        self.heat: dict[str, str] = {}
        self.stage: dict[str, str] = {}
        self.recall_map: dict[tuple[str, int], int] = {}
        self.pipeline_start: str | None = None
        self.track_writes = 0
        self.summary_map: dict[str, str] = {}

    def get_heat(self, mid: str) -> str:
        return self.heat.get(mid, "neutral")

    def set_heat(self, mid: str, h: str) -> None:
        self.heat[mid] = h

    def set_stage(self, mid: str, s: str) -> None:
        self.stage[mid] = s

    def get_recall_count_in_window(self, mid: str, days: int) -> int:
        return self.recall_map.get((mid, days), 0)

    def get_pipeline_start_time(self) -> str | None:
        return self.pipeline_start

    def track_write(self) -> None:
        self.track_writes += 1


def _engine(
    tmp_path: Path,
    state: _State | None = None,
    *,
    index_conn: sqlite3.Connection | None = None,
) -> tuple[ForgettingScreening, _State, sqlite3.Connection]:
    """构造 ForgettingScreening：主连接 + 可选 index 连接。"""
    st = state or _State()
    conn = _new_conn()
    idx = index_conn if index_conn is not None else _new_conn()
    governance_dir = tmp_path / "governance"
    governance_dir.mkdir(exist_ok=True)

    def _get_conn() -> sqlite3.Connection:
        return conn

    def _get_index_conn() -> sqlite3.Connection | None:
        return idx

    fs = ForgettingScreening(
        governance_dir=governance_dir,
        get_conn=_get_conn,
        get_index_conn=_get_index_conn,
        get_recall_count_in_window=st.get_recall_count_in_window,
        get_heat=st.get_heat,
        set_heat=st.set_heat,
        set_stage=st.set_stage,
        track_write=st.track_write,
        get_pipeline_start_time=st.get_pipeline_start_time,
    )
    return fs, st, conn


def _seed(
    conn: sqlite3.Connection,
    memory_id: str,
    *,
    stage: str = "active",
    heat: str = "neutral",
    recall_count: int = 0,
    created_days_ago: float = 2.0,
    heat_updated_days_ago: float | None = None,
    upgraded: int = 0,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO forgetting_state
           (memory_id, stage, created_at, heat_updated_at, recall_count, heat, upgraded_to_wiki)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (memory_id, stage, _now(created_days_ago),
         _now(heat_updated_days_ago) if heat_updated_days_ago is not None else None,
         recall_count, heat, upgraded),
    )
    conn.commit()


# ─── run_first_screening ───────────────────────────────────


class TestRunFirstScreening:
    def test_hot_when_density_ge_1(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", created_days_ago=2)
        st.recall_map[("m1", 7)] = 7  # density = 7/2 = 3.5
        out = fs.run_first_screening()
        assert out["hot"] == 1
        assert st.heat["m1"] == "hot"

    def test_warm_when_density_ge_03(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", created_days_ago=2)
        st.recall_map[("m1", 7)] = 2  # density = 2/2 = 1.0 → hot actually
        _seed(conn, "m2", created_days_ago=6)
        st.recall_map[("m2", 7)] = 2  # density = 2/6 ≈ 0.33 → warm
        out = fs.run_first_screening()
        assert out["warm"] >= 1

    def test_cold_when_zero_recall(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", created_days_ago=2)
        out = fs.run_first_screening()
        assert out["cold"] == 1
        assert st.heat["m1"] == "cold"

    def test_neutral_when_low_but_nonzero(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", created_days_ago=6)
        st.recall_map[("m1", 7)] = 1  # density = 1/6 ≈ 0.167 → neutral
        out = fs.run_first_screening()
        assert out["neutral"] == 1
        # old_heat 默认 "neutral" == new_heat → 不调用 set_heat
        assert "m1" not in st.heat

    def test_skips_recent_memories(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        # 创建时间距今仅 1 小时, 早于 24h cutoff
        _seed(conn, "m_new", created_days_ago=0.01)
        out = fs.run_first_screening()
        assert sum(out[k] for k in ("hot", "warm", "neutral", "cold")) == 0

    def test_pipeline_start_filters_history(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        # 冷启动: 管道 5 天前启动 → 早于此的 created_at 应被排除
        st.pipeline_start = _now(days_ago=5)
        _seed(conn, "m_old", created_days_ago=10)  # 早于 pipeline_start
        _seed(conn, "m_new", created_days_ago=2)   # 在窗口内
        out = fs.run_first_screening()
        # 只有 m_new 被处理
        assert out["cold"] + out["hot"] + out["warm"] + out["neutral"] == 1

    def test_no_redundant_set_heat(self, tmp_path):
        # old_heat == new_heat 时不调用 set_heat
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="cold", created_days_ago=2)
        st.heat["m1"] = "cold"  # 让 get_heat 返回 cold, 与 new_heat 一致
        calls: list[str] = []
        orig_set = st.set_heat

        def _spy(mid, h):
            calls.append(mid)
            orig_set(mid, h)

        fs._set_heat = _spy
        fs.run_first_screening()
        assert calls == []

    def test_exception_returns_zero_counts(self, tmp_path, monkeypatch):
        fs, _st, _conn = _engine(tmp_path)
        monkeypatch.setattr(fs, "_get_conn", lambda: _BrokenConn())
        out = fs.run_first_screening()
        assert out == {"hot": 0, "warm": 0, "neutral": 0, "cold": 0, "skipped": 0}


# ─── run_second_screening ──────────────────────────────────


class TestRunSecondScreening:
    def test_hot_with_high_recall_upgrades(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="hot", heat_updated_days_ago=10)
        st.recall_map[("m1", 7)] = 5
        out = fs.run_second_screening()
        assert out["wiki_upgrade"] == ["m1"]
        assert out["demoted_to_warm"] == 0

    def test_hot_with_low_recall_demotes(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="hot", heat_updated_days_ago=10)
        st.recall_map[("m1", 7)] = 1
        out = fs.run_second_screening()
        assert out["wiki_upgrade"] == []
        assert out["demoted_to_warm"] == 1
        assert st.heat["m1"] == "warm"

    def test_threshold_boundary_recall_2(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="hot", heat_updated_days_ago=10)
        st.recall_map[("m1", 7)] = 2  # 恰好 >= 2
        out = fs.run_second_screening()
        assert out["wiki_upgrade"] == ["m1"]

    def test_ignores_non_hot(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="warm", heat_updated_days_ago=10)
        st.recall_map[("m1", 7)] = 5
        out = fs.run_second_screening()
        assert out["wiki_upgrade"] == []
        assert out["demoted_to_warm"] == 0

    def test_ignores_recent_heat_updated(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        # heat_updated_at 距今 1 天, 早于 7d cutoff 之外
        _seed(conn, "m1", heat="hot", heat_updated_days_ago=1)
        st.recall_map[("m1", 7)] = 5
        out = fs.run_second_screening()
        assert out["wiki_upgrade"] == []

    def test_exception_returns_empty(self, tmp_path, monkeypatch):
        fs, _st, _conn = _engine(tmp_path)
        monkeypatch.setattr(fs, "_get_conn", lambda: _BrokenConn())
        out = fs.run_second_screening()
        assert out == {"wiki_upgrade": [], "demoted_to_warm": 0}


# ─── run_third_consolidation ───────────────────────────────


class TestRunThirdConsolidation:
    def test_promotes_when_refs_ge_2(self, tmp_path, monkeypatch):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="hot", recall_count=6, created_days_ago=40)
        monkeypatch.setattr(fs, "_count_wiki_references", lambda mid: 3)
        out = fs.run_third_consolidation()
        assert out["candidates"] == 1
        assert out["promoted"] == 1
        assert st.stage["m1"] == "consolidating"

    def test_monitors_when_refs_low(self, tmp_path, monkeypatch):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="hot", recall_count=6, created_days_ago=40)
        monkeypatch.setattr(fs, "_count_wiki_references", lambda mid: 1)
        out = fs.run_third_consolidation()
        assert out["promoted"] == 0
        assert out["monitored"] == 1
        # ref>0 → 不降级
        assert st.heat.get("m1") is None

    def test_demotes_to_warm_when_no_refs_and_low_recall(self, tmp_path, monkeypatch):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="hot", recall_count=6, created_days_ago=40)
        monkeypatch.setattr(fs, "_count_wiki_references", lambda mid: 0)
        out = fs.run_third_consolidation()
        assert st.heat["m1"] == "warm"

    def test_keeps_hot_when_no_refs_but_high_recall(self, tmp_path, monkeypatch):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="hot", recall_count=20, created_days_ago=40)
        monkeypatch.setattr(fs, "_count_wiki_references", lambda mid: 0)
        fs.run_third_consolidation()
        # recall >= 10 → 不降级
        assert st.heat.get("m1") is None

    def test_ignores_non_hot(self, tmp_path, monkeypatch):
        fs, _st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="warm", recall_count=10, created_days_ago=40)
        monkeypatch.setattr(fs, "_count_wiki_references", lambda mid: 3)
        out = fs.run_third_consolidation()
        assert out["candidates"] == 0

    def test_ignores_recent(self, tmp_path, monkeypatch):
        fs, _st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="hot", recall_count=6, created_days_ago=10)
        monkeypatch.setattr(fs, "_count_wiki_references", lambda mid: 3)
        out = fs.run_third_consolidation()
        assert out["candidates"] == 0

    def test_ignores_low_recall(self, tmp_path, monkeypatch):
        fs, _st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="hot", recall_count=3, created_days_ago=40)
        monkeypatch.setattr(fs, "_count_wiki_references", lambda mid: 3)
        out = fs.run_third_consolidation()
        assert out["candidates"] == 0

    def test_exception_returns_zero(self, tmp_path, monkeypatch):
        fs, _st, _conn = _engine(tmp_path)
        monkeypatch.setattr(fs, "_get_conn", lambda: _BrokenConn())
        out = fs.run_third_consolidation()
        assert out == {"promoted": 0, "monitored": 0, "candidates": 0}


# ─── run_warm_cooling ──────────────────────────────────────


class TestRunWarmCooling:
    def test_demotes_zero_recall_warm(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="warm", heat_updated_days_ago=35)
        out = fs.run_warm_cooling()
        assert out == 1
        assert st.heat["m1"] == "cold"

    def test_keeps_active_warm(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="warm", heat_updated_days_ago=35)
        st.recall_map[("m1", 30)] = 2
        out = fs.run_warm_cooling()
        assert out == 0
        assert "m1" not in st.heat  # 未调用 set_heat

    def test_ignores_non_warm(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="hot", heat_updated_days_ago=35)
        out = fs.run_warm_cooling()
        assert out == 0

    def test_exception_returns_zero(self, tmp_path, monkeypatch):
        fs, _st, _conn = _engine(tmp_path)
        monkeypatch.setattr(fs, "_get_conn", lambda: _BrokenConn())
        assert fs.run_warm_cooling() == 0


# ─── check_for_reactivation ────────────────────────────────


class TestCheckForReactivation:
    def test_consolidating_ge_3_reactivates(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", stage="consolidating")
        st.recall_map[("m1", 7)] = 3
        out = fs.check_for_reactivation()
        assert out == 1
        assert st.stage["m1"] == "active"

    def test_consolidating_below_threshold_noop(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", stage="consolidating")
        st.recall_map[("m1", 7)] = 2
        assert fs.check_for_reactivation() == 0
        assert "m1" not in st.stage

    def test_archived_ge_5_reactivates(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", stage="archived")
        st.recall_map[("m1", 7)] = 5
        out = fs.check_for_reactivation()
        assert out == 1
        assert st.stage["m1"] == "active"

    def test_archived_below_threshold_noop(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", stage="archived")
        st.recall_map[("m1", 7)] = 4
        assert fs.check_for_reactivation() == 0

    def test_active_stage_ignored(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", stage="active")
        st.recall_map[("m1", 7)] = 10
        assert fs.check_for_reactivation() == 0

    def test_counts_both_stages(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", stage="consolidating")
        _seed(conn, "m2", stage="archived")
        st.recall_map[("m1", 7)] = 5
        st.recall_map[("m2", 7)] = 10
        assert fs.check_for_reactivation() == 2

    def test_exception_returns_zero(self, tmp_path, monkeypatch):
        fs, _st, _conn = _engine(tmp_path)
        monkeypatch.setattr(fs, "_get_conn", lambda: _BrokenConn())
        assert fs.check_for_reactivation() == 0


# ─── _count_wiki_references ────────────────────────────────


class TestCountWikiReferences:
    def test_missing_wiki_dir_returns_zero(self, tmp_path):
        fs, _st, conn = _engine(tmp_path)
        # palace 目录不存在
        assert fs._count_wiki_references("m1") == 0

    def test_counts_files_containing_id(self, tmp_path):
        palace = tmp_path / "palace"
        palace.mkdir()
        (palace / "a.md").write_text("see memory_id m1 for details")
        (palace / "b.md").write_text("also references m1 here")
        (palace / "c.md").write_text("unrelated content")
        idx = _new_conn()
        idx.execute("INSERT INTO memories (id, content) VALUES (?, ?)",
                    ("m1", "some summary content"))
        idx.commit()
        fs, _st, _conn = _engine(tmp_path, index_conn=idx)
        out = fs._count_wiki_references("m1")
        assert out == 2

    def test_matches_by_summary(self, tmp_path):
        palace = tmp_path / "palace"
        palace.mkdir()
        idx = _new_conn()
        idx.execute("INSERT INTO memories (id, content) VALUES (?, ?)",
                    ("m1", "python is a great programming language"))
        idx.commit()
        (palace / "a.md").write_text("quote: python is a great programming language")
        fs, _st, _conn = _engine(tmp_path, index_conn=idx)
        assert fs._count_wiki_references("m1") == 1

    def test_ignores_non_markdown(self, tmp_path):
        palace = tmp_path / "palace"
        palace.mkdir()
        (palace / "a.txt").write_text("references m1")
        idx = _new_conn()
        idx.execute("INSERT INTO memories (id, content) VALUES (?, ?)",
                    ("m1", "summary"))
        idx.commit()
        fs, _st, _conn = _engine(tmp_path, index_conn=idx)
        assert fs._count_wiki_references("m1") == 0

    def test_no_summary_returns_zero(self, tmp_path):
        palace = tmp_path / "palace"
        palace.mkdir()
        (palace / "a.md").write_text("m1 mentioned")
        fs, _st, _conn = _engine(tmp_path)
        fs._get_index_conn = lambda: None
        # summary 空 → 提前 return 0
        assert fs._count_wiki_references("m1") == 0

    def test_walks_nested_directories(self, tmp_path):
        palace = tmp_path / "palace"
        nested = palace / "sub" / "deep"
        nested.mkdir(parents=True)
        (nested / "a.md").write_text("m1 here")
        idx = _new_conn()
        idx.execute("INSERT INTO memories (id, content) VALUES (?, ?)",
                    ("m1", "summary"))
        idx.commit()
        fs, _st, _conn = _engine(tmp_path, index_conn=idx)
        assert fs._count_wiki_references("m1") == 1


# ─── _get_memory_summary ───────────────────────────────────


class TestGetMemorySummary:
    def test_returns_first_100_chars(self, tmp_path):
        idx = _new_conn()
        content = "x" * 200
        idx.execute("INSERT INTO memories (id, content) VALUES (?, ?)", ("m1", content))
        idx.commit()
        fs, _st, _conn = _engine(tmp_path, index_conn=idx)
        assert fs._get_memory_summary("m1") == "x" * 100

    def test_missing_id_returns_empty(self, tmp_path):
        idx = _new_conn()
        fs, _st, _conn = _engine(tmp_path, index_conn=idx)
        assert fs._get_memory_summary("nope") == ""

    def test_none_conn_returns_empty(self, tmp_path):
        fs, _st, _conn = _engine(tmp_path)
        fs._get_index_conn = lambda: None
        assert fs._get_memory_summary("m1") == ""

    def test_null_content_returns_empty(self, tmp_path):
        idx = _new_conn()
        idx.execute("INSERT INTO memories (id, content) VALUES (?, ?)", ("m1", None))
        idx.commit()
        fs, _st, _conn = _engine(tmp_path, index_conn=idx)
        assert fs._get_memory_summary("m1") == ""

    def test_exception_returns_empty(self, tmp_path):
        fs, _st, _conn = _engine(tmp_path)

        class _Broken:
            def execute(self, *_a, **_kw):
                raise RuntimeError("boom")

        fs._get_index_conn = lambda: _Broken()
        assert fs._get_memory_summary("m1") == ""


# ─── _promote_to_wiki ──────────────────────────────────────


class TestPromoteToWiki:
    def test_sets_upgraded_and_stage(self, tmp_path):
        fs, st, conn = _engine(tmp_path)
        _seed(conn, "m1", heat="hot", recall_count=6, created_days_ago=40)
        assert fs._promote_to_wiki("m1") is True
        row = conn.execute(
            "SELECT upgraded_to_wiki FROM forgetting_state WHERE memory_id='m1'"
        ).fetchone()
        assert row[0] == 1
        assert st.stage["m1"] == "consolidating"
        assert st.track_writes == 1

    def test_failure_returns_false(self, tmp_path, monkeypatch):
        fs, _st, _conn = _engine(tmp_path)
        monkeypatch.setattr(fs, "_get_conn", lambda: _BrokenConn())
        assert fs._promote_to_wiki("m1") is False


# ─── 构造与依赖注入 ────────────────────────────────────────


class TestConstructor:
    def test_all_callbacks_stored(self, tmp_path):
        st = _State()
        conn = _new_conn()
        idx = _new_conn()
        fs = ForgettingScreening(
            governance_dir=tmp_path,
            get_conn=lambda: conn,
            get_index_conn=lambda: idx,
            get_recall_count_in_window=st.get_recall_count_in_window,
            get_heat=st.get_heat,
            set_heat=st.set_heat,
            set_stage=st.set_stage,
            track_write=st.track_write,
            get_pipeline_start_time=st.get_pipeline_start_time,
        )
        assert fs._governance_dir == tmp_path
        assert fs._get_conn() is conn
        assert fs._get_index_conn() is idx

    def test_missing_callbacks_type_error(self, tmp_path):
        with pytest.raises(TypeError):
            ForgettingScreening(governance_dir=tmp_path)  # type: ignore[call-arg]


# ─── 结果字典形状 ─────────────────────────────────────────


class TestResultShapes:
    def test_first_screening_keys(self, tmp_path):
        fs, _st, _conn = _engine(tmp_path)
        out = fs.run_first_screening()
        assert set(out.keys()) == {"hot", "warm", "neutral", "cold", "skipped"}

    def test_second_screening_keys(self, tmp_path):
        fs, _st, _conn = _engine(tmp_path)
        out = fs.run_second_screening()
        assert set(out.keys()) == {"wiki_upgrade", "demoted_to_warm"}
        assert isinstance(out["wiki_upgrade"], list)

    def test_third_consolidation_keys(self, tmp_path):
        fs, _st, _conn = _engine(tmp_path)
        out = fs.run_third_consolidation()
        assert set(out.keys()) == {"promoted", "monitored", "candidates"}

    def test_warm_cooling_returns_int(self, tmp_path):
        fs, _st, _conn = _engine(tmp_path)
        assert isinstance(fs.run_warm_cooling(), int)

    def test_check_for_reactivation_returns_int(self, tmp_path):
        fs, _st, _conn = _engine(tmp_path)
        assert isinstance(fs.check_for_reactivation(), int)
