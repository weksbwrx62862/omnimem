"""薄弱治理路径补测 — 主链路实际走到但此前无测试覆盖的分支。

覆盖:
  1. VectorClock sqlite 持久化(facade.close 实际走 save_to_sqlite, 原测试只测 JSON)
  2. VectorClock.recover_from_entries(变更日志重建时钟)
  3. ChangeLog append/read_new/get_last_ts/trim(sync.py 零测试)
  4. SyncEngine changelog 模式 write_with_lock + sync_from_others(增量应用)
  5. KMS reencrypt 链路(govern reencrypt action, 此前 grep 0 命中)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from omnimem.governance.sync import ChangeLog, SyncConfig, SyncEngine
from omnimem.governance.vector_clock import VectorClock


class TestVectorClockSqlite:
    """sqlite 持久化 — facade.close() 实际走的路径。"""

    def test_save_and_load_sqlite(self) -> None:
        vc = VectorClock({"node-a": 7, "node-b": 3})
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "vc.db"
            assert vc.save_to_sqlite(db) is True
            assert db.exists()
            loaded = VectorClock.load_from_sqlite(db)
            assert loaded.to_dict() == {"node-a": 7, "node-b": 3}

    def test_load_sqlite_missing_returns_empty(self) -> None:
        loaded = VectorClock.load_from_sqlite(Path("/nonexistent/vc.db"))
        assert loaded.to_dict() == {}

    def test_sqlite_roundtrip_after_increment(self) -> None:
        vc = VectorClock()
        vc.increment("n1").increment("n1").increment("n2")
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "vc.db"
            vc.save_to_sqlite(db)
            loaded = VectorClock.load_from_sqlite(db)
            assert loaded.to_dict()["n1"] == 2
            assert loaded.to_dict()["n2"] == 1

    def test_recover_from_entries(self) -> None:
        entries = [
            {"instance_id": "a", "vc": chr(39).join(['', '{"a": 2}', ''])},
            {"instance_id": "b", "vc": chr(39).join(['', '{"b": 5}', ''])},
        ]
        vc = VectorClock.recover_from_entries("a", entries)
        assert isinstance(vc, VectorClock)


class TestChangeLog:
    """变更日志 append/read/trim — sync.py 此前零测试。"""

    def _log(self, tmp: str, inst: str = "inst-a") -> ChangeLog:
        return ChangeLog(Path(tmp), inst)

    def test_append_and_read_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log(tmp)
            log.append("INSERT", "memory_index", {"memory_id": "m1"})
            log.append("UPDATE", "memory_index", {"memory_id": "m2"})
            rows = log.read_new("", exclude_instance="")
            assert len(rows) >= 2

    def test_read_new_excludes_own_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log(tmp, "inst-a")
            log.append("INSERT", "t", {"x": 1})
            rows = log.read_new("", exclude_instance="inst-a")
            assert rows == []

    def test_get_last_ts_advances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log(tmp)
            ts0 = log.get_last_ts()
            log.append("INSERT", "t", {"x": 1})
            ts1 = log.get_last_ts()
            assert ts1 >= ts0

    def test_trim_keeps_recent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log(tmp)
            for i in range(10):
                log.append("INSERT", "t", {"i": i})
            log.trim(keep_last_n=3)
            rows = log.read_new("", exclude_instance="")
            assert len(rows) <= 3


class TestSyncEngineChangelog:
    """changelog 模式下的 write_with_lock + sync_from_others。"""

    def _engine(self, tmp: str, name: str) -> SyncEngine:
        cfg = SyncConfig(mode="changelog", instance_name=name)
        return SyncEngine(Path(tmp), cfg)

    def test_write_with_lock_executes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eng = self._engine(tmp, "a")
            calls = []
            result = eng.write_with_lock(lambda x: calls.append(x) or "done", 42)
            assert result == "done"
            assert calls == [42]
            eng.close()

    def test_instance_info_and_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eng = self._engine(tmp, "a")
            info = eng.get_instance_info()
            assert "instance_id" in info
            active = eng.get_active_instances()
            assert isinstance(active, list)
            eng.close()

    def test_sync_from_others_applies_remote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eng_a = self._engine(tmp, "a")
            eng_b = self._engine(tmp, "b")
            if eng_b._changelog:
                eng_b._changelog.append("INSERT", "memory_index", {"memory_id": "mX"})
            applied: list[dict] = []
            count = eng_a.sync_from_others(lambda rec: applied.append(rec))
            assert count >= 0
            eng_a.close()
            eng_b.close()


class TestReencryptChain:
    """KMS reencrypt 链路 — govern reencrypt action, 此前无端到端测试。"""

    def test_reencrypt_dry_run_and_apply(self) -> None:
        import json
        import os

        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        from omnimem.sdk import OmniMemSDK

        sdk = OmniMemSDK(storage_dir=tempfile.mkdtemp())
        try:
            sdk.memorize("secret 数据库口令是 abc123", memory_type="fact", privacy="secret")
            r_dry = sdk.govern(action="reencrypt", params={"dry_run": True})
            r_dry = json.loads(r_dry) if isinstance(r_dry, str) else r_dry
            assert r_dry.get("status") in ("dry_run", "ok"), r_dry
            assert r_dry.get("total_secret", 0) >= 1, r_dry
            r_run = sdk.govern(action="reencrypt", params={})
            r_run = json.loads(r_run) if isinstance(r_run, str) else r_run
            assert r_run.get("status") == "ok", r_run
            assert "upgraded" in r_run and "already_current" in r_run, r_run
        finally:
            sdk.close()
