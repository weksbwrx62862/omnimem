"""P1-3 回归：index.db 的并发写入。

线上实况（2026-09-29 报告）：单个 ``check_same_thread=False`` 共享连接在并发写下抛
``InterfaceError: bad parameter or other API misuse``，而这类异常既不在重试名单里、
又被 ``add()`` 的 ``except`` 吞掉 —— 结果是主抽屉写成功、index 行静默缺失（2 596 条），
语义检索永远召不回。
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from omnimem.memory.index import ThreeLevelIndex, _retry_db_op
from omnimem.memory.meta_store import MetaStore


def _add(index: ThreeLevelIndex, memory_id: str, content: str = "内容") -> None:
    index.add(memory_id=memory_id, wing="w", hall="h", room="r", content=content)


def _palace_with_drawers(root, memory_ids) -> None:
    """建一个最小 palace 布局：``palace/<wing>/<hall>/<room>/drawer/<id>.md``。"""
    for mid in memory_ids:
        drawer_dir = root / "w" / "h" / "r" / "drawer"
        drawer_dir.mkdir(parents=True, exist_ok=True)
        (drawer_dir / f"{mid}.md").write_text(f"# {mid}\n", encoding="utf-8")
    return root


class TestConcurrentWrites:
    def test_parallel_writers_all_persist(self, tmp_path):
        """8 线程 × 25 条：一条都不能丢，也不允许出现跨线程连接异常。"""
        index = ThreeLevelIndex(tmp_path / "index")
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def worker(thread_no: int) -> None:
            barrier.wait()
            try:
                for i in range(25):
                    _add(index, f"t{thread_no}-{i}")
            except BaseException as e:  # noqa: BLE001 - 收集线程内异常供主线程断言
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        assert not any(t.is_alive() for t in threads), "并发写入挂起"
        assert errors == [], f"并发写入失败: {errors[:3]}"
        index.flush()

        rows = index.search_all_for_retrieval(limit=10_000)
        assert len(rows) == 200

    def test_flush_from_another_thread_commits_pending_batch(self, tmp_path):
        """攒批计数器/连接是按线程的，flush() 必须能把别的线程那半批一起提交。

        否则写入线程退出（或再没写过）时，那半批随连接回滚 —— 表现成「写成功却查不到」。
        """
        index = ThreeLevelIndex(tmp_path / "index")

        def writer() -> None:
            _add(index, "only-one")  # 远低于 _BATCH_THRESHOLD，不会自己 commit

        t = threading.Thread(target=writer)
        t.start()
        t.join(timeout=30)

        index.flush()
        assert index.get("only-one") is not None

    def test_writer_thread_connection_survives_thread_exit(self, tmp_path):
        index = ThreeLevelIndex(tmp_path / "index")
        t = threading.Thread(target=lambda: _add(index, "from-dead-thread"))
        t.start()
        t.join(timeout=30)
        index.flush()
        assert index.get("from-dead-thread") is not None


class TestUpsertKeepsColumns:
    def test_re_add_preserves_conflict_columns(self, tmp_path):
        """``INSERT OR REPLACE`` 是先删后插，会把 add() 没传列的列抹掉。"""
        index = ThreeLevelIndex(tmp_path / "index")
        _add(index, "m1", content="旧内容")
        assert index.update_field(
            "m1", immediate=True, conflicting_with="m0", is_superseded=1, is_updated=0
        )

        _add(index, "m1", content="新内容")
        index.flush()

        conn = index._conn
        assert conn is not None
        row = conn.execute(
            "SELECT content, conflicting_with, is_superseded FROM memory_index WHERE memory_id='m1'"
        ).fetchone()
        assert row is not None
        assert row[0] == "新内容"
        assert row[1] == "m0", "冲突标记被 REPLACE 抹掉了"
        assert row[2] == 1

    def test_fts_stays_in_sync_after_upsert(self, tmp_path):
        index = ThreeLevelIndex(tmp_path / "index")
        _add(index, "m1", content="苹果派做法")
        _add(index, "m1", content="蓝莓派做法")
        index.flush()

        conn = index._conn
        assert conn is not None
        hits = conn.execute(
            "SELECT rowid FROM memory_index_fts WHERE memory_index_fts MATCH ?", ('"苹果派做法"',)
        ).fetchall()
        assert hits == [], "FTS 残留旧内容（触发器未随 upsert 更新）"


class TestFailureIsVisible:
    def test_add_raises_instead_of_swallowing(self, tmp_path, monkeypatch):
        """索引写失败必须冒泡给 saga —— 吞掉就是 P1-1 那 2 596 条的来源。"""

        class _Boom:
            def __init__(self) -> None:
                self.calls = 0

            def execute(self, *args, **kwargs):
                self.calls += 1
                raise sqlite3.OperationalError("no such table: memory_index")

            def commit(self) -> None:
                return None

        index = ThreeLevelIndex(tmp_path / "index")
        boom = _Boom()
        monkeypatch.setattr(index._local, "conn", boom, raising=False)

        with pytest.raises(sqlite3.OperationalError):
            _add(index, "m1")
        assert boom.calls == 1, "非并发类错误不应被重试放大"

    def test_maybe_commit_rolls_back_batch_on_failure(self, tmp_path):
        """commit 反复失败时必须回滚，否则半批悬在一个不再被提交的事务里。

        回滚不彻底的后果是「下一次提交把上次失败的行一起带出来」，
        即补偿逻辑认为写失败、磁盘上却留着一条残缺行。
        """
        import omnimem.memory.index as index_module

        index = ThreeLevelIndex(tmp_path / "index")
        index._BATCH_THRESHOLD = 1  # 每次写都尝试 commit，失败点可控
        conn = index._conn
        assert conn is not None
        original_retry = index_module._retry_db_op
        fail_commit = {"on": True}

        def retry_except_commit(fn, *args, **kwargs):
            # 绑定方法每次访问都是新对象，只能用 __self__/__name__ 识别 commit 调用
            if getattr(fn, "__name__", "") == "commit" and fail_commit["on"]:
                raise sqlite3.OperationalError("database is locked")
            return original_retry(fn, *args, **kwargs)

        index_module._retry_db_op = retry_except_commit
        try:
            with pytest.raises(sqlite3.OperationalError):
                _add(index, "m1")
        finally:
            fail_commit["on"] = False
            index_module._retry_db_op = original_retry

        _add(index, "m2")
        rows = index.search_all_for_retrieval(limit=100)
        assert [r["memory_id"] for r in rows] == ["m2"]


class TestRetryClassifier:
    @pytest.mark.parametrize(
        "error",
        [
            sqlite3.OperationalError("database is locked"),
            sqlite3.OperationalError("database table is busy"),
            sqlite3.InterfaceError("bad parameter or other API misuse"),
        ],
    )
    def test_transient_errors_are_retried(self, monkeypatch, error):
        monkeypatch.setattr("omnimem.memory.index._DB_RETRY_DELAY", 0.001)
        calls: list[int] = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise error
            return "ok"

        assert _retry_db_op(flaky) == "ok"
        assert len(calls) == 3

    def test_permanent_error_raises_immediately(self, monkeypatch):
        monkeypatch.setattr("omnimem.memory.index._DB_RETRY_DELAY", 0.001)
        calls: list[int] = []

        def always_fail():
            calls.append(1)
            raise sqlite3.OperationalError("no such column: nope")

        with pytest.raises(sqlite3.OperationalError):
            _retry_db_op(always_fail)
        assert len(calls) == 1

    def test_gives_up_after_retry_budget(self, monkeypatch):
        monkeypatch.setattr("omnimem.memory.index._DB_RETRY_DELAY", 0.001)
        calls: list[int] = []

        def always_locked():
            calls.append(1)
            raise sqlite3.OperationalError("database is locked")

        with pytest.raises(sqlite3.OperationalError):
            _retry_db_op(always_locked)
        assert len(calls) == 5, "重试次数应为 _DB_RETRY_COUNT=5（原为 3）"


class TestMetaStoreSyncUnderLock:
    def test_sync_waits_for_a_concurrent_writer(self, tmp_path):
        """sync_from_index 连的是 index.db 的第二条连接，不设 busy_timeout 会直接 locked。"""
        index = ThreeLevelIndex(tmp_path / "index")
        _add(index, "existing")
        index.flush()

        # ★ P1-7：删除判据是磁盘抽屉，不是 MetaStore。给三条记忆都放好抽屉，
        #   于是「existing 有抽屉、无 meta 行」不再算幽灵，只补 meta 缺的两条。
        palace = _palace_with_drawers(tmp_path / "palace", ["existing", "m1", "m2"])

        meta = MetaStore(palace / ".meta", palace_dir=palace)
        for mid in ("m1", "m2"):
            meta.add(
                mid,
                wing="w",
                type="fact",
                room="r",
                summary="s",
                confidence=3,
                privacy="personal",
                stored_at="2026-09-29T00:00:00",
                content_preview="内容",
            )
        meta.flush()

        release = threading.Event()

        def holder() -> None:
            conn = sqlite3.connect(str(index.db_path), check_same_thread=False)
            conn.execute("BEGIN IMMEDIATE")
            release.wait(timeout=5)
            conn.commit()
            conn.close()

        t = threading.Thread(target=holder)
        t.start()
        time.sleep(0.2)  # 让 holder 真正握住写锁
        try:
            started = time.time()
            stale, missing = meta.sync_from_index(index.db_path)
            elapsed = time.time() - started
        finally:
            release.set()
            t.join(timeout=10)

        # existing 有抽屉 → 保留；meta 的 m1/m2 缺 index 行且磁盘有抽屉 → 补 2 条
        assert (stale, missing) == (0, 2), "并发写锁下的同步被静默降级成 (0, 0)"
        assert elapsed >= 0.1, "没有等待锁就直接返回，说明 busy_timeout 未生效"
