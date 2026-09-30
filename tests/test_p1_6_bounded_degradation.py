"""P1-6 回归：检索路径的失败必须是「降级」，不是「挂起」。

2026-09-29 的检索基准两次死锁（25min / 15min，``futex_do_wait``），根因不是一次慢查询，
而是若干处**无界等待**叠在一起：

- ``_CachedEmbeddingFunction._get_model`` 只在最内层 try 的 finally 里置位就绪事件，
  早期异常（import 失败等）让所有 ``wait_ready()`` 等满超时；
- 多个线程可以各起一次完整模型加载；
- ``FairReadWriteLock`` 的 ``Condition.wait()`` 不带超时，而全量重建持写锁可达分钟级；
- 共享线程池 ``release_shared_executor(wait=True)`` 在持锁状态下 join 卡住的 worker；
- ``recall_strategy == "embedding"`` 分支的向量通道完全没有超时。

本文件的每个测试都带硬超时：如果修复回退，测试会挂住并被 ``pytest-timeout``/外层
``timeout`` 判失败，而不是把套件拖死。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from omnimem.retrieval import executor as executor_module
from omnimem.retrieval.executor import (
    acquire_shared_executor,
    release_shared_executor,
)
from omnimem.retrieval.rw_lock import FairReadWriteLock, LockTimeoutError
from omnimem.retrieval.vector_store import _CachedEmbeddingFunction


@pytest.fixture
def emb_fn(tmp_path) -> _CachedEmbeddingFunction:
    return _CachedEmbeddingFunction(
        cache_path=tmp_path / "emb.db",
        model_path=str(tmp_path / "does-not-exist"),
        load_timeout=2.0,
    )


class TestModelLoadFailureIsBounded:
    def test_import_failure_still_signals_ready(self, emb_fn, monkeypatch):
        """sentence_transformers 导入失败时，等待者必须立刻被唤醒而不是等满超时。"""
        import sys

        monkeypatch.setitem(sys.modules, "sentence_transformers", None)

        with pytest.raises(Exception):
            emb_fn._get_model()

        started = time.monotonic()
        assert emb_fn.wait_ready(timeout=5.0) is False
        assert time.monotonic() - started < 1.0, "就绪事件没有被置位"

    def test_concurrent_loaders_do_not_storm(self, emb_fn, monkeypatch):
        """N 个并发 _get_model 都要在有限时间内返回，且失败结果被缓存（不再重新排队）。"""
        import sys

        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        outcomes: list[bool] = []
        lock = threading.Lock()

        def _probe() -> None:
            try:
                emb_fn._get_model()
                ready = True
            except Exception:
                ready = False
            with lock:
                outcomes.append(ready)

        threads = [threading.Thread(target=_probe) for _ in range(8)]
        started = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(10.0)
        assert not any(t.is_alive() for t in threads), "有等待者没有被唤醒"
        assert time.monotonic() - started < 5.0
        assert outcomes and not any(outcomes)
        # 冷却期内的后续调用立即失败，不再重复起加载
        started = time.monotonic()
        with pytest.raises(Exception):
            emb_fn._get_model()
        assert time.monotonic() - started < 0.5

    def test_slow_load_times_out_and_later_success_is_adopted(self, tmp_path, monkeypatch):
        """加载超过 load_timeout → 本次报错；后台线程随后的成功会被回收，不重复加载。"""
        fn = _CachedEmbeddingFunction(
            cache_path=tmp_path / "emb.db", model_path="x", load_timeout=1.0
        )
        state = {"calls": 0}

        class SlowModel:
            def __init__(self, *args, **kwargs):
                state["calls"] += 1
                time.sleep(2.5)

        module = SimpleNamespace(SentenceTransformer=SlowModel)
        monkeypatch.setitem(
            __import__("sys").modules, "sentence_transformers", module
        )

        with pytest.raises(TimeoutError):
            fn._get_model()
        assert fn.wait_ready(timeout=0.5) is False
        assert state["calls"] == 1

        # 后台线程随后加载完成 —— 下一次调用应直接采用结果，不再起第二个加载
        time.sleep(2.5)
        assert fn._get_model() is not None
        assert state["calls"] == 1, "超时后重复加载了模型"


class TestFairReadWriteLockIsBounded:
    def test_reader_times_out_against_long_writer(self):
        rw = FairReadWriteLock()
        rw.acquire_write()
        try:
            started = time.monotonic()
            with pytest.raises(LockTimeoutError):
                rw.acquire_read(timeout=0.5)
            assert time.monotonic() - started < 2.0
        finally:
            rw.release_write()

    def test_timed_out_reader_does_not_leak_queue(self):
        """超时路径必须退队，否则 readers_waiting 虚高会永久挡住后续读者。"""
        rw = FairReadWriteLock(max_readers_waiting=1)
        rw.acquire_write()
        rw.release_write()
        # 先制造一次超时
        blocked = threading.Thread(target=lambda: _try_read(rw))
        blocked.start()
        blocked.join(5.0)
        rw.acquire_read(timeout=1.0)  # 不应被残留的排队计数卡住
        rw.release_read()

    def test_read_to_write_upgrade_fails_fast(self):
        rw = FairReadWriteLock(write_timeout=30.0)
        rw.acquire_read()
        try:
            started = time.monotonic()
            with pytest.raises(LockTimeoutError):
                rw.acquire_write()
            assert time.monotonic() - started < 0.5, "锁升级应立即判死而非等待"
        finally:
            rw.release_read()

    def test_nested_reads_same_thread_still_work(self):
        rw = FairReadWriteLock()
        with rw.read_lock():
            with rw.read_lock():
                pass
        rw.acquire_write()
        rw.release_write()


def _try_read(rw: FairReadWriteLock) -> None:
    try:
        rw.acquire_read(timeout=0.2)
        rw.release_read()
    except LockTimeoutError:
        pass


class TestSharedExecutorReleaseIsBounded:
    def test_release_does_not_block_on_stuck_worker(self, monkeypatch):
        """worker 卡住时，release 必须有界返回，且不能把锁留给下一个 acquire。"""
        # 与套件内其它用例隔离：共享池是全进程单例，若有别的持有者未释放，
        # 引用计数不会归零，"关闭后换新池" 的断言就会因执行顺序而假失败。
        monkeypatch.setattr(executor_module, "_shared_executor", None, raising=False)
        monkeypatch.setattr(executor_module, "_shared_executor_refs", 0, raising=False)

        executor = acquire_shared_executor(2)
        gate = threading.Event()
        executor.submit(gate.wait, 60)

        started = time.monotonic()
        release_shared_executor(wait=True, timeout=1.0)
        elapsed = time.monotonic() - started
        gate.set()
        assert elapsed < 5.0, f"release 阻塞了 {elapsed:.1f}s"

        # acquire 不能因为上一次关闭还在 join 而挂住
        second = acquire_shared_executor(1)
        try:
            assert second is not executor, "关闭后应返回一个新的池"
        finally:
            release_shared_executor(wait=False)
            second.shutdown(wait=True, cancel_futures=True)


class TestEmbeddingStrategyChannelIsBounded:
    def test_vector_channel_without_timeout_degrades(self, monkeypatch):
        """recall_strategy=embedding 时，慢向量通道必须超时降级而不是同步挂住。"""
        from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator

        orch = object.__new__(HybridOrchestrator)
        orch._executor = ThreadPoolExecutor(max_workers=2)

        class SlowVector:
            _embedding_fn = None

            def search(self, query, top_k=10):
                time.sleep(10)
                return []

        class Breaker:
            def __init__(self):
                self.skips = 0
                self.failures = 0

            def should_skip(self):
                self.skips += 1
                return self.skips > 1  # 首次仍尝试，之后打开

            def record_success(self):
                pass

            def record_failure(self):
                self.failures += 1

        breaker = Breaker()
        orch._facade = SimpleNamespace(
            _recall_strategy="embedding",
            _channels={"vector": (None, 1.0)},
            _vector=SlowVector(),
            _vector_breaker=breaker,
            _recall_timeout_ms=300,
        )
        try:
            started = time.monotonic()
            results = orch.dispatch_channels("q", 5, None, None)
            elapsed = time.monotonic() - started
            assert elapsed < 5.0, f"embedding 分支同步等待了 {elapsed:.1f}s"
            assert results.get("vector") == []
            assert breaker.failures == 1
        finally:
            orch._executor.shutdown(wait=False, cancel_futures=True)
