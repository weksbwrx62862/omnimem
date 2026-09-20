"""共享检索线程池 acquire/release 引用计数单测（改进项 #2 覆盖，离线）。

覆盖：首次创建/规格由首个调用方决定、共享同一实例、引用计数、归零真关闭、
空池 release 幂等、全释放后再获取生成新池。autouse fixture 保证每个用例结束
时排空并关闭池，避免非守护线程残留挂住解释器退出。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import omnimem.retrieval.executor as ex
import pytest


@pytest.fixture(autouse=True)
def _clean_pool(monkeypatch):
    # 测试前：重置模块全局到干净状态
    monkeypatch.setattr(ex, "_shared_executor", None, raising=False)
    monkeypatch.setattr(ex, "_shared_executor_refs", 0, raising=False)
    yield
    # 测试后：关闭任何残留池（join 线程），再把全局复位
    while getattr(ex, "_shared_executor", None) is not None:
        ex.release_shared_executor(wait=True)


def _drain_all():
    while getattr(ex, "_shared_executor", None) is not None:
        ex.release_shared_executor(wait=True)


def test_acquire_creates_pool_first_spec_wins():
    e1 = ex.acquire_shared_executor(2)
    e2 = ex.acquire_shared_executor(99)  # 规格已由首调用方决定
    assert isinstance(e1, ThreadPoolExecutor)
    assert e1 is e2
    assert e1._max_workers == 2
    assert ex._shared_executor_refs == 2
    _drain_all()


def test_release_decrements_then_shuts_down():
    e = ex.acquire_shared_executor(2)
    ex.acquire_shared_executor(2)  # refs=2
    ex.release_shared_executor(wait=True)
    assert ex._shared_executor_refs == 1
    assert ex._shared_executor is e  # 仍存活
    ex.release_shared_executor(wait=True)
    assert ex._shared_executor is None
    assert ex._shared_executor_refs == 0


def test_release_without_pool_is_noop():
    assert ex._shared_executor is None
    ex.release_shared_executor(wait=True)  # 不应抛错
    assert ex._shared_executor is None


def test_reacquire_after_full_release_makes_new_pool():
    e1 = ex.acquire_shared_executor(1)
    _drain_all()
    e2 = ex.acquire_shared_executor(1)
    assert e2 is not e1
    assert ex._shared_executor_refs == 1
    _drain_all()


def test_concurrent_acquire_shares_single_pool():
    import threading

    barrier = threading.Barrier(4)
    got: list[ThreadPoolExecutor] = []
    lock = threading.Lock()

    def worker():
        barrier.wait(timeout=3)
        e = ex.acquire_shared_executor(4)
        with lock:
            got.append(e)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=3)
    assert len(got) == 4
    assert len({id(e) for e in got}) == 1  # 全部同一实例
    assert ex._shared_executor_refs == 4
    _drain_all()
