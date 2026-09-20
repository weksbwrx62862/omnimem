"""FairReadWriteLock 单测（改进项 #2 覆盖，离线，线程安全且有界超时）。

覆盖：读/写计数在上下文管理器前后归零、多读者并行（Barrier 证明）、写者独占
阻塞读者与读者阻塞写者、写等待计数复位、向后兼容别名。所有跨线程等待均带超时，
避免测试挂死。
"""
from __future__ import annotations

import threading

from omnimem.retrieval.rw_lock import (
    FairReadWriteLock,
    _ReadLockContext,
    _ReadWriteLock,
)


def test_read_context_updates_counters():
    rw = FairReadWriteLock()
    with rw.read_lock():
        assert rw._readers == 1
    assert rw._readers == 0


def test_write_context_manager():
    rw = FairReadWriteLock()
    with rw:
        assert rw._writers == 1
    assert rw._writers == 0
    assert rw._writer_waiting == 0  # 获取后归零


def test_nested_readers():
    rw = FairReadWriteLock()
    with rw.read_lock():
        with rw.read_lock():
            assert rw._readers == 2
        assert rw._readers == 1
    assert rw._readers == 0


def test_concurrent_readers_are_parallel():
    rw = FairReadWriteLock()
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def reader():
        try:
            rw.acquire_read()
            # 若读者不能并行，第二个读者进不来 → barrier 超时
            barrier.wait(timeout=2)
            rw.release_read()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=reader) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=3)
    assert errors == []
    assert rw._readers == 0


def test_writer_blocks_readers():
    rw = FairReadWriteLock()
    rw.acquire_write()
    done = threading.Event()

    def reader():
        rw.acquire_read()
        done.set()
        rw.release_read()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    assert not done.wait(0.1)  # 被写者阻塞
    rw.release_write()
    assert done.wait(1.0)      # 写者释放后读者通过
    t.join(timeout=2)


def test_reader_blocks_writer():
    rw = FairReadWriteLock()
    rw.acquire_read()
    done = threading.Event()

    def writer():
        rw.acquire_write()
        done.set()
        rw.release_write()

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    assert not done.wait(0.1)  # 被读者阻塞
    rw.release_read()
    assert done.wait(1.0)
    t.join(timeout=2)


def test_writer_waiting_counter_resets():
    rw = FairReadWriteLock()
    rw.acquire_write()
    rw.release_write()
    assert rw._writer_waiting == 0 and rw._writers == 0


def test_backward_compat_aliases():
    assert _ReadWriteLock is FairReadWriteLock


def test_read_lock_context_type():
    rw = FairReadWriteLock()
    ctx = rw.read_lock()
    assert isinstance(ctx, _ReadLockContext)
