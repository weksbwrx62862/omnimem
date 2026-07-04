"""公平读写锁实现。

多个读者可并行持有读锁；写者必须独占。
公平锁策略：当写者等待时，限制新读者排队数量（max_readers_waiting），
防止写者饥饿，同时避免读者完全被阻塞。
"""

from __future__ import annotations

import threading


class FairReadWriteLock:
    """公平读写锁实现。"""

    def __init__(self, max_readers_waiting: int = 10) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._readers = 0
        self._writers = 0
        self._writer_waiting = 0
        self._readers_waiting = 0
        self._max_readers_waiting = max_readers_waiting

    def acquire_read(self) -> None:
        with self._cond:
            # 公平锁：写者等待时，限制读者排队数量
            while self._writers > 0 or (
                self._writer_waiting > 0
                and self._readers_waiting >= self._max_readers_waiting
            ):
                self._cond.wait()
            self._readers_waiting += 1
            # 等待写者释放
            while self._writers > 0:
                self._cond.wait()
            self._readers_waiting -= 1
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writer_waiting += 1
            while self._readers > 0 or self._writers > 0:
                self._cond.wait()
            self._writer_waiting -= 1
            self._writers += 1

    def release_write(self) -> None:
        with self._cond:
            self._writers -= 1
            self._cond.notify_all()

    def read_lock(self) -> _ReadLockContext:
        """返回读锁上下文管理器，支持 with 语句获取读锁。"""
        return _ReadLockContext(self)

    def __enter__(self) -> FairReadWriteLock:
        self.acquire_write()
        return self

    def __exit__(self, *args: object) -> None:
        self.release_write()


class _ReadLockContext:
    """读锁上下文管理器，支持 with rw_lock.read_lock() 语法。"""

    def __init__(self, rw_lock: FairReadWriteLock) -> None:
        self._rw_lock = rw_lock

    def __enter__(self) -> _ReadLockContext:
        self._rw_lock.acquire_read()
        return self

    def __exit__(self, *args: object) -> None:
        self._rw_lock.release_read()


# 向后兼容别名
_ReadWriteLock = FairReadWriteLock
_ReadLockContext = _ReadLockContext
