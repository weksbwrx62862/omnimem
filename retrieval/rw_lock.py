"""公平读写锁实现。

多个读者可并行持有读锁；写者必须独占。
公平锁策略：当写者等待时，限制新读者排队数量（max_readers_waiting），
防止写者饥饿，同时避免读者完全被阻塞。

★ P1-6：所有等待都是有界的。原先 ``Condition.wait()`` 不带超时，一旦写者长期持锁
（例如 ``rebuild_all_from_entries`` 在整个重建期间持有写锁，可达分钟级），gateway 的
每一轮对话都会在 ``acquire_read()`` 上无限挂起 —— 表现是"检索变慢"而不是"检索失败"。
现在超时抛 ``LockTimeoutError``，由调用方决定降级；同一线程的读→写升级直接判死，
不再等待一个永远不会到来的唤醒。
"""

from __future__ import annotations

import threading
import time
from collections import Counter


class LockTimeoutError(TimeoutError):
    """等待读写锁超时/必然自死锁 —— 调用方应当降级而不是继续等。"""


class FairReadWriteLock:
    """公平读写锁实现。"""

    #: 读者默认最多等 30s（一轮对话的检索预算远小于此）
    DEFAULT_READ_TIMEOUT = 30.0
    #: 写者默认最多等 120s（写者之间需要排队，但也要有界）
    DEFAULT_WRITE_TIMEOUT = 120.0

    def __init__(
        self,
        max_readers_waiting: int = 10,
        read_timeout: float | None = None,
        write_timeout: float | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._readers = 0
        self._writers = 0
        self._writer_waiting = 0
        self._readers_waiting = 0
        self._max_readers_waiting = max_readers_waiting
        self._read_timeout = self.DEFAULT_READ_TIMEOUT if read_timeout is None else read_timeout
        self._write_timeout = self.DEFAULT_WRITE_TIMEOUT if write_timeout is None else write_timeout
        # 每线程持读锁次数（用于识别不可重入的读→写升级）
        self._read_counts: Counter[int] = Counter()
        self._write_owner: int | None = None

    def acquire_read(self, timeout: float | None = None) -> None:
        wait = self._read_timeout if timeout is None else timeout
        ident = threading.get_ident()
        deadline = time.monotonic() + wait
        with self._cond:
            if self._write_owner == ident:
                raise LockTimeoutError("同一线程持写锁再取读锁会自死锁")
            # 公平锁：写者等待时，限制读者排队数量
            self._readers_waiting += 1
            try:
                while self._writers > 0 or (
                    self._writer_waiting > 0
                    and self._readers_waiting > self._max_readers_waiting
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LockTimeoutError(
                            f"等待读锁超过 {wait:.0f}s（写者持锁中；"
                            f"readers={self._readers} writers={self._writers} "
                            f"writer_waiting={self._writer_waiting}）"
                        )
                    self._cond.wait(remaining)
                self._readers += 1
                self._read_counts[ident] += 1
            finally:
                # ★ 超时路径也必须退队，否则排队计数虚高会把后来的读者永久挡在门外
                self._readers_waiting -= 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            ident = threading.get_ident()
            if self._read_counts.get(ident):
                self._read_counts[ident] -= 1
                if not self._read_counts[ident]:
                    del self._read_counts[ident]
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self, timeout: float | None = None) -> None:
        wait = self._write_timeout if timeout is None else timeout
        ident = threading.get_ident()
        deadline = time.monotonic() + wait
        with self._cond:
            if self._write_owner == ident:
                raise LockTimeoutError("写锁不可重入：同一线程重复取写锁会自死锁")
            if self._read_counts.get(ident):
                raise LockTimeoutError("不支持读→写升级：先 release_read 再 acquire_write")
            self._writer_waiting += 1
            try:
                while self._readers > 0 or self._writers > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LockTimeoutError(
                            f"等待写锁超过 {wait:.0f}s（readers={self._readers} "
                            f"writers={self._writers}）"
                        )
                    self._cond.wait(remaining)
                self._writers += 1
                self._write_owner = ident
            finally:
                self._writer_waiting -= 1

    def release_write(self) -> None:
        with self._cond:
            self._writers -= 1
            self._write_owner = None
            self._cond.notify_all()

    def read_lock(self, timeout: float | None = None) -> _ReadLockContext:
        """返回读锁上下文管理器，支持 with 语句获取读锁。"""
        return _ReadLockContext(self, timeout)

    def __enter__(self) -> FairReadWriteLock:
        self.acquire_write()
        return self

    def __exit__(self, *args: object) -> None:
        self.release_write()


class _ReadLockContext:
    """读锁上下文管理器，支持 with rw_lock.read_lock() 语法。"""

    def __init__(self, rw_lock: FairReadWriteLock, timeout: float | None = None) -> None:
        self._rw_lock = rw_lock
        self._timeout = timeout

    def __enter__(self) -> _ReadLockContext:
        self._rw_lock.acquire_read(self._timeout)
        return self

    def __exit__(self, *args: object) -> None:
        self._rw_lock.release_read()


# 向后兼容别名
_ReadWriteLock = FairReadWriteLock
_ReadLockContext = _ReadLockContext
