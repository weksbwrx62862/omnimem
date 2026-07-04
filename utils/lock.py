"""分布式锁抽象层。

为单实例、单主机多进程、多主机分布式场景提供统一的 LockProvider 抽象。
默认使用基于文件/fcntl 的 FileLockProvider；RedisLockProvider 为分布式场景预留接口。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# fcntl 仅在 Unix 平台可用，Windows 上降级为线程锁
try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False


class LockProvider(ABC):
    """锁提供者抽象基类。

    所有实现必须支持：
      - acquire(timeout, exclusive): 获取锁
      - release(): 释放锁
      - close(): 清理资源
    """

    @abstractmethod
    def acquire(self, timeout: float = 5.0, exclusive: bool = True) -> bool:
        """获取锁。

        Args:
            timeout: 超时时间（秒）
            exclusive: True 表示排他锁，False 表示共享锁

        Returns:
            是否成功获取锁
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """释放锁。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """关闭锁并释放相关资源。"""
        ...

    def __enter__(self) -> LockProvider:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class FileLockProvider(LockProvider):
    """基于 fcntl 的跨进程文件锁（Unix）；Windows 上降级为进程内线程锁。

    适用于单主机多进程场景，防止多个进程同时写入 SQLite 等共享存储。
    """

    def __init__(self, lock_path: str | Path):
        self._lock_path = Path(lock_path)
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.touch(exist_ok=True)
        self._fd: int | None = None
        self._lock_count = 0
        self._wait_time = 0.0
        self._has_fcntl = _HAS_FCNTL
        # Windows 或无法使用 fcntl 时降级为线程锁（仅进程内有效）
        self._fallback_lock = threading.Lock()
        if not self._has_fcntl:
            logger.warning(
                "fcntl 不可用 — FileLockProvider 降级为 threading.Lock（仅进程内互斥）"
            )

    def acquire(self, timeout: float = 5.0, exclusive: bool = True) -> bool:
        if not self._has_fcntl:
            self._fallback_lock.acquire()
            self._lock_count += 1
            return True

        if self._fd is None:
            self._fd = os.open(str(self._lock_path), os.O_RDWR)

        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH  # type: ignore[attr-defined]
        start_time = time.monotonic()
        while True:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)  # type: ignore[attr-defined]
                self._lock_count += 1
                return True
            except OSError:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    self._wait_time += elapsed
                    return False
                time.sleep(0.05)

    def release(self) -> None:
        if not self._has_fcntl:
            self._fallback_lock.release()
            return
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)  # type: ignore[attr-defined]
            except OSError as e:
                logger.warning("FileLockProvider release failed: %s", e)

    def stats(self) -> dict[str, Any]:
        """获取锁统计信息。"""
        return {
            "acquisitions": self._lock_count,
            "total_wait_time_ms": round(self._wait_time * 1000, 2),
        }

    def close(self) -> None:
        self.release()
        if self._has_fcntl and self._fd is not None:
            try:
                os.close(self._fd)
            except OSError as e:
                logger.warning("FileLockProvider close failed: %s", e)
            self._fd = None


# Lua 脚本：原子 CAS 释放锁（仅当值为当前持有者标识时才删除）
_LUA_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class RedisLockProvider(LockProvider):
    """基于 Redis 的分布式锁（预留接口）。

    当前仅提供接口占位，实际 Redlock 实现可在后续迭代中补充。
    未安装 redis 时抛出明确错误。
    """

    def __init__(
        self,
        lock_name: str,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 10,
    ):
        self._lock_name = lock_name
        self._redis_url = redis_url
        self._ttl_seconds = ttl_seconds
        self._redis: Any = None
        self._locked = False
        self._identifier: str | None = None
        self._release_script: Any = None

    def _ensure_runtime(self) -> Any:
        """确保 redis 库已安装。"""
        try:
            import redis
        except ImportError as e:
            raise RuntimeError(
                "redis 未安装，无法使用 RedisLockProvider。"
                "请执行: pip install redis"
            ) from e
        return redis

    def _get_client(self) -> Any:
        if self._redis is None:
            redis = self._ensure_runtime()
            self._redis = redis.from_url(self._redis_url)
        return self._redis

    def acquire(self, timeout: float = 5.0, exclusive: bool = True) -> bool:
        """尝试获取 Redis 分布式锁（当前使用简单 SET NX 实现）。"""
        _ = exclusive  # 共享锁语义待后续扩展
        client = self._get_client()
        self._identifier = f"{os.getpid()}-{threading.current_thread().ident}"
        start_time = time.monotonic()
        while True:
            try:
                acquired = client.set(
                    self._lock_name,
                    self._identifier,
                    nx=True,
                    ex=self._ttl_seconds,
                )
                if acquired:
                    self._locked = True
                    return True
            except Exception as e:
                logger.warning("RedisLockProvider acquire failed: %s", e)

            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                return False
            time.sleep(0.05)

    def release(self) -> None:
        """原子释放锁（Lua CAS 脚本，防止误释放其他持有者的锁）。"""
        if not self._locked:
            return
        client = self._get_client()
        try:
            if self._release_script is None:
                self._release_script = client.register_script(_LUA_RELEASE_SCRIPT)
            result = self._release_script(
                keys=[self._lock_name],
                args=[self._identifier or ""],
            )
            if result == 0:
                logger.warning(
                    "RedisLockProvider: 锁已被其他进程持有或已过期 (lock=%s)",
                    self._lock_name,
                )
        except Exception as e:
            logger.warning("RedisLockProvider release failed: %s", e)
        finally:
            self._locked = False
            self._identifier = None

    def close(self) -> None:
        self.release()
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception as e:
                logger.warning("RedisLockProvider close failed: %s", e)
            self._redis = None


def create_lock_provider(
    lock_path: str | Path,
    backend: str = "file",
    **kwargs: Any,
) -> LockProvider:
    """根据后端类型构造 LockProvider。

    Args:
        lock_path: 文件锁路径（file 后端必填）
        backend: "file" 或 "redis"
        **kwargs: 后端特定参数

    Returns:
        LockProvider 实例
    """
    if backend == "file":
        return FileLockProvider(lock_path)
    if backend == "redis":
        return RedisLockProvider(**kwargs)
    raise ValueError(f"不支持的 lock backend: {backend}")
