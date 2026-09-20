"""FileLockProvider / RedisLockProvider 单测（改进项 #2 覆盖，离线）。

补齐 test_abstractions 未覆盖的具体行为：fcntl 路径的重入计数与 stats、
排他/共享锁竞争与超时、release/close 幂等、上下文管理器；Redis 后端惰性依赖。
跨进程文件锁依赖 fcntl（仅 Unix）；缺失时相关用例跳过。
"""
from __future__ import annotations

from typing import Any

import pytest
from omnimem.utils.lock import (
    _HAS_FCNTL,
    FileLockProvider,
    LockProvider,
    RedisLockProvider,
    create_lock_provider,
)

fcntl_required = pytest.mark.skipif(not _HAS_FCNTL, reason="fcntl 不可用（非 Unix 平台）")


# ── 工厂 ──────────────────────────────────────────────
def test_factory_file_backend(tmp_path: Any):
    p = create_lock_provider(tmp_path / "x.lock", backend="file")
    assert isinstance(p, FileLockProvider)
    assert isinstance(p, LockProvider)


def test_factory_redis_backend_is_lazy(tmp_path: Any):
    # 构造不应连接 Redis（仅 acquire 时才 _get_client）
    p = create_lock_provider(tmp_path / "x.lock", backend="redis", lock_name="n")
    assert isinstance(p, RedisLockProvider)


def test_factory_unsupported(tmp_path: Any):
    with pytest.raises(ValueError):
        create_lock_provider(tmp_path / "x.lock", backend="nope")


# ── FileLockProvider 计数与统计 ───────────────────────
@fcntl_required
def test_reentrant_acquire_counts_acquisitions(tmp_path: Any):
    p = FileLockProvider(tmp_path / "a.lock")
    assert p.acquire(timeout=1) is True
    assert p.acquire(timeout=1) is True  # 同一 fd 再次 flock 成功
    assert p.stats()["acquisitions"] == 2
    p.close()


@fcntl_required
def test_stats_shape(tmp_path: Any):
    p = FileLockProvider(tmp_path / "a.lock")
    p.acquire(timeout=1)
    st = p.stats()
    assert set(st) == {"acquisitions", "total_wait_time_ms"}
    assert isinstance(st["total_wait_time_ms"], float)
    p.close()


# ── 锁竞争语义（同一文件、两个独立 fd）────────────────
@fcntl_required
def test_exclusive_contention_times_out_then_succeeds(tmp_path: Any):
    path = tmp_path / "c.lock"
    p1 = FileLockProvider(path)
    p2 = FileLockProvider(path)
    assert p1.acquire(timeout=1, exclusive=True) is True
    # p1 持排他锁 → p2 立即超时失败
    assert p2.acquire(timeout=0, exclusive=True) is False
    p1.release()
    # 释放后 p2 可获取
    assert p2.acquire(timeout=0, exclusive=True) is True
    p1.close()
    p2.close()


@fcntl_required
def test_shared_locks_are_compatible(tmp_path: Any):
    path = tmp_path / "s.lock"
    p1 = FileLockProvider(path)
    p2 = FileLockProvider(path)
    assert p1.acquire(timeout=1, exclusive=False) is True
    assert p2.acquire(timeout=0, exclusive=False) is True  # 共享可并存
    p1.close()
    p2.close()


@fcntl_required
def test_exclusive_blocked_by_held_shared(tmp_path: Any):
    path = tmp_path / "hs.lock"
    p1 = FileLockProvider(path)
    p2 = FileLockProvider(path)
    assert p1.acquire(timeout=1, exclusive=False) is True
    assert p2.acquire(timeout=0, exclusive=True) is False  # 有共享者时排他失败
    p1.close()
    p2.close()


# ── release / close 幂等 & 上下文管理器 ───────────────
@fcntl_required
def test_close_then_reacquire(tmp_path: Any):
    p = FileLockProvider(tmp_path / "r.lock")
    p.acquire(timeout=1)
    p.close()
    p.close()  # 二次 close 幂等，不抛错
    assert p.acquire(timeout=1) is True  # fd 重新打开
    p.close()


@fcntl_required
def test_context_manager(tmp_path: Any):
    path = tmp_path / "cm.lock"
    with FileLockProvider(path) as p:
        assert p.acquire.__self__ is p  # 已进入
        assert p.stats()["acquisitions"] >= 1
    # 退出后同文件可再次独占获取
    p2 = FileLockProvider(path)
    assert p2.acquire(timeout=0, exclusive=True) is True
    p2.close()


def test_release_without_acquire_is_safe(tmp_path: Any):
    p = FileLockProvider(tmp_path / "z.lock")
    p.release()  # 未持锁释放不应抛错
    p.close()


# ── Redis 后端惰性依赖 ────────────────────────────────
def test_redis_acquire_without_lib_raises(tmp_path: Any):
    try:
        import redis  # noqa: F401
        pytest.skip("redis 已安装，无法验证缺失依赖路径")
    except ImportError:
        p = RedisLockProvider("lock", redis_url="redis://127.0.0.1:6399/0")
        with pytest.raises(RuntimeError):
            p.acquire(timeout=0)
