"""分布式同步协调模块。

将向量时钟与分布式同步逻辑从业务模块中独立出来，为多实例 OmniMem 提供：
  - VectorClock: 逻辑时钟、因果比较、冲突检测
  - DistributedSyncCoordinator: 协调文件锁、变更日志、向量时钟的高层组件

设计原则：
  - 零外部依赖: 默认使用文件锁与本地变更日志
  - 向后兼容: 默认 mode="none"，不改变现有行为
  - 可扩展: 锁后端可替换为 Redis 等分布式实现
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from omnimem.governance.sync import SyncConfig, SyncEngine
from omnimem.governance.vector_clock import VectorClock, detect_conflict, merge_records
from omnimem.utils.lock import FileLockProvider, LockProvider

logger = logging.getLogger(__name__)

__all__ = [
    "VectorClock",
    "detect_conflict",
    "merge_records",
    "DistributedSyncCoordinator",
]


class DistributedSyncCoordinator:
    """分布式同步协调器。

    将向量时钟、锁、变更日志组合在一起，为 facade 提供统一的同步入口。
    底层委托现有的 SyncEngine / ChangeLog，但抽象了锁后端的选择。
    """

    def __init__(
        self,
        data_dir: Path,
        config: SyncConfig | None = None,
        lock_provider: LockProvider | None = None,
    ):
        self._data_dir = data_dir
        self._config = config or SyncConfig()
        self._sync_engine = SyncEngine(data_dir, self._config)

        # 允许外部注入锁后端；默认使用文件锁
        self._lock_provider = lock_provider or FileLockProvider(
            self._data_dir / "locks" / "omnimem.lock"
        )

        # 向量时钟优先从 SQLite 加载，回退 JSON 文件
        self._vc_db_path = self._data_dir / "governance" / "vector_clock.db"
        self._vc_json_path = self._data_dir / "governance" / "vector_clock.json"
        if self._vc_db_path.exists():
            self._vector_clock = VectorClock.load_from_sqlite(self._vc_db_path)
        elif self._vc_json_path.exists():
            self._vector_clock = VectorClock.load(self._vc_json_path)
        else:
            self._vector_clock = VectorClock()

        logger.info(
            "DistributedSyncCoordinator initialized: mode=%s, instance=%s",
            self._config.mode,
            self._config.instance_id,
        )

    @property
    def vector_clock(self) -> VectorClock:
        """当前实例的向量时钟。"""
        return self._vector_clock

    @property
    def sync_engine(self) -> SyncEngine:
        """底层同步引擎。"""
        return self._sync_engine

    @property
    def lock_provider(self) -> LockProvider:
        """当前使用的锁后端。"""
        return self._lock_provider

    def increment_clock(self, node_id: str | None = None) -> VectorClock:
        """递增向量时钟并返回。"""
        node_id = node_id or self._config.instance_id
        self._vector_clock.increment(node_id)
        return self._vector_clock

    def is_conflict(self, remote_vc: VectorClock) -> bool:
        """判断远程向量时钟是否与本地方并发冲突。"""
        return detect_conflict(self._vector_clock, remote_vc)

    def merge_vector_clock(self, remote_vc: VectorClock) -> VectorClock:
        """合并远程向量时钟到本地并返回合并结果。"""
        self._vector_clock = self._vector_clock.merge(remote_vc)
        return self._vector_clock

    def write_with_lock(
        self,
        write_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """使用注入的锁后端执行写入，并委托 SyncEngine 记录变更日志。"""
        if self._config.mode == "none":
            return write_fn(*args, **kwargs)

        acquired = self._lock_provider.acquire(timeout=5.0)
        if not acquired:
            logger.warning("DistributedSyncCoordinator: failed to acquire lock")
            return None

        try:
            result = write_fn(*args, **kwargs)
            # 变更日志由 SyncEngine 统一处理
            if self._sync_engine._changelog and isinstance(result, dict) and "sync_log" in result:
                sync_info = result["sync_log"]
                self._sync_engine._changelog.append(
                    operation=sync_info.get("operation", "UPDATE"),
                    table=sync_info.get("table", "unknown"),
                    data=sync_info.get("data", {}),
                )
            return result
        finally:
            self._lock_provider.release()

    def sync_from_others(
        self,
        apply_fn: Callable[..., Any],
        since_ts: str = "",
        get_local_fn: Callable[[str], dict[str, Any] | None] | None = None,
    ) -> int:
        """从其他实例同步变更。"""
        return self._sync_engine.sync_from_others(apply_fn, since_ts, get_local_fn)

    def get_instance_info(self) -> dict[str, Any]:
        """获取当前实例信息。"""
        info = self._sync_engine.get_instance_info()
        info["vector_clock"] = self._vector_clock.to_dict()
        return info

    def get_active_instances(self) -> list[dict[str, Any]]:
        """获取活跃实例列表。"""
        return self._sync_engine.get_active_instances()

    def save_vector_clock(self) -> bool:
        """持久化向量时钟到 SQLite。"""
        try:
            return self._vector_clock.save_to_sqlite(self._vc_db_path)
        except Exception as e:
            logger.warning("DistributedSyncCoordinator save_vector_clock failed: %s", e)
            return False

    def close(self) -> None:
        """关闭同步资源并持久化向量时钟。"""
        self.save_vector_clock()
        self._sync_engine.close()
        self._lock_provider.close()
