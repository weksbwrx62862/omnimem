"""
EngramBridge - Plur 共享记忆层集成到 OmniMem
=============================================

将 Plur 的共享记忆能力集成到 OmniMem，解决跨实例记忆同步问题。

核心功能：
1. EngramBridge: 桥接层，连接本地 OmniMem 和 Plur 共享记忆
2. SharedMemorySync: 同步管理器，处理跨实例记忆同步
3. MemoryFederation: 联邦记忆查询，聚合多实例记忆
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """同步状态枚举"""
    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    CONFLICT = "conflict"
    FAILED = "failed"


@dataclass
class Engram:
    """Plur Engram 格式"""
    id: str
    content: str
    memory_type: str
    confidence: float
    source_instance: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]
    tags: list[str]
    relationships: list[str]

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "confidence": self.confidence,
            "source_instance": self.source_instance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "tags": self.tags,
            "relationships": self.relationships
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Engram':
        """从字典创建 Engram"""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=data["memory_type"],
            confidence=data["confidence"],
            source_instance=data["source_instance"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            relationships=data.get("relationships", [])
        )


@dataclass
class SyncResult:
    """同步结果"""
    success: bool
    synced_count: int
    conflict_count: int
    failed_count: int
    conflicts: list[dict[str, Any]]
    errors: list[str]
    duration_ms: float


class EngramBridge:
    """
    EngramBridge - 桥接本地 OmniMem 和 Plur 共享记忆

    核心职责：
    1. 将本地 OmniMem 记忆转换为 Plur Engram 格式
    2. 从 Plur 共享记忆获取外部 Engram
    3. 处理记忆冲突和合并
    4. 维护本地-远程记忆映射
    """

    def __init__(
        self,
        instance_id: str,
        plur_endpoint: str | None = None,
        sync_interval: int = 300,  # 5分钟
        auto_sync: bool = True
    ):
        self.instance_id = instance_id
        self.plur_endpoint = plur_endpoint or "http://localhost:8080"
        self.sync_interval = sync_interval
        self.auto_sync = auto_sync

        # 本地记忆缓存
        self._local_cache: dict[str, Engram] = {}
        self._remote_cache: dict[str, Engram] = {}

        # 同步状态
        self._last_sync: datetime | None = None
        self._sync_status: dict[str, SyncStatus] = {}

        # 冲突队列
        self._conflict_queue: list[dict[str, Any]] = []

        logger.info(f"EngramBridge initialized for instance: {instance_id}")

    def _generate_engram_id(self, content: str, source: str) -> str:
        """生成 Engram ID"""
        raw = f"{source}:{content}:{datetime.now().isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def convert_to_engram(self, omni_memory: dict[str, Any]) -> Engram:
        """将 OmniMem 记忆转换为 Plur Engram 格式"""
        return Engram(
            id=omni_memory.get("memory_id", self._generate_engram_id(
                omni_memory.get("content", ""), self.instance_id
            )),
            content=omni_memory.get("content", ""),
            memory_type=omni_memory.get("type", "fact"),
            confidence=float(omni_memory.get("confidence", 3)),
            source_instance=self.instance_id,
            created_at=datetime.fromisoformat(
                omni_memory.get("stored_at", datetime.now().isoformat())
            ),
            updated_at=datetime.now(),
            metadata={
                "wing": omni_memory.get("wing", "personal"),
                "room": omni_memory.get("room"),
                "privacy": omni_memory.get("privacy", "personal"),
                "original_content": omni_memory.get("original_content", "")
            },
            tags=self._extract_tags(omni_memory.get("content", "")),
            relationships=[]
        )

    def _extract_tags(self, content: str) -> list[str]:
        """从内容中提取标签"""
        tags = []
        # 提取常见的标签模式
        patterns = [
            ("教训", "lesson"),
            ("经验", "experience"),
            ("偏好", "preference"),
            ("纠正", "correction"),
            ("技能", "skill"),
            ("事实", "fact")
        ]

        for cn_tag, en_tag in patterns:
            if cn_tag in content:
                tags.append(en_tag)

        return tags

    async def sync_to_plur(self, memories: list[dict[str, Any]]) -> SyncResult:
        """将本地记忆同步到 Plur 共享记忆"""
        start_time = datetime.now()
        synced_count = 0
        failed_count = 0
        errors = []

        try:
            for memory in memories:
                try:
                    engram = self.convert_to_engram(memory)

                    # 调用 Plur API 存储
                    if await self._store_to_plur(engram):
                        synced_count += 1
                        self._sync_status[engram.id] = SyncStatus.SYNCED
                    else:
                        failed_count += 1
                        self._sync_status[engram.id] = SyncStatus.FAILED

                except Exception as e:
                    failed_count += 1
                    errors.append(f"Failed to sync memory {memory.get('memory_id')}: {str(e)}")
                    logger.error(f"Sync error: {e}")

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._last_sync = datetime.now()

            return SyncResult(
                success=failed_count == 0,
                synced_count=synced_count,
                conflict_count=0,
                failed_count=failed_count,
                conflicts=[],
                errors=errors,
                duration_ms=duration_ms
            )

        except Exception as e:
            logger.error(f"Sync to Plur failed: {e}")
            return SyncResult(
                success=False,
                synced_count=synced_count,
                conflict_count=0,
                failed_count=len(memories) - synced_count,
                conflicts=[],
                errors=[str(e)],
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    async def _store_to_plur(self, engram: Engram) -> bool:
        """存储 Engram 到 Plur（模拟实现）"""
        # 这里应该是实际的 Plur API 调用
        # 模拟实现：存储到本地缓存
        self._remote_cache[engram.id] = engram
        logger.debug(f"Stored engram {engram.id} to Plur")
        return True

    async def fetch_from_plur(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        since: datetime | None = None,
        limit: int = 100
    ) -> list[Engram]:
        """从 Plur 共享记忆获取 Engram"""
        try:
            # 这里应该是实际的 Plur API 调用
            # 模拟实现：从缓存返回
            engrams = list(self._remote_cache.values())

            # 过滤
            if query:
                engrams = [e for e in engrams if query.lower() in e.content.lower()]

            if tags:
                engrams = [e for e in engrams if any(t in e.tags for t in tags)]

            if since:
                engrams = [e for e in engrams if e.updated_at >= since]

            # 限制数量
            engrams = engrams[:limit]

            # 更新本地缓存
            for engram in engrams:
                self._remote_cache[engram.id] = engram

            return engrams

        except Exception as e:
            logger.error(f"Fetch from Plur failed: {e}")
            return []

    async def resolve_conflict(
        self,
        local_engram: Engram,
        remote_engram: Engram,
        strategy: str = "merge"
    ) -> Engram:
        """解决记忆冲突"""
        if strategy == "local":
            return local_engram
        elif strategy == "remote":
            return remote_engram
        elif strategy == "newest":
            return local_engram if local_engram.updated_at >= remote_engram.updated_at else remote_engram
        elif strategy == "highest_confidence":
            return local_engram if local_engram.confidence >= remote_engram.confidence else remote_engram
        elif strategy == "merge":
            # 合并策略：取最新内容，合并标签
            merged_content = local_engram.content
            if remote_engram.updated_at > local_engram.updated_at:
                merged_content = remote_engram.content

            merged_tags = list(set(local_engram.tags + remote_engram.tags))
            merged_confidence = max(local_engram.confidence, remote_engram.confidence)

            return Engram(
                id=local_engram.id,
                content=merged_content,
                memory_type=local_engram.memory_type,
                confidence=merged_confidence,
                source_instance=local_engram.source_instance,
                created_at=min(local_engram.created_at, remote_engram.created_at),
                updated_at=max(local_engram.updated_at, remote_engram.updated_at),
                metadata={**local_engram.metadata, **remote_engram.metadata},
                tags=merged_tags,
                relationships=list(set(local_engram.relationships + remote_engram.relationships))
            )
        else:
            raise ValueError(f"Unknown conflict resolution strategy: {strategy}")

    def get_sync_status(self) -> dict[str, Any]:
        """获取同步状态"""
        return {
            "instance_id": self.instance_id,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "local_cache_size": len(self._local_cache),
            "remote_cache_size": len(self._remote_cache),
            "conflict_queue_size": len(self._conflict_queue),
            "sync_statuses": {
                k: v.value for k, v in self._sync_status.items()
            }
        }


class SharedMemorySync:
    """
    SharedMemorySync - 共享记忆同步管理器

    负责：
    1. 定期同步本地记忆到 Plur
    2. 从 Plur 拉取新记忆
    3. 处理记忆冲突
    4. 维护同步状态
    """

    def __init__(
        self,
        bridge: EngramBridge,
        auto_sync_interval: int = 300,
        conflict_strategy: str = "merge"
    ):
        self.bridge = bridge
        self.auto_sync_interval = auto_sync_interval
        self.conflict_strategy = conflict_strategy

        self._sync_task: asyncio.Task | None = None
        self._is_running = False

        logger.info("SharedMemorySync initialized")

    async def start(self):
        """启动自动同步"""
        if self._is_running:
            logger.warning("Sync already running")
            return

        self._is_running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(f"Auto sync started with interval: {self.auto_sync_interval}s")

    async def stop(self):
        """停止自动同步"""
        self._is_running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        logger.info("Auto sync stopped")

    async def _sync_loop(self):
        """同步循环"""
        while self._is_running:
            try:
                await self.perform_sync()
                await asyncio.sleep(self.auto_sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(60)  # 出错后等待1分钟

    async def perform_sync(self) -> SyncResult:
        """执行一次完整同步"""
        logger.info("Starting sync...")

        # 1. 获取本地记忆（这里应该调用 OmniMem 的 API）
        local_memories = await self._get_local_memories()

        # 2. 同步到 Plur
        push_result = await self.bridge.sync_to_plur(local_memories)

        # 3. 从 Plur 拉取新记忆
        remote_engrams = await self.bridge.fetch_from_plur(
            since=self.bridge._last_sync
        )

        # 4. 处理冲突
        conflicts = []
        for engram in remote_engrams:
            if engram.id in self.bridge._local_cache:
                local = self.bridge._local_cache[engram.id]
                if local.content != engram.content:
                    conflicts.append({
                        "local": local.to_dict(),
                        "remote": engram.to_dict()
                    })

        # 5. 解决冲突
        resolved_count = 0
        for conflict in conflicts:
            local = Engram.from_dict(conflict["local"])
            remote = Engram.from_dict(conflict["remote"])
            resolved = await self.bridge.resolve_conflict(
                local, remote, self.conflict_strategy
            )
            self.bridge._local_cache[resolved.id] = resolved
            resolved_count += 1

        logger.info(f"Sync completed: pushed={push_result.synced_count}, "
                   f"pulled={len(remote_engrams)}, conflicts_resolved={resolved_count}")

        return SyncResult(
            success=push_result.success,
            synced_count=push_result.synced_count,
            conflict_count=len(conflicts),
            failed_count=push_result.failed_count,
            conflicts=conflicts,
            errors=push_result.errors,
            duration_ms=push_result.duration_ms
        )

    async def _get_local_memories(self) -> list[dict[str, Any]]:
        """获取本地记忆（模拟实现）"""
        # 这里应该调用 OmniMem 的实际 API
        # 模拟实现：返回缓存
        return [engram.to_dict() for engram in self.bridge._local_cache.values()]


class MemoryFederation:
    """
    MemoryFederation - 联邦记忆查询

    支持跨多个 OmniMem 实例查询记忆，实现分布式记忆网络。
    """

    def __init__(self):
        self._bridges: dict[str, EngramBridge] = {}
        logger.info("MemoryFederation initialized")

    def register_instance(self, instance_id: str, bridge: EngramBridge):
        """注册实例"""
        self._bridges[instance_id] = bridge
        logger.info(f"Registered instance: {instance_id}")

    async def federated_query(
        self,
        query: str,
        instances: list[str] | None = None,
        limit_per_instance: int = 10
    ) -> dict[str, list[Engram]]:
        """联邦查询：从多个实例查询记忆"""
        target_instances = instances or list(self._bridges.keys())

        results = {}
        tasks = []

        for instance_id in target_instances:
            if instance_id in self._bridges:
                bridge = self._bridges[instance_id]
                task = bridge.fetch_from_plur(query=query, limit=limit_per_instance)
                tasks.append((instance_id, task))

        # 并发查询
        for instance_id, task in tasks:
            try:
                engrams = await task
                results[instance_id] = engrams
            except Exception as e:
                logger.error(f"Federated query failed for {instance_id}: {e}")
                results[instance_id] = []

        return results

    async def aggregate_memories(
        self,
        query: str,
        merge_strategy: str = "confidence_weighted"
    ) -> list[Engram]:
        """聚合多个实例的记忆"""
        all_engrams = []

        # 获取所有实例的记忆
        federated_results = await self.federated_query(query)

        for instance_id, engrams in federated_results.items():
            all_engrams.extend(engrams)

        # 去重（基于 ID）
        seen_ids: set[str] = set()
        unique_engrams = []
        for engram in all_engrams:
            if engram.id not in seen_ids:
                seen_ids.add(engram.id)
                unique_engrams.append(engram)

        # 排序策略
        if merge_strategy == "confidence_weighted":
            unique_engrams.sort(key=lambda e: e.confidence, reverse=True)
        elif merge_strategy == "newest":
            unique_engrams.sort(key=lambda e: e.updated_at, reverse=True)
        elif merge_strategy == "most_relevant":
            # 这里可以实现更复杂的排序算法
            unique_engrams.sort(key=lambda e: e.confidence, reverse=True)

        return unique_engrams

    def get_federation_status(self) -> dict[str, Any]:
        """获取联邦状态"""
        return {
            "registered_instances": list(self._bridges.keys()),
            "instance_count": len(self._bridges),
            "bridges": {
                instance_id: bridge.get_sync_status()
                for instance_id, bridge in self._bridges.items()
            }
        }


# 工厂函数
def create_engram_bridge(
    instance_id: str,
    plur_endpoint: str | None = None,
    **kwargs
) -> EngramBridge:
    """创建 EngramBridge 实例"""
    return EngramBridge(
        instance_id=instance_id,
        plur_endpoint=plur_endpoint,
        **kwargs
    )


def create_shared_memory_sync(
    bridge: EngramBridge,
    **kwargs
) -> SharedMemorySync:
    """创建 SharedMemorySync 实例"""
    return SharedMemorySync(bridge=bridge, **kwargs)


def create_memory_federation() -> MemoryFederation:
    """创建 MemoryFederation 实例"""
    return MemoryFederation()


# 使用示例
async def example_usage():
    """使用示例"""

    # 1. 创建 EngramBridge
    bridge = create_engram_bridge(
        instance_id="hermes-instance-1",
        plur_endpoint="http://localhost:8080"
    )

    # 2. 创建同步管理器
    sync = create_shared_memory_sync(bridge=bridge, auto_sync_interval=300)

    # 3. 启动自动同步
    await sync.start()

    # 4. 手动同步一次
    result = await sync.perform_sync()
    print(f"Sync result: {result}")

    # 5. 查询共享记忆
    engrams = await bridge.fetch_from_plur(query="用户偏好")
    print(f"Found {len(engrams)} engrams")

    # 6. 创建联邦
    federation = create_memory_federation()
    federation.register_instance("instance-1", bridge)

    # 7. 联邦查询
    results = await federation.federated_query("技术知识")
    print(f"Federated results: {results}")

    # 8. 停止同步
    await sync.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
