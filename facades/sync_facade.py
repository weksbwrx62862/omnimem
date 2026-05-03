"""SyncFacade — Saga 协调 + 后台任务 + 内部化记忆。

封装: SagaCoordinator, BackgroundTaskExecutor, MemoryStoreService,
      KVCacheManager, LoRATrainer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omnimem.core.background import BackgroundTaskExecutor
from omnimem.core.saga import SagaCoordinator
from omnimem.core.store_service import MemoryStoreService
from omnimem.internalize.kv_cache import KVCacheManager
from omnimem.internalize.lora_train import LoRATrainer


class SyncFacade:
    """同步与内部化门面：Saga + 后台任务 + 存储服务 + L4 内存。"""

    def __init__(
        self,
        data_dir: Path,
        config: Any,
        session_id: str,
        storage_facade: Any,
        retrieval_facade: Any,
    ):
        gov_dir = data_dir / "governance"

        # Saga 事务协调
        self._saga = SagaCoordinator(
            pending_path=gov_dir / "saga_pending.json"
        )
        # 后台任务执行器
        self._bg_executor = BackgroundTaskExecutor(max_workers=2)

        # 存储服务层
        self._store_service = MemoryStoreService(
            store=storage_facade.store,
            perception=retrieval_facade.perception,
            provenance=None,  # 延迟绑定
            session_id=session_id,
            turn_count=0,
        )

        # L4 内化记忆（延迟初始化）
        self._kv_cache: KVCacheManager | None = None
        self._lora_trainer: LoRATrainer | None = None
        self._internalize_dir = data_dir / "internalize"
        self._config = config
        self._initialized_l4 = False

    @property
    def saga(self) -> SagaCoordinator:
        return self._saga

    @property
    def bg_executor(self) -> BackgroundTaskExecutor:
        return self._bg_executor

    @property
    def store_service(self) -> MemoryStoreService:
        return self._store_service

    @property
    def kv_cache(self) -> KVCacheManager:
        self.init_l4()
        return self._kv_cache

    @property
    def lora_trainer(self) -> LoRATrainer:
        self.init_l4()
        return self._lora_trainer

    def bind_provenance(self, provenance: Any) -> None:
        """延迟绑定溯源追踪器。"""
        self._store_service._provenance = provenance

    def init_l4(self) -> None:
        """延迟初始化 L4 内化记忆。"""
        if self._initialized_l4:
            return
        from omnimem.internalize.kv_cache import KVCacheManager
        from omnimem.internalize.lora_train import LoRATrainer

        self._kv_cache = KVCacheManager(
            self._internalize_dir,
            auto_preload_threshold=self._config.get("kv_cache_threshold", 10),
            max_cache_size=self._config.get("kv_cache_max", 100),
        )
        self._lora_trainer = LoRATrainer(
            self._internalize_dir,
            base_model=self._config.get("lora_base_model", "Qwen2.5-7B"),
            lora_rank=self._config.get("lora_rank", 16),
            lora_alpha=self._config.get("lora_alpha", 32),
        )
        self._initialized_l4 = True

    def close(self) -> None:
        """关闭同步资源。"""
        self._bg_executor.shutdown(wait=True)
        if self._kv_cache:
            self._kv_cache.close()
        if self._lora_trainer:
            self._lora_trainer.close()
