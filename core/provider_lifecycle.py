"""Provider 生命周期管理：initialize、shutdown、async 包装、会话结束等。"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from omnimem.core.memory_monitor import MemoryMonitor
from omnimem.core.warmup_manager import WarmupManager
from omnimem.handlers.memorize import shutdown_background_executor

logger = logging.getLogger(__name__)


class ProviderLifecycleMixin:
    """负责 Provider 的 initialize、shutdown、async 包装与会话生命周期钩子。"""

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        hermes_home = kwargs.get("hermes_home", str(Path.home() / ".hermes"))
        platform = kwargs.get("platform", "cli")
        agent_context = kwargs.get("agent_context", "primary")

        self._should_write = agent_context == "primary"

        self._data_dir = Path(hermes_home) / "omnimem"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id

        from omnimem.config import OmniMemConfig

        self._config = OmniMemConfig(self._data_dir)

        # 降级模式：跳过向量检索和 ChromaDB，仅 BM25 检索
        if self._degraded_mode:
            logger.warning("OmniMem: 降级模式 — 向量检索和 reranker 不可用，仅 BM25 检索")
            self._init_l1()
            logger.info(
                "OmniMem initialized (degraded): session=%s, platform=%s, data_dir=%s, BM25-only",
                session_id,
                platform,
                self._data_dir,
            )
            return

        # 阶段1: 核心同步初始化（快速返回，让 agent 尽早就绪）
        self._init_l1()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self._init_store): "store",
                executor.submit(self._init_retrieval): "retrieval",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning("Init %s failed: %s", name, e)

        try:
            self._init_governance_sync_services()
        except Exception as e:
            logger.warning("Init governance_sync_services failed: %s", e)

        logger.info(
            "OmniMem initialized: session=%s, platform=%s, data_dir=%s, L3=enabled, L4=enabled",
            session_id,
            platform,
            self._data_dir,
        )

        self._memory_monitor = MemoryMonitor(
            interval=self._config.get("memory_monitor_interval", 60.0),
            warning_mb=self._config.get("memory_warning_mb", 500.0),
        )
        self._memory_monitor.start()

        # 阶段2: 后台异步预热（重活丢到 worker 线程，不阻塞对话启动）
        self._warmup_manager = WarmupManager(
            init_reflect_fn=self._init_reflect,
            init_lora_fn=self._init_lora,
            index=self._index,
            store=self._store,
            retriever=self._retriever,
            retrieval=self._retrieval,
            auditor=self._auditor,
        )
        t_bg = threading.Thread(
            target=self._warmup_manager.run, daemon=True, name="omnimem_bg_warmup"
        )
        t_bg.start()

        # 后台回填: 同步已有 KG 三元组到 Temporal KG
        t_backfill = threading.Thread(
            target=self._backfill_temporal_kg, daemon=True, name="omnimem_backfill_tkg"
        )
        t_backfill.start()

        try:
            step_actions = {
                "three_level_index": lambda mid: self._retry_index_add(mid),
                "retriever": lambda mid: self._retry_retriever_add(mid),
                "knowledge_graph": lambda mid: self._retry_kg_extract(mid),
            }
            if self._saga is not None:
                fixed = self._saga.auto_retry_pending(step_actions)
                if fixed > 0:
                    logger.info("OmniMem: auto-retried %d pending saga records", fixed)
        except Exception as e:
            logger.warning("OmniMem: saga auto-retry failed (non-fatal): %s", e)

    def _backfill_temporal_kg(self) -> None:
        """启动时将已有 KG 三元组同步到 Temporal KG（后台执行，仅首次运行时有数据）。"""
        try:
            import time as _time

            # 等待 warmup 完成（_knowledge_graph 在 _init_reflect 中初始化）
            for _ in range(30):
                if hasattr(self, "_knowledge_graph") and self._knowledge_graph is not None:
                    break
                _time.sleep(1)
            else:
                return

            temporal_kg = getattr(self, "_temporal_kg", None)
            if temporal_kg is None:
                return

            # 检查 temporal_kg 是否已有数据，有则跳过
            existing = temporal_kg.query_current("", "")
            if existing:
                return

            from datetime import datetime, timezone

            triples = self._knowledge_graph._get_all_triples(limit=5000)
            if not triples:
                return

            now = datetime.now(timezone.utc).isoformat()
            synced = 0
            for t in triples:
                try:
                    temporal_kg.add_triple_from_kg(
                        subject=t["subject"],
                        predicate=t["predicate"],
                        obj=t["object"],
                        valid_at=now,
                        source_memory_id=t.get("source_memory_id", "startup_backfill"),
                        confidence=int(t.get("confidence", 3)),
                    )
                    synced += 1
                except Exception:
                    pass
            if synced > 0:
                logger.info("OmniMem Temporal KG backfill: synced %d triples from KG", synced)
        except Exception as e:
            logger.warning("OmniMem Temporal KG backfill failed: %s", e)

    @property
    def async_provider(self) -> Any:
        """获取异步包装器（延迟初始化）。"""
        if not hasattr(self, "_async_provider"):
            from omnimem.core.async_provider import AsyncOmniMemProvider

            self._async_provider = AsyncOmniMemProvider(self)
        return self._async_provider

    def system_prompt_block(self) -> str:
        """构建系统提示词块 — 委托给 SystemPromptBuilder。"""
        if not hasattr(self, "_system_prompt_builder"):
            return ""
        result, cache_turn, cache_value = self._system_prompt_builder.build(
            turn_count=self._turn_count,
            system_prompt_cache_turn=self._system_prompt_cache_turn,
            system_prompt_cache_value=self._system_prompt_cache_value,
            last_query=getattr(self, "_last_query", ""),
        )
        self._system_prompt_cache_turn = cache_turn
        self._system_prompt_cache_value = cache_value
        return result

    def on_session_end(self, messages: list | None = None) -> None:
        """会话结束：Consolidation + 治理归档 + 物理过期清理。"""
        if not self._should_write:
            return
        # ★ 修复 L9：会话结束时触发物理过期清理
        # 原实现仅改 forgetting.db 的 stage 标记，Drawer 文件永不物理删除，导致存储膨胀
        try:
            self._prune_expired_memories()
        except Exception as e:
            logger.warning("on_session_end prune expired memories failed: %s", e)
        if self._session_manager:
            # 委托给 SessionManager
            self._session_manager.turn_count = self._turn_count
            self._session_manager.on_session_end(messages)
            self._turn_count = self._session_manager.turn_count
            return
        # Fallback: 直接执行（测试兼容）
        try:
            config = getattr(self, "_config", None)
            interval = config.get("audit_interval_turns", 50) if config else 50
            if self._turn_count > 0 and self._turn_count % interval == 0:
                auditor = getattr(self, "_auditor", None) or getattr(
                    self._governance, "auditor", None
                )
                if auditor:
                    auditor.quick_health_check()
        except Exception as e:
            logger.warning("on_session_end audit failed: %s", e)

    def _prune_expired_memories(self) -> None:
        """物理删除已标记为 superseded 且超过指定天数的记忆。

        清理范围：
          1. UnifiedMemoryIndex / ThreeLevelIndex 中 is_superseded=1 且 stored_at < cutoff
          2. DrawerClosetStore 对应的 Drawer/Closet 文件
          3. HybridRetriever 中对应的向量/BM25 文档
          4. MetaStore 中对应的元数据

        触发频率：每次 on_session_end（可配置阈值）
        """
        config = getattr(self, "_config", None) or {}
        prune_days = config.get("prune_superseded_days", 90)
        # UnifiedMemoryIndex（若已启用统一索引重构）
        unified = getattr(self, "_unified_index", None)
        if unified is not None:
            deleted = unified.prune_expired(days=prune_days)
            if deleted > 0:
                logger.info("Pruned %d superseded memories (>=%dd)", deleted, prune_days)

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any
    ) -> None:
        """子 Agent 完成时：记录过程记忆。"""
        if not self._should_write:
            return
        self._store_service.store_delegation(task, result, child_session_id)

    def _create_backup(self) -> tuple[str, int]:
        """创建备份 — 委托给 BackupManager。"""
        result = self._backup_manager.create_backup()
        self._last_backup_time = self._backup_manager.last_backup_time
        return result

    def _cleanup_old_backups(self, max_copies: int = 3) -> None:
        """清理旧备份 — 委托给 BackupManager。"""
        self._backup_manager.cleanup_old_backups(max_copies)

    def shutdown(self) -> None:
        """清理：刷新所有缓冲到磁盘。

        关闭顺序: retriever → md_store → index → knowledge_graph →
                  consolidation → reflect_engine → kv_cache → lora_trainer →
                  provenance → sync_engine → forgetting → executors
        """
        # 幂等性保护：防止 __del__ 重复调用导致资源关闭两次
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        if hasattr(self, "_memory_monitor") and self._memory_monitor:
            self._memory_monitor.stop()
        if hasattr(self, "_feedback") and self._feedback:
            self._feedback.close()
        if hasattr(self, "_prefetch_executor") and self._prefetch_executor:
            self._prefetch_executor.shutdown(wait=True)
        if hasattr(self, "_bg_executor") and self._bg_executor:
            self._bg_executor.shutdown(wait=True)
        # 关闭 memorize 模块级后台 fallback 线程池
        shutdown_background_executor(wait=True)
        # ★ 修复 L1/L2：关闭 recall + query_planner 模块级线程池
        #   原实现仅靠 atexit 注册（planner 甚至无 atexit），进程退出前线程泄漏
        try:
            from omnimem.services.recall_service import _recall_executor
            _recall_executor.shutdown(wait=False)
        except Exception as e:
            logger.debug("recall_executor shutdown failed: %s", e)
        try:
            from omnimem.handlers.query_planner import _planner_executor
            _planner_executor.shutdown(wait=False)
        except Exception as e:
            logger.debug("planner_executor shutdown failed: %s", e)
        # ★ 修复 L2：关闭 HybridOrchestrator 检索并行线程池
        try:
            if hasattr(self, "_retriever") and self._retriever:
                self._retriever.shutdown()
        except Exception as e:
            logger.debug("HybridOrchestrator shutdown failed: %s", e)

        # 1. 存储层
        self._store.close()  # flush + 关闭 MetaStore SQLite 连接
        self._retriever.flush()
        self._md_store.flush()
        self._index.close()
        if hasattr(self, "_trace_chain") and self._trace_chain:
            self._trace_chain.close()
        if hasattr(self, "_perception") and self._perception:
            self._perception.close()
        if self._knowledge_graph:
            self._knowledge_graph.close()
        if self._consolidation:
            self._consolidation.close()
        if hasattr(self, "_reflect_engine") and self._reflect_engine:
            self._reflect_engine.close()
        if hasattr(self, "_kv_cache") and self._kv_cache:
            self._kv_cache.close()
        if hasattr(self, "_lora_trainer") and self._lora_trainer:
            self._lora_trainer.close()
        if hasattr(self, "_provenance") and self._provenance:
            self._provenance.close()
        if hasattr(self, "_sync_engine") and self._sync_engine:
            self._sync_engine.close()
        # 持久化向量时钟 + 关闭治理资源
        if hasattr(self, "_governance") and self._governance:
            self._governance.close()
        # OPT-2: 关闭 LLM 客户端
        if hasattr(self, "_llm_client_manager") and self._llm_client_manager:
            self._llm_client_manager.close()
        if hasattr(self, "_distillation_engine") and self._distillation_engine:
            self._distillation_engine.close()
        self._forgetting.close()
        if hasattr(self, "_quality_evaluator") and self._quality_evaluator:
            self._quality_evaluator.close()
        logger.info("OmniMem shutdown complete")

    def __del__(self) -> None:
        """安全网：GC 回收时自动释放资源（防止 shutdown() 未被调用时泄漏）。"""
        try:
            self.shutdown()
        except Exception:
            pass  # __del__ 中不应抛出异常
