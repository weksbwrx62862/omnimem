"""会话管理器 — 负责对话同步（sync_turn）和会话结束（on_session_end）逻辑。"""

from __future__ import annotations

import logging
import time
from typing import Any

from omnimem.core.reflection_trigger import ReflectionSignal, ReflectionTrigger
from omnimem.core.session_deps import SessionDependencies
from omnimem.utils.event_publisher import get_event_publisher
from omnimem.utils.security import SecurityValidator

logger = logging.getLogger(__name__)


# ★ PluginOrchestrator 集成：会话结束事件
def _publish_session_end_event(session_id: str) -> None:
    """会话结束时，通过 EventBus 通知其他插件。

    下游插件（如 disk-cleanup）可据此执行批量操作。
    如果 PluginOrchestrator 未安装，静默跳过。
    """
    if not session_id:
        return
    publisher = get_event_publisher(session_id)
    publisher.publish(
        "session_memory_processed",
        source_plugin="omnimem",
        session_id=session_id,
    )


class SessionManager:
    """管理对话同步和会话生命周期。

    职责:
      1. sync_turn: 每轮对话后的感知→写入→治理
      2. on_session_end: 会话结束时的 Consolidation + 治理归档 + 备份
    """

    def __init__(self, deps: SessionDependencies) -> None:
        """通过 SessionDependencies 注入全部外部依赖。

        保持实例变量名与旧签名一致，避免破坏外部访问。
        """
        self._config = deps.config
        self._perception = deps.perception
        self._store_service = deps.store_service
        self._retriever = deps.retriever
        self._bg_executor = deps.bg_executor
        self._forgetting = deps.forgetting
        self._consolidation = deps.consolidation
        self._kv_cache = deps.kv_cache
        self._lora_trainer = deps.lora_trainer
        self._store = deps.store
        self._index = deps.index
        self._auditor = deps.auditor
        self._saga = deps.saga
        self._prefetch_executor = deps.prefetch_executor
        self._pipeline_scheduler = deps.pipeline_scheduler
        self._distill_init_fn = deps.distill_init_fn
        self._distillation_engine = deps.distillation_engine
        self._session_id = deps.session_id
        self._should_write = deps.should_write
        self._strip_fn = deps.strip_system_injections_fn or SecurityValidator.strip_system_injections
        self._should_store_fn = deps.should_store_fn
        # 自动反思触发器
        self._reflection_trigger = ReflectionTrigger(
            deps.config.get("reflection_trigger", None)
        )
        self._handle_memorize_fn = deps.handle_memorize_fn
        self._retry_fns = {
            "three_level_index": deps.retry_index_add_fn,
            "retriever": deps.retry_retriever_add_fn,
            "knowledge_graph": deps.retry_kg_extract_fn,
        }
        self._create_backup_fn = deps.create_backup_fn
        self._cleanup_old_backups_fn = deps.cleanup_old_backups_fn
        # 运行时状态
        self._turn_count: int = 0
        self._last_save_turn: int = 0
        self._save_interval: int = deps.config.get("save_interval", 15)
        self._last_backup_time: float = 0.0

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @turn_count.setter
    def turn_count(self, value: int) -> None:
        self._turn_count = value

    @property
    def last_save_turn(self) -> int:
        return self._last_save_turn

    @property
    def last_backup_time(self) -> float:
        return self._last_backup_time

    @last_backup_time.setter
    def last_backup_time(self, value: float) -> None:
        self._last_backup_time = value

    def sync_turn(self, user_content: str, assistant_content: str) -> None:
        """每轮对话后：感知 → 写入 → 治理。"""
        if not self._should_write:
            return
        clean_user = self._strip_fn(user_content)
        signals = self._perception.detect_signals(clean_user, assistant_content)
        # 信号驱动的记忆写入（互斥：correction > reinforcement > fact）
        if signals.has_correction:
            self._store_service.store_correction(signals, user_content)
            self._reflection_trigger.record_correction()
        elif signals.has_reinforcement:
            self._store_service.store_reinforcement(signals, user_content)
        elif signals.should_memorize:
            self._store_service.store_fact(signals, user_content)
            self._reflection_trigger.record_new_memory()
        # 定期自动存档
        self._turn_count += 1
        self._store_service.turn_count = self._turn_count
        if self._turn_count - self._store_service.last_save_turn >= self._save_interval:
            self._store_service.auto_checkpoint(user_content, self._save_interval)
            self._last_save_turn = self._store_service.last_save_turn
        # 定期 LLM 蒸馏
        if self._config.get("distill_enabled", True):
            distill = self._distillation_engine
            if distill is None and self._distill_init_fn:
                self._distill_init_fn()
                distill = self._distillation_engine
            distill_interval = self._config.get("distill_interval", 15)
            if distill and self._turn_count % distill_interval == 0:
                self._bg_executor.submit(
                    lambda: distill.distill_recent_facts(turn_count=self._turn_count)
                )
                if self._pipeline_scheduler:
                    self._pipeline_scheduler.schedule_l2_after_l1(self._session_id)
        # 统一后台任务执行器
        self._bg_executor.submit(self._retriever.index_update, user_content, assistant_content)

    def evaluate_reflection(self) -> ReflectionSignal:
        """评估是否应该自动触发反思。供 Provider 调用。"""
        return self._reflection_trigger.evaluate(
            turn_count=self._turn_count,
            session_id=self._session_id,
        )

    def record_tool_call(self, tool_name: str) -> None:
        """记录工具调用，供反思触发器评分。"""
        self._reflection_trigger.record_tool_call(tool_name)

    def mark_reflected(self) -> None:
        """标记已反思（Agent 主动调用 omni_reflect 时）。"""
        self._reflection_trigger.mark_reflected()

    def reset_reflection_session(self) -> None:
        """重置反思触发器的会话状态。"""
        self._reflection_trigger.reset_session()

    @property
    def reflection_stats(self) -> dict[str, Any]:
        """返回反思触发器状态。"""
        return self._reflection_trigger.stats

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """会话结束：Consolidation + 治理归档。"""
        if not self._should_write:
            return
        if self._prefetch_executor:
            self._prefetch_executor.shutdown(wait=True)
        # 1. 提取遗漏的记忆
        self._store_service.extract_session_memories(
            messages, self._strip_fn, self._should_store_fn,
            lambda args: self._handle_memorize_fn(args) if self._handle_memorize_fn else "",
        )
        # 2. 遗忘曲线归档 + 清理
        if self._forgetting.run_archive_cycle() > 0:
            self._cleanup_archived_entries()
        # 3. L3 Consolidation
        if self._consolidation:
            processed = self._consolidation.process_pending()
            if processed > 0:
                logger.info("OmniMem consolidation: processed %d memories", processed)
        # 4-5. L4 KV Cache + LoRA
        self._preload_kv_cache()
        self._submit_lora_training()
        # 6. 刷新存储缓冲与索引
        self._store.flush()
        self._retriever.flush()
        # 7-8. 治理巡检 + Saga 重试
        self._run_governance_audit()
        self._retry_saga_pending()
        # 9-10. Pipeline + 后台执行器
        if self._pipeline_scheduler:
            self._pipeline_scheduler.flush_session(self._session_id)
        if self._bg_executor:
            self._bg_executor.shutdown(wait=True)
        # 11. 自动备份
        self._auto_backup()
        logger.info("OmniMem session end: processed %d messages", len(messages))

        # ★ PluginOrchestrator 集成：会话结束事件
        _publish_session_end_event(self._session_id)

    def _cleanup_archived_entries(self) -> None:
        """归档后从检索索引+文件系统中清理。"""
        try:
            archived_list = self._forgetting.get_archived_ids(limit=500)
            for mid in archived_list:
                self._retriever.delete(mid)
                for deleter in (self._index, self._store):
                    try:
                        deleter.delete(mid)
                    except Exception:
                        logger.debug("OmniMem archive cleanup: delete failed for %s", mid)
            logger.info("OmniMem: cleaned %d archived entries", len(archived_list))
        except Exception as e:
            logger.warning("OmniMem archive cleanup skipped: %s", e)

    def _preload_kv_cache(self) -> None:
        """L4 KV Cache: 将高频回忆和心智模型预填充。"""
        if not self._kv_cache or not self._consolidation:
            return
        try:
            patterns = []
            for m in self._consolidation.get_mental_models(limit=10):
                patterns.append({"key": f"model-{m.get('item_id', '')}", "content": m.get("content", ""),
                                 "metadata": {"type": "mental_model", "source": "consolidation"}})
            for o in self._consolidation.get_observations(limit=10):
                patterns.append({"key": f"obs-{o.get('item_id', '')}", "content": o.get("content", ""),
                                 "metadata": {"type": "observation", "source": "consolidation"}})
            if patterns:
                cached = self._kv_cache.preload(patterns)
                logger.info("OmniMem KV Cache: preloaded %d patterns from consolidation", cached)
        except Exception as e:
            logger.warning("OmniMem KV Cache preload failed: %s", e)

    def _submit_lora_training(self) -> None:
        """L4: 将心智模型提交到 LoRA 训练队列。"""
        if not self._consolidation or not self._lora_trainer:
            return
        try:
            models = self._consolidation.get_mental_models(limit=20)
            if models:
                self._lora_trainer.submit_training_data(models, shade="default")
                logger.info("OmniMem L4: submitted %d mental models for LoRA training", len(models))
            if self._lora_trainer._training_queue:
                train_result = self._lora_trainer.train(shade="default")
                if train_result.get("status") != "no_data":
                    logger.info("OmniMem L4 auto-train: %s", train_result.get("status"))
        except Exception as e:
            logger.warning("OmniMem L4 submit/train failed: %s", e)

    def _run_governance_audit(self) -> None:
        """治理巡检（每 N 轮执行一次一致性审计）。"""
        if not self._auditor:
            return
        if self._turn_count % self._config.get("audit_interval_turns", 50) != 0:
            return
        try:
            health = self._auditor.quick_health_check()
            if not health["healthy"]:
                audit = self._auditor.run_full_audit(limit=1000)
                if audit["total_issues"] > 0:
                    fixed = self._auditor.repair(audit)
                    logger.info("OmniMem governance audit: %d issues found, %d fixed", audit["total_issues"], fixed)
        except Exception as e:
            logger.warning("Governance audit failed: %s", e)

    def _retry_saga_pending(self) -> None:
        """Saga pending 重试（会话结束前补偿未完成的索引写入）。"""
        if not self._saga.get_pending():
            return
        if not all(self._retry_fns.values()):
            return
        try:
            fixed = self._saga.retry_pending(self._retry_fns)  # type: ignore[arg-type]
            if fixed > 0:
                logger.info("OmniMem saga retry: fixed %d pending records", fixed)
        except Exception as e:
            logger.warning("OmniMem saga retry failed: %s", e)

    def _auto_backup(self) -> None:
        """自动备份：检查距上次备份是否超过 backup_interval_hours。"""
        if not self._create_backup_fn or not self._cleanup_old_backups_fn:
            return
        try:
            backup_interval_hours = self._config.get("backup_interval_hours", 24)
            if time.time() - self._last_backup_time >= backup_interval_hours * 3600:
                self._create_backup_fn()
                self._cleanup_old_backups_fn(self._config.get("backup_max_copies", 3))
        except Exception as e:
            logger.warning("OmniMem 自动备份失败: %s", e)
