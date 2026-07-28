"""ForgettingCurve — 运维操作 Mixin。"""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger("governance.forgetting")


class _ForgettingOps:
    """运维 Mixin：筛选/FSRS/语义/状态/连接管理/flush/close。"""

    # ── 三阶段筛选（委托给 ForgettingScreening） ─────────────────────────────

    def run_first_screening(self) -> dict[str, int]:
        """T+24h 首次筛选：基于频率密度计算热度等级。委托给 ForgettingScreening。"""
        with self._lock:
            return self._screening.run_first_screening()

    def run_second_screening(self) -> dict[str, Any]:
        """T+7d 二次筛选：扫描 hot 满 7 天的记忆。委托给 ForgettingScreening。"""
        with self._lock:
            return self._screening.run_second_screening()

    def run_third_consolidation(self) -> dict[str, Any]:
        """T+30d 最终巩固：Wiki 交叉引用扫描。委托给 ForgettingScreening。"""
        with self._lock:
            return self._screening.run_third_consolidation()

    def run_warm_cooling(self) -> int:
        """warm 降温：30 天内零检索的 warm 记忆降级为 cold。委托给 ForgettingScreening。"""
        with self._lock:
            return self._screening.run_warm_cooling()

    # ── 归档周期（整合三阶段筛选） ────────────────────────────────────────────

    def run_archive_cycle(self) -> int:
        """后台运行：执行三阶段筛选 + 过期记忆降级 + access_log 清理。

        ★ 改造：
        1. Phase 1 新增：自动升级检查（高频访问记忆回到 active）
        2. T+24h 首次筛选（含冷启动保护）
        3. T+7d 二次筛选（窗口增量）
        4. warm→cold 降温（30天零检索）
        5. 原有加速遗忘逻辑（小时级精度）
        6. access_log 清理（90天前）

        Returns:
            归档的记忆数量
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            archived_count = 0

            # ★ Phase 1 优化：自动升级检查
            self._screening.check_for_reactivation()

            # 三阶段筛选 + warm 降温
            self._screening.run_first_screening()
            self._screening.run_second_screening()
            self._screening.run_third_consolidation()
            self._screening.run_warm_cooling()

            # ★ access_log 清理：90天前的旧记录
            self.prune_access_log(days=90)

            # ★ 自适应衰减：基于记忆类型和访问频率计算个性化阶段
            # 增量查询：按阶段分别查询，只处理可能需要降级的记忆
            assert self._conn is not None

            # 计算各阶段的时间阈值（保守下界，确保不遗漏候选）
            active_threshold = now - timedelta(hours=24)       # active 记忆至少 24h 才可能降级
            consolidating_threshold = now - timedelta(days=7)  # consolidating 记忆至少 7d 才可能降级
            archived_threshold = now - timedelta(days=30)      # archived 记忆至少 30d 才可能降级

            incremental_queries = [
                ("active", active_threshold),
                ("consolidating", consolidating_threshold),
                ("archived", archived_threshold),
            ]

            rows = []
            for stage_name, threshold in incremental_queries:
                try:
                    stage_rows = self._conn.execute(
                        """SELECT memory_id, created_at, stage, recall_count, memory_type
                           FROM forgetting_state
                           WHERE stage = ? AND created_at < ?""",
                        (stage_name, threshold.isoformat()),
                    ).fetchall()
                    rows.extend(stage_rows)
                except Exception as e:
                    logger.warning("Archive cycle incremental query failed for stage=%s: %s", stage_name, e)
                    return 0

            for memory_id, created_at, stage, recall_count, memory_type in rows:
                try:
                    if not created_at:
                        continue
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    if created_dt.tzinfo is None:
                        created_dt = created_dt.replace(tzinfo=timezone.utc)
                    hours_elapsed = (now - created_dt).total_seconds() / 3600
                    days = hours_elapsed / 24

                    effective_type = memory_type if memory_type else "fact"
                    effective_recall = recall_count if recall_count else 0
                    adaptive_stages = self._compute_adaptive_stages(effective_type, effective_recall)
                    expected_stage = self._get_stage_by_age_custom(days, adaptive_stages)

                    # 如果阶段比预期低，降级
                    stage_order = ["active", "consolidating", "archived", "forgotten"]
                    current_idx = stage_order.index(stage) if stage in stage_order else 0
                    expected_idx = (
                        stage_order.index(expected_stage) if expected_stage in stage_order else 0
                    )

                    if expected_idx > current_idx:
                        self._set_stage(memory_id, expected_stage)
                        archived_count += 1
                except Exception as e:
                    logger.warning("Archive cycle failed for %s: %s", memory_id, e)

            logger.info("Archive cycle: %d memories archived", archived_count)
            return archived_count

    @staticmethod
    def _get_stage_by_age_custom(days: int, stages: dict[str, tuple[int, int | None]]) -> str:
        """根据天数和自定义阶段定义计算阶段。"""
        for stage, (min_days, max_days) in stages.items():
            if max_days is None:
                if days >= min_days:
                    return str(stage)
            elif min_days <= days < max_days:
                return str(stage)
        return "active"

    # ── FSRS 相关方法（委托给 ForgettingFSRS） ────────────────────────────────

    def calculate_fsrs_retention(self, memory_id: str) -> float:
        """使用 FSRS 计算记忆保持率。委托给 ForgettingFSRS。"""
        with self._lock:
            return self._fsrs_adapter.calculate_fsrs_retention(memory_id)

    def calculate_fsrs_retention_batch(self, memory_ids: list[str]) -> dict[str, float]:
        """批量使用 FSRS 计算记忆保持率。委托给 ForgettingFSRS。"""
        with self._lock:
            return self._fsrs_adapter.calculate_fsrs_retention_batch(memory_ids)

    def suggest_review_time(self, memory_id: str, desired_retention: float = 0.9) -> datetime | None:
        """建议下次复习时间。委托给 ForgettingFSRS。"""
        with self._lock:
            return self._fsrs_adapter.suggest_review_time(memory_id, desired_retention)

    def get_fsrs_stats(self) -> dict[str, Any]:
        """获取 FSRS 统计信息。委托给 ForgettingFSRS。"""
        with self._lock:
            return self._fsrs_adapter.get_fsrs_stats()

    # ── 记忆强度评估方法（委托给 ForgettingFSRS） ──────────────────────────────

    def evaluate_memory_strength(self, memory_id: str) -> dict[str, Any]:
        """评估单个记忆的强度。委托给 ForgettingFSRS。"""
        with self._lock:
            return self._fsrs_adapter.evaluate_memory_strength(memory_id)

    def evaluate_all_memories(self, limit: int = 100) -> dict[str, Any]:
        """评估所有记忆的强度。委托给 ForgettingFSRS。"""
        with self._lock:
            return self._fsrs_adapter.evaluate_all_memories(limit)

    def get_memory_grade(self, memory_id: str) -> str:
        """获取记忆等级。委托给 ForgettingFSRS。"""
        return self._fsrs_adapter.get_memory_grade(memory_id)

    def get_strength_distribution(self) -> dict[str, Any]:
        """获取记忆强度分布统计。委托给 ForgettingFSRS。"""
        return self._fsrs_adapter.get_strength_distribution()

    # ── 语义重要性评估方法（委托给 ForgettingSemantic） ──────────────────────

    def evaluate_semantic_importance(self, memory_id: str) -> dict[str, Any]:
        """评估记忆的语义重要性。委托给 ForgettingSemantic。"""
        with self._lock:
            return self._semantic_adapter.evaluate_semantic_importance(memory_id)

    def evaluate_semantic_importance_batch(self, memory_ids: list[str]) -> dict[str, dict[str, Any]]:
        """批量评估记忆的语义重要性。委托给 ForgettingSemantic。"""
        with self._lock:
            return self._semantic_adapter.evaluate_semantic_importance_batch(memory_ids)

    def get_semantic_importance_distribution(self) -> dict[str, Any]:
        """获取语义重要性分布统计。委托给 ForgettingSemantic。"""
        with self._lock:
            return self._semantic_adapter.get_semantic_importance_distribution()

    # ── 状态查询 ──────────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """获取遗忘状态概览（含热度分类和升级候选）。"""
        with self._lock:
            counts: dict[str, int] = {"active": 0, "consolidating": 0, "archived": 0, "forgotten": 0}
            heat_counts: dict[str, int] = {"neutral": 0, "hot": 0, "warm": 0, "cold": 0}
            upgrade_candidates: list[dict[str, Any]] = []

            assert self._conn is not None
            try:
                # 阶段统计
                for stage, count in self._conn.execute(
                    "SELECT stage, COUNT(*) FROM forgetting_state GROUP BY stage"
                ).fetchall():
                    if stage in counts:
                        counts[stage] = count
                # 热度统计
                for heat, count in self._conn.execute(
                    "SELECT heat, COUNT(*) FROM forgetting_state GROUP BY heat"
                ).fetchall():
                    if heat in heat_counts:
                        heat_counts[heat] = count
                # 升级候选
                upgrade_candidates = self.get_upgrade_candidates()
            except Exception as e:
                logger.warning("Get forgetting status failed: %s", e)

            return {
                "stages": counts,
                "heat": heat_counts,
                "upgrade_candidates_count": len(upgrade_candidates),
                "upgrade_candidates": upgrade_candidates[:10],
            }

    def get_archived_ids(self, limit: int = 5000) -> list[str]:
        """获取已归档（archived 或 forgotten）的记忆 ID 列表。

        Args:
            limit: 最大返回数量

        Returns:
            memory_id 列表
        """
        with self._lock:
            assert self._conn is not None
            try:
                rows = self._conn.execute(
                    "SELECT memory_id FROM forgetting_state WHERE stage IN ('archived', 'forgotten') LIMIT ?",
                    (limit,),
                ).fetchall()
                return [r[0] for r in rows if r[0]]
            except Exception as e:
                logger.warning("Get archived ids failed: %s", e)
                return []

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _set_stage(self, memory_id: str, stage: str) -> None:
        """设置记忆的阶段。"""
        now = datetime.now(timezone.utc).isoformat()
        assert self._conn is not None
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO forgetting_state
                   (memory_id, stage, last_accessed, created_at)
                   VALUES (?, ?, ?, ?)""",
                (memory_id, stage, now, now),
            )
            self._pending_writes += 1
            self._maybe_commit()
        except Exception as e:
            logger.warning("Stage update failed: %s", e)

    def _maybe_commit(self) -> None:
        """到达阈值时提交。"""
        if self._pending_writes >= self._BATCH_THRESHOLD:
            # ★ 修复: _ensure_conn_alive 从未在本类定义(M6-8 重构遗留),
            #   连接生命周期由 GovernanceStore 管理, 直接提交即可
            self._store.commit()
            self._pending_writes = 0

    def _get_index_conn(self) -> "sqlite3.Connection | None":
        """获取只读索引连接。★ M6-8: 委托给 GovernanceStore。"""
        return self._store.get_read_conn() if hasattr(self, '_store') else None

    def flush(self) -> None:
        """显式提交所有待写入。★ M6-8: 委托给 GovernanceStore。"""
        with self._lock:
            if self._pending_writes > 0:
                self._store.commit()
                self._pending_writes = 0

    def close(self) -> None:
        """关闭数据库连接。★ M6-8: 委托给 GovernanceStore。"""
        with self._lock:
            self.flush()
            # GovernanceStore 由调用方统一关闭
