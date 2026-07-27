"""ForgettingCurve — Ebbinghaus 遗忘曲线驱动的4阶段归档 + 热度分类 + 时间窗口查询。

4个阶段:
  - active (0-7天): 完整保留，正常检索
  - consolidating (7-30天): 可能需要提示，降权但不归档
  - archived (30-90天): 仅摘要可用，原文归档
  - forgotten (90天+): 仅L0索引可用，需要显式召回

★ Phase 1 优化 (2026-05-26):
  - 热度计算: 基于频率密度 (density = recall_7d / min(7, days_alive))
    - hot: density >= 1.0 (平均每天1次以上)
    - warm: density >= 0.3 (平均3天1次)
    - neutral: 有检索但未达warm
    - cold: 7天内零检索
  - 自动升级: consolidating/archived 阶段的高频记忆自动回到 active
  - 第三阶段: T+30d Wiki 交叉引用扫描 + 自动晋升
  - 数据库索引: stage+created_at, heat+heat_updated_at, heat+recall_count

★ 拆分重构 (2026-06-14):
  - ForgettingFSRS: FSRS 算法 + 记忆强度评估 → governance/fsrs_engine.py
  - ForgettingSemantic: 语义重要性评估 → governance/semantic_importance.py
  - ForgettingScreening: 三阶段筛选引擎 → governance/screening_engine.py
  - ForgettingCurve: 阶段管理 + 热度分类 + 归档调度（本文件）

★ M6-8 合库 (2026-07):
  - 接入 GovernanceStore 统一存储，消除 _FORGETTING_DB_LOCK、类级共享连接、
    引用计数等约 500 行防御代码
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from omnimem.governance.fsrs_engine import (
    ForgettingFSRS,
    get_fsrs_engine,
)
from omnimem.governance.memory_strength import (
    get_evaluator,
)
from omnimem.governance.screening_engine import ForgettingScreening
from omnimem.governance.semantic_importance import (
    ForgettingSemantic,
    get_semantic_evaluator,
)

logger = logging.getLogger("governance.forgetting")

# 4个阶段定义
STAGES = {
    "active": (0, 7),
    "consolidating": (7, 30),
    "archived": (30, 90),
    "forgotten": (90, None),
}

# 热度分类
HEAT_LEVELS = ("neutral", "hot", "warm", "cold")


class _ForgettingCore:
    """Ebbinghaus 遗忘曲线驱动的4阶段归档 + 热度分类 + 时间窗口查询。

    通过组合持有 ForgettingFSRS、ForgettingSemantic、ForgettingScreening 实例，
    将 FSRS 算法、语义评估、三阶段筛选等职责委托给子模块。
    本类保留阶段管理、热度分类、归档调度等核心逻辑。

    批量提交优化：写操作攒到阈值或显式 flush/close 时统一提交。

    ★ M6-8: 接入 GovernanceStore 统一存储，消除 _FORGETTING_DB_LOCK、
    类级共享连接等防御代码。写锁由 store.write_lock 提供。
    """

    # ★ P2修复：批量提交阈值从5提升到20，减少频繁 commit 的 I/O 开销
    _BATCH_THRESHOLD = 20

    def __init__(self, governance_dir: Path, config: Any = None,
                 governance_store: Any = None):
        self._governance_dir = governance_dir
        self._governance_dir.mkdir(parents=True, exist_ok=True)

        # ★ M6-8: 接入 GovernanceStore（若未传入则惰性创建独立实例）
        if governance_store is not None:
            self._store = governance_store
        else:
            from omnimem.governance.governance_store import GovernanceStore
            self._store = GovernanceStore(self._governance_dir)
        self._conn = self._store.get_write_conn()
        self._lock = self._store.write_lock
        self._pending_writes = 0

        self._active_days = getattr(config, 'forgetting_active_days', 7) if config else 7
        self._consolidating_days = getattr(config, 'forgetting_consolidating_days', 30) if config else 30
        self._archived_days = getattr(config, 'forgetting_archived_days', 90) if config else 90
        self._stages: dict[str, tuple[int, int | None]] = {
            "active": (0, self._active_days),
            "consolidating": (self._active_days, self._consolidating_days),
            "archived": (self._consolidating_days, self._archived_days),
            "forgotten": (self._archived_days, None),
        }
        self._stage_config: dict[str, dict[str, int]] = {}

        # ★ 冷启动标记：首次运行时跳过历史数据
        self._ensure_pipeline_marker()

        # ★ 子模块实例化
        fsrs_engine = get_fsrs_engine()
        evaluator = get_evaluator()
        semantic_evaluator = get_semantic_evaluator()

        self._fsrs_adapter = ForgettingFSRS(
            fsrs_engine=fsrs_engine,
            memory_evaluator=evaluator,
            get_conn=self._get_conn,
        )
        self._semantic_adapter = ForgettingSemantic(
            evaluator=semantic_evaluator,
            get_conn=self._get_conn,
            get_index_conn=self._get_index_conn,
        )
        self._screening = ForgettingScreening(
            governance_dir=self._governance_dir,
            get_conn=self._get_conn,
            get_index_conn=self._get_index_conn,
            get_recall_count_in_window=self.get_recall_count_in_window,
            get_heat=self.get_heat,
            set_heat=self.set_heat,
            set_stage=self._set_stage,
            track_write=self._track_write,
            get_pipeline_start_time=self._get_pipeline_start_time,
        )

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（子模块回调用）。★ M6-8: 委托给 GovernanceStore。"""
        return self._conn

    def _get_index_conn(self) -> sqlite3.Connection:
        """获取只读索引连接。★ M6-8: 委托给 GovernanceStore。"""
        if not hasattr(self, '_store') or self._store is None:
            return self._conn
        return self._store.get_read_conn()

    def _track_write(self) -> None:
        """记录一次待写入并检查批量提交（子模块回调用）。"""
        self._pending_writes += 1
        self._maybe_commit()

    def _commit(self) -> None:
        """提交写连接。★ M6-8: 委托给 GovernanceStore。"""
        self._store.commit()

    def _maybe_commit(self) -> None:
        """批量提交：攒到阈值时自动提交。"""
        if self._pending_writes >= self._BATCH_THRESHOLD:
            self._store.commit()
            self._pending_writes = 0

    # ── 自适应衰减阈值 ──────────────────────────────────────────────────────

    def _compute_adaptive_stages(self, memory_type: str, recall_count: int) -> dict[str, tuple[int, int | None]]:
        """基于记忆类型和访问频率计算自适应衰减阶段。

        规则：
        - preference/preference 类型：active 阶段延长 2x（用户偏好不应快速遗忘）
        - reasoning 类型：active 阶段延长 1.5x（经验教训有长期价值）
        - action 类型：保持默认（操作记录时效性短）
        - recall_count >= 5：active 阶段延长 2x（高频访问记忆更重要）
        - recall_count >= 10：active 阶段延长 3x
        - recall_count == 0：active 阶段缩短 0.5x（从未被访问的记忆可加速遗忘）
        """
        base_active = self._active_days
        base_consolidating = self._consolidating_days
        base_archived = self._archived_days

        if memory_type in self._stage_config:
            cfg = self._stage_config[memory_type]
            base_active = cfg.get("active_days", base_active)
            base_consolidating = cfg.get("consolidating_days", base_consolidating)
            base_archived = cfg.get("archived_days", base_archived)

        multiplier = 1.0
        if memory_type in ("preference", "preferences"):
            multiplier = 2.0
        elif memory_type == "reasoning":
            multiplier = 1.5

        if recall_count >= 10:
            freq_multiplier = 3.0
        elif recall_count >= 5:
            freq_multiplier = 2.0
        elif recall_count == 0:
            freq_multiplier = 0.5
        else:
            freq_multiplier = 1.0

        final_multiplier = max(multiplier, freq_multiplier)

        adaptive_active = max(1, int(base_active * final_multiplier))
        adaptive_consolidating = max(adaptive_active + 1, int(base_consolidating * final_multiplier))
        adaptive_archived = max(adaptive_consolidating + 1, int(base_archived * final_multiplier))

        return {
            "active": (0, adaptive_active),
            "consolidating": (adaptive_active, adaptive_consolidating),
            "archived": (adaptive_consolidating, adaptive_archived),
            "forgotten": (adaptive_archived, None),
        }

    def set_stage_config(self, memory_type: str, active_days: int, consolidating_days: int, archived_days: int) -> None:
        """为指定记忆类型设置自定义阶段阈值。

        Args:
            memory_type: 记忆类型（如 fact, preference, reasoning, action）
            active_days: active 阶段天数上限
            consolidating_days: consolidating 阶段天数上限
            archived_days: archived 阶段天数上限
        """
        if active_days <= 0 or consolidating_days <= active_days or archived_days <= consolidating_days:
            logger.warning(
                "set_stage_config 参数无效: active=%d, consolidating=%d, archived=%d（需满足 0 < active < consolidating < archived）",
                active_days, consolidating_days, archived_days,
            )
            return
        self._stage_config[memory_type] = {
            "active_days": active_days,
            "consolidating_days": consolidating_days,
            "archived_days": archived_days,
        }
        logger.info(
            "已为记忆类型 '%s' 设置自定义阈值: active=%d, consolidating=%d, archived=%d",
            memory_type, active_days, consolidating_days, archived_days,
        )

