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
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
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
from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger("governance.forgetting")

# 模块级共享锁：防止多个 ForgettingCurve 实例并发写同一 forgetting.db
# （GovernanceFacade + MemoryAPI 各持独立实例，实例级 RLock 无法互斥）
_FORGETTING_DB_LOCK = threading.RLock()

# 引用计数
_connection_refcounts: dict[str, int] = {}

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
    """

    # ★ P2修复：批量提交阈值从5提升到20，减少频繁 commit 的 I/O 开销
    _BATCH_THRESHOLD = 20

    # ★ 类级连接共享：所有实例共用同一 forgetting.db 连接，防止多实例 SQLite 写锁冲突
    _shared_connections: dict[str, sqlite3.Connection] = {}
    _shared_index_connections: dict[str, sqlite3.Connection] = {}

    def __init__(self, governance_dir: Path, config: Any = None):
        self._governance_dir = governance_dir
        self._governance_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._governance_dir / "forgetting.db"
        self._conn: sqlite3.Connection | None = None
        self._index_conn: sqlite3.Connection | None = None
        self._lock = _FORGETTING_DB_LOCK
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
        self._init_db()
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
        """获取 forgetting_state 数据库连接（子模块回调用）。"""
        self._ensure_conn_alive()
        assert self._conn is not None
        return self._conn

    def _track_write(self) -> None:
        """记录一次待写入并检查批量提交（子模块回调用）。"""
        self._pending_writes += 1
        self._maybe_commit()

    def _init_db(self) -> None:
        """初始化遗忘数据库。"""
        db_path_str = str(self._db_path)
        if db_path_str in self._shared_connections:
            conn = self._shared_connections[db_path_str]
            # 健康检查：防止拿到已被 close() 关闭的连接
            try:
                conn.execute("SELECT 1")
                self._conn = conn
                return
            except sqlite3.ProgrammingError:
                logger.warning(
                    "ForgettingCurve: shared connection is closed, removing from cache and re-creating"
                )
                del self._shared_connections[db_path_str]
        self._conn = sqlite3.connect(db_path_str, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._shared_connections[db_path_str] = self._conn
        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="forgetting_state",
            create_sql="""
                CREATE TABLE IF NOT EXISTS forgetting_state (
                    memory_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL DEFAULT 'active',
                    last_accessed TEXT,
                    created_at TEXT,
                    archive_count INTEGER DEFAULT 0,
                    recall_count INTEGER DEFAULT 0
                )
            """,
            migrations=[],
        )
        # ★ 兼容旧表：逐列添加，已有则跳过（查 PRAGMA table_info 避免异常）
        existing_cols = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(forgetting_state)"
            ).fetchall()
        }
        _new_columns = [
            ("recall_count", "INTEGER DEFAULT 0"),
            ("heat", "TEXT NOT NULL DEFAULT 'neutral'"),
            ("heat_updated_at", "TEXT"),
            ("upgraded_to_wiki", "INTEGER DEFAULT 0"),
            ("wiki_page_path", "TEXT"),
            ("memory_type", "TEXT DEFAULT 'fact'"),
        ]
        for col_name, col_type in _new_columns:
            if col_name in existing_cols:
                continue
            # 列名来自硬编码常量，非用户输入，安全使用 f-string
            self._conn.execute(
                f"ALTER TABLE forgetting_state ADD COLUMN {col_name} {col_type}"
            )

        # ★ access_log 表 —— 记录每次检索的时间戳，支持时间窗口查询
        migrator.migrate(
            table_name="access_log",
            create_sql="""
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    accessed_at TEXT NOT NULL
                )
            """,
            migrations=[],
        )
        # 索引：按 memory_id + accessed_at 加速窗口查询
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_access_log_mid_at ON access_log(memory_id, accessed_at)"
        )

        # ★ Phase 1 优化：添加复合索引提升查询性能
        # forgetting_state 表索引
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_forgetting_stage_created ON forgetting_state(stage, created_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_forgetting_heat_updated ON forgetting_state(heat, heat_updated_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_forgetting_heat_recall ON forgetting_state(heat, recall_count)"
        )

        self._conn.commit()

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

