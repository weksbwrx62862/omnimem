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

热度分类（基于频率密度）:
  - neutral: 新记忆，未经过筛选
  - hot: 7天内平均每天检索≥1次
  - warm: 7天内平均3天检索1次
  - cold: 7天内零检索

归档操作:
  - archive(memory_id): 将记忆从 active 降级到 archived
  - reactivate(memory_id): 将记忆从 archived/forgotten 恢复到 active
  - run_archive_cycle(): 后台运行归档周期
  - set_heat/get_heat: 热度分类管理
  - get_recall_count_in_window: 时间窗口内检索计数
  - get_upgrade_candidates: Wiki 升级候选列表
  - mark_upgraded_to_wiki: 标记已升级到 Wiki
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from omnimem.utils.migration import SchemaMigrator
from omnimem.governance.fsrs_engine import FSRSEngine, FSRSItem, FSRSParameters, get_fsrs_engine
from omnimem.governance.memory_strength import MemoryStrengthEvaluator, ScoringWeights, get_evaluator
from omnimem.governance.semantic_importance import SemanticImportanceEvaluator, get_semantic_evaluator

logger = logging.getLogger(__name__)

# 4个阶段定义
STAGES = {
    "active": (0, 7),
    "consolidating": (7, 30),
    "archived": (30, 90),
    "forgotten": (90, None),
}

# 热度分类
HEAT_LEVELS = ("neutral", "hot", "warm", "cold")


class ForgettingCurve:
    """Ebbinghaus 遗忘曲线驱动的4阶段归档 + 热度分类 + 时间窗口查询。

    批量提交优化：写操作攒到阈值或显式 flush/close 时统一提交。
    """

    _BATCH_THRESHOLD = 5

    def __init__(self, governance_dir: Path, config: Any = None):
        self._governance_dir = governance_dir
        self._governance_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._governance_dir / "forgetting.db"
        self._conn: sqlite3.Connection | None = None
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

        # ★ Phase 2: FSRS 引擎
        self._fsrs = get_fsrs_engine()

        # ★ Phase 3: 记忆强度评估器
        self._evaluator = get_evaluator()

        # ★ Phase 4: 语义重要性评估器
        self._semantic_evaluator = get_semantic_evaluator()

    def _init_db(self) -> None:
        """初始化遗忘数据库。"""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
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

    # ── 冷启动 & access_log 清理 ────────────────────────────────────────────

    def _ensure_pipeline_marker(self) -> None:
        """冷启动标记：首次运行时记录时间戳，后续筛选跳过历史数据。"""
        assert self._conn is not None
        try:
            migrator = SchemaMigrator(self._conn)
            migrator.migrate(
                table_name="pipeline_meta",
                create_sql="""
                    CREATE TABLE IF NOT EXISTS pipeline_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                """,
                migrations=[],
            )
            row = self._conn.execute(
                "SELECT value FROM pipeline_meta WHERE key = 'start_time'"
            ).fetchone()
            if not row:
                now = datetime.now(timezone.utc).isoformat()
                self._conn.execute(
                    "INSERT OR IGNORE INTO pipeline_meta (key, value) VALUES ('start_time', ?)",
                    (now,),
                )
                self._conn.commit()
                logger.info("Pipeline marker set at %s — historical data will be skipped", now)
        except Exception as e:
            logger.warning("_ensure_pipeline_marker failed: %s", e)

    def _get_pipeline_start_time(self) -> str | None:
        """获取管道启动时间。"""
        assert self._conn is not None
        try:
            row = self._conn.execute(
                "SELECT value FROM pipeline_meta WHERE key = 'start_time'"
            ).fetchone()
            return str(row[0]) if row else None
        except Exception as e:
            logger.warning("ForgettingCurve _get_pipeline_start_time failed: %s", e)
            return None

    def prune_access_log(self, days: int = 90) -> int:
        """清理 access_log 中超过 N 天的旧记录。

        Args:
            days: 保留天数，默认 90 天

        Returns:
            删除的记录数
        """
        assert self._conn is not None
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            cursor = self._conn.execute(
                "DELETE FROM access_log WHERE accessed_at < ?", (cutoff,)
            )
            deleted = cursor.rowcount
            self._conn.commit()
            if deleted > 0:
                logger.info("Pruned %d access_log entries older than %d days", deleted, days)
            return deleted
        except Exception as e:
            logger.warning("prune_access_log failed: %s", e)
            return 0

    def get_stage(self, memory_id: str) -> str:
        """获取记忆的当前阶段。"""
        assert self._conn is not None
        try:
            row = self._conn.execute(
                "SELECT stage FROM forgetting_state WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row:
                return str(row[0])
        except Exception as e:
            logger.warning("Forgetting stage query failed: %s", e)
        return "active"

    def get_stage_by_age(self, days: int) -> str:
        """根据天数计算阶段。"""
        for stage, (min_days, max_days) in self._stages.items():
            if max_days is None:
                if days >= min_days:
                    return stage
            elif min_days <= days < max_days:
                return stage
        return "active"

    def archive(self, memory_id: str) -> None:
        """将记忆归档（降级到 archived）。"""
        current = self.get_stage(memory_id)
        if current == "forgotten":
            return
        new_stage = "archived"
        if current == "archived":
            new_stage = "forgotten"
        self._set_stage(memory_id, new_stage)

    def reactivate(self, memory_id: str) -> None:
        """将记忆重新激活（恢复到 active）。"""
        self._set_stage(memory_id, "active")
        # 更新最后访问时间
        now = datetime.now(timezone.utc).isoformat()
        assert self._conn is not None
        try:
            self._conn.execute(
                "UPDATE forgetting_state SET last_accessed = ? WHERE memory_id = ?",
                (now, memory_id),
            )
            self._pending_writes += 1
            self._maybe_commit()
        except Exception as e:
            logger.warning("Reactivate update failed: %s", e)

    def record_access(self, memory_id: str, memory_type: str = "fact") -> None:
        """记录记忆被访问（重置遗忘计时器 + 增加召回计数 + 写入 access_log）。

        ★ 改造：现在同时写入 access_log 表，支持时间窗口查询。
        ★ 自适应增强：同时记录 memory_type 到 forgetting_state。

        Args:
            memory_id: 记忆 ID
            memory_type: 记忆类型（如 fact, preference, reasoning, action）
        """
        now = datetime.now(timezone.utc).isoformat()
        assert self._conn is not None
        try:
            existing = self._conn.execute(
                "SELECT recall_count FROM forgetting_state WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if existing is not None:
                new_count = (existing[0] or 0) + 1
                self._conn.execute(
                    "UPDATE forgetting_state SET stage = 'active', last_accessed = ?, recall_count = ?, memory_type = ? WHERE memory_id = ?",
                    (now, new_count, memory_type, memory_id),
                )
            else:
                self._conn.execute(
                    """INSERT OR REPLACE INTO forgetting_state
                       (memory_id, stage, last_accessed, created_at, recall_count, memory_type)
                       VALUES (?, 'active', ?, ?, 1, ?)""",
                    (memory_id, now, now, memory_type),
                )
            self._conn.execute(
                "INSERT INTO access_log (memory_id, accessed_at) VALUES (?, ?)",
                (memory_id, now),
            )
            self._pending_writes += 1
            self._maybe_commit()
        except Exception as e:
            logger.warning("Access record failed: %s", e)

    # ── 热度分类 ──────────────────────────────────────────────────────────────

    def set_heat(self, memory_id: str, heat: str) -> None:
        """设置记忆的热度分类。

        Args:
            memory_id: 记忆 ID
            heat: 热度等级 (neutral/hot/warm/cold)
        """
        assert self._conn is not None
        if heat not in HEAT_LEVELS:
            logger.warning("Invalid heat level: %s", heat)
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._conn.execute(
                "UPDATE forgetting_state SET heat = ?, heat_updated_at = ? WHERE memory_id = ?",
                (heat, now, memory_id),
            )
            self._pending_writes += 1
            self._maybe_commit()
        except Exception as e:
            logger.warning("set_heat failed: %s", e)

    def get_heat(self, memory_id: str) -> str:
        """获取记忆的热度分类。"""
        assert self._conn is not None
        try:
            row = self._conn.execute(
                "SELECT heat FROM forgetting_state WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            return str(row[0]) if row else "neutral"
        except Exception as e:
            logger.warning("get_heat failed: %s", e)
            return "neutral"

    # ── 时间窗口查询 ──────────────────────────────────────────────────────────

    def get_recall_count_in_window(self, memory_id: str, days: int) -> int:
        """查询指定时间窗口内的检索次数。

        Args:
            memory_id: 记忆 ID
            days: 窗口天数（如 1=24h, 7=一周, 30=一月）

        Returns:
            窗口内检索次数
        """
        assert self._conn is not None
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            row = self._conn.execute(
                "SELECT COUNT(*) FROM access_log WHERE memory_id = ? AND accessed_at >= ?",
                (memory_id, cutoff),
            ).fetchone()
            return int(row[0]) if row else 0
        except Exception as e:
            logger.warning("get_recall_count_in_window failed: %s", e)
            return 0

    # ── 热度查询 ──────────────────────────────────────────────────────────────

    def get_candidates_by_heat(self, heat: str) -> list[dict[str, Any]]:
        """按热度分类查询记忆列表。

        Args:
            heat: 热度等级 (neutral/hot/warm/cold)

        Returns:
            包含 memory_id, created_at, recall_count, stage, heat 的字典列表
        """
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                "SELECT memory_id, created_at, recall_count, stage, heat FROM forgetting_state WHERE heat = ?",
                (heat,),
            ).fetchall()
            return [
                {
                    "memory_id": r[0],
                    "created_at": r[1],
                    "recall_count": r[2] or 0,
                    "stage": r[3],
                    "heat": r[4],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning("get_candidates_by_heat failed: %s", e)
            return []

    # ── Wiki 升级 ─────────────────────────────────────────────────────────────

    def mark_upgraded_to_wiki(self, memory_id: str, wiki_path: str) -> None:
        """标记记忆已升级到 Wiki。

        Args:
            memory_id: 记忆 ID
            wiki_path: Wiki 页面路径
        """
        assert self._conn is not None
        try:
            self._conn.execute(
                "UPDATE forgetting_state SET upgraded_to_wiki = 1, wiki_page_path = ? WHERE memory_id = ?",
                (wiki_path, memory_id),
            )
            self._pending_writes += 1
            self.flush()  # 显式 flush，防止 session 异常退出丢失标记
        except Exception as e:
            logger.warning("mark_upgraded_to_wiki failed: %s", e)

    def get_upgrade_candidates(self, min_recall: int = 2) -> list[dict[str, Any]]:
        """获取 Wiki 升级候选。

        条件：recall_count >= min_recall AND heat = 'hot' AND stage = 'active'
              AND upgraded_to_wiki = 0

        Returns:
            候选记忆列表
        """
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                """SELECT memory_id, created_at, recall_count, heat, stage
                   FROM forgetting_state
                   WHERE recall_count >= ? AND heat = 'hot' AND stage = 'active'
                   AND (upgraded_to_wiki = 0 OR upgraded_to_wiki IS NULL)
                   ORDER BY recall_count DESC""",
                (min_recall,),
            ).fetchall()
            return [
                {
                    "memory_id": r[0],
                    "created_at": r[1],
                    "recall_count": r[2] or 0,
                    "heat": r[3],
                    "stage": r[4],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning("get_upgrade_candidates failed: %s", e)
            return []

    # ── 三阶段筛选引擎 ────────────────────────────────────────────────────────

    def run_first_screening(self) -> dict[str, int]:
        """T+24h 首次筛选：基于频率密度计算热度等级。

        ★ Phase 1 优化：引入频率密度 + warm 过渡态
        热度等级:
        - hot: 7天内平均每天检索≥1次 (density >= 1.0)
        - warm: 7天内平均3天检索1次 (density >= 0.3)
        - neutral: 有检索但未达warm
        - cold: 7天内零检索

        ★ 冷启动保护：跳过管道启动前创建的历史数据。

        Returns:
            {"hot": N, "warm": N, "neutral": N, "cold": N, "skipped": N}
        """
        assert self._conn is not None
        now = datetime.now(timezone.utc)
        cutoff_24h = (now - timedelta(hours=24)).isoformat()
        counts = {"hot": 0, "warm": 0, "neutral": 0, "cold": 0, "skipped": 0}

        # ★ 冷启动：只处理管道启动后创建的记忆
        pipeline_start = self._get_pipeline_start_time()

        try:
            if pipeline_start:
                rows = self._conn.execute(
                    """SELECT memory_id, created_at FROM forgetting_state
                       WHERE created_at <= ? AND created_at >= ?""",
                    (cutoff_24h, pipeline_start),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT memory_id, created_at FROM forgetting_state
                       WHERE created_at <= ?""",
                    (cutoff_24h,),
                ).fetchall()

            for memory_id, created_at in rows:
                # 计算记忆存活天数
                created_dt = datetime.fromisoformat(created_at.replace('+00:00', ''))
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                days_alive = max(1, (now - created_dt).days)

                # 查询 7 天内检索次数
                recall_7d = self.get_recall_count_in_window(memory_id, days=7)

                # ★ 频率密度 = 检索次数 / min(7, 存活天数)
                density = recall_7d / min(7, days_alive)

                # ★ 基于频率密度判断热度
                if density >= 1.0:
                    new_heat = "hot"
                elif density >= 0.3:
                    new_heat = "warm"
                elif recall_7d == 0:
                    new_heat = "cold"
                else:
                    new_heat = "neutral"

                # 更新热度
                old_heat = self.get_heat(memory_id)
                if new_heat != old_heat:
                    self.set_heat(memory_id, new_heat)
                    counts[new_heat] += 1
                else:
                    counts[new_heat] += 1

            logger.info(
                "T+24h first screening: hot=%d, warm=%d, neutral=%d, cold=%d",
                counts["hot"], counts["warm"], counts["neutral"], counts["cold"]
            )
        except Exception as e:
            logger.warning("run_first_screening failed: %s", e)

        return counts

    def run_second_screening(self) -> dict[str, Any]:
        """T+7d 二次筛选：扫描 hot 满 7 天的记忆，按检索次数决定升级/降级。

        Returns:
            {"wiki_upgrade": [...], "demoted_to_warm": N}
        """
        assert self._conn is not None
        now = datetime.now(timezone.utc)
        cutoff_7d = (now - timedelta(days=7)).isoformat()
        result: dict[str, Any] = {"wiki_upgrade": [], "demoted_to_warm": 0}

        try:
            rows = self._conn.execute(
                """SELECT memory_id, created_at, heat_updated_at FROM forgetting_state
                   WHERE heat = 'hot' AND heat_updated_at <= ?""",
                (cutoff_7d,),
            ).fetchall()

            for memory_id, created_at, heat_updated_at in rows:
                recall_in_7d = self.get_recall_count_in_window(memory_id, days=7)
                if recall_in_7d >= 2:
                    # 持续高频 → 升级候选
                    result["wiki_upgrade"].append(memory_id)
                else:
                    # 未达阈值 → 降级为 warm
                    self.set_heat(memory_id, "warm")
                    result["demoted_to_warm"] += 1

            logger.info(
                "T+7d second screening: upgrade=%d, warm=%d",
                len(result["wiki_upgrade"]),
                result["demoted_to_warm"],
            )
        except Exception as e:
            logger.warning("run_second_screening failed: %s", e)

        return result

    def run_third_consolidation(self) -> dict[str, Any]:
        """T+30d 最终巩固：扫描 Wiki 页面交叉引用，自动晋升候选记忆。

        ★ Phase 1 优化：实现第三阶段逻辑

        流程:
        1. 查找 hot+高频且已存在30天以上的记忆
        2. 检查是否已被其他 Wiki 引用
        3. 被多次引用 → 自动晋升

        Returns:
            {"promoted": N, "monitored": N, "candidates": N}
        """
        assert self._conn is not None
        result: dict[str, Any] = {"promoted": 0, "monitored": 0, "candidates": 0}
        now = datetime.now(timezone.utc)
        cutoff_30d = (now - timedelta(days=30)).isoformat()

        try:
            # 查找候选记忆：hot + 高频访问 + 存在30天以上
            rows = self._conn.execute(
                """SELECT memory_id, recall_count, heat
                   FROM forgetting_state
                   WHERE heat = 'hot'
                     AND created_at <= ?
                     AND recall_count >= 5""",
                (cutoff_30d,)
            ).fetchall()

            result["candidates"] = len(rows)

            for memory_id, recall_count, heat in rows:
                # 检查 Wiki 引用次数
                ref_count = self._count_wiki_references(memory_id)

                if ref_count >= 2:
                    # 被多次引用 → 自动晋升
                    self._promote_to_wiki(memory_id)
                    result["promoted"] += 1
                    logger.info("Promoted %s to wiki (refs=%d)", memory_id, ref_count)
                else:
                    # 继续监控
                    result["monitored"] += 1
                    # 如果完全没有引用，可能需要降级热度
                    if ref_count == 0 and recall_count < 10:
                        self.set_heat(memory_id, "warm")

            logger.info(
                "Third consolidation: candidates=%d, promoted=%d, monitored=%d",
                result["candidates"], result["promoted"], result["monitored"]
            )

        except Exception as e:
            logger.warning("run_third_consolidation failed: %s", e)

        return result

    def _count_wiki_references(self, memory_id: str) -> int:
        """统计 Wiki 页面对该记忆的引用次数。

        搜索策略：
        1. 检查 memory_id 是否被引用
        2. 检查记忆内容的前50字符是否被引用

        Returns:
            引用次数
        """
        import os

        wiki_dir = self._governance_dir.parent / "palace"
        if not wiki_dir.exists():
            return 0

        # 获取记忆内容摘要
        memory_summary = self._get_memory_summary(memory_id)
        if not memory_summary:
            return 0

        ref_count = 0
        try:
            for root, dirs, files in os.walk(str(wiki_dir)):
                for file in files:
                    if file.endswith('.md'):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath) as f:
                                content = f.read()
                            # 检查 memory_id 或内容摘要
                            if memory_id in content or memory_summary[:50] in content:
                                ref_count += 1
                        except Exception:
                            pass
        except Exception as e:
            logger.warning("_count_wiki_references failed: %s", e)

        return ref_count

    def _get_memory_summary(self, memory_id: str) -> str:
        """获取记忆内容摘要（用于 Wiki 引用检查）。"""
        # 从 index.db 获取记忆摘要
        index_db = self._governance_dir.parent / "index" / "index.db"
        if not index_db.exists():
            return ""

        try:
            conn = sqlite3.connect(str(index_db))
            row = conn.execute(
                "SELECT content FROM memories WHERE id = ? LIMIT 1",
                (memory_id,)
            ).fetchone()
            conn.close()

            if row and row[0]:
                # 返回前 100 字符作为摘要
                return row[0][:100]
        except Exception:
            pass

        return ""

    def _promote_to_wiki(self, memory_id: str) -> bool:
        """晋升记忆到 Wiki。

        操作：
        1. 标记 upgraded_to_wiki = 1
        2. 更新阶段为 consolidating

        Returns:
            是否成功
        """
        assert self._conn is not None

        try:
            self._conn.execute(
                """UPDATE forgetting_state
                   SET upgraded_to_wiki = 1
                   WHERE memory_id = ?""",
                (memory_id,)
            )
            self._set_stage(memory_id, "consolidating")
            self._pending_writes += 1
            self._maybe_commit()

            logger.info("Promoted memory %s to wiki", memory_id)
            return True

        except Exception as e:
            logger.warning("_promote_to_wiki failed for %s: %s", memory_id, e)
            return False

    def run_warm_cooling(self) -> int:
        """warm 降温：30 天内零检索的 warm 记忆降级为 cold。

        Returns:
            降级的记忆数量
        """
        assert self._conn is not None
        demoted = 0
        try:
            rows = self._conn.execute(
                "SELECT memory_id, heat_updated_at FROM forgetting_state WHERE heat = 'warm'"
            ).fetchall()
            for memory_id, heat_updated_at in rows:
                recall_in_30d = self.get_recall_count_in_window(memory_id, days=30)
                if recall_in_30d == 0:
                    self.set_heat(memory_id, "cold")
                    demoted += 1
            if demoted > 0:
                logger.info("Warm cooling: %d memories demoted to cold", demoted)
        except Exception as e:
            logger.warning("run_warm_cooling failed: %s", e)
        return demoted

    def _check_for_reactivation(self) -> int:
        """自动升级检查：consolidating/archived 阶段的记忆如果被频繁访问，自动回到 active。

        ★ Phase 1 优化：消除单向降级限制

        规则：
        - consolidating 记忆：7天内检索 ≥ 3 次 → 升级回 active
        - archived 记忆：7天内检索 ≥ 5 次 → 升级回 active

        Returns:
            升级的记忆数量
        """
        assert self._conn is not None
        reactivated = 0

        try:
            # 检查 consolidating 记忆
            rows = self._conn.execute(
                """SELECT memory_id FROM forgetting_state
                   WHERE stage = 'consolidating'"""
            ).fetchall()

            for (memory_id,) in rows:
                recall_7d = self.get_recall_count_in_window(memory_id, days=7)
                if recall_7d >= 3:
                    self._set_stage(memory_id, "active")
                    reactivated += 1
                    logger.info("Reactivated %s from consolidating (recall_7d=%d)", memory_id, recall_7d)

            # 检查 archived 记忆
            rows = self._conn.execute(
                """SELECT memory_id FROM forgetting_state
                   WHERE stage = 'archived'"""
            ).fetchall()

            for (memory_id,) in rows:
                recall_7d = self.get_recall_count_in_window(memory_id, days=7)
                if recall_7d >= 5:
                    self._set_stage(memory_id, "active")
                    reactivated += 1
                    logger.info("Reactivated %s from archived (recall_7d=%d)", memory_id, recall_7d)

            if reactivated > 0:
                logger.info("Reactivation check: %d memories upgraded to active", reactivated)

        except Exception as e:
            logger.warning("_check_for_reactivation failed: %s", e)

        return reactivated

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
        now = datetime.now(timezone.utc)
        archived_count = 0

        # ★ Phase 1 优化：自动升级检查
        # consolidating/archived 阶段的记忆如果被频繁访问，自动回到 active
        self._check_for_reactivation()

        # 三阶段筛选 + warm 降温
        self.run_first_screening()
        self.run_second_screening()
        self.run_third_consolidation()  # ★ Phase 1 新增：T+30d Wiki 晋升
        self.run_warm_cooling()

        # ★ access_log 清理：90天前的旧记录
        self.prune_access_log(days=90)

        # ★ 自适应衰减：基于记忆类型和访问频率计算个性化阶段
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                "SELECT memory_id, created_at, stage, recall_count, memory_type FROM forgetting_state"
            ).fetchall()
        except Exception as e:
            logger.warning("Archive cycle query failed: %s", e)
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

        logger.warning("Archive cycle: %d memories archived", archived_count)
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

    # ── Phase 2: FSRS 相关方法 ──────────────────────────────────────────────────

    def calculate_fsrs_retention(self, memory_id: str) -> float:
        """使用 FSRS 计算记忆保持率

        Args:
            memory_id: 记忆 ID

        Returns:
            保持率 (0-1)
        """
        assert self._conn is not None

        try:
            row = self._conn.execute(
                """SELECT recall_count, created_at, last_accessed
                   FROM forgetting_state WHERE memory_id = ?""",
                (memory_id,)
            ).fetchone()

            if not row:
                return 1.0

            recall_count, created_at, last_accessed = row
            now = datetime.now(timezone.utc)

            # 计算创建天数
            if created_at:
                created_dt = datetime.fromisoformat(created_at.replace('+00:00', ''))
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                days_since_creation = max(1, (now - created_dt).days)
            else:
                days_since_creation = 1

            # 计算最后检索距今天数
            if last_accessed:
                accessed_dt = datetime.fromisoformat(last_accessed.replace('+00:00', ''))
                if accessed_dt.tzinfo is None:
                    accessed_dt = accessed_dt.replace(tzinfo=timezone.utc)
                last_recall_days_ago = max(0, (now - accessed_dt).days)
            else:
                last_recall_days_ago = days_since_creation

            # 使用 FSRS 计算保持率
            return self._fsrs.calculate_retention_from_recall(
                recall_count or 0, days_since_creation, last_recall_days_ago
            )

        except Exception as e:
            logger.warning("calculate_fsrs_retention failed for %s: %s", memory_id, e)
            return 0.5

    def suggest_review_time(self, memory_id: str, desired_retention: float = 0.9) -> Optional[datetime]:
        """建议下次复习时间

        Args:
            memory_id: 记忆 ID
            desired_retention: 目标保持率

        Returns:
            建议复习时间，失败返回 None
        """
        assert self._conn is not None

        try:
            row = self._conn.execute(
                """SELECT recall_count, created_at, last_accessed
                   FROM forgetting_state WHERE memory_id = ?""",
                (memory_id,)
            ).fetchone()

            if not row:
                return None

            recall_count, created_at, last_accessed = row
            now = datetime.now(timezone.utc)

            # 创建 FSRSItem
            item = FSRSItem()
            item.reps = recall_count or 0

            if created_at:
                created_dt = datetime.fromisoformat(created_at.replace('+00:00', ''))
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)

            if last_accessed:
                accessed_dt = datetime.fromisoformat(last_accessed.replace('+00:00', ''))
                if accessed_dt.tzinfo is None:
                    accessed_dt = accessed_dt.replace(tzinfo=timezone.utc)
                item.last_review = accessed_dt

            # 估算稳定性
            if recall_count and recall_count > 0:
                item.stability = min(100.0, recall_count * 2.0)
            else:
                item.stability = 0.5

            # 计算建议复习时间
            return self._fsrs.suggest_review(item, now, desired_retention)

        except Exception as e:
            logger.warning("suggest_review_time failed for %s: %s", memory_id, e)
            return None

    def get_fsrs_stats(self) -> dict[str, Any]:
        """获取 FSRS 统计信息

        Returns:
            包含保持率分布、平均稳定性等统计信息
        """
        assert self._conn is not None

        stats = {
            "total_memories": 0,
            "avg_retention": 0.0,
            "avg_stability": 0.0,
            "retention_distribution": {
                "high": 0,    # > 0.8
                "medium": 0,  # 0.5 - 0.8
                "low": 0,     # < 0.5
            }
        }

        try:
            rows = self._conn.execute(
                "SELECT memory_id, recall_count FROM forgetting_state"
            ).fetchall()

            stats["total_memories"] = len(rows)
            retentions = []
            stabilities = []

            for memory_id, recall_count in rows:
                # 计算保持率
                retention = self.calculate_fsrs_retention(memory_id)
                retentions.append(retention)

                # 估算稳定性
                stability = min(100.0, (recall_count or 0) * 2.0)
                stabilities.append(stability)

                # 分类
                if retention > 0.8:
                    stats["retention_distribution"]["high"] += 1
                elif retention > 0.5:
                    stats["retention_distribution"]["medium"] += 1
                else:
                    stats["retention_distribution"]["low"] += 1

            if retentions:
                stats["avg_retention"] = sum(retentions) / len(retentions)
            if stabilities:
                stats["avg_stability"] = sum(stabilities) / len(stabilities)

        except Exception as e:
            logger.warning("get_fsrs_stats failed: %s", e)

        return stats

    # ── Phase 3: 记忆强度评估方法 ──────────────────────────────────────────────

    def evaluate_memory_strength(self, memory_id: str) -> dict[str, Any]:
        """评估单个记忆的强度

        Args:
            memory_id: 记忆 ID

        Returns:
            包含强度向量、综合评分、等级的字典
        """
        assert self._conn is not None

        try:
            row = self._conn.execute(
                """SELECT recall_count, created_at, last_accessed
                   FROM forgetting_state WHERE memory_id = ?""",
                (memory_id,)
            ).fetchone()

            if not row:
                return {"memory_id": memory_id, "error": "not found"}

            recall_count, created_at, last_accessed = row

            # 获取 FSRS 保持率
            fsrs_retention = self.calculate_fsrs_retention(memory_id)

            # 估算 FSRS 稳定性和难度
            fsrs_stability = min(100.0, (recall_count or 0) * 2.0)
            fsrs_difficulty = 0.5  # 默认中等难度

            # 使用评估器
            return self._evaluator.evaluate_memory(
                memory_id=memory_id,
                recall_count=recall_count or 0,
                created_at=created_at,
                last_accessed=last_accessed,
                fsrs_retention=fsrs_retention,
                fsrs_stability=fsrs_stability,
                fsrs_difficulty=fsrs_difficulty,
            )

        except Exception as e:
            logger.warning("evaluate_memory_strength failed for %s: %s", memory_id, e)
            return {"memory_id": memory_id, "error": str(e)}

    def evaluate_all_memories(self, limit: int = 100) -> dict[str, Any]:
        """评估所有记忆的强度

        Args:
            limit: 最大评估数量

        Returns:
            包含评估结果和分布统计的字典
        """
        assert self._conn is not None

        results = []

        try:
            rows = self._conn.execute(
                """SELECT memory_id, recall_count, created_at, last_accessed
                   FROM forgetting_state
                   ORDER BY recall_count DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()

            for memory_id, recall_count, created_at, last_accessed in rows:
                # 获取 FSRS 保持率
                fsrs_retention = self.calculate_fsrs_retention(memory_id)

                # 估算 FSRS 稳定性和难度
                fsrs_stability = min(100.0, (recall_count or 0) * 2.0)
                fsrs_difficulty = 0.5

                result = self._evaluator.evaluate_memory(
                    memory_id=memory_id,
                    recall_count=recall_count or 0,
                    created_at=created_at,
                    last_accessed=last_accessed,
                    fsrs_retention=fsrs_retention,
                    fsrs_stability=fsrs_stability,
                    fsrs_difficulty=fsrs_difficulty,
                )
                results.append(result)

            # 获取分布统计
            distribution = self._evaluator.get_distribution(results)

            return {
                "evaluated": len(results),
                "distribution": distribution,
                "top_memories": sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:10],
                "bottom_memories": sorted(results, key=lambda x: x.get("score", 0))[:10],
            }

        except Exception as e:
            logger.warning("evaluate_all_memories failed: %s", e)
            return {"error": str(e)}

    def get_memory_grade(self, memory_id: str) -> str:
        """获取记忆等级

        Args:
            memory_id: 记忆 ID

        Returns:
            等级 (S/A/B/C/D)
        """
        result = self.evaluate_memory_strength(memory_id)
        return result.get("grade", "D")

    def get_strength_distribution(self) -> dict[str, Any]:
        """获取记忆强度分布统计

        Returns:
            包含等级分布、平均分数等统计
        """
        result = self.evaluate_all_memories(limit=1000)
        return result.get("distribution", {})

    # ── Phase 4: 语义重要性评估方法 ──────────────────────────────────────────────

    def evaluate_semantic_importance(self, memory_id: str) -> dict[str, Any]:
        """评估记忆的语义重要性

        Args:
            memory_id: 记忆 ID

        Returns:
            包含语义特征和综合重要性的字典
        """
        assert self._conn is not None

        try:
            # 获取记忆内容
            content = self._get_memory_content(memory_id)

            # 使用语义评估器
            return self._semantic_evaluator.evaluate_importance(memory_id, content)

        except Exception as e:
            logger.warning("evaluate_semantic_importance failed for %s: %s", memory_id, e)
            return {"memory_id": memory_id, "error": str(e)}

    def _get_memory_content(self, memory_id: str) -> Optional[str]:
        """获取记忆内容

        Args:
            memory_id: 记忆 ID

        Returns:
            记忆内容，失败返回 None
        """
        # 从 index.db 获取记忆内容
        index_db = self._governance_dir.parent / "index" / "index.db"
        if not index_db.exists():
            return None

        try:
            conn = sqlite3.connect(str(index_db))
            row = conn.execute(
                "SELECT content FROM memories WHERE id = ? LIMIT 1",
                (memory_id,)
            ).fetchone()
            conn.close()

            if row and row[0]:
                return row[0]
        except Exception:
            pass

        return None

    def get_semantic_importance_distribution(self) -> dict[str, Any]:
        """获取语义重要性分布统计

        Returns:
            包含重要性分布的字典
        """
        assert self._conn is not None

        distribution = {
            "high": 0,    # > 0.7
            "medium": 0,  # 0.4 - 0.7
            "low": 0,     # < 0.4
            "total": 0,
            "avg_importance": 0.0,
        }

        try:
            rows = self._conn.execute(
                "SELECT memory_id FROM forgetting_state"
            ).fetchall()

            distribution["total"] = len(rows)
            importances = []

            for (memory_id,) in rows:
                result = self.evaluate_semantic_importance(memory_id)
                importance = result.get("importance", 0.5)
                importances.append(importance)

                if importance > 0.7:
                    distribution["high"] += 1
                elif importance > 0.4:
                    distribution["medium"] += 1
                else:
                    distribution["low"] += 1

            if importances:
                distribution["avg_importance"] = sum(importances) / len(importances)

        except Exception as e:
            logger.warning("get_semantic_importance_distribution failed: %s", e)

        return distribution

    def get_status(self) -> dict[str, Any]:
        """获取遗忘状态概览（含热度分类和升级候选）。"""
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
            assert self._conn is not None
            self._conn.commit()
            self._pending_writes = 0

    def flush(self) -> None:
        """显式提交所有待写入。"""
        if self._conn and self._pending_writes > 0:
            try:
                self._conn.commit()
                self._pending_writes = 0
            except Exception as e:
                logger.warning("Forgetting flush failed: %s", e)

    def close(self) -> None:
        """关闭数据库连接。"""
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None
