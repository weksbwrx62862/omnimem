"""ForgettingCurve — Ebbinghaus 遗忘曲线驱动的4阶段归档。

4个阶段:
  - active (0-7天): 完整保留，正常检索
  - consolidating (7-30天): 可能需要提示，降权但不归档
  - archived (30-90天): 仅摘要可用，原文归档
  - forgotten (90天+): 仅L0索引可用，需要显式召回

归档操作:
  - archive(memory_id): 将记忆从 active 降级到 archived
  - reactivate(memory_id): 将记忆从 archived/forgotten 恢复到 active
  - run_archive_cycle(): 后台运行归档周期
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 4个阶段定义
STAGES = {
    "active": (0, 7),
    "consolidating": (7, 30),
    "archived": (30, 90),
    "forgotten": (90, None),
}


class ForgettingCurve:
    """Ebbinghaus 遗忘曲线驱动的4阶段归档。

    批量提交优化：写操作攒到阈值或显式 flush/close 时统一提交。
    """

    _BATCH_THRESHOLD = 5

    def __init__(self, governance_dir: Path):
        self._governance_dir = governance_dir
        self._governance_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._governance_dir / "forgetting.db"
        self._conn: sqlite3.Connection | None = None
        self._pending_writes = 0
        self._init_db()

    def _init_db(self) -> None:
        """初始化遗忘数据库。"""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forgetting_state (
                memory_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL DEFAULT 'active',
                last_accessed TEXT,
                created_at TEXT,
                archive_count INTEGER DEFAULT 0,
                recall_count INTEGER DEFAULT 0
            )
        """
        )
        # ★ 兼容旧表：如果 recall_count 列不存在则添加
        try:
            self._conn.execute(
                "ALTER TABLE forgetting_state ADD COLUMN recall_count INTEGER DEFAULT 0"
            )
        except Exception:
            pass  # 列已存在
        self._conn.commit()

    def get_stage(self, memory_id: str) -> str:
        """获取记忆的当前阶段。"""
        try:
            row = self._conn.execute(
                "SELECT stage FROM forgetting_state WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row:
                return row[0]
        except Exception as e:
            logger.debug("Forgetting stage query failed: %s", e)
        return "active"

    def get_stage_by_age(self, days: int) -> str:
        """根据天数计算阶段。"""
        for stage, (min_days, max_days) in STAGES.items():
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
        try:
            self._conn.execute(
                "UPDATE forgetting_state SET last_accessed = ? WHERE memory_id = ?",
                (now, memory_id),
            )
            self._pending_writes += 1
            self._maybe_commit()
        except Exception as e:
            logger.debug("Reactivate update failed: %s", e)

    def record_access(self, memory_id: str) -> None:
        """记录记忆被访问（重置遗忘计时器 + 增加召回计数）。"""
        now = datetime.now(timezone.utc).isoformat()
        try:
            existing = self._conn.execute(
                "SELECT recall_count FROM forgetting_state WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if existing is not None:
                new_count = (existing[0] or 0) + 1
                self._conn.execute(
                    "UPDATE forgetting_state SET stage = 'active', last_accessed = ?, recall_count = ? WHERE memory_id = ?",
                    (now, new_count, memory_id),
                )
            else:
                self._conn.execute(
                    """INSERT OR REPLACE INTO forgetting_state
                       (memory_id, stage, last_accessed, created_at, recall_count)
                       VALUES (?, 'active', ?, ?, 1)""",
                    (memory_id, now, now),
                )
            self._pending_writes += 1
            self._maybe_commit()
        except Exception as e:
            logger.debug("Access record failed: %s", e)

    def run_archive_cycle(self) -> int:
        """后台运行：将过期记忆降级。

        ★ 访问衰减：从未被召回（recall_count=0）的记忆加速遗忘。
        有召回的记忆按正常阶段走，没召回的记忆阶段阈值减半。

        Returns:
            归档的记忆数量
        """
        now = datetime.now(timezone.utc)
        archived_count = 0

        try:
            rows = self._conn.execute(
                "SELECT memory_id, created_at, stage, recall_count FROM forgetting_state"
            ).fetchall()
        except Exception:
            return 0

        for memory_id, created_at, stage, recall_count in rows:
            try:
                if not created_at:
                    continue
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                days = (now - created_dt).days

                # ★ 访问衰减：recall_count=0 的记忆，阈值减半
                if recall_count is None or recall_count == 0:
                    # 加速遗忘：active→3天, consolidating→15天, archived→45天
                    accelerated_stages = {
                        "active": (0, 3),
                        "consolidating": (3, 15),
                        "archived": (15, 45),
                        "forgotten": (45, None),
                    }
                    expected_stage = self._get_stage_by_age_custom(days, accelerated_stages)
                else:
                    expected_stage = self.get_stage_by_age(days)

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
                logger.debug("Archive cycle failed for %s: %s", memory_id, e)

        logger.debug("Archive cycle: %d memories archived", archived_count)
        return archived_count

    @staticmethod
    def _get_stage_by_age_custom(days: int, stages: dict) -> str:
        """根据天数和自定义阶段定义计算阶段。"""
        for stage, (min_days, max_days) in stages.items():
            if max_days is None:
                if days >= min_days:
                    return stage
            elif min_days <= days < max_days:
                return stage
        return "active"

    def get_status(self) -> dict[str, Any]:
        """获取遗忘状态概览。"""
        counts = {"active": 0, "consolidating": 0, "archived": 0, "forgotten": 0}
        try:
            rows = self._conn.execute(
                "SELECT stage, COUNT(*) FROM forgetting_state GROUP BY stage"
            ).fetchall()
            for stage, count in rows:
                if stage in counts:
                    counts[stage] = count
        except Exception:
            pass
        return counts

    def get_archived_ids(self, limit: int = 5000) -> list[str]:
        """获取已归档（archived 或 forgotten）的记忆 ID 列表。

        Args:
            limit: 最大返回数量

        Returns:
            memory_id 列表
        """
        try:
            rows = self._conn.execute(
                "SELECT memory_id FROM forgetting_state WHERE stage IN ('archived', 'forgotten') LIMIT ?",
                (limit,),
            ).fetchall()
            return [r[0] for r in rows if r[0]]
        except Exception as e:
            logger.debug("Get archived ids failed: %s", e)
            return []

    def _set_stage(self, memory_id: str, stage: str) -> None:
        """设置记忆的阶段。"""
        now = datetime.now(timezone.utc).isoformat()
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
            logger.debug("Stage update failed: %s", e)

    def _maybe_commit(self) -> None:
        """到达阈值时提交。"""
        if self._pending_writes >= self._BATCH_THRESHOLD:
            self._conn.commit()
            self._pending_writes = 0

    def flush(self) -> None:
        """显式提交所有待写入。"""
        if self._conn and self._pending_writes > 0:
            try:
                self._conn.commit()
                self._pending_writes = 0
            except Exception as e:
                logger.debug("Forgetting flush failed: %s", e)

    def close(self) -> None:
        """关闭数据库连接。"""
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None
