"""ForgettingCurve — 阶段管理 Mixin。"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger("governance.forgetting")
from omnimem.utils.migration import SchemaMigrator
from omnimem.governance.forgetting_core import HEAT_LEVELS, STAGES


class _ForgettingStages:
    """阶段管理 Mixin：冷启动、access_log、阶段查询、访问记录、热度分类、Wiki升级。"""

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
        with self._lock:
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

    # ── 阶段管理 ──────────────────────────────────────────────────────────────

    def get_stage(self, memory_id: str) -> str:
        """获取记忆的当前阶段。"""
        with self._lock:
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
        with self._lock:
            current = self.get_stage(memory_id)
            if current == "forgotten":
                return
            new_stage = "archived"
            if current == "archived":
                new_stage = "forgotten"
            self._set_stage(memory_id, new_stage)

    def reactivate(self, memory_id: str) -> None:
        """将记忆重新激活（恢复到 active）。"""
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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

