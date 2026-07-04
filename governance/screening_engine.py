"""ForgettingScreening — 三阶段筛选引擎。

从 ForgettingCurve 拆分的子模块，负责：
1. T+24h 首次筛选：基于频率密度计算热度等级
2. T+7d 二次筛选：hot 记忆升级/降级决策
3. T+30d 最终巩固：Wiki 交叉引用扫描 + 自动晋升
4. warm 降温：30 天零检索降级为 cold
5. 自动升级检查：高频访问记忆回到 active

子模块内部不加锁，由 ForgettingCurve._lock 保护。
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ForgettingScreening:
    """三阶段筛选引擎。

    通过回调访问 ForgettingCurve 的数据库连接和操作方法，
    避免与 ForgettingCurve 产生循环依赖。
    """

    def __init__(
        self,
        governance_dir: Path,
        get_conn: Callable[[], sqlite3.Connection],
        get_index_conn: Callable[[], sqlite3.Connection | None],
        get_recall_count_in_window: Callable[[str, int], int],
        get_heat: Callable[[str], str],
        set_heat: Callable[[str, str], None],
        set_stage: Callable[[str, str], None],
        track_write: Callable[[], None],
        get_pipeline_start_time: Callable[[], str | None],
    ):
        """
        Args:
            governance_dir: 治理目录路径
            get_conn: 回调，返回 forgetting_state 数据库连接
            get_index_conn: 回调，返回 index.db 数据库连接（可返回 None）
            get_recall_count_in_window: 回调，查询时间窗口内检索次数
            get_heat: 回调，获取记忆热度
            set_heat: 回调，设置记忆热度
            set_stage: 回调，设置记忆阶段
            track_write: 回调，记录一次待写入并检查批量提交
            get_pipeline_start_time: 回调，获取管道启动时间
        """
        self._governance_dir = governance_dir
        self._get_conn = get_conn
        self._get_index_conn = get_index_conn
        self._get_recall_count_in_window = get_recall_count_in_window
        self._get_heat = get_heat
        self._set_heat = set_heat
        self._set_stage = set_stage
        self._track_write = track_write
        self._get_pipeline_start_time = get_pipeline_start_time

    # ── 第一阶段：T+24h 首次筛选 ──────────────────────────────────────────

    def run_first_screening(self) -> dict[str, int]:
        """T+24h 首次筛选：基于频率密度计算热度等级。

        热度等级:
        - hot: 7天内平均每天检索≥1次 (density >= 1.0)
        - warm: 7天内平均3天检索1次 (density >= 0.3)
        - neutral: 有检索但未达warm
        - cold: 7天内零检索

        冷启动保护：跳过管道启动前创建的历史数据。

        Returns:
            {"hot": N, "warm": N, "neutral": N, "cold": N, "skipped": N}
        """
        conn = self._get_conn()
        now = datetime.now(timezone.utc)
        cutoff_24h = (now - timedelta(hours=24)).isoformat()
        counts: dict[str, int] = {"hot": 0, "warm": 0, "neutral": 0, "cold": 0, "skipped": 0}

        # 冷启动：只处理管道启动后创建的记忆
        pipeline_start = self._get_pipeline_start_time()

        try:
            if pipeline_start:
                rows = conn.execute(
                    """SELECT memory_id, created_at FROM forgetting_state
                       WHERE created_at <= ? AND created_at >= ?""",
                    (cutoff_24h, pipeline_start),
                ).fetchall()
            else:
                rows = conn.execute(
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
                recall_7d = self._get_recall_count_in_window(memory_id, days=7)

                # 频率密度 = 检索次数 / min(7, 存活天数)
                density = recall_7d / min(7, days_alive)

                # 基于频率密度判断热度
                if density >= 1.0:
                    new_heat = "hot"
                elif density >= 0.3:
                    new_heat = "warm"
                elif recall_7d == 0:
                    new_heat = "cold"
                else:
                    new_heat = "neutral"

                # 更新热度
                old_heat = self._get_heat(memory_id)
                if new_heat != old_heat:
                    self._set_heat(memory_id, new_heat)
                counts[new_heat] += 1

            logger.info(
                "T+24h first screening: hot=%d, warm=%d, neutral=%d, cold=%d",
                counts["hot"], counts["warm"], counts["neutral"], counts["cold"]
            )
        except Exception as e:
            logger.warning("run_first_screening failed: %s", e)

        return counts

    # ── 第二阶段：T+7d 二次筛选 ──────────────────────────────────────────

    def run_second_screening(self) -> dict[str, Any]:
        """T+7d 二次筛选：扫描 hot 满 7 天的记忆，按检索次数决定升级/降级。

        Returns:
            {"wiki_upgrade": [...], "demoted_to_warm": N}
        """
        conn = self._get_conn()
        now = datetime.now(timezone.utc)
        cutoff_7d = (now - timedelta(days=7)).isoformat()
        result: dict[str, Any] = {"wiki_upgrade": [], "demoted_to_warm": 0}

        try:
            rows = conn.execute(
                """SELECT memory_id, created_at, heat_updated_at FROM forgetting_state
                   WHERE heat = 'hot' AND heat_updated_at <= ?""",
                (cutoff_7d,),
            ).fetchall()

            for memory_id, created_at, heat_updated_at in rows:
                recall_in_7d = self._get_recall_count_in_window(memory_id, days=7)
                if recall_in_7d >= 2:
                    # 持续高频 → 升级候选
                    result["wiki_upgrade"].append(memory_id)
                else:
                    # 未达阈值 → 降级为 warm
                    self._set_heat(memory_id, "warm")
                    result["demoted_to_warm"] += 1

            logger.info(
                "T+7d second screening: upgrade=%d, warm=%d",
                len(result["wiki_upgrade"]),
                result["demoted_to_warm"],
            )
        except Exception as e:
            logger.warning("run_second_screening failed: %s", e)

        return result

    # ── 第三阶段：T+30d 最终巩固 ──────────────────────────────────────────

    def run_third_consolidation(self) -> dict[str, Any]:
        """T+30d 最终巩固：扫描 Wiki 页面交叉引用，自动晋升候选记忆。

        流程:
        1. 查找 hot+高频且已存在30天以上的记忆
        2. 检查是否已被其他 Wiki 引用
        3. 被多次引用 → 自动晋升

        Returns:
            {"promoted": N, "monitored": N, "candidates": N}
        """
        conn = self._get_conn()
        result: dict[str, Any] = {"promoted": 0, "monitored": 0, "candidates": 0}
        now = datetime.now(timezone.utc)
        cutoff_30d = (now - timedelta(days=30)).isoformat()

        try:
            # 查找候选记忆：hot + 高频访问 + 存在30天以上
            rows = conn.execute(
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
                        self._set_heat(memory_id, "warm")

            logger.info(
                "Third consolidation: candidates=%d, promoted=%d, monitored=%d",
                result["candidates"], result["promoted"], result["monitored"]
            )

        except Exception as e:
            logger.warning("run_third_consolidation failed: %s", e)

        return result

    # ── warm 降温 ──────────────────────────────────────────────────────────

    def run_warm_cooling(self) -> int:
        """warm 降温：30 天内零检索的 warm 记忆降级为 cold。

        Returns:
            降级的记忆数量
        """
        conn = self._get_conn()
        demoted = 0
        try:
            rows = conn.execute(
                "SELECT memory_id, heat_updated_at FROM forgetting_state WHERE heat = 'warm'"
            ).fetchall()
            for memory_id, heat_updated_at in rows:
                recall_in_30d = self._get_recall_count_in_window(memory_id, days=30)
                if recall_in_30d == 0:
                    self._set_heat(memory_id, "cold")
                    demoted += 1
            if demoted > 0:
                logger.info("Warm cooling: %d memories demoted to cold", demoted)
        except Exception as e:
            logger.warning("run_warm_cooling failed: %s", e)
        return demoted

    # ── 自动升级检查 ──────────────────────────────────────────────────────

    def check_for_reactivation(self) -> int:
        """自动升级检查：consolidating/archived 阶段的记忆如果被频繁访问，自动回到 active。

        规则：
        - consolidating 记忆：7天内检索 ≥ 3 次 → 升级回 active
        - archived 记忆：7天内检索 ≥ 5 次 → 升级回 active

        Returns:
            升级的记忆数量
        """
        conn = self._get_conn()
        reactivated = 0

        try:
            # 检查 consolidating 记忆
            rows = conn.execute(
                """SELECT memory_id FROM forgetting_state
                   WHERE stage = 'consolidating'"""
            ).fetchall()

            for (memory_id,) in rows:
                recall_7d = self._get_recall_count_in_window(memory_id, days=7)
                if recall_7d >= 3:
                    self._set_stage(memory_id, "active")
                    reactivated += 1
                    logger.info("Reactivated %s from consolidating (recall_7d=%d)", memory_id, recall_7d)

            # 检查 archived 记忆
            rows = conn.execute(
                """SELECT memory_id FROM forgetting_state
                   WHERE stage = 'archived'"""
            ).fetchall()

            for (memory_id,) in rows:
                recall_7d = self._get_recall_count_in_window(memory_id, days=7)
                if recall_7d >= 5:
                    self._set_stage(memory_id, "active")
                    reactivated += 1
                    logger.info("Reactivated %s from archived (recall_7d=%d)", memory_id, recall_7d)

            if reactivated > 0:
                logger.info("Reactivation check: %d memories upgraded to active", reactivated)

        except Exception as e:
            logger.warning("check_for_reactivation failed: %s", e)

        return reactivated

    # ── 内部辅助方法 ──────────────────────────────────────────────────────

    def _count_wiki_references(self, memory_id: str) -> int:
        """统计 Wiki 页面对该记忆的引用次数。

        搜索策略：
        1. 检查 memory_id 是否被引用
        2. 检查记忆内容的前50字符是否被引用

        Returns:
            引用次数
        """
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
                            logger.debug("Forgetting: failed to read wiki file %s", filepath, exc_info=True)
        except Exception as e:
            logger.warning("_count_wiki_references failed: %s", e)

        return ref_count

    def _get_memory_summary(self, memory_id: str) -> str:
        """获取记忆内容摘要（用于 Wiki 引用检查）。"""
        try:
            conn = self._get_index_conn()
            if conn is None:
                return ""
            row = conn.execute(
                "SELECT content FROM memories WHERE id = ? LIMIT 1",
                (memory_id,)
            ).fetchone()

            if row and row[0]:
                # 返回前 100 字符作为摘要
                return row[0][:100]
        except Exception:
            logger.warning("ForgettingScreening: get_memory_summary DB query failed", exc_info=True)

        return ""

    def _promote_to_wiki(self, memory_id: str) -> bool:
        """晋升记忆到 Wiki。

        操作：
        1. 标记 upgraded_to_wiki = 1
        2. 更新阶段为 consolidating

        Returns:
            是否成功
        """
        conn = self._get_conn()

        try:
            conn.execute(
                """UPDATE forgetting_state
                   SET upgraded_to_wiki = 1
                   WHERE memory_id = ?""",
                (memory_id,)
            )
            self._set_stage(memory_id, "consolidating")
            self._track_write()

            logger.info("Promoted memory %s to wiki", memory_id)
            return True

        except Exception as e:
            logger.warning("_promote_to_wiki failed for %s: %s", memory_id, e)
            return False
