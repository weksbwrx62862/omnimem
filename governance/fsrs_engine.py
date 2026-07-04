"""
FSRS (Free Spaced Repetition Scheduler) v4 引擎实现。

参考: https://github.com/open-spaced-repetition/fsrs4anki

FSRS 是目前最优的间隔重复算法：
- 保持率预测精度: 85-90% (vs SM-2 的 60-70%)
- 19 个可学习参数
- 已被 Anki 采用

核心公式:
- 遗忘曲线: R(t,S) = (1 + t/(S * α))^(-β)
- 间隔计算: I = S * (R^(-1/β) - 1)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FSRSItem:
    """FSRS 记忆项状态"""
    stability: float = 0.0      # 稳定性 (天)
    difficulty: float = 0.0     # 难度 (0-1)
    due: datetime | None = None
    last_review: datetime | None = None
    elapsed_days: int = 0
    scheduled_days: int = 0
    reps: int = 0               # 复习次数
    lapses: int = 0             # 遗忘次数
    state: int = 0              # 0=New, 1=Learning, 2=Review, 3=Relearning


@dataclass
class FSRSParameters:
    """FSRS 参数（可学习）

    基于大量用户数据训练的默认参数
    """
    w: list = field(default_factory=lambda: [
        # 初始稳定性 (Again, Hard, Good, Easy)
        0.4, 0.6, 2.4, 5.8,
        # 初始难度参数
        4.93, 0.94,
        # 稳定性增长参数
        0.86, 0.01, 1.49, 0.14, 0.94, 2.18,
        # 难度调整参数
        0.05, 0.34, 1.26, 0.29, 2.61,
        # 遗忘曲线参数 (α, β)
        0.0, 0.0
    ])

    # 默认 α 和 β（当 w[17], w[18] 为 0 时使用）
    default_alpha: float = 9.0
    default_beta: float = 0.5


class FSRSEngine:
    """FSRS 引擎

    提供:
    - 遗忘曲线计算
    - 间隔预测
    - 稳定性/难度更新
    - 个性化参数学习
    """

    def __init__(self, parameters: FSRSParameters | None = None):
        self.p = parameters or FSRSParameters()
        self._alpha = self.p.w[17] if self.p.w[17] > 0 else self.p.default_alpha
        self._beta = self.p.w[18] if self.p.w[18] > 0 else self.p.default_beta

    def forgetting_curve(self, t: float, s: float) -> float:
        """FSRS 遗忘曲线模型

        R(t,S) = (1 + t/(S * α))^(-β)

        Args:
            t: 经过的时间（天）
            s: 稳定性（天）

        Returns:
            保持率 (0-1)
        """
        if s <= 0 or t < 0:
            return 1.0 if t == 0 else 0.0

        return (1 + t / (s * self._alpha)) ** (-self._beta)

    def next_interval(self, s: float, desired_retention: float = 0.9) -> int:
        """计算下次复习间隔

        I = S * (R^(-1/β) - 1)

        Args:
            s: 当前稳定性
            desired_retention: 目标保持率

        Returns:
            间隔天数
        """
        if s <= 0 or desired_retention <= 0 or desired_retention >= 1:
            return 1

        interval = s * (desired_retention ** (-1 / self._beta) - 1)
        return max(1, round(interval))

    def init_stability(self, rating: int) -> float:
        """初始化稳定性

        Args:
            rating: 评分 (1=Again, 2=Hard, 3=Good, 4=Easy)
        """
        rating = max(1, min(4, rating))
        return max(0.1, self.p.w[rating - 1])

    def init_difficulty(self, rating: int) -> float:
        """初始化难度

        Args:
            rating: 评分 (1=Again, 2=Hard, 3=Good, 4=Easy)
        """
        rating = max(1, min(4, rating))
        return max(0.0, min(1.0, self.p.w[4] - self.p.w[5] * (rating - 3)))

    def update_stability(self, d: float, s: float, r: float, rating: int) -> float:
        """更新稳定性

        Args:
            d: 难度 (0-1)
            s: 当前稳定性
            r: 当前保持率
            rating: 评分 (1=Again, 2=Hard, 3=Good, 4=Easy)
        """
        if s <= 0:
            return self.init_stability(rating)

        rating = max(1, min(4, rating))

        # 稳定性增长率
        if rating == 1:  # Again - 遗忘
            growth = 0.0
        else:
            # 基础增长
            growth = self.p.w[6] * (11 - d) * s ** (-self.p.w[7]) * (
                math.exp(self.p.w[8] * (1 - r)) - 1
            )

            # 困难/容易调整
            if rating == 2:  # Hard
                growth *= self.p.w[9]
            elif rating == 4:  # Easy
                growth *= self.p.w[10]

        new_stability = s * (1 + growth)
        return max(0.1, new_stability)

    def update_difficulty(self, d: float, rating: int) -> float:
        """更新难度

        Args:
            d: 当前难度
            rating: 评分 (1=Again, 2=Hard, 3=Good, 4=Easy)
        """
        rating = max(1, min(4, rating))

        # 难度调整
        delta_d = -self.p.w[11] * (rating - 3)
        new_d = d + delta_d

        return max(0.0, min(1.0, new_d))

    def predict_retention(self, item: FSRSItem, now: datetime) -> float:
        """预测当前保持率

        Args:
            item: 记忆项
            now: 当前时间

        Returns:
            保持率 (0-1)
        """
        if item.stability <= 0 or item.last_review is None:
            return 1.0 if item.state == 0 else 0.0

        # 计算经过的时间
        elapsed = (now - item.last_review).total_seconds() / 86400

        return self.forgetting_curve(elapsed, item.stability)

    def suggest_review(self, item: FSRSItem, now: datetime, desired_retention: float = 0.9) -> datetime:
        """建议下次复习时间

        Args:
            item: 记忆项
            now: 当前时间
            desired_retention: 目标保持率

        Returns:
            建议复习时间
        """
        interval = self.next_interval(item.stability, desired_retention)

        if item.last_review is None:
            return now + timedelta(days=interval)

        return item.last_review + timedelta(days=interval)

    def review(self, item: FSRSItem, rating: int, now: datetime) -> FSRSItem:
        """执行一次复习

        Args:
            item: 记忆项
            rating: 评分 (1=Again, 2=Hard, 3=Good, 4=Easy)
            now: 复习时间

        Returns:
            更新后的记忆项
        """
        rating = max(1, min(4, rating))

        # 计算当前保持率
        if item.last_review is not None:
            elapsed = (now - item.last_review).total_seconds() / 86400
            r = self.forgetting_curve(elapsed, item.stability)
        else:
            r = 1.0

        # 更新状态
        if item.state == 0:  # New
            item.stability = self.init_stability(rating)
            item.difficulty = self.init_difficulty(rating)
            item.state = 1 if rating < 4 else 2  # Learning or Review
        elif item.state == 1:  # Learning
            if rating == 1:  # Again
                item.stability = self.init_stability(1)
                item.lapses += 1
            else:
                item.stability = self.update_stability(item.difficulty, item.stability, r, rating)
                item.state = 2  # Review
        elif item.state == 2:  # Review
            if rating == 1:  # Again - 遗忘
                item.stability = self.init_stability(1)
                item.difficulty = self.update_difficulty(item.difficulty, rating)
                item.state = 3  # Relearning
                item.lapses += 1
            else:
                item.stability = self.update_stability(item.difficulty, item.stability, r, rating)
                item.difficulty = self.update_difficulty(item.difficulty, rating)
        elif item.state == 3:  # Relearning
            if rating == 1:  # Again
                item.stability = self.init_stability(1)
                item.lapses += 1
            else:
                item.stability = self.update_stability(item.difficulty, item.stability, r, rating)
                item.state = 2  # Review

        # 更新时间
        if item.last_review is not None:
            item.elapsed_days = int((now - item.last_review).total_seconds() / 86400)
        item.last_review = now
        item.reps += 1

        # 计算下次复习间隔
        item.scheduled_days = self.next_interval(item.stability, 0.9)
        item.due = now + timedelta(days=item.scheduled_days)

        return item

    def calculate_retention_from_recall(
        self,
        recall_count: int,
        days_since_creation: int,
        last_recall_days_ago: int = 0
    ) -> float:
        """基于访问历史估算保持率

        用于将现有数据迁移到 FSRS

        Args:
            recall_count: 总检索次数
            days_since_creation: 创建天数
            last_recall_days_ago: 最后检索距今天数

        Returns:
            估算的保持率
        """
        # 估算稳定性（基于检索次数）
        if recall_count == 0:
            estimated_stability = 0.5
        elif recall_count == 1:
            estimated_stability = 2.0
        elif recall_count <= 3:
            estimated_stability = 5.0
        elif recall_count <= 5:
            estimated_stability = 10.0
        else:
            estimated_stability = min(100.0, recall_count * 2.0)

        # 计算保持率
        return self.forgetting_curve(last_recall_days_ago, estimated_stability)

    def estimate_parameters_from_data(self, review_data: list[dict]) -> FSRSParameters:
        """从历史数据估算 FSRS 参数

        Args:
            review_data: 复习数据列表
                [{"rating": int, "elapsed_days": int, "stability": float}, ...]

        Returns:
            估算的参数
        """
        if not review_data:
            return self.p

        # 简单的参数估算（实际应该用机器学习优化）
        # 这里使用统计方法估算 α 和 β

        ratings = [d["rating"] for d in review_data]
        avg_rating = sum(ratings) / len(ratings)

        # 根据平均评分调整参数
        if avg_rating >= 3.5:  # 用户记得很好
            self.p.w[17] = 12.0  # α 更大，衰减更慢
            self.p.w[18] = 0.4   # β 更小，曲线更平缓
        elif avg_rating >= 2.5:  # 一般
            self.p.w[17] = 9.0   # 默认
            self.p.w[18] = 0.5
        else:  # 用户经常忘记
            self.p.w[17] = 6.0   # α 更小，衰减更快
            self.p.w[18] = 0.6   # β 更大，曲线更陡峭

        self._alpha = self.p.w[17] if self.p.w[17] > 0 else self.p.default_alpha
        self._beta = self.p.w[18] if self.p.w[18] > 0 else self.p.default_beta

        logger.info("Estimated parameters: α=%.1f, β=%.2f", self._alpha, self._beta)

        return self.p


# 全局实例
_engine: FSRSEngine | None = None


def get_fsrs_engine(parameters: FSRSParameters | None = None) -> FSRSEngine:
    """获取全局 FSRS 引擎实例"""
    global _engine
    if _engine is None or parameters is not None:
        _engine = FSRSEngine(parameters)
    return _engine


def calculate_retention(
    recall_count: int,
    days_since_creation: int,
    last_recall_days_ago: int = 0
) -> float:
    """便捷函数：计算保持率"""
    engine = get_fsrs_engine()
    return engine.calculate_retention_from_recall(
        recall_count, days_since_creation, last_recall_days_ago
    )


# ── ForgettingFSRS 适配器 ─────────────────────────────────────────────────


class ForgettingFSRS:
    """FSRS 算法适配器 — 桥接 FSRSEngine / MemoryStrengthEvaluator 与 forgetting_state 数据库。

    从 ForgettingCurve 拆分的子模块，负责：
    - FSRS 保持率计算
    - 复习时间建议
    - FSRS 统计信息
    - 记忆强度评估

    子模块内部不加锁，由 ForgettingCurve._lock 保护。
    """

    def __init__(
        self,
        fsrs_engine: FSRSEngine,
        memory_evaluator: Any,
        get_conn: Any,
    ):
        """
        Args:
            fsrs_engine: FSRS 引擎实例
            memory_evaluator: MemoryStrengthEvaluator 实例
            get_conn: 回调函数，返回 forgetting_state 数据库连接
        """
        self._fsrs = fsrs_engine
        self._evaluator = memory_evaluator
        self._get_conn = get_conn

    # ── 内部工具方法 ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_datetime(dt_str: str | None) -> datetime:
        """解析 ISO 格式日期时间字符串，失败返回当前 UTC 时间。"""
        if not dt_str:
            return datetime.now(timezone.utc)
        dt = datetime.fromisoformat(dt_str.replace('+00:00', ''))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _get_memory_state(self, memory_id: str) -> tuple | None:
        """获取记忆的遗忘状态行 (recall_count, created_at, last_accessed)。"""
        conn = self._get_conn()
        return conn.execute(
            """SELECT recall_count, created_at, last_accessed
               FROM forgetting_state WHERE memory_id = ?""",
            (memory_id,)
        ).fetchone()

    # ── FSRS 保持率与复习 ────────────────────────────────────────────────

    def calculate_fsrs_retention(self, memory_id: str) -> float:
        """使用 FSRS 计算记忆保持率。

        Args:
            memory_id: 记忆 ID

        Returns:
            保持率 (0-1)
        """
        try:
            row = self._get_memory_state(memory_id)
            if not row:
                return 1.0

            recall_count, created_at, last_accessed = row
            now = datetime.now(timezone.utc)

            # 计算创建天数
            if created_at:
                created_dt = self._parse_datetime(created_at)
                days_since_creation = max(1, (now - created_dt).days)
            else:
                days_since_creation = 1

            # 计算最后检索距今天数
            if last_accessed:
                accessed_dt = self._parse_datetime(last_accessed)
                last_recall_days_ago = max(0, (now - accessed_dt).days)
            else:
                last_recall_days_ago = days_since_creation

            return self._fsrs.calculate_retention_from_recall(
                recall_count or 0, days_since_creation, last_recall_days_ago
            )

        except Exception as e:
            logger.warning("calculate_fsrs_retention failed for %s: %s", memory_id, e)
            return 0.5

    def calculate_fsrs_retention_batch(self, memory_ids: list[str]) -> dict[str, float]:
        """批量使用 FSRS 计算记忆保持率。

        将 N 次独立 SQLite 查询合并为 1 次批量查询，在 Python 中批量计算保持率。

        Args:
            memory_ids: 记忆 ID 列表

        Returns:
            {memory_id: 保持率} 字典，查询失败的记忆返回 0.5
        """
        if not memory_ids:
            return {}

        conn = self._get_conn()
        result: dict[str, float] = {}

        try:
            # 一次性批量查询所有记忆的 FSRS 参数
            placeholders = ",".join("?" * len(memory_ids))
            rows = conn.execute(
                f"""SELECT memory_id, recall_count, created_at, last_accessed
                    FROM forgetting_state WHERE memory_id IN ({placeholders})""",
                memory_ids
            ).fetchall()

            # 构建 memory_id → (recall_count, created_at, last_accessed) 映射
            state_map: dict[str, tuple] = {}
            for row in rows:
                state_map[row[0]] = (row[1], row[2], row[3])

            now = datetime.now(timezone.utc)

            for mid in memory_ids:
                state = state_map.get(mid)
                if not state:
                    result[mid] = 1.0
                    continue

                recall_count, created_at, last_accessed = state

                # 计算创建天数
                if created_at:
                    created_dt = self._parse_datetime(created_at)
                    days_since_creation = max(1, (now - created_dt).days)
                else:
                    days_since_creation = 1

                # 计算最后检索距今天数
                if last_accessed:
                    accessed_dt = self._parse_datetime(last_accessed)
                    last_recall_days_ago = max(0, (now - accessed_dt).days)
                else:
                    last_recall_days_ago = days_since_creation

                result[mid] = self._fsrs.calculate_retention_from_recall(
                    recall_count or 0, days_since_creation, last_recall_days_ago
                )

        except Exception as e:
            logger.warning("calculate_fsrs_retention_batch failed: %s", e)
            # 对未计算的记忆填充默认值
            for mid in memory_ids:
                if mid not in result:
                    result[mid] = 0.5

        return result

    def suggest_review_time(self, memory_id: str, desired_retention: float = 0.9) -> datetime | None:
        """建议下次复习时间。

        Args:
            memory_id: 记忆 ID
            desired_retention: 目标保持率

        Returns:
            建议复习时间，失败返回 None
        """
        try:
            row = self._get_memory_state(memory_id)
            if not row:
                return None

            recall_count, created_at, last_accessed = row
            now = datetime.now(timezone.utc)

            # 构建 FSRSItem
            item = FSRSItem()
            item.reps = recall_count or 0

            if last_accessed:
                accessed_dt = self._parse_datetime(last_accessed)
                item.last_review = accessed_dt

            # 估算稳定性
            if recall_count and recall_count > 0:
                item.stability = min(100.0, recall_count * 2.0)
            else:
                item.stability = 0.5

            return self._fsrs.suggest_review(item, now, desired_retention)

        except Exception as e:
            logger.warning("suggest_review_time failed for %s: %s", memory_id, e)
            return None

    def get_fsrs_stats(self) -> dict[str, Any]:
        """获取 FSRS 统计信息。

        Returns:
            包含保持率分布、平均稳定性等统计信息
        """
        conn = self._get_conn()
        stats: dict[str, Any] = {
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
            rows = conn.execute(
                "SELECT memory_id, recall_count FROM forgetting_state"
            ).fetchall()

            stats["total_memories"] = len(rows)

            if not rows:
                return stats

            # 批量计算保持率：1 次 SQL 查询替代 N 次
            memory_ids = [r[0] for r in rows]
            retention_map = self.calculate_fsrs_retention_batch(memory_ids)

            retentions = []
            stabilities = []

            for memory_id, recall_count in rows:
                retention = retention_map.get(memory_id, 0.5)
                retentions.append(retention)

                stability = min(100.0, (recall_count or 0) * 2.0)
                stabilities.append(stability)

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

    # ── 记忆强度评估 ─────────────────────────────────────────────────────

    def evaluate_memory_strength(self, memory_id: str) -> dict[str, Any]:
        """评估单个记忆的强度。

        Args:
            memory_id: 记忆 ID

        Returns:
            包含强度向量、综合评分、等级的字典
        """
        try:
            row = self._get_memory_state(memory_id)
            if not row:
                return {"memory_id": memory_id, "error": "not found"}

            recall_count, created_at, last_accessed = row

            fsrs_retention = self.calculate_fsrs_retention(memory_id)
            fsrs_stability = min(100.0, (recall_count or 0) * 2.0)
            fsrs_difficulty = 0.5  # 默认中等难度

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
        """评估所有记忆的强度。

        优化：使用批量 FSRS 保持率计算，将 N 次 SQLite 查询合并为 1 次。

        Args:
            limit: 最大评估数量

        Returns:
            包含评估结果和分布统计的字典
        """
        conn = self._get_conn()
        results = []

        try:
            rows = conn.execute(
                """SELECT memory_id, recall_count, created_at, last_accessed
                   FROM forgetting_state
                   ORDER BY recall_count DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()

            if not rows:
                return {"evaluated": 0, "distribution": {}, "top_memories": [], "bottom_memories": []}

            # 批量计算保持率：1 次 SQL 查询替代 N 次
            memory_ids = [r[0] for r in rows]
            retention_map = self.calculate_fsrs_retention_batch(memory_ids)

            for memory_id, recall_count, created_at, last_accessed in rows:
                fsrs_retention = retention_map.get(memory_id, 0.5)
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
        """获取记忆等级。

        Args:
            memory_id: 记忆 ID

        Returns:
            等级 (S/A/B/C/D)
        """
        result = self.evaluate_memory_strength(memory_id)
        return result.get("grade", "D")

    def get_strength_distribution(self) -> dict[str, Any]:
        """获取记忆强度分布统计。"""
        result = self.evaluate_all_memories(limit=1000)
        return result.get("distribution", {})
