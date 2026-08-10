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
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FSRSItem:
    """FSRS 记忆项状态"""

    stability: float = 0.0  # 稳定性 (天)
    difficulty: float = 0.0  # 难度 (0-1)
    due: Optional[datetime] = None
    last_review: Optional[datetime] = None
    elapsed_days: int = 0
    scheduled_days: int = 0
    reps: int = 0  # 复习次数
    lapses: int = 0  # 遗忘次数
    state: int = 0  # 0=New, 1=Learning, 2=Review, 3=Relearning


@dataclass
class FSRSParameters:
    """FSRS 参数（可学习）

    基于大量用户数据训练的默认参数
    """

    w: list = field(
        default_factory=lambda: [
            # 初始稳定性 (Again, Hard, Good, Easy)
            0.4,
            0.6,
            2.4,
            5.8,
            # 初始难度参数
            4.93,
            0.94,
            # 稳定性增长参数
            0.86,
            0.01,
            1.49,
            0.14,
            0.94,
            2.18,
            # 难度调整参数
            0.05,
            0.34,
            1.26,
            0.29,
            2.61,
            # 遗忘曲线参数 (α, β)
            0.0,
            0.0,
        ]
    )

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

    def __init__(self, parameters: Optional[FSRSParameters] = None):
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
            growth = (
                self.p.w[6] * (11 - d) * s ** (-self.p.w[7]) * (math.exp(self.p.w[8] * (1 - r)) - 1)
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

    def suggest_review(
        self, item: FSRSItem, now: datetime, desired_retention: float = 0.9
    ) -> datetime:
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
        self, recall_count: int, days_since_creation: int, last_recall_days_ago: int = 0
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
            self.p.w[18] = 0.4  # β 更小，曲线更平缓
        elif avg_rating >= 2.5:  # 一般
            self.p.w[17] = 9.0  # 默认
            self.p.w[18] = 0.5
        else:  # 用户经常忘记
            self.p.w[17] = 6.0  # α 更小，衰减更快
            self.p.w[18] = 0.6  # β 更大，曲线更陡峭

        self._alpha = self.p.w[17] if self.p.w[17] > 0 else self.p.default_alpha
        self._beta = self.p.w[18] if self.p.w[18] > 0 else self.p.default_beta

        logger.info("Estimated parameters: α=%.1f, β=%.2f", self._alpha, self._beta)

        return self.p


# 全局实例
_engine: Optional[FSRSEngine] = None


def get_fsrs_engine(parameters: Optional[FSRSParameters] = None) -> FSRSEngine:
    """获取全局 FSRS 引擎实例"""
    global _engine
    if _engine is None or parameters is not None:
        _engine = FSRSEngine(parameters)
    return _engine


def calculate_retention(
    recall_count: int, days_since_creation: int, last_recall_days_ago: int = 0
) -> float:
    """便捷函数：计算保持率"""
    engine = get_fsrs_engine()
    return engine.calculate_retention_from_recall(
        recall_count, days_since_creation, last_recall_days_ago
    )
