"""
ReviewScheduler — 自动复习调度模块。

根据 FSRS 预测自动提醒复习：
1. 计算所有记忆的建议复习时间
2. 按优先级排序
3. 生成每日复习计划
4. 复习效果追踪

核心功能:
- 智能复习计划生成
- 优先级排序
- 效果追踪
- 自适应调整
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReviewItem:
    """复习项目"""

    memory_id: str
    due_date: datetime
    priority: float  # 优先级 (0-1)
    retention: float  # 当前保持率
    stability: float  # 稳定性
    days_overdue: int  # 逾期天数


@dataclass
class DailyPlan:
    """每日复习计划"""

    date: str
    items: list[ReviewItem]
    total_count: int
    estimated_time: int  # 预计时间（分钟）


class ReviewScheduler:
    """复习调度器

    提供:
    - 复习计划生成
    - 优先级排序
    - 效果追踪
    - 自适应调整
    """

    def __init__(
        self,
        governance_dir: Path | None = None,
        desired_retention: float = 0.9,
    ):
        self._governance_dir = governance_dir or Path.home() / ".hermes" / "omnimem" / "governance"
        self._desired_retention = desired_retention
        self._db_path = self._governance_dir / "forgetting.db"

    def generate_review_plan(
        self,
        days: int = 7,
        max_items_per_day: int = 50,
    ) -> list[DailyPlan]:
        """生成复习计划

        Args:
            days: 计划天数
            max_items_per_day: 每天最大复习数量

        Returns:
            每日复习计划列表
        """
        # 获取所有需要复习的记忆
        review_items = self._get_due_items()

        if not review_items:
            return []

        # 按日期分组
        plans = []
        today = datetime.now(timezone.utc).date()

        for day_offset in range(days):
            target_date = today + timedelta(days=day_offset)
            target_date_str = target_date.isoformat()

            # 筛选该日期的复习项
            day_items = [item for item in review_items if item.due_date.date() <= target_date]

            # 按优先级排序
            day_items.sort(key=lambda x: x.priority, reverse=True)

            # 限制数量
            day_items = day_items[:max_items_per_day]

            if day_items:
                # 估算时间（每个记忆约 1-2 分钟）
                estimated_time = len(day_items) * 2

                plans.append(
                    DailyPlan(
                        date=target_date_str,
                        items=day_items,
                        total_count=len(day_items),
                        estimated_time=estimated_time,
                    )
                )

        return plans

    def _get_due_items(self) -> list[ReviewItem]:
        """获取需要复习的记忆"""
        if not self._db_path.exists():
            return []

        try:
            conn = sqlite3.connect(str(self._db_path))
            now = datetime.now(timezone.utc)

            # 查询所有记忆
            rows = conn.execute(
                """SELECT memory_id, recall_count, created_at, last_accessed
                   FROM forgetting_state"""
            ).fetchall()

            conn.close()

            items = []
            for memory_id, recall_count, created_at, last_accessed in rows:
                # 计算建议复习时间
                due_date = self._calculate_due_date(
                    recall_count or 0,
                    created_at,
                    last_accessed,
                )

                # 计算当前保持率
                retention = self._estimate_retention(
                    recall_count or 0,
                    last_accessed,
                    now,
                )

                # 计算优先级
                priority = self._calculate_priority(
                    retention,
                    due_date,
                    now,
                )

                # 计算逾期天数
                days_overdue = max(0, (now - due_date).days)

                items.append(
                    ReviewItem(
                        memory_id=memory_id,
                        due_date=due_date,
                        priority=priority,
                        retention=retention,
                        stability=min(100.0, (recall_count or 0) * 2.0),
                        days_overdue=days_overdue,
                    )
                )

            return items

        except Exception as e:
            logger.warning("Failed to get due items: %s", e)
            return []

    def _calculate_due_date(
        self,
        recall_count: int,
        created_at: str | None,
        last_accessed: str | None,
    ) -> datetime:
        """计算建议复习时间"""
        now = datetime.now(timezone.utc)

        # 估算稳定性
        stability = min(100.0, recall_count * 2.0) if recall_count else 1.0

        # 计算间隔（简化版 FSRS）
        # I = S * (R^(-1/β) - 1)
        beta = 0.5
        interval_days = stability * (self._desired_retention ** (-1 / beta) - 1)
        interval_days = max(1, round(interval_days))

        # 基于最后访问时间计算
        if last_accessed:
            try:
                last_dt = datetime.fromisoformat(last_accessed.replace("+00:00", ""))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                return last_dt + timedelta(days=interval_days)
            except:
                pass

        # 基于创建时间计算
        if created_at:
            try:
                created_dt = datetime.fromisoformat(created_at.replace("+00:00", ""))
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                return created_dt + timedelta(days=interval_days)
            except:
                pass

        # 默认明天
        return now + timedelta(days=1)

    def _estimate_retention(
        self,
        recall_count: int,
        last_accessed: str | None,
        now: datetime,
    ) -> float:
        """估算当前保持率"""
        if not last_accessed:
            return 0.5

        try:
            last_dt = datetime.fromisoformat(last_accessed.replace("+00:00", ""))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)

            elapsed_days = (now - last_dt).total_seconds() / 86400

            # 估算稳定性
            stability = min(100.0, recall_count * 2.0) if recall_count else 1.0

            # 遗忘曲线
            alpha = 9.0
            beta = 0.5
            retention = (1 + elapsed_days / (stability * alpha)) ** (-beta)

            return max(0.0, min(1.0, retention))

        except:
            return 0.5

    def _calculate_priority(
        self,
        retention: float,
        due_date: datetime,
        now: datetime,
    ) -> float:
        """计算优先级

        优先级因素:
        - 保持率越低，优先级越高
        - 逾期天数越多，优先级越高
        """
        # 保持率因子 (保持率越低，优先级越高)
        retention_factor = 1.0 - retention

        # 逾期因子
        days_overdue = (now - due_date).days
        overdue_factor = min(1.0, days_overdue / 7.0) if days_overdue > 0 else 0.0

        # 综合优先级
        priority = 0.6 * retention_factor + 0.4 * overdue_factor

        return max(0.0, min(1.0, priority))

    def get_today_plan(self, max_items: int = 30) -> DailyPlan:
        """获取今日复习计划"""
        plans = self.generate_review_plan(days=1, max_items_per_day=max_items)
        return (
            plans[0]
            if plans
            else DailyPlan(
                date=datetime.now(timezone.utc).date().isoformat(),
                items=[],
                total_count=0,
                estimated_time=0,
            )
        )

    def get_review_stats(self) -> dict[str, Any]:
        """获取复习统计"""
        items = self._get_due_items()

        if not items:
            return {
                "total_due": 0,
                "overdue": 0,
                "avg_retention": 1.0,
                "urgent": 0,
            }

        overdue = sum(1 for item in items if item.days_overdue > 0)
        avg_retention = sum(item.retention for item in items) / len(items)
        urgent = sum(1 for item in items if item.days_overdue > 3)

        return {
            "total_due": len(items),
            "overdue": overdue,
            "avg_retention": avg_retention,
            "urgent": urgent,
        }


# 全局实例
_scheduler: ReviewScheduler | None = None


def get_scheduler(
    governance_dir: Path | None = None,
    desired_retention: float = 0.9,
) -> ReviewScheduler:
    """获取全局调度器实例"""
    global _scheduler
    if _scheduler is None:
        _scheduler = ReviewScheduler(governance_dir, desired_retention)
    return _scheduler


def get_today_plan(max_items: int = 30) -> dict[str, Any]:
    """便捷函数：获取今日复习计划"""
    scheduler = get_scheduler()
    plan = scheduler.get_today_plan(max_items)

    return {
        "date": plan.date,
        "total_count": plan.total_count,
        "estimated_time": plan.estimated_time,
        "items": [
            {
                "memory_id": item.memory_id,
                "due_date": item.due_date.isoformat(),
                "priority": item.priority,
                "retention": item.retention,
                "days_overdue": item.days_overdue,
            }
            for item in plan.items
        ],
    }
