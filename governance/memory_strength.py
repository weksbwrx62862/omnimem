"""
MemoryStrength — 多维度记忆强度评估系统。

六维记忆强度:
1. stability: 稳定性 (FSRS)
2. retrievability: 可提取性 (当前保持率)
3. difficulty: 难度 (FSRS)
4. recency: 新近性 (距离上次访问)
5. frequency: 访问频率
6. semantic_importance: 语义重要性 (基于向量聚类)

综合评分公式:
Score = w1*√S + w2*R + w3*e^(-λt) + w4*log(F+1) + w5*SI
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryStrengthVector:
    """六维记忆强度向量"""

    stability: float = 0.0  # 稳定性 (天)
    retrievability: float = 1.0  # 可提取性 (0-1)
    difficulty: float = 0.5  # 难度 (0-1)
    recency: float = 0.0  # 新近性 (天)
    frequency: int = 0  # 访问频率
    semantic_importance: float = 0.5  # 语义重要性 (0-1)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "stability": self.stability,
            "retrievability": self.retrievability,
            "difficulty": self.difficulty,
            "recency": self.recency,
            "frequency": self.frequency,
            "semantic_importance": self.semantic_importance,
        }


@dataclass
class ScoringWeights:
    """评分权重配置"""

    stability: float = 0.25  # 稳定性权重
    retrievability: float = 0.25  # 可提取性权重
    recency: float = 0.20  # 新近性权重
    frequency: float = 0.15  # 频率权重
    semantic: float = 0.15  # 语义重要性权重

    # 衰减参数
    recency_lambda: float = 0.1  # 新近性衰减系数

    def normalize(self) -> ScoringWeights:
        """归一化权重"""
        total = self.stability + self.retrievability + self.recency + self.frequency + self.semantic

        if total == 0:
            return ScoringWeights()

        return ScoringWeights(
            stability=self.stability / total,
            retrievability=self.retrievability / total,
            recency=self.recency / total,
            frequency=self.frequency / total,
            semantic=self.semantic / total,
            recency_lambda=self.recency_lambda,
        )


class MemoryStrengthEvaluator:
    """记忆强度评估器

    提供:
    - 六维强度计算
    - 综合评分
    - 语义重要性评估
    - 批量评估
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        weights: Optional[ScoringWeights] = None,
    ):
        self._db_path = db_path
        self._weights = (weights or ScoringWeights()).normalize()

    def calculate_strength(
        self,
        memory_id: str,
        recall_count: int,
        created_at: Optional[str] = None,
        last_accessed: Optional[str] = None,
        fsrs_retention: float = 1.0,
        fsrs_stability: float = 0.0,
        fsrs_difficulty: float = 0.5,
    ) -> MemoryStrengthVector:
        """计算记忆的六维强度向量

        Args:
            memory_id: 记忆 ID
            recall_count: 检索次数
            created_at: 创建时间 (ISO format)
            last_accessed: 最后访问时间 (ISO format)
            fsrs_retention: FSRS 保持率
            fsrs_stability: FSRS 稳定性
            fsrs_difficulty: FSRS 难度

        Returns:
            六维强度向量
        """
        now = datetime.now(timezone.utc)

        # 1. 稳定性 (FSRS)
        stability = fsrs_stability if fsrs_stability > 0 else min(100.0, (recall_count or 0) * 2.0)

        # 2. 可提取性 (FSRS 保持率)
        retrievability = fsrs_retention

        # 3. 难度 (FSRS)
        difficulty = fsrs_difficulty

        # 4. 新近性 (距离上次访问的天数)
        recency = self._calculate_recency(last_accessed, now)

        # 5. 频率
        frequency = recall_count or 0

        # 6. 语义重要性 (默认 0.5，可由外部计算)
        semantic_importance = 0.5

        return MemoryStrengthVector(
            stability=stability,
            retrievability=retrievability,
            difficulty=difficulty,
            recency=recency,
            frequency=frequency,
            semantic_importance=semantic_importance,
        )

    def _calculate_recency(self, last_accessed: Optional[str], now: datetime) -> float:
        """计算新近性（距离上次访问的天数）"""
        if not last_accessed:
            return 365.0  # 从未访问，视为很久以前

        try:
            accessed_dt = datetime.fromisoformat(last_accessed.replace("+00:00", ""))
            if accessed_dt.tzinfo is None:
                accessed_dt = accessed_dt.replace(tzinfo=timezone.utc)
            return max(0.0, (now - accessed_dt).total_seconds() / 86400)
        except Exception:
            return 365.0

    def calculate_score(self, strength: MemoryStrengthVector) -> float:
        """计算综合评分

        Score = w1*√S + w2*R + w3*e^(-λt) + w4*log(F+1) + w5*SI

        Args:
            strength: 六维强度向量

        Returns:
            综合评分 (0-100)
        """
        w = self._weights

        # 1. 稳定性评分 (0-100)
        # 使用平方根压缩，避免极端值影响
        stability_score = math.sqrt(min(strength.stability, 100.0)) * 10.0

        # 2. 可提取性评分 (0-100)
        retrievability_score = strength.retrievability * 100.0

        # 3. 新近性评分 (0-100)
        # 使用指数衰减，越新分数越高
        recency_score = math.exp(-w.recency_lambda * strength.recency) * 100.0

        # 4. 频率评分 (0-100)
        # 使用对数压缩，避免高频记忆分数过高
        frequency_score = math.log(strength.frequency + 1) * 20.0
        frequency_score = min(100.0, frequency_score)

        # 5. 语义重要性评分 (0-100)
        semantic_score = strength.semantic_importance * 100.0

        # 加权综合
        total_score = (
            w.stability * stability_score
            + w.retrievability * retrievability_score
            + w.recency * recency_score
            + w.frequency * frequency_score
            + w.semantic * semantic_score
        )

        return round(min(100.0, max(0.0, total_score)), 2)

    def evaluate_memory(
        self,
        memory_id: str,
        recall_count: int,
        created_at: Optional[str] = None,
        last_accessed: Optional[str] = None,
        fsrs_retention: float = 1.0,
        fsrs_stability: float = 0.0,
        fsrs_difficulty: float = 0.5,
        semantic_importance: float = 0.5,
    ) -> dict[str, Any]:
        """评估单个记忆

        Returns:
            包含强度向量、综合评分、等级的字典
        """
        strength = self.calculate_strength(
            memory_id=memory_id,
            recall_count=recall_count,
            created_at=created_at,
            last_accessed=last_accessed,
            fsrs_retention=fsrs_retention,
            fsrs_stability=fsrs_stability,
            fsrs_difficulty=fsrs_difficulty,
        )

        # 设置语义重要性
        strength.semantic_importance = semantic_importance

        # 计算综合评分
        score = self.calculate_score(strength)

        # 确定等级
        grade = self._score_to_grade(score)

        return {
            "memory_id": memory_id,
            "strength": strength.to_dict(),
            "score": score,
            "grade": grade,
        }

    def _score_to_grade(self, score: float) -> str:
        """评分转等级

        Args:
            score: 综合评分 (0-100)

        Returns:
            等级 (S/A/B/C/D)
        """
        if score >= 90:
            return "S"
        elif score >= 80:
            return "A"
        elif score >= 60:
            return "B"
        elif score >= 40:
            return "C"
        else:
            return "D"

    def evaluate_batch(
        self,
        memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """批量评估记忆

        Args:
            memories: 记忆数据列表
                [{"memory_id": str, "recall_count": int, ...}, ...]

        Returns:
            评估结果列表
        """
        results = []
        for mem in memories:
            result = self.evaluate_memory(
                memory_id=mem.get("memory_id", ""),
                recall_count=mem.get("recall_count", 0),
                created_at=mem.get("created_at"),
                last_accessed=mem.get("last_accessed"),
                fsrs_retention=mem.get("fsrs_retention", 1.0),
                fsrs_stability=mem.get("fsrs_stability", 0.0),
                fsrs_difficulty=mem.get("fsrs_difficulty", 0.5),
                semantic_importance=mem.get("semantic_importance", 0.5),
            )
            results.append(result)

        return results

    def get_distribution(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """获取评估结果分布

        Args:
            results: evaluate_batch 的结果

        Returns:
            分布统计
        """
        if not results:
            return {"total": 0, "grades": {}, "avg_score": 0.0}

        grades = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0}
        scores = []

        for r in results:
            grade = r.get("grade", "D")
            if grade in grades:
                grades[grade] += 1
            scores.append(r.get("score", 0.0))

        return {
            "total": len(results),
            "grades": grades,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
        }


# 全局实例
_evaluator: Optional[MemoryStrengthEvaluator] = None


def get_evaluator(
    db_path: Optional[str] = None,
    weights: Optional[ScoringWeights] = None,
) -> MemoryStrengthEvaluator:
    """获取全局评估器实例"""
    global _evaluator
    if _evaluator is None or weights is not None:
        _evaluator = MemoryStrengthEvaluator(db_path, weights)
    return _evaluator


def evaluate_memory(
    memory_id: str,
    recall_count: int,
    **kwargs,
) -> dict[str, Any]:
    """便捷函数：评估单个记忆"""
    evaluator = get_evaluator()
    return evaluator.evaluate_memory(
        memory_id=memory_id,
        recall_count=recall_count,
        **kwargs,
    )
