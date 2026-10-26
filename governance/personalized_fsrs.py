"""
PersonalizedFSRS — 个性化 FSRS 参数学习模块。

根据用户行为自动调整 FSRS 参数：
1. 收集历史复习数据
2. 使用梯度下降优化参数
3. 验证参数收敛
4. 保存个性化参数

核心算法:
- 参数优化: 最小化预测保持率与实际保持率的差异
- 收敛检测: 参数变化 < epsilon 时停止
- A/B 测试: 对比个性化参数与默认参数的效果
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReviewRecord:
    """复习记录"""

    memory_id: str
    rating: int  # 评分 (1=Again, 2=Hard, 3=Good, 4=Easy)
    elapsed_days: int  # 距上次复习的天数
    retention_before: float  # 复习前的保持率
    retention_after: float  # 复习后的保持率
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ParameterHistory:
    """参数历史记录"""

    timestamp: datetime
    parameters: list[float]
    loss: float
    sample_count: int


class PersonalizedFSRS:
    """个性化 FSRS 参数学习器

    提供:
    - 从复习数据学习参数
    - 参数收敛检测
    - A/B 测试框架
    - 参数持久化
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        convergence_threshold: float = 0.001,
        max_iterations: int = 1000,
        param_file: str | None = None,
    ):
        self._learning_rate = learning_rate
        self._convergence_threshold = convergence_threshold
        self._max_iterations = max_iterations
        self._param_file = param_file or os.path.expanduser(
            "~/.hermes/omnimem/governance/personalized_params.json"
        )

        # 默认参数 (FSRS v4)
        self._default_params = [
            0.4,
            0.6,
            2.4,
            5.8,  # 初始稳定性
            4.93,
            0.94,  # 初始难度参数
            0.86,
            0.01,
            1.49,
            0.14,
            0.94,
            2.18,  # 稳定性增长
            0.05,
            0.34,
            1.26,
            0.29,
            2.61,  # 难度调整
            9.0,
            0.5,  # 遗忘曲线参数 (α, β)
        ]

        # 当前参数
        self._current_params = self._default_params.copy()

        # 参数历史
        self._history: list[ParameterHistory] = []

        # 加载已保存的参数
        self._load_params()

    def _load_params(self) -> None:
        """加载已保存的参数"""
        try:
            if os.path.exists(self._param_file):
                with open(self._param_file) as f:
                    data = json.load(f)
                    if "parameters" in data:
                        self._current_params = data["parameters"]
                        logger.info("Loaded personalized parameters")
        except Exception as e:
            logger.warning("Failed to load parameters: %s", e)

    def _save_params(self) -> None:
        """保存参数"""
        try:
            os.makedirs(os.path.dirname(self._param_file), exist_ok=True)
            with open(self._param_file, "w") as f:
                json.dump(
                    {
                        "parameters": self._current_params,
                        "updated_at": datetime.now().isoformat(),
                        "history_count": len(self._history),
                    },
                    f,
                    indent=2,
                )
            logger.info("Saved personalized parameters")
        except Exception as e:
            logger.warning("Failed to save parameters: %s", e)

    def get_parameters(self) -> list[float]:
        """获取当前参数"""
        return self._current_params.copy()

    def set_parameters(self, params: list[float]) -> None:
        """设置参数"""
        if len(params) == len(self._default_params):
            self._current_params = params
        else:
            logger.warning("Invalid parameter count: %d", len(params))

    def calculate_loss(self, review_data: list[ReviewRecord]) -> float:
        """计算损失函数

        损失 = mean((predicted_retention - actual_retention)^2)

        Args:
            review_data: 复习数据列表

        Returns:
            损失值
        """
        if not review_data:
            return 0.0

        total_loss = 0.0
        for record in review_data:
            # 预测保持率
            predicted = self._predict_retention(
                record.elapsed_days,
                record.rating,
            )

            # 计算平方误差
            error = predicted - record.retention_before
            total_loss += error * error

        return total_loss / len(review_data)

    def _predict_retention(self, elapsed_days: int, rating: int) -> float:
        """预测保持率

        Args:
            elapsed_days: 经过天数
            rating: 评分

        Returns:
            预测保持率
        """
        # 使用当前参数计算
        alpha = self._current_params[17] if self._current_params[17] > 0 else 9.0
        beta = self._current_params[18] if self._current_params[18] > 0 else 0.5

        # 估算稳定性
        stability = self._current_params[rating - 1] if 1 <= rating <= 4 else 2.0

        # 遗忘曲线
        if stability <= 0:
            return 1.0

        retention = (1 + elapsed_days / (stability * alpha)) ** (-beta)
        return max(0.0, min(1.0, retention))

    def learn_from_reviews(
        self,
        review_data: list[ReviewRecord],
        verbose: bool = False,
    ) -> dict[str, Any]:
        """从复习数据学习参数

        Args:
            review_data: 复习数据列表
            verbose: 是否输出详细信息

        Returns:
            学习结果
        """
        if not review_data:
            return {"status": "no_data", "iterations": 0}

        # 记录初始损失
        initial_loss = self.calculate_loss(review_data)

        # 梯度下降优化
        best_params = self._current_params.copy()
        best_loss = initial_loss
        no_improve_count = 0

        for iteration in range(self._max_iterations):
            # 计算梯度
            gradients = self._calculate_gradients(review_data)

            # 更新参数
            new_params = self._current_params.copy()
            for i in range(len(new_params)):
                new_params[i] -= self._learning_rate * gradients[i]
                # 限制参数范围
                new_params[i] = max(0.01, min(100.0, new_params[i]))

            # 计算新损失
            self._current_params = new_params
            new_loss = self.calculate_loss(review_data)

            # 检查是否改进
            if new_loss < best_loss:
                best_loss = new_loss
                best_params = new_params.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 收敛检测
            if abs(initial_loss - new_loss) < self._convergence_threshold:
                if verbose:
                    logger.info("Converged at iteration %d", iteration)
                break

            # 早停
            if no_improve_count >= 50:
                if verbose:
                    logger.info("Early stopping at iteration %d", iteration)
                break

        # 使用最佳参数
        self._current_params = best_params

        # 保存参数
        self._save_params()

        # 记录历史
        self._history.append(
            ParameterHistory(
                timestamp=datetime.now(),
                parameters=self._current_params.copy(),
                loss=best_loss,
                sample_count=len(review_data),
            )
        )

        return {
            "status": "success",
            "iterations": iteration + 1,
            "initial_loss": initial_loss,
            "final_loss": best_loss,
            "improvement": initial_loss - best_loss,
            "sample_count": len(review_data),
        }

    def _calculate_gradients(self, review_data: list[ReviewRecord]) -> list[float]:
        """计算梯度

        使用数值梯度（简化实现）

        Args:
            review_data: 复习数据

        Returns:
            梯度列表
        """
        gradients = []
        epsilon = 0.001

        for i in range(len(self._current_params)):
            # 保存原参数
            original = self._current_params[i]

            # 计算正向损失
            self._current_params[i] = original + epsilon
            loss_plus = self.calculate_loss(review_data)

            # 计算负向损失
            self._current_params[i] = original - epsilon
            loss_minus = self.calculate_loss(review_data)

            # 恢复原参数
            self._current_params[i] = original

            # 计算梯度
            gradient = (loss_plus - loss_minus) / (2 * epsilon)
            gradients.append(gradient)

        return gradients

    def get_history(self) -> list[dict[str, Any]]:
        """获取参数历史"""
        return [
            {
                "timestamp": h.timestamp.isoformat(),
                "loss": h.loss,
                "sample_count": h.sample_count,
            }
            for h in self._history
        ]

    def reset_to_default(self) -> None:
        """重置为默认参数"""
        self._current_params = self._default_params.copy()
        self._save_params()
        logger.info("Reset to default parameters")

    def compare_with_default(self, review_data: list[ReviewRecord]) -> dict[str, Any]:
        """与默认参数对比

        Args:
            review_data: 复习数据

        Returns:
            对比结果
        """
        # 当前参数损失
        current_loss = self.calculate_loss(review_data)

        # 默认参数损失
        original_params = self._current_params.copy()
        self._current_params = self._default_params.copy()
        default_loss = self.calculate_loss(review_data)
        self._current_params = original_params

        return {
            "current_loss": current_loss,
            "default_loss": default_loss,
            "improvement": default_loss - current_loss,
            "improvement_pct": ((default_loss - current_loss) / default_loss * 100)
            if default_loss > 0
            else 0,
        }


# 全局实例
_learner: PersonalizedFSRS | None = None


def get_learner(
    learning_rate: float = 0.01,
    convergence_threshold: float = 0.001,
) -> PersonalizedFSRS:
    """获取全局学习器实例"""
    global _learner
    if _learner is None:
        _learner = PersonalizedFSRS(learning_rate, convergence_threshold)
    return _learner


def learn_from_reviews(review_data: list[ReviewRecord]) -> dict[str, Any]:
    """便捷函数：从复习数据学习"""
    learner = get_learner()
    return learner.learn_from_reviews(review_data)
