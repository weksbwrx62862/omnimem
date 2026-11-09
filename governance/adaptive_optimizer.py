"""
AdaptiveFSRSOptimizer — 自适应 FSRS 优化器

监控预测精度，在精度下降时自动引入大模型优化

核心机制:
1. 实时监控预测精度
2. 检测精度下降趋势
3. 自动触发大模型优化
4. 验证优化效果
5. 智能回退策略
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PrecisionRecord:
    """精度记录"""

    timestamp: datetime
    precision: float
    sample_count: int
    mode: str  # gradient / llm / hybrid


@dataclass
class OptimizationResult:
    """优化结果"""

    mode: str
    precision_before: float
    precision_after: float
    improvement: float
    parameters: list[float]
    cost: float  # 计算成本
    timestamp: datetime = field(default_factory=datetime.now)


class AdaptiveFSRSOptimizer:
    """自适应 FSRS 优化器

    监控预测精度，在精度下降时自动引入大模型优化

    优化策略:
    1. 正常模式: 梯度下降优化 (快速、低成本)
    2. 降级模式: 大模型优化 (慢速、高成本、高精度)
    3. 混合模式: 两者结合 (平衡)
    """

    def __init__(
        self,
        precision_threshold: float = 0.85,
        decline_threshold: float = 0.05,
        check_interval: int = 100,
        llm_caller: Callable | None = None,
        cache_dir: str | None = None,
    ):
        """
        Args:
            precision_threshold: 精度阈值，低于此值触发优化
            decline_threshold: 下降阈值，连续下降超过此值触发优化
            check_interval: 检查间隔（数据量）
            llm_caller: 大模型调用函数
            cache_dir: 缓存目录
        """
        # 阈值配置
        self._precision_threshold = precision_threshold
        self._decline_threshold = decline_threshold
        self._check_interval = check_interval

        # 优化器
        self._llm_caller = llm_caller

        # 缓存
        self._cache_dir = Path(
            cache_dir or os.path.expanduser("~/.hermes/omnimem/governance/adaptive_cache")
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # 状态
        self._current_mode = "gradient"
        self._current_precision = 1.0
        self._precision_history: deque[PrecisionRecord] = deque(maxlen=100)
        self._optimization_history: list[OptimizationResult] = []

        # 统计
        self._total_calls = 0
        self._gradient_calls = 0
        self._llm_calls = 0
        self._mode_switches = 0

        # 加载历史状态
        self._load_state()

    def _load_state(self) -> None:
        """加载历史状态"""
        state_file = self._cache_dir / "optimizer_state.json"
        try:
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                    self._current_mode = state.get("mode", "gradient")
                    self._current_precision = state.get("precision", 1.0)
                    self._total_calls = state.get("total_calls", 0)
                    self._gradient_calls = state.get("gradient_calls", 0)
                    self._llm_calls = state.get("llm_calls", 0)
                    self._mode_switches = state.get("mode_switches", 0)
                logger.info(
                    "Loaded optimizer state: mode=%s, precision=%.2f%%",
                    self._current_mode,
                    self._current_precision * 100,
                )
        except Exception as e:
            logger.warning("Failed to load optimizer state: %s", e)

    def _save_state(self) -> None:
        """保存状态"""
        state_file = self._cache_dir / "optimizer_state.json"
        try:
            state = {
                "mode": self._current_mode,
                "precision": self._current_precision,
                "total_calls": self._total_calls,
                "gradient_calls": self._gradient_calls,
                "llm_calls": self._llm_calls,
                "mode_switches": self._mode_switches,
                "updated_at": datetime.now().isoformat(),
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save optimizer state: %s", e)

    def optimize(
        self,
        review_data: list[dict[str, Any]],
        current_params: list[float],
    ) -> tuple[list[float], str]:
        """优化参数

        Args:
            review_data: 复习数据
            current_params: 当前参数

        Returns:
            (优化后参数, 使用的模式)
        """
        self._total_calls += 1

        # 计算当前精度
        precision = self._calculate_precision(review_data, current_params)

        # 记录精度
        self._record_precision(precision, len(review_data))

        # 检查是否需要切换模式
        self._check_and_switch_mode(precision, len(review_data))

        # 根据模式选择优化器
        if self._current_mode == "llm":
            new_params = self._optimize_with_llm(review_data, current_params)
            self._llm_calls += 1
        elif self._current_mode == "hybrid":
            new_params = self._optimize_hybrid(review_data, current_params)
            self._gradient_calls += 1
            self._llm_calls += 1
        else:
            new_params = self._optimize_with_gradient(review_data, current_params)
            self._gradient_calls += 1

        # 验证优化效果
        new_precision = self._calculate_precision(review_data, new_params)

        # 记录优化结果
        result = OptimizationResult(
            mode=self._current_mode,
            precision_before=precision,
            precision_after=new_precision,
            improvement=new_precision - precision,
            parameters=new_params,
            cost=self._estimate_cost(self._current_mode),
        )
        self._optimization_history.append(result)

        # 更新状态
        self._current_precision = new_precision
        self._save_state()

        logger.info(
            "Optimization complete: mode=%s, precision=%.2f%% → %.2f%%, improvement=%.2f%%",
            self._current_mode,
            precision * 100,
            new_precision * 100,
            result.improvement * 100,
        )

        return new_params, self._current_mode

    def _calculate_precision(
        self,
        review_data: list[dict[str, Any]],
        params: list[float],
    ) -> float:
        """计算预测精度"""
        if not review_data:
            return 1.0

        from governance.fsrs_engine import FSRSEngine, FSRSParameters

        # 创建参数对象
        fsrs_params = FSRSParameters(w=params)
        engine = FSRSEngine(parameters=fsrs_params)

        errors = []
        for record in review_data:
            # 预测保持率
            predicted = engine.forgetting_curve(
                record.get("elapsed_days", 0), record.get("stability", 1.0)
            )

            # 实际保持率
            actual = record.get("retention_before", 0.5)

            # 计算误差
            error = abs(predicted - actual)
            errors.append(error)

        # 精度 = 1 - 平均误差
        avg_error = sum(errors) / len(errors)
        precision = 1.0 - avg_error

        return max(0.0, min(1.0, precision))

    def _record_precision(self, precision: float, sample_count: int) -> None:
        """记录精度"""
        record = PrecisionRecord(
            timestamp=datetime.now(),
            precision=precision,
            sample_count=sample_count,
            mode=self._current_mode,
        )
        self._precision_history.append(record)

    def _check_and_switch_mode(
        self,
        current_precision: float,
        sample_count: int,
    ) -> None:
        """检查并切换模式"""

        # 策略 1: 精度低于阈值
        if current_precision < self._precision_threshold:
            logger.warning(
                "Precision below threshold: %.2f%% < %.2f%%",
                current_precision * 100,
                self._precision_threshold * 100,
            )
            self._switch_mode("llm")
            return

        # 策略 2: 连续下降
        if len(self._precision_history) >= 3:
            recent = list(self._precision_history)[-3:]
            precisions = [r.precision for r in recent]

            # 检查是否连续下降
            if all(precisions[i] > precisions[i + 1] for i in range(len(precisions) - 1)):
                decline = precisions[0] - precisions[-1]
                if decline > self._decline_threshold:
                    logger.warning(
                        "Precision declining: %.2f%% → %.2f%% (decline=%.2f%%)",
                        precisions[0] * 100,
                        precisions[-1] * 100,
                        decline * 100,
                    )
                    self._switch_mode("hybrid")
                    return

        # 策略 3: 精度恢复，回退到梯度下降
        if self._current_mode in ["llm", "hybrid"]:
            if current_precision > self._precision_threshold + 0.05:
                logger.info("Precision recovered, switching back to gradient mode")
                self._switch_mode("gradient")
                return

    def _switch_mode(self, new_mode: str) -> None:
        """切换模式"""
        if new_mode == self._current_mode:
            return

        old_mode = self._current_mode
        self._current_mode = new_mode
        self._mode_switches += 1

        logger.info("Mode switched: %s → %s", old_mode, new_mode)

    def _optimize_with_gradient(
        self,
        review_data: list[dict[str, Any]],
        current_params: list[float],
    ) -> list[float]:
        """梯度下降优化"""
        from governance.personalized_fsrs import PersonalizedFSRS

        learner = PersonalizedFSRS()
        learner.set_parameters(current_params)

        # 转换数据格式
        from governance.personalized_fsrs import ReviewRecord

        records = [
            ReviewRecord(
                memory_id=r.get("memory_id", ""),
                rating=r.get("rating", 3),
                elapsed_days=r.get("elapsed_days", 0),
                retention_before=r.get("retention_before", 0.5),
                retention_after=r.get("retention_after", 0.9),
            )
            for r in review_data
        ]

        result = learner.learn_from_reviews(records)
        return learner.get_parameters()

    def _optimize_with_llm(
        self,
        review_data: list[dict[str, Any]],
        current_params: list[float],
    ) -> list[float]:
        """大模型优化"""
        if not self._llm_caller:
            logger.warning("No LLM caller configured, falling back to gradient")
            return self._optimize_with_gradient(review_data, current_params)

        # 准备提示
        prompt = self._build_llm_prompt(review_data, current_params)

        try:
            # 调用大模型
            response = self._llm_caller(prompt)

            # 解析参数
            new_params = self._parse_llm_response(response, current_params)

            # 验证参数有效性
            if self._validate_params(new_params):
                return new_params
            else:
                logger.warning("Invalid LLM params, falling back to gradient")
                return self._optimize_with_gradient(review_data, current_params)

        except Exception as e:
            logger.error("LLM optimization failed: %s", e)
            return self._optimize_with_gradient(review_data, current_params)

    def _optimize_hybrid(
        self,
        review_data: list[dict[str, Any]],
        current_params: list[float],
    ) -> list[float]:
        """混合优化"""
        # 先用梯度下降快速优化
        gradient_params = self._optimize_with_gradient(review_data, current_params)

        # 再用大模型精细调整
        if self._llm_caller:
            try:
                llm_params = self._optimize_with_llm(review_data, gradient_params)

                # 对比效果
                gradient_precision = self._calculate_precision(review_data, gradient_params)
                llm_precision = self._calculate_precision(review_data, llm_params)

                # 选择更好的
                if llm_precision > gradient_precision:
                    logger.info(
                        "Hybrid: LLM better (%.2f%% > %.2f%%)",
                        llm_precision * 100,
                        gradient_precision * 100,
                    )
                    return llm_params
                else:
                    logger.info(
                        "Hybrid: Gradient better (%.2f%% >= %.2f%%)",
                        gradient_precision * 100,
                        llm_precision * 100,
                    )
                    return gradient_params

            except Exception as e:
                logger.warning("Hybrid LLM failed, using gradient: %s", e)
                return gradient_params

        return gradient_params

    def _build_llm_prompt(
        self,
        review_data: list[dict[str, Any]],
        current_params: list[float],
    ) -> str:
        """构建大模型提示"""
        # 统计数据
        total_reviews = len(review_data)
        avg_rating = sum(r.get("rating", 3) for r in review_data) / total_reviews
        avg_retention = sum(r.get("retention_before", 0.5) for r in review_data) / total_reviews

        prompt = f"""你是一个记忆科学专家，需要优化 FSRS (Free Spaced Repetition Scheduler) 参数。

## 当前参数
```python
parameters = {current_params}
```

## 用户学习数据
- 总复习次数: {total_reviews}
- 平均评分: {avg_rating:.2f} (1=Again, 2=Hard, 3=Good, 4=Easy)
- 平均保持率: {avg_retention:.2%}

## 近期复习记录
```json
{json.dumps(review_data[:10], indent=2, ensure_ascii=False)}
```

## 任务
请分析用户的学习模式，优化 FSRS 参数。

参数含义:
- w[0:4]: 初始稳定性 (Again, Hard, Good, Easy)
- w[4:6]: 初始难度参数
- w[6:12]: 稳定性增长参数
- w[12:17]: 难度调整参数
- w[17:19]: 遗忘曲线参数 (α, β)

请返回优化后的参数列表 (19 个浮点数)。

输出格式:
```python
[0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61, 9.0, 0.5]
```
"""
        return prompt

    def _parse_llm_response(
        self,
        response: str,
        current_params: list[float],
    ) -> list[float]:
        """解析大模型响应"""
        import re

        # 尝试提取参数列表
        pattern = r"\[[\d\.,\s]+\]"
        match = re.search(pattern, response)

        if match:
            try:
                params_str = match.group(0)
                params = json.loads(params_str)

                if len(params) == len(current_params):
                    return params
            except:
                pass

        # 解析失败，返回当前参数
        logger.warning("Failed to parse LLM response")
        return current_params

    def _validate_params(self, params: list[float]) -> bool:
        """验证参数有效性"""
        if len(params) != 19:
            return False

        # 检查参数范围
        for i, p in enumerate(params):
            if not isinstance(p, (int, float)):
                return False
            if p < 0 or p > 100:
                return False

        return True

    def _estimate_cost(self, mode: str) -> float:
        """估算计算成本"""
        if mode == "gradient":
            return 0.001  # 极低成本
        elif mode == "llm":
            return 0.1  # 中等成本 (API 调用)
        elif mode == "hybrid":
            return 0.101  # 混合成本
        return 0.0

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        return {
            "current_mode": self._current_mode,
            "current_precision": self._current_precision,
            "total_calls": self._total_calls,
            "gradient_calls": self._gradient_calls,
            "llm_calls": self._llm_calls,
            "mode_switches": self._mode_switches,
            "precision_threshold": self._precision_threshold,
            "decline_threshold": self._decline_threshold,
        }

    def get_history(self) -> dict[str, Any]:
        """获取历史记录"""
        return {
            "precision_history": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "precision": r.precision,
                    "sample_count": r.sample_count,
                    "mode": r.mode,
                }
                for r in self._precision_history
            ],
            "optimization_history": [
                {
                    "mode": r.mode,
                    "precision_before": r.precision_before,
                    "precision_after": r.precision_after,
                    "improvement": r.improvement,
                    "cost": r.cost,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self._optimization_history[-10:]  # 最近 10 条
            ],
        }

    def reset(self) -> None:
        """重置状态"""
        self._current_mode = "gradient"
        self._current_precision = 1.0
        self._precision_history.clear()
        self._optimization_history.clear()
        self._total_calls = 0
        self._gradient_calls = 0
        self._llm_calls = 0
        self._mode_switches = 0
        self._save_state()
        logger.info("Optimizer reset")


# 全局实例
_optimizer: AdaptiveFSRSOptimizer | None = None


def get_adaptive_optimizer(
    precision_threshold: float = 0.85,
    decline_threshold: float = 0.05,
    llm_caller: Callable | None = None,
) -> AdaptiveFSRSOptimizer:
    """获取全局自适应优化器"""
    global _optimizer
    if _optimizer is None:
        _optimizer = AdaptiveFSRSOptimizer(
            precision_threshold=precision_threshold,
            decline_threshold=decline_threshold,
            llm_caller=llm_caller,
        )
    return _optimizer


def adaptive_optimize(
    review_data: list[dict[str, Any]],
    current_params: list[float],
) -> tuple[list[float], str]:
    """便捷函数：自适应优化"""
    optimizer = get_adaptive_optimizer()
    return optimizer.optimize(review_data, current_params)
