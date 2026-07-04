"""自动反思触发器 — 多因子评分决定何时触发反思。

借鉴 QoderWork 的设计理念，在会话结束后自动评估是否值得反思，
而不是依赖 Agent 主动调用 omni_reflect。

评分因子：
  1. 对话深度（turns）
  2. 工具调用多样性
  3. 用户纠正信号
  4. 任务复杂度（工具调用总数）
  5. 距上次反思的时间间隔
  6. 新记忆产生数量

防抖机制：
  - 冷却窗口：反思后 N 分钟内不再触发
  - 会话级去重：同一会话最多反思 1 次
  - 置信度阈值：评分低于阈值不触发
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReflectionSignal:
    """反思触发信号。"""
    should_reflect: bool = False
    score: float = 0.0
    factors: dict[str, float] = field(default_factory=dict)
    query_hint: str = ""  # 建议的反思主题
    reason: str = ""


class ReflectionTrigger:
    """多因子反思触发器。

    在每个会话结束后评估是否应该自动触发反思。
    """

    # 默认配置
    DEFAULT_CONFIG = {
        "enabled": True,
        "score_threshold": 0.6,          # 触发阈值 (0-1)
        "cooldown_seconds": 1800,         # 冷却窗口 30 分钟
        "max_reflections_per_session": 1, # 每会话最多反思次数
        "min_turns": 3,                   # 最少对话轮数
        "weights": {
            "turn_depth": 0.15,           # 对话深度权重
            "tool_diversity": 0.15,       # 工具调用多样性
            "correction_signals": 0.25,   # 用户纠正信号权重
            "task_complexity": 0.15,      # 任务复杂度
            "time_since_last": 0.10,      # 距上次反思间隔
            "new_memories": 0.10,         # 新记忆产生数量
            "session_duration": 0.10,     # 会话持续时间
        },
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._last_reflection_time: float = 0.0
        self._reflections_this_session: int = 0
        self._session_tool_calls: list[str] = []
        self._session_corrections: int = 0
        self._session_new_memories: int = 0
        self._session_start_time: float = time.time()

    def reset_session(self) -> None:
        """新会话开始时重置计数器。"""
        self._reflections_this_session = 0
        self._session_tool_calls = []
        self._session_corrections = 0
        self._session_new_memories = 0
        self._session_start_time = time.time()

    def record_tool_call(self, tool_name: str) -> None:
        """记录工具调用。"""
        self._session_tool_calls.append(tool_name)

    def record_correction(self) -> None:
        """记录用户纠正信号。"""
        self._session_corrections += 1

    def record_new_memory(self) -> None:
        """记录新记忆产生。"""
        self._session_new_memories += 1

    def evaluate(
        self,
        turn_count: int,
        signals: Any = None,
        session_id: str = "",
    ) -> ReflectionSignal:
        """评估是否应该触发反思。

        Args:
            turn_count: 当前对话轮数
            signals: PerceptionEngine 检测到的信号
            session_id: 会话 ID

        Returns:
            ReflectionSignal 包含是否触发、评分、原因
        """
        if not self._config.get("enabled", True):
            return ReflectionSignal(reason="反思触发器已禁用")

        # 防抖检查
        now = time.time()

        # 冷却窗口
        cooldown = self._config["cooldown_seconds"]
        if now - self._last_reflection_time < cooldown:
            remaining = int(cooldown - (now - self._last_reflection_time))
            return ReflectionSignal(
                reason=f"冷却中，还需等待 {remaining} 秒"
            )

        # 每会话次数限制
        if self._reflections_this_session >= self._config["max_reflections_per_session"]:
            return ReflectionSignal(
                reason=f"本会话已反思 {self._reflections_this_session} 次"
            )

        # 最少轮数
        if turn_count < self._config["min_turns"]:
            return ReflectionSignal(
                reason=f"对话轮数不足 ({turn_count} < {self._config['min_turns']})"
            )

        # ── 多因子评分 ──
        weights = self._config["weights"]
        factors = {}

        # 1. 对话深度 (turns / 20, capped at 1.0)
        factors["turn_depth"] = min(turn_count / 20.0, 1.0)

        # 2. 工具调用多样性 (unique_tools / 8, capped at 1.0)
        unique_tools = len(set(self._session_tool_calls))
        total_tools = len(self._session_tool_calls)
        factors["tool_diversity"] = min(unique_tools / 8.0, 1.0)

        # 3. 用户纠正信号 (corrections / 3, capped at 1.0)
        correction_count = self._session_corrections
        if signals and hasattr(signals, "has_correction") and signals.has_correction:
            correction_count += 1
        factors["correction_signals"] = min(correction_count / 3.0, 1.0)

        # 4. 任务复杂度 (total_tools / 15, capped at 1.0)
        factors["task_complexity"] = min(total_tools / 15.0, 1.0)

        # 5. 距上次反思的时间间隔 (hours_since_last / 4, capped at 1.0)
        hours_since = (now - self._last_reflection_time) / 3600.0
        factors["time_since_last"] = min(hours_since / 4.0, 1.0)

        # 6. 新记忆产生数量 (new_memories / 5, capped at 1.0)
        factors["new_memories"] = min(self._session_new_memories / 5.0, 1.0)

        # 7. 会话持续时间 (minutes / 30, capped at 1.0)
        duration_min = (now - self._session_start_time) / 60.0
        factors["session_duration"] = min(duration_min / 30.0, 1.0)

        # 加权求和
        total_score = sum(
            factors[k] * weights.get(k, 0) for k in factors
        )

        # 生成反思主题提示
        query_hint = self._generate_query_hint(
            turn_count, unique_tools, correction_count, total_tools
        )

        # 决策
        threshold = self._config["score_threshold"]
        should_reflect = total_score >= threshold

        if should_reflect:
            self._last_reflection_time = now
            self._reflections_this_session += 1

        reason_parts = []
        if factors["correction_signals"] > 0.3:
            reason_parts.append(f"用户纠正{correction_count}次")
        if factors["task_complexity"] > 0.5:
            reason_parts.append(f"复杂任务({total_tools}次工具调用)")
        if factors["turn_depth"] > 0.5:
            reason_parts.append(f"深度对话({turn_count}轮)")
        if factors["new_memories"] > 0.3:
            reason_parts.append(f"产生{self._session_new_memories}条新记忆")

        reason = "触发反思: " + ", ".join(reason_parts) if should_reflect else f"评分 {total_score:.2f} < 阈值 {threshold}"

        signal = ReflectionSignal(
            should_reflect=should_reflect,
            score=total_score,
            factors=factors,
            query_hint=query_hint,
            reason=reason,
        )

        if should_reflect:
            logger.info(
                "自动反思触发 [score=%.2f]: %s", total_score, reason
            )

        return signal

    def _generate_query_hint(
        self,
        turn_count: int,
        unique_tools: int,
        corrections: int,
        total_tools: int,
    ) -> str:
        """根据会话特征生成反思主题提示。"""
        parts = []

        if corrections > 0:
            parts.append(f"用户在本次会话中纠正了{corrections}次，分析纠正原因并提炼经验")

        if unique_tools >= 5:
            parts.append(f"本次任务使用了{unique_tools}种工具，总结工具组合模式")

        if total_tools >= 10:
            parts.append("本次是复杂多步骤任务，提炼可复用的工作流程")

        if turn_count >= 10:
            parts.append("长对话中的关键决策点和转折")

        if not parts:
            parts.append("总结本次会话的经验和可改进之处")

        return "；".join(parts)

    def mark_reflected(self) -> None:
        """手动标记已反思（用于 Agent 主动调用 omni_reflect 时）。"""
        self._last_reflection_time = time.time()
        self._reflections_this_session += 1

    @property
    def stats(self) -> dict[str, Any]:
        """返回当前状态统计。"""
        return {
            "enabled": self._config.get("enabled", True),
            "last_reflection_ago_seconds": int(time.time() - self._last_reflection_time) if self._last_reflection_time else None,
            "reflections_this_session": self._reflections_this_session,
            "session_tool_calls": len(self._session_tool_calls),
            "session_unique_tools": len(set(self._session_tool_calls)),
            "session_corrections": self._session_corrections,
            "session_new_memories": self._session_new_memories,
            "cooldown_seconds": self._config["cooldown_seconds"],
            "score_threshold": self._config["score_threshold"],
        }
