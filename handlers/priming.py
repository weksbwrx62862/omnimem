"""PrimingCache — 启动效应缓存。

模拟人类"想到A → 更容易想到B"的语义启动效应（priming effect）。
基于最近检索的实体，给当前检索中匹配的结果加权。

工作原理：
  1. 每次 recall 后，将结果中的实体记录到 PrimingCache
  2. 下次 recall 时，检查结果实体是否与缓存的实体重叠
  3. 每重叠一个实体 +15% 分数加成
  4. 缓存随时间衰减（每次新记录衰减到 80%）
  5. Session 隔离：不同 session 的启动状态独立

集成点（handle_recall）:
  - 检索后 enrich 阶段：调用 get_priming_state(sid).apply_boost(results)
  - 返回结果前：调用 get_priming_state(sid).record(entities)
"""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# ── 模块级缓存：session_id -> PrimingState ─────────────────────
# 所有 session 的启动状态共享同一个模块级字典，
# 保证同一 session 内多次 recall 调用共享上下文。
_priming_cache: dict[str, PrimingState] = {}

# ── 默认参数 ───────────────────────────────────────────────────
# 每次新记录时，已有权重乘以此系数
_DEFAULT_DECAY: float = 0.8
# 跟踪最近 N 次 recall 的实体（超过的被弹出）
_DEFAULT_MAX_EVENTS: int = 5
# 每重叠一个实体的分数加成比例
_DEFAULT_BOOST_PER_ENTITY: float = 0.15
# 权重的静默阈值（低于此值被清理）
_WEIGHT_EPSILON: float = 0.01


class PrimingState:
    """单个 session 的启动状态。

    Attributes:
        max_events: 跟踪的最近 recall 次数
        decay: 每次新记录时的衰减系数
        weights: entity -> cumulative_weight
        recent_events: 最近 N 次 recall 的实体列表（用于诊断）
    """

    def __init__(
        self,
        max_events: int = _DEFAULT_MAX_EVENTS,
        decay: float = _DEFAULT_DECAY,
    ) -> None:
        self.max_events = max_events
        self.decay = decay
        self.weights: dict[str, float] = {}
        self.recent_events: deque[list[str]] = deque(maxlen=max_events)

    def record(self, entities: list[str], weight: float = 1.0) -> None:
        """记录本次 recall 命中的实体。

        1. 已有所有权重指数衰减 (× decay)
        2. 低于 ε 的权重被清理
        3. 新实体累加到权重中
        4. 加入 recent_events 队列（自动管理长度）

        Args:
            entities: 本次 recall 结果中的实体列表
            weight: 本次记录的基础权重（默认 1.0）
        """
        if not entities:
            return

        # 加入事件队列（自动管理 maxlen）
        self.recent_events.append(list(entities))

        # 已有权重衰减
        stale_keys = []
        for k in self.weights:
            new_w = self.weights[k] * self.decay
            if new_w < _WEIGHT_EPSILON:
                stale_keys.append(k)
            else:
                self.weights[k] = new_w
        for k in stale_keys:
            del self.weights[k]

        # 新实体累加（去重，避免同一 recall 实体内重复计数）
        for e in sorted(set(entities)):  # sorted 保证确定性
            self.weights[e] = self.weights.get(e, 0) + weight

        logger.debug(
            "Priming record: %d entities added, total primed: %d",
            len(set(entities)),
            len(self.weights),
        )

    def apply_boost(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """对检索结果应用启动效应加成。

        对每条结果，检查其 entities 与缓存实体的重叠度，
        每重叠一个实体，score 乘以 (1 + _DEFAULT_BOOST_PER_ENTITY)。

        Args:
            results: 检索结果列表（每项需含 entities/metadata.entities）

        Returns:
            加成后的结果列表（原地修改并返回）
        """
        if not self.weights:
            return results

        primed = set(self.weights.keys())
        if not primed:
            return results

        for r in results:
            ents = r.get("entities", [])
            if isinstance(ents, str):
                try:
                    ents = json.loads(ents)
                except (json.JSONDecodeError, TypeError):
                    ents = [ents] if ents else []

            overlap = len(set(ents) & primed)
            if overlap > 0:
                boost = 1.0 + _DEFAULT_BOOST_PER_ENTITY * overlap
                old_score = r.get("score", 0.0)
                r["score"] = old_score * boost
                r["_priming_boost"] = round(boost, 4)
                logger.debug(
                    "Priming boost: overlap=%d boost=%.4f old=%.4f new=%.4f",
                    overlap,
                    boost,
                    old_score,
                    r["score"],
                )

        return results

    def get_primed_entities(self) -> list[str]:
        """获取当前所有缓存的实体（按权重降序）。"""
        return sorted(self.weights, key=lambda e: -self.weights[e])

    def clear(self) -> None:
        """清空启动状态。"""
        self.weights.clear()
        self.recent_events.clear()

    def __repr__(self) -> str:
        n_ents = len(self.weights)
        top = self.get_primed_entities()[:3]
        return (
            f"PrimingState(entities={n_ents}, decay={self.decay}, "
            f"events={len(self.recent_events)}, top={top})"
        )


def get_priming_state(session_id: str) -> PrimingState:
    """获取指定 session 的启动状态（惰性创建）。

    Args:
        session_id: 会话标识

    Returns:
        PrimingState 实例（模块级缓存，同一 session 共享）
    """
    if session_id not in _priming_cache:
        _priming_cache[session_id] = PrimingState()
        logger.debug("PrimingCache: created new state for session %s", session_id[:16])
    return _priming_cache[session_id]


def clear_session(session_id: str) -> None:
    """清除指定 session 的启动状态。

    用于 session 结束或重置。
    """
    state = _priming_cache.pop(session_id, None)
    if state is not None:
        logger.debug("PrimingCache: cleared state for session %s", session_id[:16])


def get_stats() -> dict[str, Any]:
    """返回 PrimingCache 的全局统计（用于诊断）。"""
    return {
        "active_sessions": len(_priming_cache),
        "sessions": {
            sid: {
                "entities": len(state.weights),
                "events": len(state.recent_events),
                "top_entities": state.get_primed_entities()[:5],
            }
            for sid, state in _priming_cache.items()
        },
    }
