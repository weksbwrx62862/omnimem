"""HMS Session-local 证据算子 — 会话内位置邻近增强。

对应 HMS 论文的 Session-local (L) 算子：碎片化上下文问题的答案往往紧邻
高置信度入口点（Entry Point），但因表面不相关而被传统检索忽略。

实现：
  1. 在融合后的候选集上运行（不需要额外检索通道，零额外召回成本）
  2. 利用记忆的 stored_at 时间戳（时间邻近 ≈ 会话内位置邻近的代理）
  3. 对与高置信度结果（入口点）时间邻近的记忆做高斯核加权提升
  4. 高斯核: w = exp(-(Δt/σ)² / 2)，σ 默认 24h（同一天内的记忆视为邻近）

设计约束：
  - 纯后处理增强，不改变检索通道本身 → 无侵入、可独立开关
  - 只在候选集存在"入口点"（高分结果）时启用
  - 时间差 Δt 取绝对值，单位小时；无时间戳的记忆跳过
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# 高斯核带宽：24 小时（同一天内的记忆视为会话内邻近）
DEFAULT_SIGMA_HOURS = 24.0
# 启用阈值：候选集中必须有至少一个"入口点"（高分结果）才启用
ENTRY_POINT_SCORE_THRESHOLD = 0.6
# 提升上限（避免过度抬高低质量结果）
MAX_BOOST = 1.5


def _parse_time(value: Any) -> float | None:
    """解析记忆中的时间戳为 Unix 秒。支持多种格式，失败返回 None。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # 秒级时间戳
        return float(value) if value > 1e9 else float(value) * 1000 if value > 1e6 else None
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            pass
        try:
            from datetime import datetime as _dt
            return _dt.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
        except (ValueError, TypeError):
            pass
    return None


def _gaussian_weight(dt_hours: float, sigma_hours: float = DEFAULT_SIGMA_HOURS) -> float:
    """高斯核权重：时间差越小权重越高，范围 (0, 1]。"""
    if dt_hours < 0:
        dt_hours = -dt_hours
    return math.exp(-(dt_hours / sigma_hours) ** 2 / 2.0)


class SessionLocalOperator:
    """会话内位置邻近增强算子。"""

    def __init__(self, sigma_hours: float = DEFAULT_SIGMA_HOURS, enabled: bool = True) -> None:
        self.sigma_hours = sigma_hours
        self.enabled = enabled

    def enhance(
        self,
        results: list[dict[str, Any]],
        *,
        entry_point_scores: list[float] | None = None,
        time_field: str = "stored_at",
    ) -> list[dict[str, Any]]:
        """对候选结果做会话内位置邻近增强（原地修改 score 字段）。

        Args:
            results: 检索结果列表（每项含 score/content 等字段）
            entry_point_scores: 入口点（高置信度结果）的分数列表。
                为 None 时取 results 中 score >= ENTRY_POINT_SCORE_THRESHOLD 的项。
            time_field: 记忆时间字段名

        Returns:
            增强后的结果列表（新列表，不修改入参）
        """
        if not self.enabled or len(results) < 2:
            return list(results)

        # 确定入口点：高分结果的时间作为锚点
        entry_times: list[float] = []
        if entry_point_scores is not None:
            for score in entry_point_scores:
                if score >= ENTRY_POINT_SCORE_THRESHOLD:
                    entry_times.append(float(score))
        # 若未提供分数，从结果自身推断（score 字段）
        if not entry_times:
            for r in results:
                score = float(r.get("score", 0.0))
                if score >= ENTRY_POINT_SCORE_THRESHOLD:
                    t = _parse_time(r.get(time_field) or r.get("timestamp") or r.get("created_at"))
                    if t is not None:
                        entry_times.append(t)
            if not entry_times:
                return list(results)  # 无入口点时间，跳过

        # 对每个结果计算与最近入口点的时间邻近加权
        boosted = []
        for r in results:
            score = float(r.get("score", 0.0))
            t = _parse_time(r.get(time_field) or r.get("timestamp") or r.get("created_at"))
            if t is not None:
                # 到最近入口点的最小时间差
                min_dt = min(abs(t - et) for et in entry_times)
                w = _gaussian_weight(min_dt / 3600.0, self.sigma_hours)
                boost = 1.0 + w * (MAX_BOOST - 1.0)  # 提升 1.0 ~ 1.5 倍
                new_score = score * boost
                r = dict(r)
                r["score"] = new_score
                r["_session_local_boost"] = round(boost, 3)
                r["_session_local_dt_hours"] = round(min_dt / 3600.0, 2)
            boosted.append(r)

        # 按新分数降序
        boosted.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return boosted

    def __repr__(self) -> str:
        return f"SessionLocalOperator(sigma={self.sigma_hours}h, enabled={self.enabled})"
