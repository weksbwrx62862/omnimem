"""HMS Verifier — 证据充分性验证器。

对应 HMS 论文的 Evidence Sufficiency Verifier：验证候选证据是否覆盖了
查询需要回答的关键槽位（slots）。若覆盖不足，返回"信息不足"信号，
让上层 Agent 明确知道缺什么，而不是硬着头皮生成（解决"缺失无告警"）。

实现：
  - 基于 Planner 的查询意图（temporal/entity/count/general）推导期望证据槽位
  - 槽位推导为纯规则：时间类查询期望"时间锚点"，实体类期望"实体+属性"，
    COUNT 类期望"可计数条目"，general 期望"相关内容覆盖"
  - 输出 coverage 报告：filled/total/missing + 置信度
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 覆盖阈值：filled/total 低于此值 → 信息不足
COVERAGE_THRESHOLD = 0.4
# 最小结果数：少于该数量视为信息不足（即使类型匹配）
MIN_RESULTS_FOR_ADEQUATE = 3


@dataclass
class CoverageReport:
    """证据覆盖报告。"""

    filled_slots: int = 0
    total_slots: int = 0
    missing_slots: list[str] = field(default_factory=list)
    coverage: float = 0.0
    adequate: bool = True
    verdict: str = "adequate"  # adequate | insufficient
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "filled_slots": self.filled_slots,
            "total_slots": self.total_slots,
            "missing_slots": self.missing_slots,
            "coverage": round(self.coverage, 2),
            "adequate": self.adequate,
            "verdict": self.verdict,
            "reason": self.reason,
        }


class EvidenceVerifier:
    """证据充分性验证器：根据查询意图检查检索结果的覆盖度。"""

    def __init__(self, coverage_threshold: float = COVERAGE_THRESHOLD, enabled: bool = True) -> None:
        self.coverage_threshold = coverage_threshold
        self.enabled = enabled

    def verify(
        self,
        query: str,
        results: list[dict[str, Any]],
        *,
        intent: str = "general",
    ) -> CoverageReport:
        """验证检索结果对查询的证据覆盖。

        Args:
            query: 原始查询
            results: 检索结果列表
            intent: Planner 输出的查询意图（temporal/entity/count/preference/conversational/general）

        Returns:
            CoverageReport
        """
        if not self.enabled:
            return CoverageReport(adequate=True, verdict="adequate")

        if not results:
            return CoverageReport(
                filled_slots=0, total_slots=1, missing_slots=["任何相关记忆"],
                coverage=0.0, adequate=False, verdict="insufficient",
                reason="未检索到任何记忆",
            )

        # 按意图推导槽位
        slots = self._derive_slots(query, intent, results)
        filled = self._count_filled(slots, query, results, intent)

        total = max(len(slots), 1)
        coverage = filled / total
        missing = [s["name"] for s in slots if not s["filled"]]

        adequate = coverage >= self.coverage_threshold and len(results) >= MIN_RESULTS_FOR_ADEQUATE
        verdict = "adequate" if adequate else "insufficient"
        reason = ""
        if not adequate:
            reason = (
                f"证据覆盖不足: 期望 {total} 个槽位, 仅满足 {filled} 个"
                f"{('（缺失: ' + ', '.join(missing) + '）') if missing else ''}"
                f", 检索到 {len(results)} 条记忆"
            )

        return CoverageReport(
            filled_slots=filled,
            total_slots=total,
            missing_slots=missing,
            coverage=coverage,
            adequate=adequate,
            verdict=verdict,
            reason=reason,
        )

    # ── 内部 ──

    @staticmethod
    def _derive_slots(query: str, intent: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按查询意图推导期望的证据槽位。"""
        slots: list[dict[str, Any]] = []
        q = query.lower()

        if intent == "temporal":
            # 时间类：期望时间锚点 + 相关内容
            time_anchor = bool(re.search(
                r"\d{4}年|\d{1,2}月|\d{1,2}日|今天|昨天|上周|上个月|去年|ago|yesterday|last (week|month|year)",
                q,
            ))
            slots.append({"name": "时间锚点", "filled": time_anchor})
            slots.append({"name": "时间段内内容", "filled": False})
        elif intent == "entity":
            # 实体类：期望实体提及 + 属性/行为描述
            slots.append({"name": "目标实体", "filled": False})
            slots.append({"name": "实体行为/属性", "filled": False})
        elif intent == "count":
            # COUNT 类：期望可计数条目 + 数值信号
            slots.append({"name": "可计数条目", "filled": False})
            slots.append({"name": "数值信号", "filled": False})
        elif intent == "preference":
            slots.append({"name": "偏好描述", "filled": False})
            slots.append({"name": "候选/推荐", "filled": False})
        elif intent == "conversational":
            # 寒暄：无需证据
            return []
        else:
            # general：期望内容与查询相关 + 至少有一定覆盖
            slots.append({"name": "相关内容", "filled": False})

        # 补充：结果数量本身是证据（越多覆盖越好，最多计 3 个内容槽）
        n = len(results)
        if n >= 3:
            slots.append({"name": "多来源印证", "filled": True})
        elif n >= 1:
            slots.append({"name": "至少一条记忆", "filled": True})

        return slots

    @staticmethod
    def _count_filled(
        slots: list[dict[str, Any]], query: str, results: list[dict[str, Any]], _intent: str
    ) -> int:
        """统计已填充的槽位（规则启发式）。"""
        filled_count = 0
        for s in slots:
            if s["filled"]:
                filled_count += 1
                continue
            name = s["name"]
            # 联合内容文本（前若干条）
            blob = " ".join(
                (r.get("content", "") or r.get("summary", "") or r.get("text", "") or "")[:200]
                for r in results[:5]
            ).lower()
            q = query.lower()

            if name == "时间锚点":
                filled_count += 1  # 已由 derive 判定
            elif name == "时间段内内容":
                # 结果中含时间类词 → 认为有内容
                filled_count += 1 if re.search(r"\d{4}年|\d{1,2}月|\d{1,2}日|周|月|年|day|week|month", blob) else 0
            elif name == "目标实体":
                # 结果含查询中的实体词（去停用词后的 token）
                tokens = [t for t in re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", q) if t not in ("什么", "怎么", "为什么", "怎么样")]
                filled_count += 1 if any(t in blob for t in tokens) else 0
            elif name == "实体行为/属性":
                filled_count += 1 if any(k in blob for k in ("说", "做", "去", "买", "吃", "看", "计划", "喜欢", "是", "有")) else 0
            elif name == "可计数条目":
                filled_count += 1 if any(k in blob for k in ("个", "次", "条", "笔", "项", "天", "人", "家", "款")) else 0
            elif name == "数值信号":
                filled_count += 1 if re.search(r"\d+", blob) else 0
            elif name == "偏好描述":
                filled_count += 1 if any(k in blob for k in ("喜欢", "偏好", "推荐", "prefer", "like", "best")) else 0
            elif name == "候选/推荐":
                filled_count += 1 if len(results) >= 2 else 0
            elif name == "相关内容":
                tokens = [t for t in re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", q) if t not in ("什么", "怎么", "为什么", "怎么样", "一下", "这个", "那个")]
                filled_count += 1 if any(t in blob for t in tokens) else 0

        return filled_count

    def __repr__(self) -> str:
        return f"EvidenceVerifier(threshold={self.coverage_threshold}, enabled={self.enabled})"
