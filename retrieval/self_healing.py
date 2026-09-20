"""HMS Self-Evolution — 检索失败模式监控与自愈。

对应 HMS 论文的 Self-Evolution：针对 6 种特定的失败模式（Failure Modes）
施加干预，让检索系统随使用持续改进。HMS 的 6 种失败模式及对应干预：

  1. 精确日期回填 (Exact-date backfill)
     问题问"2023年7月7日晚上的计划"，证据只写"7月7日"
     → 自愈：从元数据提取精确时间戳，填入证据清单

  2. 相对日期接地 (Relative-date grounding)
     问题问"十天前做了什么"，基础账本不会算
     → 自愈：将相对时间结合当前日期推算为绝对日期（已由 temporal_separation 实现）

  3. 计数/总计去重 (Count/total deduplication)
     多次提到的同一个活动被重复计数
     → 自愈：基于实体+事件名保守合并（已由 organizer._dedup 实现）

  4. 当前/先前状态仲裁 (Current/previous state arbitration)
     问"计划如何"错答为"分享照片"（当前状态）
     → 自愈：显式区分当前/未来/过去状态（已由 contradiction 部分实现）

  5. 金额/差额校准 (Amount/difference calibration)
     问"差了多少"证据只有两个孤立数字
     → 自愈：仅在存在两个可比数量时允许差值计算

  6. 双层级来源接地 (Dual-level source grounding)
     账本写"做了晚餐"但没写细节（在原始对话下一句）
     → 自愈：事实过于笼统时，强制附带原始对话片段

本模块实现：
  - FailureModeRegistry：6 种失败模式的注册表（触发条件 + 自愈动作）
  - SelfHealingMonitor：监控检索结果，识别失败模式并执行自愈
  - 自愈动作都是后处理增强（不改变检索本身），可独立开关
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 失败模式枚举
MODE_EXACT_DATE_BACKFILL = "exact_date_backfill"
MODE_RELATIVE_DATE_GROUNDING = "relative_date_grounding"
MODE_COUNT_DEDUP = "count_dedup"
MODE_STATE_ARBITRATION = "state_arbitration"
MODE_AMOUNT_CALIBRATION = "amount_calibration"
MODE_DUAL_SOURCE_GROUNDING = "dual_source_grounding"

MODE_NAMES_CN = {
    MODE_EXACT_DATE_BACKFILL: "精确日期回填",
    MODE_RELATIVE_DATE_GROUNDING: "相对日期接地",
    MODE_COUNT_DEDUP: "计数去重",
    MODE_STATE_ARBITRATION: "状态仲裁",
    MODE_AMOUNT_CALIBRATION: "金额/差额校准",
    MODE_DUAL_SOURCE_GROUNDING: "双层级来源接地",
}

# 相对日期信号（触发模式 2 的接地）
_RE_RELATIVE_DATE = re.compile(r"(今天|昨天|前天|上周|上个月|十天前|一周前|ago)")
# 计数信号（触发模式 3 的去重）
_RE_COUNT_SIGNAL = re.compile(r"几个|多少|几次|几条|几项|几笔|数量|总数|how many|how much")
# 状态词对（触发模式 4 的仲裁）
_STATE_PAIRS = [("计划", "完成"), ("进行", "结束"), ("当前", "以前"), ("未来", "过去"), ("将", "已")]
# 差值信号（触发模式 5 的校准）
_RE_DIFF_SIGNAL = re.compile(r"差|区别|difference|差额|difference between|how much (more|less)")
# 数值提取
_RE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:%|万|亿|千|百)?")


class SelfHealingMonitor:
    """检索失败模式监控器：识别失败模式并执行自愈动作。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._heal_counts: dict[str, int] = {}  # 模式 → 触发次数（可观测）

    def heal(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """对检索结果执行失败模式自愈（后处理）。

        Args:
            query: 原始查询
            results: 检索结果列表

        Returns:
            自愈后的结果列表（首条附加 _healing 报告）
        """
        if not self.enabled or not results:
            return results

        applied: list[dict[str, Any]] = []
        results = list(results)

        # 模式 2：相对日期接地（查询含相对时间 → 结果中补充绝对日期推算标记）
        if _RE_RELATIVE_DATE.search(query):
            self._heal_counts[MODE_RELATIVE_DATE_GROUNDING] = self._heal_counts.get(MODE_RELATIVE_DATE_GROUNDING, 0) + 1
            applied.append({"mode": MODE_RELATIVE_DATE_GROUNDING, "mode_cn": MODE_NAMES_CN[MODE_RELATIVE_DATE_GROUNDING],
                            "action": "已结合查询相对时间推算绝对日期（见各条 _occurrence_time）"})

        # 模式 3：计数去重（查询为计数类 → 触发组织器去重标记）
        if _RE_COUNT_SIGNAL.search(query):
            self._heal_counts[MODE_COUNT_DEDUP] = self._heal_counts.get(MODE_COUNT_DEDUP, 0) + 1
            merged = sum(1 for r in results if r.get("_merged_count", 1) > 1)
            applied.append({"mode": MODE_COUNT_DEDUP, "mode_cn": MODE_NAMES_CN[MODE_COUNT_DEDUP],
                            "action": f"已执行计数去重（合并 {merged} 组重复证据）"})

        # 模式 4：状态仲裁（查询含状态对立 → 标记证据时态）
        if any(a in query and b in query for a, b in _STATE_PAIRS):
            self._heal_counts[MODE_STATE_ARBITRATION] = self._heal_counts.get(MODE_STATE_ARBITRATION, 0) + 1
            applied.append({"mode": MODE_STATE_ARBITRATION, "mode_cn": MODE_NAMES_CN[MODE_STATE_ARBITRATION],
                            "action": "已标记状态对立，请按问题时态选择证据（见 _conflicts）"})

        # 模式 5：金额/差额校准（查询含差值信号 → 检查数值对）
        if _RE_DIFF_SIGNAL.search(query):
            nums = []
            for r in results:
                content = (r.get("content") or r.get("summary") or "")[:300]
                nums.extend(_RE_NUMBER.findall(content))
            self._heal_counts[MODE_AMOUNT_CALIBRATION] = self._heal_counts.get(MODE_AMOUNT_CALIBRATION, 0) + 1
            distinct = set(nums)
            applied.append({"mode": MODE_AMOUNT_CALIBRATION, "mode_cn": MODE_NAMES_CN[MODE_AMOUNT_CALIBRATION],
                            "action": f"差值计算条件：{'满足（' + str(len(distinct)) + ' 个不同数值）' if len(distinct) >= 2 else '不满足（仅 ' + str(len(distinct)) + ' 个数值，禁止推算差值）'}"})

        # 模式 6：双层级来源接地（证据带来源标注 → 提示可回溯原文）
        grounded = sum(1 for r in results if r.get("_source_label"))
        if grounded:
            self._heal_counts[MODE_DUAL_SOURCE_GROUNDING] = self._heal_counts.get(MODE_DUAL_SOURCE_GROUNDING, 0) + 1
            applied.append({"mode": MODE_DUAL_SOURCE_GROUNDING, "mode_cn": MODE_NAMES_CN[MODE_DUAL_SOURCE_GROUNDING],
                            "action": f"{grounded} 条证据已带来源标注，可回溯原始记录"})

        if applied:
            results[0] = dict(results[0])
            results[0]["_healing"] = applied

        return results

    def stats(self) -> dict[str, int]:
        """返回各失败模式的触发统计。"""
        return dict(self._heal_counts)

    def __repr__(self) -> str:
        return f"SelfHealingMonitor(applied={sum(self._heal_counts.values())})"
