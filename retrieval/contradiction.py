"""HMS Contradiction 算子 — 冲突检测与裁决。

对应 HMS 论文的 Contradiction (C) 算子：不同来源/时期的记录对同一事实
描述相互矛盾（如年龄、决定、状态）。不判断对错、不合并，而是将互相矛盾
的整组记录全部打包召回，并高亮标注各自的时间和来源——让下游生成器同时
看到双方证据，避免偏听偏信。

实现：
  1. 从检索结果中检测"同一主体 + 属性值分歧"的模式
  2. 数值冲突：同一实体相关记忆中的数字不一致（年龄/价格/数量）
  3. 状态冲突：布尔/状态类词汇对立（是/否、开始/结束、计划/完成）
  4. 输出：冲突组（grouped evidence）+ 高亮标注（时间/来源）
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 数值提取：支持整数/小数/百分比
_RE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?%?")
# 状态对立词对
_STATE_PAIRS = [
    ("是", "否"), ("有", "没有"), ("会", "不会"), ("能", "不能"),
    ("开始", "结束"), ("完成", "未完成"), ("计划", "完成"),
    ("同意", "拒绝"), ("喜欢", "讨厌"), ("支持", "反对"),
    ("true", "false"), ("yes", "no"), ("open", "closed"),
    ("active", "inactive"), ("开启", "关闭"), ("上线", "下线"),
]


class ContradictionDetector:
    """冲突检测器：检测同主体属性值分歧，打包返回冲突证据组。"""

    def __init__(self, enabled: bool = True, min_evidence: int = 2) -> None:
        self.enabled = enabled
        self.min_evidence = min_evidence  # 至少 N 条结果才值得检测

    def detect(
        self,
        results: list[dict[str, Any]],
        *,
        entity_field: str = "entities",
    ) -> list[dict[str, Any]]:
        """检测检索结果中的冲突。

        Args:
            results: 检索结果
            entity_field: 结果中的实体字段名

        Returns:
            冲突组列表：[{entity, conflict_type, values, evidences}]
        """
        if not self.enabled or len(results) < self.min_evidence:
            return []

        conflicts: list[dict[str, Any]] = []
        seen_keys: set[tuple] = set()

        # 按共享实体分组
        entity_groups: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            entities = r.get(entity_field) or r.get("metadata", {}).get("entities", []) or []
            content = (r.get("content") or r.get("summary") or "")[:300]
            # 从内容中粗提实体（引号专名 + 中文称谓）
            for ent in re.findall(r'["“「『]([^"”」』]{2,12})["”」』]', content):
                entities.append(ent)
            for ent in re.findall(r"[\u4e00-\u9fff]{2,6}(?:公司|集团|银行|医院|大学|项目|系统)", content):
                entities.append(ent)
            for ent in entities[:3]:
                entity_groups.setdefault(ent, []).append(r)

        # 1. 数值冲突：同实体相关记忆中出现不同数值
        for entity, group in entity_groups.items():
            if len(group) < 2:
                continue
            key = (entity, "numeric")
            if key in seen_keys:
                continue
            numeric = self._detect_numeric_conflict(entity, group)
            if numeric:
                seen_keys.add(key)
                conflicts.append(numeric)
            state = self._detect_state_conflict(entity, group)
            if state:
                s_key = (entity, "state")
                if s_key not in seen_keys:
                    seen_keys.add(s_key)
                    conflicts.append(state)

        # 2. 内容级冲突（无共享实体时）：跨全部结果检测数值/状态对立
        if not conflicts:
            content_level_numeric = self._detect_numeric_conflict("(全局)", results)
            if content_level_numeric:
                conflicts.append(content_level_numeric)
            content_level_state = self._detect_state_conflict("(全局)", results)
            if content_level_state:
                conflicts.append(content_level_state)

        return conflicts

    # ── 内部 ──

    @staticmethod
    def _detect_numeric_conflict(entity: str, group: list[dict[str, Any]]) -> dict[str, Any] | None:
        """检测数值冲突：不同记忆中的数值不一致。"""
        value_sources: list[tuple[str, str, str]] = []  # (value, time, source)
        for r in group:
            content = (r.get("content") or r.get("summary") or "")[:300]
            nums = _RE_NUMBER.findall(content)
            if nums:
                t = r.get("stored_at") or r.get("timestamp") or r.get("created_at") or ""
                src = (
                    r.get("wing") or r.get("metadata", {}).get("wing", "")
                ) or "unknown"
                # 取最后一个数值（通常是最具体的）
                value_sources.append((nums[-1], str(t)[:10], src))

        if len(value_sources) >= 2:
            distinct = {v for v, _, _ in value_sources}
            if len(distinct) >= 2:
                return {
                    "entity": entity,
                    "conflict_type": "numeric",
                    "values": value_sources,
                    "evidences": group,
                    "note": "同一主体的数值记录不一致，请同时参考双方证据",
                }
        return None

    @staticmethod
    def _detect_state_conflict(entity: str, group: list[dict[str, Any]]) -> dict[str, Any] | None:
        """检测状态冲突：对立状态词同时出现。"""
        blob = " ".join(
            (r.get("content") or r.get("summary") or "")[:200] for r in group
        ).lower()
        for a, b in _STATE_PAIRS:
            if a in blob and b in blob:
                return {
                    "entity": entity,
                    "conflict_type": "state",
                    "pairs": [(a, b)],
                    "evidences": group,
                    "note": f"检测到状态对立（{a}/{b}），请同时参考双方证据",
                }
        return None

    def __repr__(self) -> str:
        return f"ContradictionDetector(min_evidence={self.min_evidence}, enabled={self.enabled})"
