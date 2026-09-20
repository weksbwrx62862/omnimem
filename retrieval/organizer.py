"""HMS GatedOrganization — 证据组织器。

对应 HMS 论文的 Organization & Construction：证据往往已全部找到，但按
"检索分数"乱序排列——同一事件被提取两次（重复）、计划/已完成混在一起、
发生/提及时间混在一起。让生成器对着乱麻做"计数/排序"是人为制造难度。

实现：
  1. 去重：基于内容相似度（SimHash/Jaccard）合并重复记忆
  2. 时间排序：按 stored_at 升序排列（时间线清晰）
  3. 来源标注：为每条证据附加 source 信息（wing/hall/room/session）
  4. Gated 触发：仅当查询包含聚合类信号（计数/排序/比较/时序）时启用，
     日常查询跳过（节省算力）
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 触发 GatedOrganization 的查询信号（计数/排序/比较/时序）
_GATE_SIGNALS = re.compile(
    r"几个|多少|几次|几条|几项|几笔|数量|总数|统计|频率|"
    r"how many|how much|count|total|"
    r"排序|顺序|最早|最晚|最新|先后|"
    r"比较|对比|区别|差异|哪个|谁|"
    r"变化|趋势|进展|过程|时间线|时间轴|"
    r"上周|上周|昨天|今天|最近|之前|当时|"
    r"第一次|最后一次|然后|接着|后来"
)


class EvidenceOrganizer:
    """证据组织器：去重 + 时间排序 + 来源标注（Gated 触发）。"""

    def __init__(self, enabled: bool = True, dedup_threshold: float = 0.85) -> None:
        self.enabled = enabled
        self.dedup_threshold = dedup_threshold

    def should_organize(self, query: str) -> bool:
        """Gated 判定：查询是否属于聚合类（计数/排序/比较/时序）。"""
        if not query:
            return False
        return bool(_GATE_SIGNALS.search(query))

    def organize(
        self,
        results: list[dict[str, Any]],
        *,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """组织证据清单（去重/排序/标注）。

        Args:
            results: 检索结果
            force: 跳过 Gated 判定强制组织

        Returns:
            组织后的结果列表（每项带 _source_label 标注；重复项被合并）
        """
        if not self.enabled:
            return list(results)
        if not results:
            return []

        # 1. 去重（SimHash 近似）
        deduped = self._dedup(results)

        # 2. 时间排序（无时间戳的排在最后）
        deduped.sort(key=lambda r: self._sort_key(r))

        # 3. 来源标注（新列表对象，不动原列表）
        out = []
        for r in deduped:
            r = dict(r)
            r["_source_label"] = self._source_label(r)
            r["_organized"] = True
            out.append(r)

        return out

    # ── 内部 ──

    @staticmethod
    def _sort_key(r: dict[str, Any]) -> tuple:
        """排序键：时间戳（无则用最大，排最后）。"""
        t = (
            r.get("stored_at")
            or r.get("timestamp")
            or r.get("created_at")
            or r.get("metadata", {}).get("stored_at")
            or ""
        )
        return (0, str(t)) if t else (1, "")

    @staticmethod
    def _source_label(r: dict[str, Any]) -> str:
        """生成来源标注。"""
        wing = r.get("wing") or r.get("metadata", {}).get("wing", "")
        hall = r.get("hall") or r.get("metadata", {}).get("hall", "")
        room = r.get("room") or r.get("metadata", {}).get("room", "")
        session = r.get("session_id") or r.get("metadata", {}).get("session_id", "")
        parts = [p for p in (wing, hall, room) if p]
        if session:
            parts.append(f"会话:{session[:8]}")
        return "/".join(parts) if parts else "未知来源"

    def _dedup(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """基于内容指纹去重：指纹相似度超阈值视为重复，保留分数高的。"""
        if len(results) < 2:
            return list(results)

        buckets: list[list[dict[str, Any]]] = []  # 每组重复项
        fingerprints: list[str] = []

        for r in results:
            content = (r.get("content") or r.get("summary") or r.get("text") or "")[:500]
            fp = self._simhash(content)
            placed = False
            for i, group_fp in enumerate(fingerprints):
                if self._hamming_similarity(fp, group_fp) >= self.dedup_threshold:
                    buckets[i].append(r)
                    placed = True
                    break
            if not placed:
                buckets.append([r])
                fingerprints.append(fp)

        # 每组取分数最高的
        out = []
        for group in buckets:
            best = max(group, key=lambda x: float(x.get("score", 0.0)))
            best = dict(best)
            if len(group) > 1:
                best["_merged_count"] = len(group)
            out.append(best)
        return out

    @staticmethod
    def _simhash(text: str) -> str:
        """简化 SimHash：按 4-gram 特征哈希取指纹（64bit 位串的十六进制）。"""
        if not text:
            return "0" * 16
        grams = [text[i:i+4] for i in range(len(text) - 3)] if len(text) >= 4 else [text]
        if not grams:
            return "0" * 16
        bits = [0] * 64
        for g in grams:
            h = hash(g) & 0xFFFFFFFFFFFFFFFF
            for i in range(64):
                if (h >> i) & 1:
                    bits[i] += 1
                else:
                    bits[i] -= 1
        return "".join("1" if b > 0 else "0" for b in bits)

    @staticmethod
    def _hamming_similarity(a: str, b: str) -> float:
        """汉明相似度 = 1 - 汉明距离/长度。"""
        if len(a) != len(b):
            return 0.0
        if not a:
            return 1.0
        diff = sum(1 for x, y in zip(a, b) if x != y)
        return 1.0 - diff / len(a)

    def __repr__(self) -> str:
        return f"EvidenceOrganizer(dedup_threshold={self.dedup_threshold}, enabled={self.enabled})"
