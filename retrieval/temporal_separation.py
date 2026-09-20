"""HMS Temporal 算子增强 — 提及时间 / 发生时间分离。

对应 HMS 论文 Temporal (T) 算子的关键注意事项：区分"提及时间
(MentionTime)"和"发生时间 (OccurrenceTime)"，避免"现在聊过去"造成的
时间错乱。

实现：
  1. mention_time：记忆被记录的时间（stored_at，已有）
  2. occurrence_time：从记忆内容中提取事件实际发生的时间
     - 显式日期：2023年7月7日 / 7月7日 / 2023-07-07
     - 相对日期：昨天/上周/上个月/十天前（结合 mention_time 推算绝对日期）
  3. 为每条记忆生成时间标注：{mention_time, occurrence_time}
  4. 供组织器/检索排序使用：按 occurrence_time 排序而非 mention_time
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# 显式日期：2023年7月7日 / 2023-07-07 / 7月7日 / 2023/07/07
_RE_FULL_DATE = re.compile(
    r"(\d{4})\s*[年/\-.]\s*(\d{1,2})\s*[月/\-.]\s*(\d{1,2})\s*[日号]?"
)
_RE_MONTH_DAY = re.compile(
    r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]"
)
# 相对时间词
_RE_RELATIVE = re.compile(
    r"(今天|昨天|前天|上周|上上周|这个月|上个月|两个月前|三个月前|"
    r"十天前|一周前|两周前|三周前|一年前|半年前)"
)


def parse_occurrence_time(
    content: str,
    mention_time: str | None = None,
) -> str | None:
    """从内容中提取事件发生时间（ISO 格式），提取不到返回 None。

    Args:
        content: 记忆内容
        mention_time: 提及时间（ISO 字符串），用于相对时间推算

    Returns:
        ISO 时间字符串，或 None
    """
    if not content:
        return None

    # 1. 完整日期（优先）
    m = _RE_FULL_DATE.search(content)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d, tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass

    # 2. 月-日（年份取提及时间）
    m = _RE_MONTH_DAY.search(content)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        base_year = _mention_year(mention_time)
        try:
            return datetime(base_year, mo, d, tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass

    # 3. 相对时间（结合 mention_time 推算）
    base = _parse_mention_time(mention_time)
    m = _RE_RELATIVE.search(content)
    if m and base:
        word = m.group(1)
        delta = {
            "今天": timedelta(days=0),
            "昨天": timedelta(days=1),
            "前天": timedelta(days=2),
            "上周": timedelta(days=7),
            "上上周": timedelta(days=14),
            "这个月": timedelta(days=0),
            "上个月": timedelta(days=30),
            "两个月前": timedelta(days=60),
            "三个月前": timedelta(days=90),
            "十天前": timedelta(days=10),
            "一周前": timedelta(days=7),
            "两周前": timedelta(days=14),
            "三周前": timedelta(days=21),
            "一年前": timedelta(days=365),
            "半年前": timedelta(days=182),
        }.get(word)
        if delta:
            return (base - delta).isoformat()

    return None


def annotate_time(r: dict[str, Any]) -> dict[str, Any]:
    """为单条记忆生成时间标注（原地修改并返回）。

    标注字段：
      _mention_time: 提及时间（stored_at）
      _occurrence_time: 发生时间（内容提取）
    """
    mention = (
        r.get("stored_at") or r.get("timestamp") or r.get("created_at") or ""
    )
    content = (r.get("content") or r.get("summary") or r.get("text") or "")
    occ = parse_occurrence_time(content, mention or None)
    r["_mention_time"] = mention
    r["_occurrence_time"] = occ
    return r


def annotate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """批量时间标注。"""
    return [annotate_time(dict(r)) for r in results]


def sort_by_occurrence(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按发生时间排序（无发生时间的按提及时间，均无的排最后）。"""
    def key(r: dict[str, Any]) -> tuple:
        occ = r.get("_occurrence_time")
        mention = r.get("_mention_time")
        if occ:
            return (0, occ)
        if mention:
            return (1, mention)
        return (2, "")

    return sorted(results, key=key)


# ── 内部 ──

def _mention_year(mention_time: str | None) -> int:
    """从提及时间提取年份（默认 2026）。"""
    dt = _parse_mention_time(mention_time)
    return dt.year if dt else datetime.now(timezone.utc).year


def _parse_mention_time(mention_time: str | None) -> datetime | None:
    """解析提及时间为 datetime（支持 ISO 格式）。"""
    if not mention_time:
        return None
    try:
        return datetime.fromisoformat(str(mention_time).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
