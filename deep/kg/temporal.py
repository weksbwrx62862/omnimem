"""时序三元组处理。

职责：
  - 获取实体相关三元组的时间线
  - 生成时间线可读文本
  - 获取最近 N 天的三元组变更
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from omnimem.deep.kg.entity import _normalize_kg_entity

logger = logging.getLogger(__name__)


def get_timeline(self, entity: str, limit: int = 50) -> list[dict[str, Any]]:
    """获取实体的时间线：与其相关的所有三元组按创建时间排序。

    对标 Zep/kektordb 的时序图谱能力 — 追踪实体关系的演变。

    Returns:
        按 created_at 升序的三元组列表，形成实体关系演变时间线
    """
    if not self._conn:
        return []
    try:
        normalized = _normalize_kg_entity(entity)
        rows = self._conn.execute(
            "SELECT * FROM triples "
            "WHERE (subject = ? OR object = ?) AND (valid_to = '' OR valid_to IS NULL) "
            "ORDER BY created_at ASC LIMIT ?",
            (normalized, normalized, limit),
        ).fetchall()
        return self._rows_to_dicts(rows)
    except Exception as e:
        logger.warning("Timeline query failed: %s", e)
        return []


def get_entity_timeline_text(self, entity: str, limit: int = 20) -> str:
    """生成实体时间线的可读文本，适合注入 LLM 上下文。

    Returns:
        格式化的时间线文本，无结果返回空字符串
    """
    timeline = self.get_timeline(_normalize_kg_entity(entity), limit=limit)
    if not timeline:
        return ""

    relation_labels = {
        "uses": "开始使用", "belongs_to": "归属", "causes": "引起",
        "replaces": "取代", "connects_to": "关联到", "contains": "包含",
        "located_in": "位于", "better_than": "优于",
    }
    lines = [f"[{entity} 时间线]"]
    for t in timeline:
        created = t.get("created_at", "")[:10]  # 只取日期
        subj = t.get("subject", "")
        obj = t.get("object", "")
        pred = t.get("predicate", "")
        label = relation_labels.get(pred, pred)
        lines.append(f"  {created}: {subj} {label} {obj}")
    return "\n".join(lines)


def get_recent_changes(self, since_days: int = 7, limit: int = 50) -> list[dict[str, Any]]:
    """获取最近N天的三元组变更。

    Args:
        since_days: 最近多少天
        limit: 最大返回数

    Returns:
        按创建时间降序的三元组列表
    """
    if not self._conn:
        return []
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
        rows = self._conn.execute(
            "SELECT * FROM triples WHERE created_at >= ? "
            "AND (valid_to = '' OR valid_to IS NULL) "
            "ORDER BY created_at DESC LIMIT ?",
            (since, limit),
        ).fetchall()
        return self._rows_to_dicts(rows)
    except Exception as e:
        logger.warning("Recent changes query failed: %s", e)
        return []
