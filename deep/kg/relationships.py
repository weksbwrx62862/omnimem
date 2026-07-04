"""关系边同步与批量维护。

职责：
  - 将新增三元组同步到 relationships 关系表
  - 批量从已有 triples 回填 relationships
  - 关系强度累计与去重
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def get_stats(self) -> dict[str, Any]:
    """获取知识图谱统计信息。"""
    stats: dict[str, Any] = {
        "triples": 0,
        "entities": 0,
        "relationships": 0,
    }
    if not self._conn:
        return stats
    try:
        row = self._conn.execute("SELECT COUNT(*) FROM triples").fetchone()
        stats["triples"] = row[0] if row else 0
        row = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()
        stats["entities"] = row[0] if row else 0
        row = self._conn.execute("SELECT COUNT(*) FROM relationships").fetchone()
        stats["relationships"] = row[0] if row else 0
    except Exception as e:
        logger.warning("KnowledgeGraph get_stats failed: %s", e)
    return stats


def _sync_relationship_locked(
    self,
    source: str,
    target: str,
    relation_type: str,
    strength: float = 1.0,
) -> None:
    """将三元组同步写入 relationships 表（内部已持有锁时调用）。"""
    try:
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        # UPSERT: 新关系插入，已有关系累加 strength
        existing = self._conn.execute(
            "SELECT id, strength FROM relationships WHERE source_id = ? AND target_id = ? AND relation_type = ?",
            (source, target, relation_type),
        ).fetchone()
        if existing:
            new_strength = min(existing[1] + 0.1, 10.0)
            self._conn.execute(
                "UPDATE relationships SET strength = ? WHERE id = ?",
                (new_strength, existing[0]),
            )
        else:
            self._conn.execute(
                "INSERT OR IGNORE INTO relationships (source_id, target_id, relation_type, strength, created_at) VALUES (?, ?, ?, ?, ?)",
                (source, target, relation_type, strength, now),
            )
        self._conn.commit()
    except Exception as e:
        logger.debug("Relationship sync failed: %s", e)


def sync_relationships_from_triples(self) -> int:
    """从已有 triples 批量构建 relationships（用于迁移/回填）。

    Returns:
        新插入的 relationship 数量
    """
    with self._lock:
        assert self._conn is not None
        triples = self._conn.execute(
            "SELECT subject, object, predicate, confidence FROM triples WHERE (valid_to = '' OR valid_to IS NULL)"
        ).fetchall()
        count = 0
        for subj, obj, pred, conf in triples:
            existing = self._conn.execute(
                "SELECT id FROM relationships WHERE source_id = ? AND target_id = ? AND relation_type = ?",
                (subj, obj, pred),
            ).fetchone()
            if not existing:
                now = datetime.now(timezone.utc).isoformat()
                try:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO relationships (source_id, target_id, relation_type, strength, created_at) VALUES (?, ?, ?, ?, ?)",
                        (subj, obj, pred, conf or 1.0, now),
                    )
                    count += 1
                except Exception:
                    pass
        self._conn.commit()
        logger.info("sync_relationships_from_triples: inserted %d relationships from %d triples", count, len(triples))
        return count
