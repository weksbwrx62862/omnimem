"""反思结果持久化。

职责：
  - 将 ReflectResult 写入 SQLite reflections 表
  - 生成 reflection_id 并处理并发写入
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from omnimem.deep.reflect.disposition import ReflectResult

logger = logging.getLogger(__name__)


def _persist_reflection(self, result: ReflectResult) -> None:
    """持久化反思结果。"""
    if not self._conn:
        return
    with self._lock:
        now = datetime.now(timezone.utc).isoformat()
        reflection_id = f"ref-{self._reflection_count:04d}-{now[:10]}"
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO reflections
                   (reflection_id, query, observation, mental_model, confidence,
                    disposition, source_ids, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    reflection_id,
                    result.query,
                    result.observation,
                    result.mental_model,
                    result.confidence,
                    json.dumps(result.disposition_used, ensure_ascii=False)
                    if result.disposition_used
                    else "",
                    json.dumps(result.sources, ensure_ascii=False),
                    now,
                ),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning("Reflect persist failed: %s", e)
