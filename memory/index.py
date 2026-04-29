"""ThreeLevelIndex — 三层索引 L0/L1/L2。

参考 OpenViking 的三层索引设计：
  - L0 (目录索引): Wing/Hall/Room 结构索引，最小化加载
  - L1 (摘要索引): Closet 摘要，中等粒度
  - L2 (全文索引): Drawer 原文，最大精度

索引存储在 SQLite 中，支持快速查找和范围查询。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ThreeLevelIndex:
    """三层索引 L0/L1/L2。

    批量提交优化：add() 不立即 commit，攒到阈值或显式 flush() 时统一提交，
    减少磁盘 fsync 次数。
    """

    _BATCH_THRESHOLD = 5  # 每 5 次写入 commit 一次

    def __init__(self, index_dir: Path):
        self._index_dir = index_dir
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._index_dir / "index.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._pending_writes = 0
        self._init_db()

    def _init_db(self) -> None:
        """初始化 SQLite 数据库。"""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_index (
                memory_id TEXT PRIMARY KEY,
                wing TEXT NOT NULL,
                hall TEXT NOT NULL,
                room TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                type TEXT NOT NULL,
                confidence INTEGER DEFAULT 3,
                privacy TEXT DEFAULT 'personal',
                scope TEXT DEFAULT 'personal',
                stored_at TEXT,
                provenance TEXT,
                metadata TEXT
            )
        """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_wing ON memory_index(wing)
        """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_type ON memory_index(type)
        """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stored_at ON memory_index(stored_at)
        """
        )
        self._conn.commit()

    def _maybe_commit(self) -> None:
        """检查待写入数是否达到阈值，达到则提交事务。"""
        self._pending_writes += 1
        if self._pending_writes >= self._BATCH_THRESHOLD:
            self._conn.commit()
            self._pending_writes = 0

    def add(
        self,
        memory_id: str,
        wing: str,
        hall: str,
        room: str,
        content: str,
        summary: str = "",
        type: str = "fact",
        confidence: int = 3,
        privacy: str = "personal",
        scope: str = "personal",
        stored_at: str = "",
        provenance: str = "",
        metadata: str = "",
    ) -> None:
        """添加一条索引记录。"""
        if not stored_at:
            stored_at = datetime.now().isoformat()
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO memory_index
                   (memory_id, wing, hall, room, content, summary, type,
                    confidence, privacy, scope, stored_at, provenance, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory_id,
                    wing,
                    hall,
                    room,
                    content,
                    summary,
                    type,
                    confidence,
                    privacy,
                    scope,
                    stored_at,
                    provenance,
                    metadata,
                ),
            )
            self._maybe_commit()
        except Exception as e:
            logger.debug("Index add failed for %s: %s", memory_id, e)

    def get(self, memory_id: str) -> Optional[dict[str, Any]]:
        """根据 ID 获取索引记录。"""
        try:
            row = self._conn.execute(
                "SELECT * FROM memory_index WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row:
                return self._row_to_dict(row)
        except Exception as e:
            logger.debug("Index get failed for %s: %s", memory_id, e)
        return None

    def delete(self, memory_id: str) -> bool:
        """从索引中删除记录。"""
        try:
            self._conn.execute(
                "DELETE FROM memory_index WHERE memory_id = ?",
                (memory_id,),
            )
            self._maybe_commit()
            return True
        except Exception as e:
            logger.debug("Index delete failed for %s: %s", memory_id, e)
            return False

    def search_l0(self, wing: str = "", hall: str = "") -> list[str]:
        """L0 目录索引：返回匹配的 Room 列表。"""
        query = "SELECT DISTINCT room FROM memory_index WHERE 1=1"
        params = []
        if wing:
            query += " AND wing = ?"
            params.append(wing)
        if hall:
            query += " AND hall = ?"
            params.append(hall)
        try:
            rows = self._conn.execute(query, params).fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            logger.debug("L0 search failed: %s", e)
            return []

    def search_l1(self, wing: str = "", type: str = "", limit: int = 50) -> list[dict[str, Any]]:
        """L1 摘要索引：返回摘要记录（含 content 用于 warm_up）。"""
        query = "SELECT memory_id, wing, hall, room, summary, type, confidence, privacy, stored_at, content FROM memory_index WHERE 1=1"
        params = []
        if wing:
            query += " AND wing = ?"
            params.append(wing)
        if type:
            query += " AND type = ?"
            params.append(type)
        query += " ORDER BY stored_at DESC LIMIT ?"
        params.append(limit)
        try:
            rows = self._conn.execute(query, params).fetchall()
            return [
                {
                    "memory_id": r[0],
                    "wing": r[1],
                    "hall": r[2],
                    "room": r[3],
                    "summary": r[4],
                    "type": r[5],
                    "confidence": r[6],
                    "privacy": r[7],
                    "stored_at": r[8],
                    "content": r[9] if len(r) > 9 else "",
                }
                for r in rows
            ]
        except Exception as e:
            logger.debug("L1 search failed: %s", e)
            return []

    def search_l2(
        self,
        keyword: str = "",
        wing: str = "",
        type: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """L2 全文索引：返回完整记录。"""
        query = "SELECT * FROM memory_index WHERE 1=1"
        params = []
        if keyword:
            escaped = keyword.replace("%", "\\%").replace("_", "\\_")
            query += " AND content LIKE ? ESCAPE '\\'"
            params.append(f"%{escaped}%")
        if wing:
            query += " AND wing = ?"
            params.append(wing)
        if type:
            query += " AND type = ?"
            params.append(type)
        query += " ORDER BY stored_at DESC LIMIT ?"
        params.append(limit)
        try:
            rows = self._conn.execute(query, params).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.debug("L2 search failed: %s", e)
            return []

    def search_all_for_retrieval(self, limit: int = 1000) -> list[dict[str, Any]]:
        """获取所有记录（用于检索引擎全量索引）。"""
        try:
            rows = self._conn.execute(
                "SELECT * FROM memory_index ORDER BY stored_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.debug("Full index scan failed: %s", e)
            return []

    def update_privacy(self, memory_id: str, privacy: str) -> bool:
        """更新隐私级别。"""
        try:
            self._conn.execute(
                "UPDATE memory_index SET privacy = ? WHERE memory_id = ?",
                (privacy, memory_id),
            )
            self._maybe_commit()
            return True
        except Exception as e:
            logger.debug("Privacy update failed: %s", e)
            return False

    def update_field(self, memory_id: str, **fields) -> bool:
        """更新索引中的指定字段。"""
        if not fields:
            return False
        try:
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            values = list(fields.values()) + [memory_id]
            self._conn.execute(
                f"UPDATE memory_index SET {set_clause} WHERE memory_id = ?",
                values,
            )
            self._maybe_commit()
            return True
        except Exception as e:
            logger.debug("Field update failed: %s", e)
            return False

    def remove(self, memory_id: str) -> bool:
        """删除索引记录。"""
        try:
            self._conn.execute(
                "DELETE FROM memory_index WHERE memory_id = ?",
                (memory_id,),
            )
            self._maybe_commit()
            return True
        except Exception as e:
            logger.debug("Index remove failed: %s", e)
            return False

    def close(self) -> None:
        """关闭数据库连接。"""
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None

    def flush(self) -> None:
        """显式提交所有待写入。"""
        if self._conn and self._pending_writes > 0:
            try:
                self._conn.commit()
                self._pending_writes = 0
            except Exception as e:
                logger.debug("Index flush failed: %s", e)

    def _row_to_dict(self, row) -> dict[str, Any]:
        """将数据库行转为字典。"""
        keys = [
            "memory_id",
            "wing",
            "hall",
            "room",
            "content",
            "summary",
            "type",
            "confidence",
            "privacy",
            "scope",
            "stored_at",
            "provenance",
            "metadata",
        ]
        result = {}
        for i, key in enumerate(keys):
            if i < len(row):
                val = row[i]
                if key == "provenance" and val:
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if key == "metadata" and val:
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
                result[key] = val
        return result
