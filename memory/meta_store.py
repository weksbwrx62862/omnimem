"""MetaStore — SQLite 元数据存储。

P0方案一：将 DrawerClosetStore 的元数据管理从文件系统迁移到 SQLite，
保留 Drawer 文件作为原始内容冷备份。

核心设计：
  1. 元数据（wing/room/type/summary 等）存 SQLite，利用 B-tree 索引加速查询
  2. 完整原文仍存 Drawer 文件，get() 需要时按需读取
  3. 提供 FTS5 全文搜索（若可用），回退到 LIKE
  4. 与 DrawerClosetStore 接口兼容，便于渐进式切换
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetaStore:
    """SQLite 元数据存储引擎。

    表结构：
      memories      — 核心元数据表
      memories_fts  — 可选 FTS5 虚拟表（全文搜索）
    """

    def __init__(self, db_dir: Path):
        self._db_dir = db_dir
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._db_dir / "meta_store.db"
        self._conn: sqlite3.Connection | None = None
        self._fts_enabled = False
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库表结构和索引。"""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # 核心元数据表
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                wing TEXT,
                hall TEXT,
                room TEXT,
                type TEXT,
                confidence INTEGER DEFAULT 3,
                privacy TEXT DEFAULT 'personal',
                stored_at TEXT,
                summary TEXT,
                content_preview TEXT,
                drawer_path TEXT,
                vc TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 向后兼容：旧表无 vc 列时自动迁移
        try:
            self._conn.execute("SELECT vc FROM memories LIMIT 1")
        except sqlite3.OperationalError:
            self._conn.execute("ALTER TABLE memories ADD COLUMN vc TEXT")
            logger.info("MetaStore migrated: added vc column")

        # 单列索引
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_wing ON memories(wing)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_privacy ON memories(privacy)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_stored_at ON memories(stored_at)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_room ON memories(room)")

        # 尝试创建 FTS5 虚拟表（全文搜索）
        try:
            self._conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id UNINDEXED,
                    summary,
                    content_preview,
                    content='memories',
                    content_rowid='rowid'
                )
            """)
            # 创建触发器保持 FTS 表同步
            self._conn.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, summary, content_preview)
                    VALUES (new.rowid, new.summary, new.content_preview);
                END
            """)
            self._conn.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, summary, content_preview)
                    VALUES ('delete', old.rowid, old.summary, old.content_preview);
                END
            """)
            self._conn.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, summary, content_preview)
                    VALUES ('delete', old.rowid, old.summary, old.content_preview);
                    INSERT INTO memories_fts(rowid, summary, content_preview)
                    VALUES (new.rowid, new.summary, new.content_preview);
                END
            """)
            self._fts_enabled = True
            logger.debug("MetaStore FTS5 enabled")
        except Exception:
            logger.debug("MetaStore FTS5 not available, falling back to LIKE search")
            self._fts_enabled = False

        self._conn.commit()

    # ─── CRUD ─────────────────────────────────────────────────

    def add(self, memory_id: str, **fields) -> None:
        """添加或替换一条元数据记录。"""
        if not self._conn:
            return
        cols = ["memory_id"] + [k for k in fields if k != "memory_id"]
        vals = [memory_id] + [fields.get(k, "") for k in cols[1:]]
        placeholders = ",".join("?" * len(cols))
        try:
            self._conn.execute(
                f"INSERT OR REPLACE INTO memories ({','.join(cols)}) VALUES ({placeholders})",
                vals,
            )
            self._conn.commit()
        except Exception as e:
            logger.debug("MetaStore add failed for %s: %s", memory_id, e)

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """根据 ID 获取元数据。"""
        if not self._conn:
            return None
        try:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if row:
                return self._row_to_dict(row)
        except Exception as e:
            logger.debug("MetaStore get failed for %s: %s", memory_id, e)
        return None

    def update_privacy(self, memory_id: str, privacy: str, new_wing: str = "") -> bool:
        """更新隐私级别和可选 wing。"""
        if not self._conn:
            return False
        try:
            if new_wing:
                self._conn.execute(
                    "UPDATE memories SET privacy = ?, wing = ? WHERE memory_id = ?",
                    (privacy, new_wing, memory_id),
                )
            else:
                self._conn.execute(
                    "UPDATE memories SET privacy = ? WHERE memory_id = ?",
                    (privacy, memory_id),
                )
            self._conn.commit()
            return True
        except Exception as e:
            logger.debug("MetaStore update_privacy failed for %s: %s", memory_id, e)
            return False

    def delete(self, memory_id: str) -> bool:
        """删除元数据记录。"""
        if not self._conn:
            return False
        try:
            self._conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            self._conn.commit()
            return True
        except Exception as e:
            logger.debug("MetaStore delete failed for %s: %s", memory_id, e)
            return False

    # ─── 搜索 ─────────────────────────────────────────────────

    def search(
        self,
        wing: str = "",
        room: str = "",
        memory_type: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """按条件搜索元数据。利用 SQLite 索引加速。"""
        if not self._conn:
            return []
        conditions: list[str] = []
        params: list[Any] = []
        if memory_type:
            conditions.append("type = ?")
            params.append(memory_type)
        if wing:
            conditions.append("wing = ?")
            params.append(wing)
        if room:
            conditions.append("room = ?")
            params.append(room)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        try:
            rows = self._conn.execute(
                f"SELECT * FROM memories {where} ORDER BY stored_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.debug("MetaStore search failed: %s", e)
            return []

    def search_by_content(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """按内容关键词搜索。

        优先使用 FTS5（若可用），否则回退到 LIKE。
        """
        if not self._conn:
            return []
        try:
            if self._fts_enabled:
                # FTS5 查询
                rows = self._conn.execute(
                    """SELECT m.* FROM memories_fts f
                       JOIN memories m ON m.rowid = f.rowid
                       WHERE memories_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (query, limit),
                ).fetchall()
                return [self._row_to_dict(r) for r in rows]
            else:
                # LIKE 回退
                q = f"%{query}%"
                rows = self._conn.execute(
                    """SELECT * FROM memories
                       WHERE summary LIKE ? OR content_preview LIKE ?
                       ORDER BY stored_at DESC
                       LIMIT ?""",
                    (q, q, limit),
                ).fetchall()
                return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.debug("MetaStore search_by_content failed: %s", e)
            return []

    def get_all(self, limit: int = 5000) -> list[dict[str, Any]]:
        """获取所有元数据记录。"""
        if not self._conn:
            return []
        try:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY stored_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.debug("MetaStore get_all failed: %s", e)
            return []

    def count(self) -> int:
        """返回记录总数。"""
        if not self._conn:
            return 0
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    def warm_up(self, entries: list[dict[str, Any]]) -> int:
        """批量预热：将现有条目导入 SQLite。"""
        if not self._conn:
            return 0
        added = 0
        try:
            for entry in entries:
                mid = entry.get("memory_id", "")
                if not mid:
                    continue
                self.add(
                    memory_id=mid,
                    wing=entry.get("wing", ""),
                    hall=entry.get("hall", entry.get("type", "fact")),
                    room=entry.get("room", ""),
                    type=entry.get("type", "fact"),
                    confidence=entry.get("confidence", 3),
                    privacy=entry.get("privacy", "personal"),
                    stored_at=entry.get("stored_at", ""),
                    summary=entry.get("summary", ""),
                    content_preview=entry.get("content", "")[:500],
                    vc=entry.get("vc", ""),
                )
                added += 1
            self._conn.commit()
            logger.info("MetaStore warmed up %d entries", added)
        except Exception as e:
            logger.debug("MetaStore warm_up failed: %s", e)
        return added

    # ─── 内部方法 ─────────────────────────────────────────────

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """将 SQLite 行转为字典。"""
        keys = [
            "memory_id",
            "wing",
            "hall",
            "room",
            "type",
            "confidence",
            "privacy",
            "stored_at",
            "summary",
            "content_preview",
            "drawer_path",
            "vc",
            "created_at",
        ]
        return {k: row[i] for i, k in enumerate(keys) if i < len(row)}

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None
