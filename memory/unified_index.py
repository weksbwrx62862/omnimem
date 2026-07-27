"""UnifiedMemoryIndex — 合并 ThreeLevelIndex + MetaStore 的统一索引。

重构动机：
  原架构中 ThreeLevelIndex 与 MetaStore 表结构高度重叠（memory_id/wing/hall/room/
  type/confidence/privacy/stored_at），均带 FTS5 + 触发器，造成：
    1. 存储空间浪费（双份 FTS5 索引）
    2. 双写 Saga 一致性负担
    3. 两处 SQLite 连接并发竞争磁盘 IO

重构方案：
  合并为单一 SQLite 数据库，单连接 + RLock 保护，支持读写分离的连接池模式。
  表结构兼容原 ThreeLevelIndex 全字段，新增 content_preview/drawer_path/vc 字段
  （原 MetaStore 独有），消除双写。

并发模型：
  - 写操作：持 _write_lock，串行化 commit
  - 读操作：使用单独的只读连接，无锁并发
  - WAL 模式天然支持读不阻塞写
"""

from __future__ import annotations

import logging
import sqlite3

try:
    import jieba as _jieba

    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)


def _tokenize_for_fts(text: str) -> str:
    """M6-7: jieba pre-tokenize for FTS5 Chinese matching (fallback: raw text)."""
    if not text or not _HAS_JIEBA:
        return text or ""
    return " ".join(_jieba.cut(text))

_DB_RETRY_COUNT = 5
_DB_RETRY_DELAY = 0.15
_BUSY_TIMEOUT_MS = 10000


class UnifiedMemoryIndex:
    """统一记忆索引 — 替代 ThreeLevelIndex + MetaStore 双写。

    特性：
      1. 单一 SQLite 数据库，消除双写一致性负担
      2. 读写连接分离：写连接持锁串行，读连接无锁并发（WAL 优势）
      3. 批量提交（_BATCH_THRESHOLD=10），减少 fsync 次数
      4. 显式 Python 级 _write_lock，修复原 ThreeLevelIndex 无锁并发问题
      5. 连接健康检查 + 自动重连，修复原 ForgettingCurve 连接失效问题
    """

    _BATCH_THRESHOLD = 10

    def __init__(self, index_dir: Path):
        self._index_dir = index_dir
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._index_dir / "unified_index.db"

        # 读写分离连接
        self._write_conn: sqlite3.Connection | None = None
        self._read_conn: sqlite3.Connection | None = None

        # ★ 修复 C3：显式 Python 级写锁，串行化写操作
        self._write_lock = threading.RLock()
        # 读操作无需 Python 锁（WAL 模式下读不阻塞写）
        self._pending_writes = 0
        self._closed = False

        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _new_connection(self, *, read_only: bool = False) -> sqlite3.Connection:
        """创建新的 SQLite 连接，统一配置 WAL + busy_timeout。"""
        uri = f"file:{self._db_path}?mode=ro" if read_only else str(self._db_path)
        conn = sqlite3.connect(
            uri if read_only else str(self._db_path),
            uri=read_only,
            check_same_thread=False,
            timeout=_BUSY_TIMEOUT_MS / 1000,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        if read_only:
            conn.execute("PRAGMA query_only=1")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """初始化数据库 schema。"""
        self._write_conn = self._new_connection(read_only=False)
        self._read_conn = self._new_connection(read_only=True)

        migrator = SchemaMigrator(self._write_conn)
        migrator.migrate(
            table_name="memory_index",
            create_sql="""
                CREATE TABLE IF NOT EXISTS memory_index (
                    memory_id TEXT PRIMARY KEY,
                    wing TEXT NOT NULL DEFAULT '',
                    hall TEXT NOT NULL DEFAULT '',
                    room TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    summary TEXT,
                    content_preview TEXT,
                    drawer_path TEXT,
                    vc TEXT,
                    type TEXT NOT NULL DEFAULT 'fact',
                    confidence INTEGER DEFAULT 3,
                    privacy TEXT DEFAULT 'personal',
                    scope TEXT DEFAULT 'personal',
                    stored_at TEXT,
                    provenance TEXT,
                    metadata TEXT,
                    conflicting_with TEXT,
                    conflict_type TEXT,
                    is_updated INTEGER DEFAULT 0,
                    is_superseded INTEGER DEFAULT 0,
                    created_at TEXT
                )
            """,
            migrations=[
                (2, "ALTER TABLE memory_index ADD COLUMN content_tok TEXT"),
            ],
        )

        # 索引
        self._write_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wing ON memory_index(wing)"
        )
        self._write_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_type ON memory_index(type)"
        )
        self._write_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stored_at ON memory_index(stored_at)"
        )
        self._write_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stage ON memory_index(is_superseded, stored_at)"
        )

        # FTS5 全文索引（content_tok + summary + content_preview）
        # ★ M6-7 修复：unicode61 对中文整句不分词，查询侧 jieba 多词 MATCH 无法命中
        #   （基准实测 recall@5 仅 40.8% vs BM25 98.7%）。改为主表 content_tok 预分词列，
        #   旧库虚表缺列时 DROP 重建并回填。
        for trig in ("memory_index_fts_ai", "memory_index_fts_ad", "memory_index_fts_au"):
            self._write_conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
        fts_cols: list[str] = []
        try:
            fts_cols = [r[1] for r in self._write_conn.execute(
                "PRAGMA table_info(memory_index_fts)").fetchall()]
        except sqlite3.Error:
            fts_cols = []
        rebuilt = bool(fts_cols) and "content_tok" not in fts_cols
        if rebuilt:
            self._write_conn.execute("DROP TABLE IF EXISTS memory_index_fts")
        self._write_conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_index_fts USING fts5("
            "content_tok, summary, content_preview)"
        )
        # 回填预分词列（幂等：仅处理 NULL 行）
        rows = self._write_conn.execute(
            "SELECT rowid, content FROM memory_index WHERE content_tok IS NULL"
        ).fetchall()
        for rid, content in rows:
            self._write_conn.execute(
                "UPDATE memory_index SET content_tok = ? WHERE rowid = ?",
                (_tokenize_for_fts(content or ""), rid),
            )
        if rebuilt:
            self._write_conn.execute(
                "INSERT INTO memory_index_fts(rowid, content_tok, summary, content_preview) "
                "SELECT rowid, COALESCE(content_tok, content), COALESCE(summary,''), "
                "COALESCE(content_preview,'') FROM memory_index"
            )
        self._write_conn.execute("""
            CREATE TRIGGER memory_index_fts_ai AFTER INSERT ON memory_index BEGIN
                INSERT INTO memory_index_fts(rowid, content_tok, summary, content_preview)
                VALUES (new.rowid, COALESCE(new.content_tok, new.content),
                        COALESCE(new.summary,''), COALESCE(new.content_preview,''));
            END
        """)
        self._write_conn.execute("""
            CREATE TRIGGER memory_index_fts_ad AFTER DELETE ON memory_index BEGIN
                DELETE FROM memory_index_fts WHERE rowid = old.rowid;
            END
        """)
        self._write_conn.execute("""
            CREATE TRIGGER memory_index_fts_au AFTER UPDATE ON memory_index BEGIN
                DELETE FROM memory_index_fts WHERE rowid = old.rowid;
                INSERT INTO memory_index_fts(rowid, content_tok, summary, content_preview)
                VALUES (new.rowid, COALESCE(new.content_tok, new.content),
                        COALESCE(new.summary,''), COALESCE(new.content_preview,''));
            END
        """)
        self._write_conn.commit()

    def _retry_db_op(self, fn, *args, **kwargs):
        """★ 修复 C3：增强重试逻辑，5 次指数退避。"""
        last_err = None
        for attempt in range(_DB_RETRY_COUNT):
            try:
                return fn(*args, **kwargs)
            except sqlite3.OperationalError as e:
                last_err = e
                if "locked" in str(e).lower() and attempt < _DB_RETRY_COUNT - 1:
                    time.sleep(_DB_RETRY_DELAY * (2 ** attempt))
                    continue
                raise
        raise last_err  # type: ignore[misc]

    def _ensure_conn_alive(self) -> None:
        """★ 修复 C10：连接健康检查 + 自动重连（持锁，避免 pop 竞态）。"""
        with self._write_lock:
            try:
                self._write_conn.execute("SELECT 1")
            except sqlite3.Error:
                logger.warning("UnifiedMemoryIndex: 写连接失效，重连中...")
                try:
                    self._write_conn.close()
                except Exception:
                    pass
                self._write_conn = self._new_connection(read_only=False)
            try:
                self._read_conn.execute("SELECT 1")
            except sqlite3.Error:
                logger.warning("UnifiedMemoryIndex: 读连接失效，重连中...")
                try:
                    self._read_conn.close()
                except Exception:
                    pass
                self._read_conn = self._new_connection(read_only=True)

    def add(
        self,
        memory_id: str,
        wing: str,
        hall: str,
        room: str,
        content: str,
        summary: str = "",
        content_preview: str = "",
        drawer_path: str = "",
        vc: str = "",
        type: str = "fact",
        confidence: int = 3,
        privacy: str = "personal",
        scope: str = "personal",
        stored_at: str = "",
        provenance: str = "",
        metadata: str = "",
        conflicting_with: str = "",
        conflict_type: str = "",
        is_updated: int = 0,
        is_superseded: int = 0,
    ) -> None:
        """添加或替换一条索引记录（原子写入，批量提交）。"""
        if not stored_at:
            stored_at = datetime.now().isoformat()
        created_at = stored_at
        with self._write_lock:
            self._ensure_conn_alive()
            self._retry_db_op(
                self._write_conn.execute,
                """INSERT OR REPLACE INTO memory_index
                   (memory_id, wing, hall, room, content, summary, content_preview,
                    drawer_path, vc, type, confidence, privacy, scope, stored_at,
                    provenance, metadata, conflicting_with, conflict_type,
                    is_updated, is_superseded, created_at, content_tok)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    memory_id, wing, hall, room, content, summary, content_preview,
                    drawer_path, vc, type, confidence, privacy, scope, stored_at,
                    provenance, metadata, conflicting_with, conflict_type,
                    is_updated, is_superseded, created_at,
                    _tokenize_for_fts(content),
                ),
            )
            self._maybe_commit()

    def _maybe_commit(self) -> None:
        self._pending_writes += 1
        if self._pending_writes >= self._BATCH_THRESHOLD:
            self._retry_db_op(self._write_conn.commit)
            self._pending_writes = 0

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """按 ID 查询（读连接，无锁并发）。"""
        self._ensure_conn_alive()
        row = self._read_conn.execute(
            "SELECT * FROM memory_index WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return dict(row) if row else None

    def search_l1(
        self, wing: str = "", type: str = "", limit: int = 50, memory_type: str = ""
    ) -> list[dict[str, Any]]:
        """L1 摘要索引查询（★ 签名与 ThreeLevelIndex 对齐，type 为主参数名）。"""
        self._ensure_conn_alive()
        mtype = type or memory_type
        sql = "SELECT * FROM memory_index WHERE 1=1"
        params: list[Any] = []
        if wing:
            sql += " AND wing = ?"
            params.append(wing)
        if mtype:
            sql += " AND type = ?"
            params.append(mtype)
        sql += " ORDER BY stored_at DESC LIMIT ?"
        params.append(limit)
        rows = self._read_conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def search_l2(
        self, keyword: str = "", wing: str = "", type: str = "", limit: int = 20,
        query: str = "", memory_type: str = "",
    ) -> list[dict[str, Any]]:
        """L2 全文检索（★ 签名与 ThreeLevelIndex 对齐；中文走 LIKE 回退）。

        FTS5 unicode61 分词器将中文按单字切分，短语查询无法命中，
        与 ThreeLevelIndex 保持一致：含非 ASCII 字符时降级 LIKE。
        """
        self._ensure_conn_alive()
        kw = keyword or query
        mtype = type or memory_type
        params: list[Any] = []

        has_non_ascii = any(ord(c) > 127 for c in kw) if kw else False
        if kw and not has_non_ascii:
            sanitized = " ".join(
                f'"{w}"' for w in kw.replace('"', '""').split() if w.strip()
            ) or '""'
            sql = """
                SELECT m.* FROM memory_index m
                JOIN memory_index_fts f ON m.rowid = f.rowid
                WHERE memory_index_fts MATCH ?
            """
            params.append(sanitized)
            if wing:
                sql += " AND m.wing = ?"
                params.append(wing)
            if mtype:
                sql += " AND m.type = ?"
                params.append(mtype)
            sql += " ORDER BY rank LIMIT ?"
        elif kw:
            escaped = kw.replace("%", "\\%").replace("_", "\\_")
            sql = "SELECT * FROM memory_index WHERE content LIKE ? ESCAPE '\\'"
            params.append(f"%{escaped}%")
            if wing:
                sql += " AND wing = ?"
                params.append(wing)
            if mtype:
                sql += " AND type = ?"
                params.append(mtype)
            sql += " ORDER BY stored_at DESC LIMIT ?"
        else:
            sql = "SELECT * FROM memory_index WHERE 1=1"
            if wing:
                sql += " AND wing = ?"
                params.append(wing)
            if mtype:
                sql += " AND type = ?"
                params.append(mtype)
            sql += " ORDER BY stored_at DESC LIMIT ?"
        params.append(limit)
        rows = self._read_conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── ThreeLevelIndex API 兼容层（feature-flag 落地替换所需） ──

    def search_l0(self, wing: str = "", hall: str = "") -> list[str]:
        """L0 目录索引：返回匹配的 Room 列表。"""
        self._ensure_conn_alive()
        sql = "SELECT DISTINCT room FROM memory_index WHERE 1=1"
        params: list[Any] = []
        if wing:
            sql += " AND wing = ?"
            params.append(wing)
        if hall:
            sql += " AND hall = ?"
            params.append(hall)
        rows = self._read_conn.execute(sql, params).fetchall()
        return [r["room"] for r in rows]

    def search_by_directory(
        self, wing: str = "", hall: str = "", room: str = ""
    ) -> list[dict[str, Any]]:
        """按 Wing/Hall/Room 目录结构查询索引条目。"""
        self._ensure_conn_alive()
        sql = (
            "SELECT memory_id, wing, hall, room, summary, type, confidence "
            "FROM memory_index WHERE 1=1"
        )
        params: list[Any] = []
        if wing:
            sql += " AND wing = ?"
            params.append(wing)
        if hall:
            sql += " AND hall = ?"
            params.append(hall)
        if room:
            sql += " AND room = ?"
            params.append(room)
        rows = self._read_conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def search_all_for_retrieval(self, limit: int = 1000) -> list[dict[str, Any]]:
        """获取所有记录（用于检索引擎全量索引）。"""
        self._ensure_conn_alive()
        rows = self._read_conn.execute(
            "SELECT * FROM memory_index ORDER BY stored_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def update_privacy(self, memory_id: str, privacy: str) -> bool:
        """更新隐私级别。"""
        return self.update_field(memory_id, privacy=privacy)

    def update_field(self, memory_id: str, immediate: bool = False, **fields: Any) -> bool:
        """更新索引中的指定字段（列名白名单校验，防注入）。"""
        if not fields:
            return False
        allowed = {
            "wing", "hall", "room", "content", "summary", "content_preview",
            "drawer_path", "vc", "type", "confidence", "privacy", "scope",
            "stored_at", "provenance", "metadata", "conflicting_with",
            "conflict_type", "is_updated", "is_superseded",
        }
        bad = set(fields) - allowed
        if bad:
            logger.warning("update_field rejected unknown columns: %s", bad)
            return False
        try:
            with self._write_lock:
                self._ensure_conn_alive()
                set_clause = ", ".join(f"{k} = ?" for k in fields)
                values = list(fields.values()) + [memory_id]
                self._retry_db_op(
                    self._write_conn.execute,
                    f"UPDATE memory_index SET {set_clause} WHERE memory_id = ?",
                    values,
                )
                if immediate:
                    self._retry_db_op(self._write_conn.commit)
                    self._pending_writes = 0
                else:
                    self._maybe_commit()
            return True
        except Exception as e:
            logger.warning("Field update failed: %s", e)
            return False

    def remove(self, memory_id: str) -> bool:
        """删除索引记录（delete 的别名，兼容 ThreeLevelIndex）。"""
        return self.delete(memory_id)

    def delete(self, memory_id: str) -> bool:
        """删除记录（FTS5 触发器自动清理全文索引）。"""
        with self._write_lock:
            self._ensure_conn_alive()
            cur = self._retry_db_op(
                self._write_conn.execute,
                "DELETE FROM memory_index WHERE memory_id = ?",
                (memory_id,),
            )
            self._maybe_commit()
            return cur.rowcount > 0

    def mark_superseded(self, memory_id: str, conflicting_with: str = "") -> bool:
        """标记记忆被新记忆取代（用于冲突仲裁）。"""
        with self._write_lock:
            self._ensure_conn_alive()
            cur = self._retry_db_op(
                self._write_conn.execute,
                """UPDATE memory_index SET is_superseded = 1,
                   conflicting_with = ? WHERE memory_id = ?""",
                (conflicting_with, memory_id),
            )
            self._maybe_commit()
            return cur.rowcount > 0

    def prune_expired(self, days: int = 90) -> int:
        """★ 修复 L9：物理删除超过指定天数且 is_superseded 的记忆。

        Returns:
            删除的记录数
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with self._write_lock:
            self._ensure_conn_alive()
            cur = self._retry_db_op(
                self._write_conn.execute,
                """DELETE FROM memory_index
                   WHERE is_superseded = 1 AND stored_at < ?""",
                (cutoff,),
            )
            self._retry_db_op(self._write_conn.commit)
            self._pending_writes = 0
            deleted = cur.rowcount
            if deleted > 0:
                logger.info("UnifiedMemoryIndex: 物理清理过期记忆 %d 条", deleted)
            return deleted

    def count(self, wing: str = "") -> int:
        self._ensure_conn_alive()
        if wing:
            row = self._read_conn.execute(
                "SELECT COUNT(*) AS c FROM memory_index WHERE wing = ?", (wing,)
            ).fetchone()
        else:
            row = self._read_conn.execute(
                "SELECT COUNT(*) AS c FROM memory_index"
            ).fetchone()
        return int(row["c"]) if row else 0

    def flush(self) -> None:
        with self._write_lock:
            if self._write_conn and self._pending_writes > 0:
                self._retry_db_op(self._write_conn.commit)
                self._pending_writes = 0

    def close(self) -> None:
        with self._write_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.flush()
            except Exception as e:
                logger.warning("UnifiedMemoryIndex flush on close failed: %s", e)
            for conn_attr in ("_write_conn", "_read_conn"):
                conn = getattr(self, conn_attr, None)
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                setattr(self, conn_attr, None)
