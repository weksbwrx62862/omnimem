"""ThreeLevelIndex — 三层索引 L0/L1/L2。

参考 OpenViking 的三层索引设计：
  - L0 (目录索引): Wing/Hall/Room 结构索引，最小化加载
  - L1 (摘要索引): Closet 摘要，中等粒度
  - L2 (全文索引): Drawer 原文，最大精度

索引存储在 SQLite 中，支持快速查找和范围查询。
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)

# ★ P1-3：重试口径对齐 governance/forgetting_stages（那里已验证 5 次线性退避够用），
#   3 次在并发写高峰期不足，第 4 个等待者就直接抛错给上层。
_DB_RETRY_COUNT = 5
_DB_RETRY_DELAY = 0.05


def _is_retryable(exc: BaseException) -> bool:
    """判断一次 SQLite 失败是否值得重试。

    只认 "locked" 是漏的：并发写还常见 "database is busy"，而连接跨线程复用时
    sqlite3 抛的是 InterfaceError("bad parameter or other API misuse") ——
    原实现里它既不是 OperationalError 也不在重试名单，直接掀掉整条写入路径。
    """
    if isinstance(exc, sqlite3.InterfaceError):
        return True
    if isinstance(exc, sqlite3.OperationalError):
        msg = str(exc).lower()
        return "locked" in msg or "busy" in msg
    return False


def _retry_db_op(fn, *args, **kwargs):
    """★ P2修复Minor-4：SQLite 操作重试，解决并发锁超时问题。"""
    last_error: BaseException | None = None
    for attempt in range(_DB_RETRY_COUNT):
        try:
            return fn(*args, **kwargs)
        except (sqlite3.OperationalError, sqlite3.InterfaceError) as e:
            if not _is_retryable(e):
                raise
            last_error = e
            if attempt < _DB_RETRY_COUNT - 1:
                time.sleep(_DB_RETRY_DELAY * (attempt + 1))
    assert last_error is not None
    raise last_error


class ThreeLevelIndex:
    """三层索引 L0/L1/L2。

    批量提交优化：add() 不立即 commit，攒到阈值或显式 flush() 时统一提交，
    减少磁盘 fsync 次数。
    """

    # ★ P1-3 追加：每 1 次写入就 commit。批处理阈值 >1 与「每线程一条连接」不兼容 ——
    #   未提交的事务对**其它线程的连接不可见**（WAL 只隔离已提交数据），于是 A 线程
    #   写完、B 线程立刻召回放不出来，线上表现是「写入成功但检索偶发漏一条」。
    #   WAL + synchronous=NORMAL 下 commit 不 fsync，代价可忽略。
    _BATCH_THRESHOLD = 1  # 每次写入立即 commit

    def __init__(self, index_dir: Path):
        self._index_dir = index_dir
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._index_dir / "index.db"
        # ★ P1-3：一进程一条共享连接（即便 check_same_thread=False）在并发写下会
        #   触发 InterfaceError/SQLITE_BUSY，且长读会挡住写。改为每线程一条连接，
        #   WAL 模式下读写并发不互斥；写锁仍由 SQLite 自身串行化保证。
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()
        self._closed = False
        self._init_db()

    @property
    def db_path(self) -> Path:
        """公开访问数据库文件路径。"""
        return self._db_path

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        with self._connections_lock:
            self._connections.append(conn)
        return conn

    @property
    def _conn(self) -> sqlite3.Connection | None:
        """当前线程的数据库连接（惰性创建）；close() 之后为 None。"""
        if self._closed:
            return None
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        """初始化 SQLite 数据库。"""
        conn = self._conn
        assert conn is not None
        migrator = SchemaMigrator(conn)
        migrator.migrate(
            table_name="memory_index",
            create_sql="""
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
                    metadata TEXT,
                    project TEXT DEFAULT ''
                )
            """,
            migrations=[],
        )

        # ★ 旧 schema 迁移：补全缺失的列
        _migrate_columns = [
            ("wing", "TEXT NOT NULL DEFAULT ''"),
            ("hall", "TEXT NOT NULL DEFAULT ''"),
            ("room", "TEXT NOT NULL DEFAULT ''"),
            ("summary", "TEXT"),
            ("confidence", "INTEGER DEFAULT 3"),
            ("privacy", "TEXT DEFAULT 'personal'"),
            ("scope", "TEXT DEFAULT 'personal'"),
            ("stored_at", "TEXT"),
            ("provenance", "TEXT"),
            ("metadata", "TEXT"),
            ("conflicting_with", "TEXT"),
            ("conflict_type", "TEXT"),
            ("is_updated", "INTEGER DEFAULT 0"),
            ("is_superseded", "INTEGER DEFAULT 0"),
            ("project", "TEXT DEFAULT ''"),
        ]
        for col_name, col_def in _migrate_columns:
            try:
                self._conn.execute(f"SELECT {col_name} FROM memory_index LIMIT 1")
            except sqlite3.OperationalError:
                # 列名来自硬编码常量，非用户输入，安全使用 f-string
                self._conn.execute(f"ALTER TABLE memory_index ADD COLUMN {col_name} {col_def}")
                logger.info("Index migrated: added %s column", col_name)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_wing ON memory_index(wing)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_type ON memory_index(type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_stored_at ON memory_index(stored_at)
        """)
        # FTS5 fulltext index (replaces LIKE '%keyword%' full-scan in search_l2)
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_index_fts USING fts5(content)
        """)
        # ★ 修复根因 A:改用 DELETE FROM 替代 FTS5 'delete' 特殊命令
        #   'delete' 命令在 contentful FTS5 表上不可用(仅 external content/contentless 表可用)
        #   用标准 DELETE FROM 替代,且先 DROP 旧触发器(IF NOT EXISTS 不会替换已存在的)
        self._conn.execute("DROP TRIGGER IF EXISTS memory_index_fts_ai")
        self._conn.execute("DROP TRIGGER IF EXISTS memory_index_fts_ad")
        self._conn.execute("DROP TRIGGER IF EXISTS memory_index_fts_au")
        self._conn.execute("""
            CREATE TRIGGER memory_index_fts_ai AFTER INSERT ON memory_index BEGIN
                INSERT INTO memory_index_fts(rowid, content) VALUES (new.rowid, new.content);
            END
        """)
        self._conn.execute("""
            CREATE TRIGGER memory_index_fts_ad AFTER DELETE ON memory_index BEGIN
                DELETE FROM memory_index_fts WHERE rowid = old.rowid;
            END
        """)
        self._conn.execute("""
            CREATE TRIGGER memory_index_fts_au AFTER UPDATE ON memory_index BEGIN
                DELETE FROM memory_index_fts WHERE rowid = old.rowid;
                INSERT INTO memory_index_fts(rowid, content) VALUES (new.rowid, new.content);
            END
        """)

        self._conn.commit()

    def _maybe_commit(self) -> None:
        """检查本线程待写入数是否达到阈值，达到则提交。

        计数必须挂在线程本地：每条连接只提交自己的事务，共享计数器会让
        A 线程的 commit 阈值被 B 线程的写入消耗掉，A 的行迟迟不落盘。
        """
        conn = self._conn
        assert conn is not None
        pending = getattr(self._local, "pending_writes", 0) + 1
        if pending >= self._BATCH_THRESHOLD:
            try:
                _retry_db_op(conn.commit)
            except Exception:
                # 半批持久化比丢批更糟：失败后必须回滚，否则阈值计数清零了，
                # 上一批里没提交成功的行会永远悬在一个不再被提交的事务里。
                with contextlib.suppress(Exception):
                    conn.rollback()
                raise
            pending = 0
        self._local.pending_writes = pending

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
        project: str = "",
    ) -> None:
        """添加或更新一条索引记录（失败向上抛出，不静默吞）。

        ★ P1-3 两处关键改动：
          1. ``INSERT OR REPLACE`` 是「先删后插」，会丢掉本方法没有传列的
             ``conflicting_with`` / ``conflict_type`` / ``is_updated`` / ``is_superseded``，
             并换掉 rowid（连带 FTS 触发器重写）。改为 ``ON CONFLICT DO UPDATE``
             只覆盖显式给出的列，其余保持原值。
          2. 原来 ``except`` 只打一条 warning 就当成功。写抽屉成功、写索引失败的
             记忆就此变成「有 drawer 无 index 行」的不可召回数据（线上 2 596 条），
             必须让 saga 看见异常并走补偿。
        """
        conn = self._conn
        assert conn is not None
        if not stored_at:
            stored_at = datetime.now().isoformat()
        try:
            _retry_db_op(
                conn.execute,
                """INSERT INTO memory_index
                   (memory_id, wing, hall, room, content, summary, type,
                    confidence, privacy, scope, stored_at, provenance, metadata, project)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(memory_id) DO UPDATE SET
                    wing=excluded.wing, hall=excluded.hall, room=excluded.room,
                    content=excluded.content, summary=excluded.summary, type=excluded.type,
                    confidence=excluded.confidence, privacy=excluded.privacy,
                    scope=excluded.scope, stored_at=excluded.stored_at,
                    provenance=excluded.provenance, metadata=excluded.metadata,
                    project=excluded.project""",
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
                    project,
                ),
            )
            self._maybe_commit()
        except Exception as e:
            logger.error("Index add failed for %s: %s", memory_id, e)
            raise

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """根据 ID 获取索引记录。"""
        assert self._conn is not None
        try:
            row = self._conn.execute(
                "SELECT * FROM memory_index WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row:
                return self._row_to_dict(row)
        except Exception as e:
            logger.warning("Index get failed for %s: %s", memory_id, e)
        return None

    def delete(self, memory_id: str) -> bool:
        """从索引中删除记录。"""
        assert self._conn is not None
        try:
            self._conn.execute(
                "DELETE FROM memory_index WHERE memory_id = ?",
                (memory_id,),
            )
            self._maybe_commit()
            return True
        except Exception as e:
            logger.warning("Index delete failed for %s: %s", memory_id, e)
            return False

    def search_l0(self, wing: str = "", hall: str = "") -> list[str]:
        """L0 目录索引：返回匹配的 Room 列表。"""
        assert self._conn is not None
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
            logger.warning("L0 search failed: %s", e)
            raise

    def search_by_directory(
        self,
        wing: str = "",
        hall: str = "",
        room: str = "",
    ) -> list[dict[str, Any]]:
        """按目录结构查询索引条目。

        内化 OpenViking 的目录定位能力：
        通过 Wing/Hall/Room 三级目录缩小搜索空间，
        返回目录内所有条目的 memory_id 和摘要。

        Args:
            wing: Wing 名称（personal/team/public）
            hall: Hall 名称（facts/preferences/...）
            room: Room 名称（话题）

        Returns:
            匹配的索引条目列表
        """
        assert self._conn is not None
        query = "SELECT memory_id, wing, hall, room, summary, type, confidence FROM memory_index WHERE 1=1"
        params = []
        if wing:
            query += " AND wing = ?"
            params.append(wing)
        if hall:
            query += " AND hall = ?"
            params.append(hall)
        if room:
            query += " AND room = ?"
            params.append(room)
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
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning("Directory search failed: %s", e)
            raise

    def search_l1(self, wing: str = "", type: str = "", limit: int = 50) -> list[dict[str, Any]]:
        """L1 摘要索引：返回摘要记录（含 content 用于 warm_up）。"""
        assert self._conn is not None
        query = "SELECT memory_id, wing, hall, room, summary, type, confidence, privacy, stored_at, content, conflicting_with, conflict_type, project FROM memory_index WHERE is_superseded=0"
        params = []
        if wing:
            query += " AND wing = ?"
            params.append(wing)
        if type:
            query += " AND type = ?"
            params.append(type)
        query += " ORDER BY stored_at DESC LIMIT ?"
        params.append(str(limit))
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
                    "conflicting_with": r[10] if len(r) > 10 else "",
                    "conflict_type": r[11] if len(r) > 11 else "",
                    "project": r[12] if len(r) > 12 else "",
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning("L1 search failed: %s", e)
            raise

    def search_l2(
        self,
        keyword: str = "",
        wing: str = "",
        type: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """L2 全文索引：返回完整记录（FTS5 优先，LIKE 回退）。"""
        assert self._conn is not None
        use_fts5 = False
        fts_keyword = ""
        if keyword:
            # ★ 修复根因 B:FTS5 默认 unicode61 分词器把中文当单个 token,
            #   双引号短语查询无法匹配短关键字,中文强制走 LIKE 回退
            has_non_ascii = any(ord(c) > 127 for c in keyword) if keyword else False
            if has_non_ascii:
                use_fts5 = False
            else:
                # 检测 FTS5 表是否可用
                try:
                    self._conn.execute(
                        "SELECT 1 FROM memory_index_fts LIMIT 0"
                    )
                    use_fts5 = True
                except sqlite3.OperationalError:
                    pass

            if use_fts5:
                # FTS5: 转义特殊字符并构造 MATCH 查询
                fts_keyword = self._escape_fts5(keyword)
                base_query = """
                    SELECT memory_index.* FROM memory_index
                    JOIN memory_index_fts ON memory_index.rowid = memory_index_fts.rowid
                    WHERE memory_index_fts MATCH ?
                """
                params: list[Any] = [fts_keyword]
                if wing:
                    base_query += " AND memory_index.wing = ?"
                    params.append(wing)
                if type:
                    base_query += " AND memory_index.type = ?"
                    params.append(type)
                base_query += " ORDER BY memory_index.stored_at DESC LIMIT ?"
                params.append(limit)
            else:
                # LIKE 回退（旧 schema 或 FTS5 不可用）
                escaped = keyword.replace("%", "\\%").replace("_", "\\_")
                base_query = (
                    "SELECT * FROM memory_index WHERE content LIKE ? ESCAPE '\\'"
                )
                params = [f"%{escaped}%"]
                if wing:
                    base_query += " AND wing = ?"
                    params.append(wing)
                if type:
                    base_query += " AND type = ?"
                    params.append(type)
                base_query += " ORDER BY stored_at DESC LIMIT ?"
                params.append(limit)
        else:
            base_query = "SELECT * FROM memory_index WHERE 1=1"
            params = []
            if wing:
                base_query += " AND wing = ?"
                params.append(wing)
            if type:
                base_query += " AND type = ?"
                params.append(type)
            base_query += " ORDER BY stored_at DESC LIMIT ?"
            params.append(limit)

        try:
            rows = self._conn.execute(base_query, params).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.warning("L2 search failed: %s", e)
            raise

    @staticmethod
    def _escape_fts5(keyword: str) -> str:
        """转义 FTS5 特殊字符，构造安全的 MATCH 查询。

        FTS5 中需要转义的特殊字符：*  \"  -  (  ) 以及前后缀引号。
        简单关键字直接包裹在双引号中作为短语查询。
        """
        # 简单处理：用双引号包裹作为短语匹配
        # 先移除已有的双引号，再包裹
        clean = keyword.replace('"', '')
        return f'"{clean}"'

    def search_all_for_retrieval(self, limit: int = 1000) -> list[dict[str, Any]]:
        """获取所有记录（用于检索引擎全量索引）。"""
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                "SELECT * FROM memory_index ORDER BY stored_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.warning("Full index scan failed: %s", e)
            raise

    def update_privacy(self, memory_id: str, privacy: str) -> bool:
        """更新隐私级别。"""
        assert self._conn is not None
        try:
            self._conn.execute(
                "UPDATE memory_index SET privacy = ? WHERE memory_id = ?",
                (privacy, memory_id),
            )
            self._maybe_commit()
            return True
        except Exception as e:
            logger.warning("Privacy update failed: %s", e)
            return False

    def update_field(self, memory_id: str, immediate: bool = False, **fields: Any) -> bool:
        """更新索引中的指定字段。

        Args:
            memory_id: 记忆 ID
            immediate: 为 True 时直接 commit 而非走 _maybe_commit 批处理，
                       适用于 governance 等需要跨组件一致性的场景
        """
        conn = self._conn
        assert conn is not None
        if not fields:
            return False
        try:
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            values = list(fields.values()) + [memory_id]
            _retry_db_op(
                conn.execute,
                f"UPDATE memory_index SET {set_clause} WHERE memory_id = ?",
                values,
            )
            if immediate:
                _retry_db_op(conn.commit)
                self._local.pending_writes = 0
            else:
                self._maybe_commit()
            return True
        except Exception as e:
            logger.warning("Field update failed: %s", e)
            return False

    def remove(self, memory_id: str) -> bool:
        """删除索引记录。"""
        assert self._conn is not None
        try:
            self._conn.execute(
                "DELETE FROM memory_index WHERE memory_id = ?",
                (memory_id,),
            )
            self._maybe_commit()
            return True
        except Exception as e:
            logger.warning("Index remove failed: %s", e)
            return False

    def close(self) -> None:
        """提交并关闭本进程内所有线程的数据库连接。"""
        self.flush()
        self._closed = True
        with self._connections_lock:
            conns, self._connections = self._connections, []
        for conn in conns:
            try:
                conn.close()
            except Exception:
                logger.debug("Index 连接关闭失败", exc_info=True)

    def flush(self) -> None:
        """显式提交所有线程连接的待写入。

        每线程一条连接后，只提交调用线程自己那条会把别的线程的半批留在未提交
        事务里；线程一退出，那半批整体回滚 —— 表现为「写成功却查不到」。
        """
        with self._connections_lock:
            conns = list(self._connections)
        for conn in conns:
            try:
                _retry_db_op(conn.commit)
            except Exception as e:
                # 其他线程正握着未完成语句时会拒绝提交，留给它的下一次写入。
                logger.debug("Index flush 跳过一条连接: %s", e)
                continue
            if conn is getattr(self._local, "conn", None):
                self._local.pending_writes = 0

    def _row_to_dict(self, row: tuple[Any, ...]) -> dict[str, Any]:
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
            "conflicting_with",
            "conflict_type",
        ]
        result = {}
        for i, key in enumerate(keys):
            if i < len(row):
                val = row[i]
                if key == "provenance" and val:
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        val = json.loads(val)
                if key == "metadata" and val:
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        val = json.loads(val)
                result[key] = val
        return result
