import sqlite3
import json
import logging
import time
from pathlib import Path
from typing import Any
import threading

from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)

class AuditLogger:
    def __init__(self, governance_dir: Path, max_rows: int = 100000):
        self._db_path = governance_dir / "audit_log.db"
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._max_rows = max_rows
        self._ensure_table()

    def _ensure_table(self) -> None:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        migrator = SchemaMigrator(conn)
        # P1-13 修复：在 schema 中增加 actor 字段，并通过 migrations 为旧库升级。
        # 新库 create_sql 已含 actor 列，无需 ALTER TABLE；
        # 旧库（无 actor 列）通过 migrations 中的 ALTER TABLE 升级。
        # SQLite 不支持 ADD COLUMN IF NOT EXISTS，先用 PRAGMA 预检查避免 "duplicate column name" 噪声，
        # 并用 try/except 兜底应对并发竞态。
        create_sql = """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                operation TEXT NOT NULL,
                memory_id TEXT,
                details TEXT,
                result TEXT,
                instance_id TEXT,
                actor TEXT
            )
        """
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(audit_log)").fetchall()}
        try:
            if existing_cols and "actor" not in existing_cols:
                # 旧库升级：create_sql 因 IF NOT EXISTS 不重建表，migrations 添加 actor 列
                migrator.migrate(
                    table_name="audit_log",
                    create_sql=create_sql,
                    migrations=[
                        (1, "ALTER TABLE audit_log ADD COLUMN actor TEXT"),
                    ],
                )
            else:
                # 新库（create_sql 已含 actor）或已迁移库：仅执行 create_sql，不触发 ALTER
                migrator.migrate(
                    table_name="audit_log",
                    create_sql=create_sql,
                    migrations=[],
                )
                if existing_cols:
                    # actor 列已存在但版本未记录（如 _schema_versions 丢失），补登记版本
                    migrator.set_version("audit_log", 1)
        except sqlite3.OperationalError as e:
            # 兜底：并发场景下可能两个实例同时 ALTER TABLE，"duplicate column name" 视为成功
            if "duplicate column name" not in str(e):
                raise
            logger.debug("audit_log: actor 列已存在，跳过 ALTER TABLE 迁移")
            migrator.set_version("audit_log", 1)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_operation ON audit_log(operation)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_memory_id ON audit_log(memory_id)")
        # P1-11/P1-12 修复：审计日志只追加，禁止 DELETE/UPDATE，保障不可篡改性
        conn.execute("DROP TRIGGER IF EXISTS audit_log_no_delete")
        conn.execute("DROP TRIGGER IF EXISTS audit_log_no_update")
        conn.execute("""
            CREATE TRIGGER audit_log_no_delete
            BEFORE DELETE ON audit_log
            BEGIN
                SELECT RAISE(ABORT, 'audit_log is append-only (DELETE blocked)');
            END
        """)
        conn.execute("""
            CREATE TRIGGER audit_log_no_update
            BEFORE UPDATE ON audit_log
            BEGIN
                SELECT RAISE(ABORT, 'audit_log is append-only (UPDATE blocked)');
            END
        """)
        conn.commit()
        conn.close()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")

    def log(self, operation: str, memory_id: str | None = None, details: dict | None = None, result: str = "success", instance_id: str | None = None, actor: str | None = None) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO audit_log (timestamp, operation, memory_id, details, result, instance_id, actor) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (time.time(), operation, memory_id, json.dumps(details, ensure_ascii=False) if details else None, result, instance_id, actor),
                )
                self._conn.commit()
                self._rotate_if_needed()
            except Exception as e:
                logger.warning("Audit log write failed: %s", e)

    def _rotate_if_needed(self):
        try:
            count = self._conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            if count > self._max_rows:
                # P1-11 修复：DELETE 触发器会阻止普通 DELETE，轮转时先临时 DROP 触发器，
                # 完成清理后立即重建，保证轮转后审计日志仍然不可篡改。
                self._conn.execute("DROP TRIGGER IF EXISTS audit_log_no_delete")
                self._conn.execute("DELETE FROM audit_log WHERE rowid IN (SELECT rowid FROM audit_log ORDER BY rowid ASC LIMIT ?)", (count - self._max_rows,))
                self._conn.execute("""
                    CREATE TRIGGER audit_log_no_delete
                    BEFORE DELETE ON audit_log
                    BEGIN
                        SELECT RAISE(ABORT, 'audit_log is append-only (DELETE blocked)');
                    END
                """)
                self._conn.commit()
        except Exception as e:
            logger.warning("AuditLog _rotate_if_needed failed: %s", e)

    def query(self, operation: str | None = None, memory_id: str | None = None, from_time: float | None = None, to_time: float | None = None, limit: int = 100) -> list[dict]:
        conditions = []
        params: list[Any] = []
        if operation:
            conditions.append("operation = ?")
            params.append(operation)
        if memory_id:
            conditions.append("memory_id = ?")
            params.append(memory_id)
        if from_time:
            conditions.append("timestamp >= ?")
            params.append(from_time)
        if to_time:
            conditions.append("timestamp <= ?")
            params.append(to_time)
        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        with self._lock:
            cursor = self._conn.execute(
                f"SELECT id, timestamp, operation, memory_id, details, result, instance_id, actor FROM audit_log WHERE {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            )
            rows = cursor.fetchall()
        return [
            {"id": r[0], "timestamp": r[1], "operation": r[2], "memory_id": r[3], "details": json.loads(r[4]) if r[4] else None, "result": r[5], "instance_id": r[6], "actor": r[7] if len(r) > 7 else None}
            for r in rows
        ]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
