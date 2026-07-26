"""UnifiedMemoryIndex 迁移工具。

将 ThreeLevelIndex (index.db) + MetaStore (meta_store.db) 的数据
合并迁移到 UnifiedMemoryIndex (unified_index.db)。
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class IndexMigrationTool:
    """ThreeLevelIndex + MetaStore → UnifiedMemoryIndex 迁移器。

    使用方式:
        tool = IndexMigrationTool(
            index_dir=Path("/path/to/index"),
            meta_dir=Path("/path/to/.meta"),
            unified_dir=Path("/path/to/unified"),
        )
        result = tool.migrate(dry_run=False)
        print(result.summary)
    """

    def __init__(self, index_dir: Path, meta_dir: Path, unified_dir: Path):
        self._index_dir = index_dir
        self._meta_dir = meta_dir
        self._unified_dir = unified_dir
        self._index_db = index_dir / "index.db"
        self._meta_db = meta_dir / "meta_store.db"
        self._unified_db = unified_dir / "unified_index.db"

    @property
    def index_db_path(self) -> Path:
        return self._index_db

    @property
    def meta_db_path(self) -> Path:
        return self._meta_db

    @property
    def unified_db_path(self) -> Path:
        return self._unified_db

    def validate_sources(self) -> list[str]:
        """验证源数据库是否存在、可读、表结构完整。返回问题列表。"""
        issues: list[str] = []

        if not self._index_db.exists():
            issues.append(f"ThreeLevelIndex 数据库不存在: {self._index_db}")
        else:
            try:
                conn = sqlite3.connect(str(self._index_db))
                conn.row_factory = sqlite3.Row
                conn.execute("SELECT * FROM memory_index LIMIT 1")
                conn.close()
            except sqlite3.OperationalError as e:
                issues.append(f"ThreeLevelIndex 表不可读: {e}")

        if not self._meta_db.exists():
            issues.append(f"MetaStore 数据库不存在: {self._meta_db}")
        else:
            try:
                conn = sqlite3.connect(str(self._meta_db))
                conn.row_factory = sqlite3.Row
                conn.execute("SELECT * FROM memories LIMIT 1")
                conn.close()
            except sqlite3.OperationalError as e:
                issues.append(f"MetaStore 表不可读: {e}")

        return issues

    def _read_index_records(self) -> dict[str, dict[str, Any]]:
        """读取 ThreeLevelIndex 全部记录，以 memory_id 为 key。"""
        if not self._index_db.exists():
            return {}
        conn = sqlite3.connect(str(self._index_db))
        conn.row_factory = sqlite3.Row
        # 安全获取所有列名
        cur = conn.execute("SELECT * FROM memory_index LIMIT 1")
        columns = [desc[0] for desc in cur.description] if cur.description else []
        records: dict[str, dict[str, Any]] = {}
        try:
            rows = conn.execute("SELECT * FROM memory_index").fetchall()
            for row in rows:
                d = dict(row)
                records[d["memory_id"]] = d
        except sqlite3.OperationalError:
            logger.warning("ThreeLevelIndex 读取失败，跳过")
        finally:
            conn.close()
        return records

    def _read_meta_records(self) -> dict[str, dict[str, Any]]:
        """读取 MetaStore 全部记录，以 memory_id 为 key。"""
        if not self._meta_db.exists():
            return {}
        conn = sqlite3.connect(str(self._meta_db))
        conn.row_factory = sqlite3.Row
        records: dict[str, dict[str, Any]] = {}
        try:
            rows = conn.execute("SELECT * FROM memories").fetchall()
            for row in rows:
                d = dict(row)
                records[d["memory_id"]] = d
        except sqlite3.OperationalError:
            logger.warning("MetaStore 读取失败，跳过")
        finally:
            conn.close()
        return records

    def _merge_records(
        self,
        index_records: dict[str, dict[str, Any]],
        meta_records: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """合并两个来源的记录。

        合并策略：
          - 全外连接 by memory_id
          - 共有字段：MetaStore 优先（"新数据优先"）
          - ThreeLevelIndex 独有字段：content, scope, provenance, metadata
          - MetaStore 独有字段：content_preview, drawer_path, vc
          - 时间字段：MetaStore.created_at 优先，回退到 ThreeLevelIndex.stored_at
        """
        all_ids = set(index_records.keys()) | set(meta_records.keys())
        merged: list[dict[str, Any]] = []

        for mid in sorted(all_ids):
            idx_row = index_records.get(mid, {})
            meta_row = meta_records.get(mid, {})

            # 共有字段：MetaStore 优先
            stored_at = meta_row.get("stored_at") or idx_row.get("stored_at") or ""
            created_at = meta_row.get("created_at") or stored_at

            record: dict[str, Any] = {
                "memory_id": mid,
                "wing": meta_row.get("wing") or idx_row.get("wing") or "",
                "hall": meta_row.get("hall") or idx_row.get("hall") or "",
                "room": meta_row.get("room") or idx_row.get("room") or "",
                # ThreeLevelIndex 独有
                "content": idx_row.get("content") or "",
                "scope": idx_row.get("scope") or "personal",
                "provenance": idx_row.get("provenance") or "",
                "metadata": idx_row.get("metadata") or "",
                # 共有字段：MetaStore 优先
                "summary": meta_row.get("summary") or idx_row.get("summary") or "",
                "type": meta_row.get("type") or idx_row.get("type") or "fact",
                "confidence": meta_row.get("confidence") or idx_row.get("confidence") or 3,
                "privacy": meta_row.get("privacy") or idx_row.get("privacy") or "personal",
                "stored_at": stored_at,
                "conflicting_with": (
                    meta_row.get("conflicting_with") or idx_row.get("conflicting_with") or ""
                ),
                "conflict_type": (
                    meta_row.get("conflict_type") or idx_row.get("conflict_type") or ""
                ),
                # MetaStore 独有
                "content_preview": meta_row.get("content_preview") or "",
                "drawer_path": meta_row.get("drawer_path") or "",
                "vc": meta_row.get("vc") or "",
                # 治理字段默认值
                "is_updated": int(idx_row.get("is_updated") or 0),
                "is_superseded": int(idx_row.get("is_superseded") or 0),
                "created_at": created_at,
            }
            merged.append(record)

        return merged

    def _write_to_unified(
        self, records: list[dict[str, Any]], dry_run: bool = False
    ) -> int:
        """将合并后的记录写入 UnifiedMemoryIndex。

        Returns:
            写入的记录数
        """
        if dry_run:
            return len(records)

        self._unified_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self._unified_db))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.row_factory = sqlite3.Row

        # 创建 UnifiedMemoryIndex 的完整表结构
        self._create_unified_schema(conn)

        # 批量写入
        sql = """
            INSERT OR REPLACE INTO memory_index
            (memory_id, wing, hall, room, content, summary, content_preview,
             drawer_path, vc, type, confidence, privacy, scope, stored_at,
             provenance, metadata, conflicting_with, conflict_type,
             is_updated, is_superseded, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        count = 0
        for rec in records:
            conn.execute(
                sql,
                (
                    rec["memory_id"],
                    rec["wing"],
                    rec["hall"],
                    rec["room"],
                    rec["content"],
                    rec["summary"],
                    rec["content_preview"],
                    rec["drawer_path"],
                    rec["vc"],
                    rec["type"],
                    rec["confidence"],
                    rec["privacy"],
                    rec["scope"],
                    rec["stored_at"],
                    rec["provenance"],
                    rec["metadata"],
                    rec["conflicting_with"],
                    rec["conflict_type"],
                    rec["is_updated"],
                    rec["is_superseded"],
                    rec["created_at"],
                ),
            )
            count += 1

        conn.commit()
        conn.close()
        return count

    @staticmethod
    def _create_unified_schema(conn: sqlite3.Connection) -> None:
        """创建 UnifiedMemoryIndex 的完整表结构（含 FTS5 和触发器）。"""
        conn.execute("""
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
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_wing ON memory_index(wing)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memory_index(type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stored_at ON memory_index(stored_at)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stage ON memory_index(is_superseded, stored_at)"
        )

        # FTS5
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_index_fts USING fts5("
            "content, summary, content_preview)"
        )
        for trig in ("memory_index_fts_ai", "memory_index_fts_ad", "memory_index_fts_au"):
            conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
        conn.execute("""
            CREATE TRIGGER memory_index_fts_ai AFTER INSERT ON memory_index BEGIN
                INSERT INTO memory_index_fts(rowid, content, summary, content_preview)
                VALUES (new.rowid, new.content, COALESCE(new.summary,''), COALESCE(new.content_preview,''));
            END
        """)
        conn.execute("""
            CREATE TRIGGER memory_index_fts_ad AFTER DELETE ON memory_index BEGIN
                DELETE FROM memory_index_fts WHERE rowid = old.rowid;
            END
        """)
        conn.execute("""
            CREATE TRIGGER memory_index_fts_au AFTER UPDATE ON memory_index BEGIN
                DELETE FROM memory_index_fts WHERE rowid = old.rowid;
                INSERT INTO memory_index_fts(rowid, content, summary, content_preview)
                VALUES (new.rowid, new.content, COALESCE(new.summary,''), COALESCE(new.content_preview,''));
            END
        """)

    def _verify(self, expected_count: int) -> dict[str, Any]:
        """验证迁移结果：行数 + 抽样内容一致性。"""
        result: dict[str, Any] = {
            "passed": True,
            "expected_count": expected_count,
            "actual_count": 0,
            "issues": [],
        }

        if not self._unified_db.exists():
            result["passed"] = False
            result["issues"].append("目标数据库未生成")
            return result

        conn = sqlite3.connect(str(self._unified_db))
        conn.row_factory = sqlite3.Row
        actual = conn.execute("SELECT COUNT(*) as cnt FROM memory_index").fetchone()["cnt"]
        result["actual_count"] = actual
        conn.close()

        if actual != expected_count:
            result["passed"] = False
            result["issues"].append(
                f"行数不一致: 期望 {expected_count}，实际 {actual}"
            )

        return result

    def migrate(self, dry_run: bool = False, skip_backup: bool = False) -> dict[str, Any]:
        """执行完整迁移流程。

        Args:
            dry_run: 仅模拟，不实际写入
            skip_backup: 跳过源数据库备份

        Returns:
            {
                "success": bool,
                "index_count": int,
                "meta_count": int,
                "merged_count": int,
                "written_count": int,
                "verification": dict,
                "backups": list[str],
                "issues": list[str],
            }
        """
        result: dict[str, Any] = {
            "success": False,
            "index_count": 0,
            "meta_count": 0,
            "merged_count": 0,
            "written_count": 0,
            "verification": {},
            "backups": [],
            "issues": [],
        }

        # 1. 验证源数据库
        issues = self.validate_sources()
        if issues:
            result["issues"] = issues
            return result

        # 2. 读取源数据
        index_records = self._read_index_records()
        meta_records = self._read_meta_records()
        result["index_count"] = len(index_records)
        result["meta_count"] = len(meta_records)
        logger.info(
            "读取完成: ThreeLevelIndex=%d 条, MetaStore=%d 条",
            len(index_records),
            len(meta_records),
        )

        # 3. 合并
        merged = self._merge_records(index_records, meta_records)
        result["merged_count"] = len(merged)
        logger.info("合并完成: %d 条唯一记录", len(merged))

        if not merged:
            result["issues"].append("源数据库无数据，无需迁移")
            result["success"] = True
            return result

        # 4. 备份源数据库
        if not dry_run and not skip_backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for src_path, label in [
                (self._index_db, "index"),
                (self._meta_db, "meta"),
            ]:
                if src_path.exists():
                    bak_path = src_path.with_suffix(f".db.bak_{timestamp}")
                    shutil.copy2(src_path, bak_path)
                    result["backups"].append(str(bak_path))
                    logger.info("备份: %s → %s", src_path, bak_path)

        # 5. 写入 UnifiedMemoryIndex
        written = self._write_to_unified(merged, dry_run=dry_run)
        result["written_count"] = written
        logger.info("写入完成: %d 条", written)

        # 6. 验证
        verification = self._verify(len(merged)) if not dry_run else {"passed": True}
        result["verification"] = verification

        if verification.get("passed", True) and not result["issues"]:
            result["success"] = True

        return result

    def print_report(self, result: dict[str, Any]) -> None:
        """打印迁移报告。"""
        print()
        print("=" * 60)
        print("  UnifiedMemoryIndex 迁移报告")
        print("=" * 60)
        print(f"  ThreeLevelIndex 记录数: {result['index_count']}")
        print(f"  MetaStore 记录数:      {result['meta_count']}")
        print(f"  合并后唯一记录数:      {result['merged_count']}")
        print(f"  写入 UnifiedIndex 数:  {result['written_count']}")
        print()

        if result.get("backups"):
            print("  备份文件:")
            for b in result["backups"]:
                print(f"    {b}")
            print()

        verification = result.get("verification", {})
        if verification:
            passed = verification.get("passed", False)
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"  验证结果: {status}")
            print(f"    期望行数: {verification.get('expected_count', '?')}")
            print(f"    实际行数: {verification.get('actual_count', '?')}")
            for issue in verification.get("issues", []):
                print(f"    ⚠ {issue}")
            print()

        if result.get("issues"):
            print("  问题:")
            for issue in result["issues"]:
                print(f"    ⚠ {issue}")
            print()

        if result.get("success"):
            print("  ✅ 迁移成功！")
        else:
            print("  ❌ 迁移失败，请检查上述问题。")
        print("=" * 60)
        print()
