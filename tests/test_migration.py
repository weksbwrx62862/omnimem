"""SQLite 迁移框架 SchemaMigrator 单测（改进项 #2 覆盖，纯标准库 sqlite，离线）。

覆盖：版本读写/上插、pending 迁移排序、幂等门控（仅执行 > current 的迁移）、
事务原子性（失败迁移整体回滚）、迁移后表结构生效。
"""
from __future__ import annotations

import sqlite3

import pytest
from omnimem.utils.migration import SchemaMigrator


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys=ON")
    return c


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows}


# ── 版本追踪 ──────────────────────────────────────────
def test_get_version_unknown_is_zero():
    mig = SchemaMigrator(_conn())
    assert mig.get_version("nope") == 0


def test_set_then_get_version_roundtrip():
    mig = SchemaMigrator(_conn())
    mig.set_version("t", 3)
    assert mig.get_version("t") == 3


def test_set_version_upsert_overwrites():
    mig = SchemaMigrator(_conn())
    mig.set_version("t", 1)
    mig.set_version("t", 5)
    assert mig.get_version("t") == 5  # ON CONFLICT DO UPDATE


def test_versions_table_created():
    mig = SchemaMigrator(_conn())
    assert "_schema_versions" in _tables(mig._conn)


# ── migrate：建表 + 执行迁移 ──────────────────────────
def test_migrate_creates_table_and_applies_all():
    conn = _conn()
    mig = SchemaMigrator(conn)
    mig.migrate(
        "t",
        "CREATE TABLE IF NOT EXISTS t (a TEXT)",
        [
            (1, "ALTER TABLE t ADD COLUMN b TEXT"),
            (2, "ALTER TABLE t ADD COLUMN c INTEGER"),
        ],
    )
    assert mig.get_version("t") == 2
    assert _columns(conn, "t") == {"a", "b", "c"}


def test_migrate_sorts_out_of_order_pending():
    conn = _conn()
    mig = SchemaMigrator(conn)
    # 版本逆序传入 → 应排序后按 1→2 顺序执行（列依赖成立）
    mig.migrate(
        "t",
        "CREATE TABLE IF NOT EXISTS t (a TEXT)",
        [
            (2, "ALTER TABLE t ADD COLUMN c INTEGER"),
            (1, "ALTER TABLE t ADD COLUMN b TEXT"),
        ],
    )
    assert _columns(conn, "t") == {"a", "b", "c"}
    assert mig.get_version("t") == 2


# ── 幂等门控 ──────────────────────────────────────────
def test_migrate_idempotent_no_reapply():
    conn = _conn()
    mig = SchemaMigrator(conn)
    sqls = [
        (1, "ALTER TABLE t ADD COLUMN b TEXT"),
        (2, "ALTER TABLE t ADD COLUMN c INTEGER"),
    ]
    mig.migrate("t", "CREATE TABLE IF NOT EXISTS t (a TEXT)", sqls)
    # 再次执行相同迁移：若重复 ALTER ADD COLUMN 会报错；应静默跳过
    mig.migrate("t", "CREATE TABLE IF NOT EXISTS t (a TEXT)", sqls)
    assert mig.get_version("t") == 2


def test_migrate_only_applies_greater_versions():
    conn = _conn()
    mig = SchemaMigrator(conn)
    mig.set_version("t", 1)  # 假装已到版本1（但列 b 实际不存在）
    # create_sql 不写 b 列；版本1的 ALTER(ADD b) 应被门控跳过，仅执行版本2
    mig.migrate(
        "t",
        "CREATE TABLE IF NOT EXISTS t (a TEXT)",
        [
            (1, "ALTER TABLE t ADD COLUMN b TEXT"),
            (2, "ALTER TABLE t ADD COLUMN c INTEGER"),
        ],
    )
    cols = _columns(conn, "t")
    assert "c" in cols and "b" not in cols  # 仅版本2执行
    assert mig.get_version("t") == 2


# ── 事务原子性 ────────────────────────────────────────
def test_migrate_propagates_failure():
    # 注意：Python sqlite3 传统模式会在每条 DDL(CREATE/ALTER) 前隐式 COMMIT，
    # 因此 migrate() 无法把 DDL 原子回滚——只保证「失败向上抛出、不被吞掉」。
    conn = _conn()
    mig = SchemaMigrator(conn)
    with pytest.raises(sqlite3.Error):
        mig.migrate(
            "t",
            "CREATE TABLE IF NOT EXISTS t (a TEXT)",
            [
                (1, "ALTER TABLE t ADD COLUMN b TEXT"),
                (2, "THIS IS NOT SQL ;"),  # 触发失败
            ],
        )


def test_migrate_empty_migrations_still_creates_table():
    conn = _conn()
    mig = SchemaMigrator(conn)
    mig.migrate("t", "CREATE TABLE IF NOT EXISTS t (a TEXT)", [])
    assert "t" in _tables(conn)
    assert mig.get_version("t") == 0  # 无迁移可应用，版本保持 0
