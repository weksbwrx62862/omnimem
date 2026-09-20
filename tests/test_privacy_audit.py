"""governance/privacy.py + governance/audit_log.py 单元测试。"""

from __future__ import annotations

import time

import pytest
from omnimem.governance.audit_log import AuditLogger
from omnimem.governance.privacy import PrivacyManager

# ─── PrivacyManager ──────────────────────────────────────


@pytest.fixture
def pm():
    return PrivacyManager(session_id="s-priv")


def test_privacy_set_and_get_override(pm):
    assert pm.get("m1") == "personal"  # 默认
    pm.set("m1", "public")
    assert pm.get("m1") == "public"
    # 未知级别被忽略
    pm.set("m1", "bogus")
    assert pm.get("m1") == "public"


def test_privacy_custom_default_level():
    pm = PrivacyManager(default_level="team", session_id="x")
    assert pm.get("nope") == "team"


def test_privacy_filter_secret_kept_but_marked(pm):
    results = [
        {"memory_id": "s", "privacy": "secret", "content": "敏感数据"},
        {"memory_id": "p", "privacy": "personal", "content": "普通数据"},
    ]
    out = pm.filter(results, max_privacy="personal")
    by_id = {r["memory_id"]: r for r in out}
    assert by_id["s"]["_encrypted"] is True
    assert "加密" in by_id["s"]["content"]
    assert by_id["p"]["content"] == "普通数据"


def test_privacy_filter_drops_above_max_level(pm):
    results = [
        {"memory_id": "pub", "privacy": "public"},
        {"memory_id": "team", "privacy": "team"},
        {"memory_id": "pers", "privacy": "personal"},
    ]
    out = pm.filter(results, max_privacy="team")
    ids = {r["memory_id"] for r in out}
    # personal 高于 team → 被过滤
    assert ids == {"pub", "team"}


def test_privacy_filter_missing_level_uses_default(pm):
    out = pm.filter([{"memory_id": "x"}], max_privacy="public")
    # 缺 privacy → 默认 personal，比 public 高 → 过滤
    assert out == []


def test_privacy_filter_unknown_level_falls_back_to_personal_order(pm):
    # 未知 privacy 字符串 → _PRIVACY_ORDER.get 默认 2 = personal
    out = pm.filter([{"memory_id": "y", "privacy": "bogus"}], max_privacy="public")
    assert out == []
    out2 = pm.filter([{"memory_id": "y", "privacy": "bogus"}], max_privacy="personal")
    assert len(out2) == 1


def test_privacy_bind_store_persists_set():
    """绑定 store 后 set 应同步 update_privacy。"""

    class _Store:
        def __init__(self):
            self.updates = []

        def update_privacy(self, mid, level, new_wing=None):
            self.updates.append((mid, level, new_wing))

        def get(self, mid):
            return None

    store = _Store()
    pm = PrivacyManager(session_id="x")
    pm.bind_store(store)
    pm.set("m", "team", new_wing="w")
    assert store.updates == [("m", "team", "w")]


def test_privacy_get_falls_back_to_store_on_cache_miss():
    class _Store:
        def get(self, mid):
            return {"privacy": "public"}

    pm = PrivacyManager(session_id="x")
    pm.bind_store(_Store())
    assert pm.get("unknown") == "public"
    # 命中后应写入覆盖表，第二次不再访问 store
    assert pm._overrides["unknown"] == "public"


# ─── AuditLogger ─────────────────────────────────────────


@pytest.fixture
def audit(tmp_path):
    a = AuditLogger(tmp_path, max_rows=5)
    try:
        yield a
    finally:
        a.close()


def test_audit_log_and_query_roundtrip(audit):
    audit.log("memorize", memory_id="m-1", details={"k": "v"}, result="success", actor="alice")
    rows = audit.query(memory_id="m-1")
    assert len(rows) == 1
    r = rows[0]
    assert r["operation"] == "memorize"
    assert r["details"] == {"k": "v"}
    assert r["result"] == "success"
    assert r["actor"] == "alice"
    assert isinstance(r["timestamp"], float)


def test_audit_query_filters_by_operation_and_time(audit):
    t0 = time.time()
    audit.log("memorize", memory_id="m-1")
    audit.log("recall", memory_id="m-2")
    time.sleep(0.01)
    audit.log("reflect", memory_id="m-3")

    only_mem = audit.query(operation="memorize")
    assert [r["operation"] for r in only_mem] == ["memorize"]

    later = audit.query(from_time=t0 + 0.005)
    assert all(r["operation"] != "memorize" and r["operation"] != "recall" for r in later)


def test_audit_query_limit_and_order_desc(audit):
    for i in range(4):
        audit.log(f"op{i}", memory_id=f"m{i}")
    rows = audit.query(limit=2)
    assert len(rows) == 2
    # 按 timestamp 倒序 → 最后写入的 op3 应在最前
    assert rows[0]["operation"] == "op3"


def test_audit_details_none_stored_as_null(audit):
    audit.log("noop", memory_id="m-none", details=None)
    rows = audit.query(memory_id="m-none")
    assert rows[0]["details"] is None


def test_audit_rotate_prunes_oldest_rows(audit):
    # max_rows=5，插入 8 条 → 轮转后应剩最后 5 条（按 rowid 递增删除最老）
    for i in range(8):
        audit.log(f"op{i}", memory_id=f"m{i}")
    rows = audit.query(limit=100)
    assert len(rows) == 5
    ops = {r["operation"] for r in rows}
    # 前 3 条 op0/op1/op2 被清理
    assert "op0" not in ops
    assert "op7" in ops


def test_audit_append_only_blocks_delete_and_update(audit):
    import sqlite3

    audit.log("op", memory_id="m-locked")
    with pytest.raises(sqlite3.IntegrityError):
        audit._conn.execute("DELETE FROM audit_log WHERE memory_id = 'm-locked'")
    with pytest.raises(sqlite3.IntegrityError):
        audit._conn.execute("UPDATE audit_log SET result = 'x' WHERE memory_id = 'm-locked'")


def test_audit_query_combines_all_filter_dimensions(audit):
    now = time.time()
    audit.log("memorize", memory_id="m-A", actor="bob")
    audit.log("memorize", memory_id="m-B", actor="bob")
    audit.log("recall", memory_id="m-A", actor="bob")
    rows = audit.query(operation="memorize", memory_id="m-A", from_time=now - 1, to_time=now + 10, limit=10)
    assert len(rows) == 1
    assert rows[0]["memory_id"] == "m-A"
    assert rows[0]["operation"] == "memorize"
