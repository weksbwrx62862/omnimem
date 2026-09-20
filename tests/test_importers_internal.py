"""改进项 #8 导入器内部单元测（纯函数、离线）。

覆盖 test_importers.py 未触及的底层原语与来源适配器的字段优先级/兜底分支。
"""
from __future__ import annotations

from datetime import datetime, timezone

from omnimem.importers import _base, sources

# ─── _base 原语 ────────────────────────────────────────

def test_coerce_text_scalar_and_container():
    assert _base._coerce_text(None) == ""
    assert _base._coerce_text("  hi \n") == "hi"
    assert _base._coerce_text(42) == "42"
    assert _base._coerce_text(True) == "True"
    # dict 优先取正文键
    assert _base._coerce_text({"content": "inner"}) == "inner"
    # dict 无正文键 -> json 序列化
    assert _base._coerce_text({"a": 1}) == '{"a": 1}'
    # list 连接非空片段
    assert _base._coerce_text(["a", None, "b"]) == "a b"
    # 嵌套 list 中的 dict 也取正文
    assert _base._coerce_text([{"memory": "x"}, "y"]) == "x y"


def test_first_text_respects_precedence():
    mapping = {"memory": "second", "content": "first"}
    assert _base._first_text(mapping, _base._CONTENT_KEYS) == "first"


def test_flatten_tags_variants():
    assert _base._flatten_tags([]) == []
    assert _base._flatten_tags("solo") == ["solo"]
    tags = _base._flatten_tags({"k": "v", "n": 3})
    assert "k" in tags and "k:v" in tags and "n:3" in tags
    lst = _base._flatten_tags([{"a": "1"}, "b"])
    assert "1" in lst and "b" in lst


def test_normalize_time_branches():
    # 毫秒 epoch
    ms = _base._normalize_time(1_700_000_000_000)
    assert datetime.fromisoformat(ms).year == 2023
    # 秒 epoch
    sec = _base._normalize_time(1_700_000_000)
    assert datetime.fromisoformat(sec).tzinfo is not None
    # ISO 带 Z -> UTC
    iso = _base._normalize_time("2024-05-01T00:00:00Z")
    assert iso.startswith("2024-05-01")
    # 非法字符串与空值回落当前 UTC（近 1 分钟内）
    now = datetime.now(timezone.utc)
    bad = datetime.fromisoformat(_base._normalize_time("not-a-date"))
    assert abs((now - bad).total_seconds()) < 60
    empty = datetime.fromisoformat(_base._normalize_time(""))
    assert abs((now - empty).total_seconds()) < 60


def test_first_present_skips_falsy():
    rec = {"a": None, "b": "", "c": [], "d": {}, "e": "hit", "f": "later"}
    assert _base._first_present(rec, ("a", "b", "c", "d", "e", "f")) == "hit"
    assert _base._first_present({}, ("x",)) is None


def test_extract_records_variants():
    # 裸列表保留 dict/str（含空串），丢弃无正文条目（None -> ""）
    assert _base.extract_records([{"memory": "m"}, "s", "", None]) == [{"memory": "m"}, "s", ""]
    # 已知包裹键
    assert _base.extract_records({"results": [{"a": 1}]}) == [{"a": 1}]
    # 未知键的 dict -> 视作单条记录
    assert _base.extract_records({"foo": "bar"}) == [{"foo": "bar"}]
    # 标量 -> 空
    assert _base.extract_records(123) == []


def test_normalize_record_string_and_defaults():
    rec = _base.normalize_record("just a string", source="mem0", wing="w", room="")
    assert rec and rec["content"] == "just a string"
    # room 空 -> 回落 source
    assert rec["room"] == "mem0"
    # 缺 id -> 生成 12 位 hex
    assert len(rec["memory_id"]) == 12
    # 无正文记录被丢弃
    assert _base.normalize_record({"note": ""}, source="x", wing="w", room="r") is None


# ─── sources 适配器字段怪癖 ──────────────────────────────

def test_mem0_adapter_maps_categories_to_tags():
    out = sources.mem0({"results": [{"memory": "x", "metadata": {"categories": ["preference"]}}]})
    assert out[0]["tags"] == ["preference"]
    # 裸字符串 -> {"memory": ...}
    assert sources.mem0(["hello"])[0] == {"memory": "hello"}


def test_zep_adapter_field_precedence():
    out = sources.zep({"facts": [{"id": "f1", "fact": "via-fact", "type": "entity", "timestamp": 1}]})
    assert out[0]["content"] == "via-fact"
    assert out[0]["memory_type"] == "entity"


def test_graphiti_adapter_entity_and_fallback():
    data = {
        "edges": [{"uuid": "e1", "fact": "rel", "valid_at": "2024-01-01T00:00:00Z", "name": "knows"}],
        "nodes": [{"uuid": "n1", "name": "Acme", "summary": "co", "labels": ["org"]}],
    }
    out = sources.graphiti(data)
    assert any(r.get("memory_type") == "entity" for r in out)
    # 空正文被过滤掉
    assert all(_base._coerce_text(r.get("content")) for r in out)
    # 无 nodes/edges 时回落 zep
    fb = sources.graphiti({"facts": [{"text": "fallback"}]})
    assert fb[0]["content"] == "fallback"


def test_cognee_adapter_delegates_and_maps():
    # 带 nodes -> 走 graphiti
    via_graph = sources.cognee({"nodes": [{"name": "N", "summary": "S"}]})
    assert any(r.get("memory_type") == "entity" for r in via_graph)
    # data_points 直接映射，过滤空正文
    pts = sources.cognee({"data_points": [{"text": "t"}, {"description": ""}]})
    assert [r["content"] for r in pts] == ["t"]


def test_letta_adapter_core_block_type():
    out = sources.letta({"memory_blocks": [{"label": "persona", "value": "I help"}]})
    assert out[0]["memory_type"] == "preference"


def test_auto_adapter_passthrough_list():
    assert sources.SOURCE_ADAPTERS["auto"]([{"a": 1}]) == [{"a": 1}]
