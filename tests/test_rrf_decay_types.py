"""retrieval/rrf.py + governance/decay.py + memory/types.py 单元测试。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from omnimem.governance.decay import HALF_LIVES, TemporalDecay
from omnimem.memory.types import MemoryEntry, MemoryType, PrivacyLevel
from omnimem.retrieval.rrf import RRFFusion

# ─── RRFFusion ────────────────────────────────────────────


def test_rrf_default_weights_and_ordering():
    fusion = RRFFusion(k=60, min_rrf=0.0)
    vec = [{"memory_id": "a", "score": 0.9}, {"memory_id": "b", "score": 0.4}]
    bm = [{"memory_id": "b"}, {"memory_id": "c"}]
    out = fusion.merge([vec, bm])
    ids = [r["memory_id"] for r in out]
    # a 只在向量 rank1，权重 3.0，score>0.5 触发 1.5x → 3*1.5/61 ≈ 0.0738
    # b 双通道：向量 rank2(3.0/62，score<0.5 不加), BM25 rank1(1.0/61)
    # 综合看 a 应排首位
    assert ids[0] == "a"
    assert set(ids) == {"a", "b", "c"}


def test_rrf_high_score_vector_bonus():
    """向量通道 score>0.5 触发 1.5x 加权。"""
    fusion = RRFFusion(k=60, min_rrf=0.0)
    hi = [{"memory_id": "hi", "score": 0.9}]
    lo = [{"memory_id": "lo", "score": 0.3}]
    [a] = fusion.merge([hi, []])
    [b] = fusion.merge([lo, []])
    # 同为 rank1 时，高分向量应获得 1.5x 加权
    assert a["rrf_score"] == pytest.approx(3.0 * 1.5 / 61)
    assert b["rrf_score"] == pytest.approx(3.0 / 61)


def test_rrf_min_rrf_threshold_filters_noise():
    fusion = RRFFusion(k=60, min_rrf=0.035)
    out = fusion.merge([[{"memory_id": "x", "score": 0.1}], [{"memory_id": "y"}]])
    ids = {r["memory_id"] for r in out}
    # x 只有向量 rank1 → 3.0/61 ≈ 0.0492 ≥ 0.035 通过
    # y 只有 BM25 rank1 → 1.0/61 ≈ 0.0164 < 0.035 被过滤
    assert ids == {"x"}


def test_rrf_custom_weights_and_min_rrf_override():
    fusion = RRFFusion(k=60, min_rrf=0.5)  # 高默认阈值
    out = fusion.merge(
        [[{"memory_id": "a"}], [{"memory_id": "a"}]],
        weights=[2.0, 2.0],
        min_rrf=0.0,
    )
    # 双通道 rank1 → 2/61 + 2/61 = 4/61
    assert out[0]["rrf_score"] == pytest.approx(4 / 61)


def test_rrf_fallback_hash_id_when_missing():
    """memory_id 缺失时用内容哈希做内部去重键；返回条目本身不被注入 memory_id。"""
    fusion = RRFFusion(k=60, min_rrf=0.0)
    out = fusion.merge([[{"content": "hello"}]])
    assert len(out) == 1
    assert out[0]["content"] == "hello"
    assert "memory_id" not in out[0]
    assert out[0]["rrf_score"] > 0


# ─── TemporalDecay ────────────────────────────────────────


def test_decay_get_and_set_half_life():
    td = TemporalDecay()
    assert td.get_half_life("preference") == 180
    assert td.get_half_life("unknown_type") == 365  # 默认兜底
    td.set_half_life("unknown_type", 100)
    assert td.get_half_life("unknown_type") == 100


def test_decay_custom_half_lives_in_constructor():
    td = TemporalDecay(custom_half_lives={"event": 30})
    assert td.get_half_life("event") == 30


def test_decay_skips_missing_stored_at_and_future_dates():
    td = TemporalDecay()
    results = [
        {"memory_id": "a", "score": 1.0, "type": "event"},  # 无 stored_at
        {
            "memory_id": "b",
            "score": 1.0,
            "type": "event",
            "stored_at": (datetime.now(timezone.utc) + timedelta(days=5)).isoformat(),
        },  # 未来
    ]
    out = td.apply(results)
    # 两项都应保持原 score
    for r in out:
        assert r["score"] == 1.0
        assert "decay_factor" not in r


def test_decay_applies_expected_factor_and_sorts():
    td = TemporalDecay()
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=90)).isoformat()
    results = [
        {"memory_id": "old", "score": 1.0, "type": "event", "stored_at": old},
        {"memory_id": "new", "score": 1.0, "type": "event", "stored_at": now.isoformat()},
    ]
    out = td.apply(results)
    # event 半衰期 90 天 → 90 天后因子 0.5
    old_entry = next(r for r in out if r["memory_id"] == "old")
    new_entry = next(r for r in out if r["memory_id"] == "new")
    assert old_entry["decay_factor"] == pytest.approx(0.5, rel=0.01)
    assert old_entry["score"] < new_entry["score"]
    assert out[0]["memory_id"] == "new"


def test_decay_fact_type_unchanged():
    td = TemporalDecay()
    past = (datetime.now(timezone.utc) - timedelta(days=365 * 5)).isoformat()
    out = td.apply([{"memory_id": "f", "score": 0.7, "type": "fact", "stored_at": past}])
    assert out[0]["score"] == pytest.approx(0.7)
    assert "decay_factor" not in out[0]


def test_decay_accepts_datetime_and_ignores_bad_types():
    td = TemporalDecay()
    past_dt = datetime.now(timezone.utc) - timedelta(days=180)
    results = [
        {"memory_id": "d", "score": 0.8, "type": "preference", "stored_at": past_dt},
        {"memory_id": "bad", "score": 0.5, "type": "preference", "stored_at": 12345},
    ]
    out = td.apply(results)
    d_entry = next(r for r in out if r["memory_id"] == "d")
    bad_entry = next(r for r in out if r["memory_id"] == "bad")
    # preference 半衰期 180 → 因子 0.5
    assert d_entry["decay_factor"] == pytest.approx(0.5, rel=0.01)
    assert bad_entry["score"] == 0.5
    assert "decay_factor" not in bad_entry


def test_decay_parse_cache_limits_growth():
    td = TemporalDecay()
    td._parse_cache_max = 2
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for i in range(5):
        ts = (base + timedelta(days=i)).isoformat()
        td.apply([{"memory_id": f"m{i}", "score": 1.0, "type": "event", "stored_at": ts}])
    assert len(td._parse_cache) <= 2


def test_half_lives_defaults():
    assert HALF_LIVES["fact"] == float("inf")
    assert HALF_LIVES["event"] == 90
    assert math.isclose(HALF_LIVES["preference"], 180)


# ─── memory/types.py ──────────────────────────────────────


def test_memory_entry_to_dict_basic_shape():
    entry = MemoryEntry(
        memory_id="m-1",
        content="用户偏好深色主题",
        memory_type=MemoryType.PREFERENCE,
        confidence=4,
        privacy=PrivacyLevel.TEAM,
        scope="team",
        wing="w",
        room="r",
        stored_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        trust=0.75,
        heat="hot",
    )
    d = entry.to_dict()
    assert d["memory_id"] == "m-1"
    assert d["type"] == "preference"
    assert d["privacy"] == "team"
    assert d["confidence"] == 4
    assert d["trust"] == 0.75
    assert d["heat"] == "hot"
    assert d["stored_at"] == "2026-01-01T00:00:00+00:00"
    # 未升级到 wiki → 相关字段不出现
    assert "upgraded_to_wiki" not in d
    assert "wiki_page_path" not in d
    assert "provenance" not in d


def test_memory_entry_to_dict_includes_optional_sections():
    entry = MemoryEntry(
        memory_id="m-2",
        content="x",
        provenance={"source": "chat"},
        upgraded_to_wiki=True,
        wiki_page_path="/wiki/x",
        metadata={"tag": "important"},
    )
    d = entry.to_dict()
    assert d["provenance"] == {"source": "chat"}
    assert d["upgraded_to_wiki"] is True
    assert d["wiki_page_path"] == "/wiki/x"
    assert d["tag"] == "important"  # metadata 平铺到顶层


def test_memory_entry_defaults_and_enum_values():
    entry = MemoryEntry(memory_id="m-3", content="y")
    assert entry.memory_type is MemoryType.FACT
    assert entry.privacy is PrivacyLevel.PERSONAL
    assert entry.confidence == 3
    assert entry.trust == 0.5
    assert entry.heat == "neutral"
    assert entry.metadata == {}
    # MemoryType 是 str Enum
    assert MemoryType.FACT == "fact"
    assert PrivacyLevel.SECRET.value == "secret"


def test_memory_entry_to_dict_stored_at_none_becomes_null():
    entry = MemoryEntry(memory_id="m-4", content="z")
    d = entry.to_dict()
    assert d["stored_at"] is None
