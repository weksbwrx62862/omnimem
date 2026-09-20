"""HMS T/L 算子：retrieval/temporal_separation.py + retrieval/session_local.py 单元测试。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from omnimem.retrieval import session_local as sl_mod
from omnimem.retrieval import temporal_separation as ts_mod
from omnimem.retrieval.session_local import (
    DEFAULT_SIGMA_HOURS,
    ENTRY_POINT_SCORE_THRESHOLD,
    MAX_BOOST,
    SessionLocalOperator,
    _gaussian_weight,
    _parse_time,
)
from omnimem.retrieval.temporal_separation import (
    annotate_results,
    annotate_time,
    parse_occurrence_time,
    sort_by_occurrence,
)

# ─── temporal_separation: parse_occurrence_time ─────────


def test_parse_full_date_iso():
    assert parse_occurrence_time("会议于 2023-07-07 举行").startswith("2023-07-07T00:00:00")


def test_parse_full_date_chinese():
    assert parse_occurrence_time("2024年3月15日发生").startswith("2024-03-15T00:00:00")


def test_parse_month_day_uses_mention_year():
    out = parse_occurrence_time("7月7日发布", mention_time="2023-01-01T00:00:00+00:00")
    assert out is not None
    assert out.startswith("2023-07-07T00:00:00")


def test_parse_month_day_defaults_to_current_year_when_no_mention():
    out = parse_occurrence_time("7月7日发布")
    assert out is not None
    assert out.startswith(str(datetime.now(timezone.utc).year))


def test_parse_relative_yesterday():
    base = datetime(2026, 5, 10, tzinfo=timezone.utc)
    out = parse_occurrence_time("昨天完成了任务", mention_time=base.isoformat())
    assert out is not None
    expected = (base - timedelta(days=1)).isoformat()
    assert out == expected


def test_parse_relative_without_mention_returns_none():
    assert parse_occurrence_time("上周开始") is None


def test_parse_returns_none_for_no_time_signal():
    assert parse_occurrence_time("用户喜欢深色主题") is None


def test_parse_returns_none_for_empty_content():
    assert parse_occurrence_time("") is None


def test_parse_invalid_full_date_falls_through():
    # 2 月 30 日 → ValueError；因未匹配月-日 / 相对时间 → 返回 None
    assert parse_occurrence_time("2023-02-30") is None


# ─── temporal_separation: 内部工具 ───────────────────────


def test_mention_year_with_valid_and_invalid():
    assert ts_mod._mention_year("2024-01-01T00:00:00+00:00") == 2024
    # 无效输入 → 当前年
    assert ts_mod._mention_year(None) == datetime.now(timezone.utc).year
    assert ts_mod._mention_year("not-a-date") == datetime.now(timezone.utc).year


def test_parse_mention_time_z_suffix():
    dt = ts_mod._parse_mention_time("2024-01-01T00:00:00Z")
    assert dt is not None
    assert dt.year == 2024
    assert ts_mod._parse_mention_time(None) is None
    assert ts_mod._parse_mention_time("garbage") is None


# ─── temporal_separation: annotate / sort ────────────────


def test_annotate_time_reads_multiple_keys():
    r = {"stored_at": "2024-01-02T00:00:00Z", "content": "2023年7月7日会议"}
    out = annotate_time(r)
    assert out is r  # 原地修改
    assert out["_mention_time"] == "2024-01-02T00:00:00Z"
    assert out["_occurrence_time"].startswith("2023-07-07")


def test_annotate_time_falls_back_to_summary_and_text():
    r1 = {"timestamp": "2024-01-01T00:00:00Z", "summary": "2020-06-01 事件"}
    assert annotate_time(r1)["_occurrence_time"].startswith("2020-06-01")
    r2 = {"created_at": "2024-01-01T00:00:00Z", "text": "2021-01-01 事件"}
    assert annotate_time(r2)["_occurrence_time"].startswith("2021-01-01")


def test_annotate_results_does_not_mutate_originals():
    original = {"stored_at": "2024-01-01T00:00:00Z", "content": "2023-05-05"}
    out = annotate_results([original])
    assert "_occurrence_time" not in original  # 原对象未被污染
    assert out[0]["_occurrence_time"].startswith("2023-05-05")


def test_sort_by_occurrence_three_tiers():
    a = {"_occurrence_time": "2024-01-01T00:00:00+00:00", "_mention_time": "2024-01-02"}
    b = {"_occurrence_time": None, "_mention_time": "2024-01-03"}
    c = {}  # 无时间
    out = sort_by_occurrence([c, b, a])
    # 优先级：occurrence > mention > none；每档内按时间字符串排序
    assert [id(x) for x in out] == [id(out[0]), id(out[1]), id(out[2])]
    assert out[0].get("_occurrence_time") == "2024-01-01T00:00:00+00:00"
    assert out[1].get("_mention_time") == "2024-01-03"
    assert out[2] == {}


# ─── session_local: _parse_time ──────────────────────────


def test_parse_time_int_seconds():
    assert _parse_time(1_700_000_000) == 1_700_000_000.0


def test_parse_time_millis():
    # 1e6 < v <= 1e9 → 视作毫秒 × 1000
    assert _parse_time(500_000_000) == 500_000_000.0 * 1000


def test_parse_time_small_int_returns_none():
    assert _parse_time(1000) is None


def test_parse_time_iso_string_with_z():
    ts = _parse_time("2024-01-01T00:00:00Z")
    assert ts is not None
    assert ts == datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()


def test_parse_time_garbage_or_none():
    assert _parse_time("not-a-date") is None
    assert _parse_time(None) is None


# ─── session_local: _gaussian_weight ─────────────────────


def test_gaussian_weight_zero_is_one():
    assert _gaussian_weight(0) == pytest.approx(1.0)


def test_gaussian_weight_at_sigma():
    w = _gaussian_weight(DEFAULT_SIGMA_HOURS, DEFAULT_SIGMA_HOURS)
    assert w == pytest.approx(math.exp(-0.5))


def test_gaussian_weight_symmetric_for_negative():
    assert _gaussian_weight(-3) == _gaussian_weight(3)


# ─── session_local: SessionLocalOperator.enhance ────────


def _mk(mid: str, score: float, hours_ago: float) -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return {"memory_id": mid, "score": score, "stored_at": ts}


def test_enhance_disabled_returns_copy():
    op = SessionLocalOperator(enabled=False)
    rs = [_mk("a", 0.9, 1), _mk("b", 0.5, 2)]
    out = op.enhance(rs)
    assert out == rs
    assert out is not rs


def test_enhance_single_result_passthrough():
    op = SessionLocalOperator()
    rs = [_mk("only", 0.9, 0)]
    out = op.enhance(rs)
    assert out[0]["score"] == 0.9


def test_enhance_no_entry_point_passthrough():
    op = SessionLocalOperator()
    rs = [_mk("a", 0.3, 1), _mk("b", 0.2, 2)]  # 都低于 ENTRY_POINT_SCORE_THRESHOLD
    out = op.enhance(rs)
    assert [r["score"] for r in out] == [0.3, 0.2]
    assert all("_session_local_boost" not in r for r in out)


def test_enhance_boosts_nearby_results_and_sorts():
    op = SessionLocalOperator()
    entry = _mk("entry", 0.9, 0)
    near = _mk("near", 0.3, 1)  # 1h 邻近
    far = _mk("far", 0.3, 240)  # 240h 远离
    out = op.enhance([entry, near, far])
    by_id = {r["memory_id"]: r for r in out}
    # 邻近项 boost 更接近 MAX_BOOST；远离项 boost 接近 1.0
    assert by_id["near"]["_session_local_boost"] > by_id["far"]["_session_local_boost"]
    # 三项都应获得 >= 1.0 的 boost 因子（入口点自身 Δt=0 → boost=MAX_BOOST）
    assert by_id["entry"]["score"] == pytest.approx(0.9 * MAX_BOOST, rel=1e-2)
    # 排序后：入口点分数最高，应在首位
    assert out[0]["memory_id"] == "entry"


def test_enhance_with_explicit_entry_point_scores():
    """entry_point_scores 分支：把 score 列表当作锚点，实际会误用 score 值当时间戳。

    这是当前实现语义：显式入口点路径把 score 数值当作 unix 时间使用（源码
    `entry_times.append(float(score))`）。测试锁定该行为，避免无声回归。
    """
    op = SessionLocalOperator()
    results = [_mk("a", 0.7, 0), _mk("b", 0.2, 1)]
    out = op.enhance(results, entry_point_scores=[0.8, 0.9])
    # 因锚点数值极小（<1e6）实际是"1970 附近秒"，与 stored_at 差数百亿秒
    # 高斯核 → 0，boost=1.0 → 分数保持不变
    scores = {r["memory_id"]: r["score"] for r in out}
    assert scores["a"] == pytest.approx(0.7)
    assert scores["b"] == pytest.approx(0.2)


def test_enhance_handles_missing_timestamp_gracefully():
    op = SessionLocalOperator()
    entry = {"memory_id": "e", "score": 0.9, "stored_at": datetime.now(timezone.utc).isoformat()}
    no_time = {"memory_id": "nt", "score": 0.5}  # 无任何时间字段
    out = op.enhance([entry, no_time])
    by_id = {r["memory_id"]: r for r in out}
    # 无时间戳项保持原分数、不带 boost 标记
    assert by_id["nt"]["score"] == 0.5
    assert "_session_local_boost" not in by_id["nt"]


def test_session_local_operator_repr():
    op = SessionLocalOperator(sigma_hours=48, enabled=True)
    text = repr(op)
    assert "sigma=48" in text and "enabled=True" in text


def test_entry_threshold_constant():
    assert ENTRY_POINT_SCORE_THRESHOLD == 0.6
    assert MAX_BOOST == 1.5
    assert DEFAULT_SIGMA_HOURS == 24.0
    # 保证模块内引用与私有别名一致（防止重构破坏）
    assert sl_mod.ENTRY_POINT_SCORE_THRESHOLD == ENTRY_POINT_SCORE_THRESHOLD
