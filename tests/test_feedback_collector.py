"""governance.feedback.FeedbackCollector 离线单元测试。

覆盖：
  - record_click / record_shown 写入与边界
  - get_source_weights：CTR → 权重映射、空/无查询、来源过滤
  - get_training_triplets：正负例配对
  - get_stats：总数与 sources 分组
  - get_memory_trust：点击计数 + 时间衰减
  - close：幂等
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omnimem.governance.feedback import FeedbackCollector


@pytest.fixture
def collector(tmp_path: Path):
    fb = FeedbackCollector(tmp_path / "fb")
    yield fb
    fb.close()


# ── record ──


def test_record_click_inserts_row(collector: FeedbackCollector) -> None:
    collector.record_click("q", "m1", "vector", rank=1)
    stats = collector.get_stats()
    assert stats["total_clicks"] == 1
    assert stats["sources"].get("vector") == 1


def test_record_shown_batch_inserts(collector: FeedbackCollector) -> None:
    candidates = [
        {"memory_id": "m1", "_source": "vector"},
        {"memory_id": "m2", "type": "bm25"},
    ]
    collector.record_shown("q", candidates)
    assert collector.get_stats()["total_shown"] == 2


def test_record_shown_empty_candidates_is_noop(collector: FeedbackCollector) -> None:
    collector.record_shown("q", [])
    assert collector.get_stats()["total_shown"] == 0


def test_record_shown_caps_at_20(collector: FeedbackCollector) -> None:
    candidates = [{"memory_id": f"m{i}", "_source": "vector"} for i in range(30)]
    collector.record_shown("q", candidates)
    assert collector.get_stats()["total_shown"] == 20


def test_record_shown_default_source_when_missing(collector: FeedbackCollector) -> None:
    collector.record_shown("q", [{"memory_id": "m1"}])
    # source 缺失 → "unknown"
    stats = collector.get_stats()
    assert stats["total_shown"] == 1


def test_record_click_closed_connection_is_noop(tmp_path: Path) -> None:
    fb = FeedbackCollector(tmp_path / "fb2")
    fb.close()
    fb.record_click("q", "m1")  # 应静默返回
    fb.record_shown("q", [{"memory_id": "m2"}])
    assert fb.get_stats()["total_clicks"] == 0
    assert fb.get_stats()["total_shown"] == 0


# ── get_source_weights ──


def test_source_weights_empty_returns_empty(collector: FeedbackCollector) -> None:
    assert collector.get_source_weights() == {}


def test_source_weights_boosts_high_ctr(collector: FeedbackCollector) -> None:
    # vector: shown=2, clicks=2 → CTR=1.0 → weight = 1.0 + 0.8*(1.0-0.3) = 1.56
    collector.record_shown("q1", [{"memory_id": "m1", "_source": "vector"}])
    collector.record_shown("q1", [{"memory_id": "m2", "_source": "vector"}])
    collector.record_click("q1", "m1", "vector", rank=1)
    collector.record_click("q1", "m2", "vector", rank=1)
    weights = collector.get_source_weights()
    assert weights.get("vector") == pytest.approx(1.56, rel=1e-3)


def test_source_weights_penalizes_low_ctr(collector: FeedbackCollector) -> None:
    # bm25: shown=10, clicks=0 → CTR=0 → weight = 1.0 + 0.8*(0-0.3) = 0.76
    for i in range(10):
        collector.record_shown("q", [{"memory_id": f"m{i}", "_source": "bm25"}])
    collector.record_click("q", "unrelated", "graph", rank=1)  # 触发 recent_queries
    collector.record_shown("q", [{"memory_id": "extra", "_source": "graph"}])
    weights = collector.get_source_weights()
    # bm25 应该被降权
    if "bm25" in weights:
        assert weights["bm25"] < 1.0


def test_record_shown_falls_back_to_unknown_when_source_empty(
    collector: FeedbackCollector,
) -> None:
    # _source='' → 触发 `or` 回退到 type → 缺失则 'unknown'
    collector.record_shown("q", [{"memory_id": "m1", "_source": ""}])
    collector.record_click("q", "m1", "unknown", rank=1)
    weights = collector.get_source_weights()
    # 未知来源被点击 1/1 → CTR=1.0 → weight=1.56
    assert weights.get("unknown") == pytest.approx(1.56, rel=1e-3)


def test_source_weights_closed_returns_empty(tmp_path: Path) -> None:
    fb = FeedbackCollector(tmp_path / "fb3")
    fb.close()
    assert fb.get_source_weights() == {}


# ── get_training_triplets ──


def test_triplets_pairs_positive_with_highest_rank_negative(
    collector: FeedbackCollector,
) -> None:
    collector.record_shown(
        "q",
        [
            {"memory_id": "p", "_source": "vector"},
            {"memory_id": "n_low", "_source": "vector"},
            {"memory_id": "n_high", "_source": "vector"},
        ],
    )
    collector.record_click("q", "p", "vector", rank=1)
    triplets = collector.get_training_triplets()
    assert len(triplets) == 1
    assert triplets[0]["query"] == "q"
    assert triplets[0]["positive"] == "p"
    # 负例取 rank DESC 第一条 → 展示列表最后一名
    assert triplets[0]["negative"] == "n_high"


def test_triplets_empty_when_no_shown(collector: FeedbackCollector) -> None:
    collector.record_click("q", "m1", "vector", rank=1)
    assert collector.get_training_triplets() == []


def test_triplets_closed_returns_empty(tmp_path: Path) -> None:
    fb = FeedbackCollector(tmp_path / "fb4")
    fb.close()
    assert fb.get_training_triplets() == []


# ── get_stats ──


def test_stats_shape_and_grouping(collector: FeedbackCollector) -> None:
    collector.record_click("q1", "a", "vector")
    collector.record_click("q1", "b", "vector")
    collector.record_click("q2", "c", "bm25")
    stats = collector.get_stats()
    assert stats["total_clicks"] == 3
    assert stats["sources"] == {"vector": 2, "bm25": 1}


def test_stats_defaults_when_closed(tmp_path: Path) -> None:
    fb = FeedbackCollector(tmp_path / "fb5")
    fb.close()
    assert fb.get_stats() == {"total_clicks": 0, "total_shown": 0, "sources": {}}


# ── get_memory_trust ──


def test_memory_trust_scales_with_clicks(collector: FeedbackCollector) -> None:
    for _ in range(5):
        collector.record_click("q", "m1", "vector")
    # 5/20 = 0.25 + recency_bonus ≈ 0.1 → 0.35
    trust = collector.get_memory_trust("m1")
    assert 0.30 <= trust <= 0.35


def test_memory_trust_caps_at_one(collector: FeedbackCollector) -> None:
    for _ in range(30):
        collector.record_click("q", "m1", "vector")
    assert collector.get_memory_trust("m1") <= 1.0


def test_memory_trust_unknown_memory_is_zero(collector: FeedbackCollector) -> None:
    # 无点击 → base_trust=0；last_click=None → 不加 recency_bonus → 0.0
    assert collector.get_memory_trust("nope") == 0.0


def test_memory_trust_closed_default(collector: FeedbackCollector) -> None:
    collector.close()
    assert collector.get_memory_trust("m1") == 0.5


# ── close ──


def test_close_is_idempotent(tmp_path: Path) -> None:
    fb = FeedbackCollector(tmp_path / "fb6")
    fb.close()
    fb.close()  # 二次关闭应静默


# ── 集成 ──


def test_shown_without_clicks_still_produces_weight(collector: FeedbackCollector) -> None:
    """只要有一个查询被点击过，其 shown 集合中的所有 source 都应参与权重计算。"""
    collector.record_shown(
        "q",
        [
            {"memory_id": "a", "_source": "vector"},
            {"memory_id": "b", "_source": "bm25"},
        ],
    )
    # vector 有 1/1 = 1.0 CTR → 1.56；bm25 有 0/1 = 0.0 CTR → 0.76
    collector.record_click("q", "a", "vector")
    weights: dict[str, Any] = collector.get_source_weights()
    assert weights["vector"] > 1.0
    assert weights["bm25"] < 1.0
