"""governance/memory_strength.py 单元测试。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from omnimem.governance import memory_strength as ms_mod
from omnimem.governance.memory_strength import (
    MemoryStrengthEvaluator,
    MemoryStrengthVector,
    ScoringWeights,
)


@pytest.fixture
def evaluator():
    return MemoryStrengthEvaluator()


# ─── MemoryStrengthVector ─────────────────────────────────


def test_vector_to_dict_shape():
    v = MemoryStrengthVector(
        stability=10, retrievability=0.9, difficulty=0.3,
        recency=1.5, frequency=7, semantic_importance=0.6,
    )
    d = v.to_dict()
    assert d == {
        "stability": 10,
        "retrievability": 0.9,
        "difficulty": 0.3,
        "recency": 1.5,
        "frequency": 7,
        "semantic_importance": 0.6,
    }


def test_vector_defaults():
    v = MemoryStrengthVector()
    assert v.stability == 0.0
    assert v.retrievability == 1.0
    assert v.difficulty == 0.5
    assert v.recency == 0.0
    assert v.frequency == 0
    assert v.semantic_importance == 0.5


# ─── ScoringWeights.normalize ─────────────────────────────


def test_normalize_sums_to_one_and_preserves_lambda():
    w = ScoringWeights(stability=2, retrievability=2, recency=1, frequency=1, semantic=4, recency_lambda=0.7)
    n = w.normalize()
    total = n.stability + n.retrievability + n.recency + n.frequency + n.semantic
    assert total == pytest.approx(1.0)
    assert n.recency_lambda == 0.7  # 归一化不影响衰减系数
    assert n.stability == pytest.approx(0.2)


def test_normalize_all_zero_returns_defaults():
    w = ScoringWeights(0, 0, 0, 0, 0)
    n = w.normalize()
    assert n.stability == 0.25  # 默认配置
    assert n.recency == 0.20


def test_default_weights_already_normalized():
    n = ScoringWeights().normalize()
    assert n.stability == pytest.approx(0.25)
    assert n.retrievability == pytest.approx(0.25)
    assert n.recency == pytest.approx(0.20)
    assert n.frequency == pytest.approx(0.15)
    assert n.semantic == pytest.approx(0.15)


# ─── calculate_strength ──────────────────────────────────


def test_calculate_strength_uses_fsrs_when_positive(evaluator):
    v = evaluator.calculate_strength("m", recall_count=3, fsrs_stability=42.0, fsrs_retention=0.7, fsrs_difficulty=0.2)
    assert v.stability == 42.0
    assert v.retrievability == 0.7
    assert v.difficulty == 0.2
    assert v.frequency == 3


def test_calculate_strength_falls_back_to_recall_count(evaluator):
    v = evaluator.calculate_strength("m", recall_count=10, fsrs_stability=0.0)
    # min(100, 10 * 2.0) = 20
    assert v.stability == 20.0


def test_calculate_strength_stability_capped_at_100(evaluator):
    v = evaluator.calculate_strength("m", recall_count=100, fsrs_stability=0.0)
    assert v.stability == 100.0


def test_calculate_recency_missing_returns_365(evaluator):
    v = evaluator.calculate_strength("m", recall_count=1, last_accessed=None)
    assert v.recency == 365.0


def test_calculate_recency_parses_iso_with_offset(evaluator):
    now = datetime.now(timezone.utc)
    two_days_ago = (now - timedelta(days=2)).isoformat()
    v = evaluator.calculate_strength("m", recall_count=1, last_accessed=two_days_ago)
    assert 1.9 <= v.recency <= 2.1


def test_calculate_recency_bad_string_falls_back(evaluator):
    v = evaluator.calculate_strength("m", recall_count=1, last_accessed="not-a-date")
    assert v.recency == 365.0


# ─── calculate_score ─────────────────────────────────────


def test_calculate_score_bounds_and_monotonic(evaluator):
    low = MemoryStrengthVector(
        stability=0, retrievability=0, difficulty=0.5,
        recency=1000, frequency=0, semantic_importance=0,
    )
    high = MemoryStrengthVector(
        stability=100, retrievability=1.0, difficulty=0.5,
        recency=0, frequency=99, semantic_importance=1.0,
    )
    s_low = evaluator.calculate_score(low)
    s_high = evaluator.calculate_score(high)
    assert 0.0 <= s_low <= 100.0
    assert 0.0 <= s_high <= 100.0
    assert s_high > s_low


def test_calculate_score_recency_decay_formula(evaluator):
    """recency=ln(2)/λ 时新近性因子应为 0.5，其他分量为 0 → 只贡献 0.5*λ_recency*100。"""
    lam = evaluator._weights.recency_lambda
    r = math.log(2) / lam
    v = MemoryStrengthVector(
        stability=0, retrievability=0, difficulty=0.5,
        recency=r, frequency=0, semantic_importance=0,
    )
    # 只有 recency 分量非零：w.recency * exp(-λ * r) * 100 = 0.2 * 0.5 * 100 = 10
    assert evaluator.calculate_score(v) == pytest.approx(10.0, rel=0.01)


def test_calculate_score_frequency_log_compression(evaluator):
    """frequency 分量使用 log(F+1)*20，其他分量为 0。"""
    v = MemoryStrengthVector(
        stability=0, retrievability=0, difficulty=0.5,
        recency=0, frequency=0, semantic_importance=0,
    )
    # recency=0 贡献 0.2*100=20；freq=0 → log(1)*20=0；retr/sem/stab=0
    # 期望分：仅 recency 权重 0.2 * 100 = 20
    assert evaluator.calculate_score(v) == pytest.approx(20.0, rel=0.01)


# ─── grade 分档 ──────────────────────────────────────────


@pytest.mark.parametrize(
    "score,expected",
    [
        (95, "S"),
        (85, "A"),
        (70, "B"),
        (50, "C"),
        (10, "D"),
    ],
)
def test_score_to_grade_thresholds(evaluator, score, expected):
    assert evaluator._score_to_grade(score) == expected


# ─── evaluate_memory / batch ─────────────────────────────


def test_evaluate_memory_returns_dict_with_expected_keys(evaluator):
    out = evaluator.evaluate_memory(
        memory_id="m",
        recall_count=5,
        fsrs_retention=0.8,
        fsrs_stability=20.0,
        fsrs_difficulty=0.4,
        semantic_importance=0.7,
    )
    assert {"memory_id", "strength", "score", "grade"}.issubset(out.keys())
    assert out["memory_id"] == "m"
    assert 0.0 <= out["score"] <= 100.0
    assert out["grade"] in {"S", "A", "B", "C", "D"}
    assert out["strength"]["semantic_importance"] == 0.7


def test_evaluate_batch_aligns_with_input(evaluator):
    mems = [
        {"memory_id": "a", "recall_count": 1},
        {"memory_id": "b", "recall_count": 2, "semantic_importance": 0.9},
    ]
    out = evaluator.evaluate_batch(mems)
    assert [o["memory_id"] for o in out] == ["a", "b"]
    assert out[1]["strength"]["semantic_importance"] == 0.9


# ─── get_distribution ────────────────────────────────────


def test_get_distribution_empty_returns_zero_shape(evaluator):
    d = evaluator.get_distribution([])
    assert d == {"total": 0, "grades": {}, "avg_score": 0.0}


def test_get_distribution_counts_grades_and_stats(evaluator):
    results = [
        {"grade": "S", "score": 95},
        {"grade": "A", "score": 85},
        {"grade": "A", "score": 80},
        {"grade": "D", "score": 10},
        {"grade": "unknown", "score": 0},
    ]
    d = evaluator.get_distribution(results)
    assert d["total"] == 5
    assert d["grades"]["S"] == 1
    assert d["grades"]["A"] == 2
    assert d["grades"]["D"] == 1
    # unknown 不计入 grades 但计入 total / scores
    assert d["max_score"] == 95
    assert d["min_score"] == 0
    assert d["avg_score"] == pytest.approx((95 + 85 + 80 + 10 + 0) / 5)


# ─── 单例与便捷函数 ──────────────────────────────────────


def test_get_evaluator_singleton_when_no_weights(monkeypatch):
    monkeypatch.setattr(ms_mod, "_evaluator", None)
    e1 = ms_mod.get_evaluator()
    e2 = ms_mod.get_evaluator()
    assert e1 is e2


def test_get_evaluator_replaces_when_weights_given(monkeypatch):
    monkeypatch.setattr(ms_mod, "_evaluator", None)
    e1 = ms_mod.get_evaluator()
    e2 = ms_mod.get_evaluator(weights=ScoringWeights(1, 1, 1, 1, 1))
    assert e1 is not e2
    assert ms_mod._evaluator is e2


def test_module_level_evaluate_memory_delegates(monkeypatch):
    monkeypatch.setattr(ms_mod, "_evaluator", None)
    out = ms_mod.evaluate_memory(memory_id="x", recall_count=2, fsrs_stability=5.0)
    assert out["memory_id"] == "x"
    assert out["strength"]["stability"] == 5.0
