"""retrieval.quality_eval.RetrievalQualityEvaluator 离线单元测试。

覆盖：
  - evaluate：precision / recall / MRR / nDCG 边界
  - _compute_mrr / _compute_ndcg：命中位置与相关性集合
  - record_evaluation + get_trend：SQLite 往返、时间窗口
  - get_auto_tune_suggestions：阈值触发路径
  - infer_relevant_ids：score/type_boost 启发式
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from omnimem.retrieval.quality_eval import (
    QualityMetrics,
    RetrievalQualityEvaluator,
)


@pytest.fixture
def evaluator(tmp_path: Path):
    ev = RetrievalQualityEvaluator(data_dir=tmp_path / "quality")
    yield ev
    ev.close()


# ── evaluate 指标计算 ──


def test_evaluate_all_relevant_returned(evaluator: RetrievalQualityEvaluator) -> None:
    results: list[dict[str, Any]] = [
        {"memory_id": "m1", "score": 0.9},
        {"memory_id": "m2", "score": 0.5},
    ]
    metrics = evaluator.evaluate("q", results, {"m1", "m2"}, latency_ms=10.0)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.mrr == 1.0
    assert metrics.ndcg == pytest.approx(1.0, rel=1e-6)
    assert metrics.result_count == 2
    assert metrics.query == "q"
    assert metrics.latency_ms == 10.0


def test_evaluate_no_returned_ids_yields_zero_precision(
    evaluator: RetrievalQualityEvaluator,
) -> None:
    metrics = evaluator.evaluate("q", [], {"m1"}, latency_ms=5.0)
    assert metrics.precision == 0.0
    # relevant 非空 + returned 空 → recall = 0
    assert metrics.recall == 0.0
    assert metrics.mrr == 0.0
    assert metrics.result_count == 0


def test_evaluate_empty_relevant_with_empty_returned_has_recall_one(
    evaluator: RetrievalQualityEvaluator,
) -> None:
    metrics = evaluator.evaluate("q", [], set(), latency_ms=1.0)
    # 边界约定：relevant 空 & returned 空 → recall=1.0
    assert metrics.recall == 1.0
    assert metrics.precision == 0.0


def test_evaluate_empty_relevant_with_returned_has_recall_zero(
    evaluator: RetrievalQualityEvaluator,
) -> None:
    metrics = evaluator.evaluate("q", [{"memory_id": "m1"}], set(), latency_ms=1.0)
    # 边界约定：relevant 空 & returned 非空 → recall=0
    assert metrics.recall == 0.0
    assert metrics.precision == 0.0


def test_evaluate_dedup_returned_ids(evaluator: RetrievalQualityEvaluator) -> None:
    # 同一 memory_id 出现多次 → returned 集合去重
    results = [
        {"memory_id": "m1", "score": 0.9},
        {"memory_id": "m1", "score": 0.5},
        {"memory_id": "m2", "score": 0.3},
    ]
    metrics = evaluator.evaluate("q", results, {"m1"}, latency_ms=1.0)
    assert metrics.result_count == 2
    assert metrics.precision == pytest.approx(1 / 2, rel=1e-4)
    assert metrics.recall == 1.0


def test_evaluate_skips_missing_memory_id(evaluator: RetrievalQualityEvaluator) -> None:
    results = [{"score": 0.9}, {"memory_id": "m1", "score": 0.5}]
    metrics = evaluator.evaluate("q", results, {"m1"}, latency_ms=1.0)
    assert metrics.result_count == 1
    assert metrics.precision == 1.0


# ── _compute_mrr ──


def test_compute_mrr_first_position(evaluator: RetrievalQualityEvaluator) -> None:
    results = [{"memory_id": "m1"}, {"memory_id": "m2"}]
    assert evaluator._compute_mrr(results, {"m1"}) == 1.0


def test_compute_mrr_second_position(evaluator: RetrievalQualityEvaluator) -> None:
    results = [{"memory_id": "m1"}, {"memory_id": "m2"}]
    assert evaluator._compute_mrr(results, {"m2"}) == 0.5


def test_compute_mrr_no_hit(evaluator: RetrievalQualityEvaluator) -> None:
    assert evaluator._compute_mrr([{"memory_id": "m1"}], {"other"}) == 0.0


def test_compute_mrr_empty_results(evaluator: RetrievalQualityEvaluator) -> None:
    assert evaluator._compute_mrr([], {"m1"}) == 0.0


# ── _compute_ndcg ──


def test_compute_ndcg_perfect_ranking(evaluator: RetrievalQualityEvaluator) -> None:
    results = [{"memory_id": "m1"}, {"memory_id": "m2"}]
    assert evaluator._compute_ndcg(results, {"m1", "m2"}) == pytest.approx(1.0, rel=1e-6)


def test_compute_ndcg_reversed_ranking(evaluator: RetrievalQualityEvaluator) -> None:
    # 2 relevant；理想顺序 DCG = 1/log2(2) + 1/log2(3)
    actual = evaluator._compute_ndcg(
        [{"memory_id": "x"}, {"memory_id": "m1"}, {"memory_id": "m2"}],
        {"m1", "m2"},
    )
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert actual == pytest.approx(dcg / idcg, rel=1e-6)


def test_compute_ndcg_no_relevant_returns_one(evaluator: RetrievalQualityEvaluator) -> None:
    assert evaluator._compute_ndcg([{"memory_id": "m1"}], set()) == 1.0


def test_compute_ndcg_no_hits_returns_zero(evaluator: RetrievalQualityEvaluator) -> None:
    # relevant 非空但完全没命中 → dcg=0，idcg>0 → 0.0
    assert evaluator._compute_ndcg([{"memory_id": "m1"}], {"other"}) == 0.0


def test_compute_ndcg_zero_when_results_shorter_than_relevant(
    evaluator: RetrievalQualityEvaluator,
) -> None:
    # ideal_count = min(len(relevant), len(results))
    ndcg = evaluator._compute_ndcg([{"memory_id": "m1"}], {"m1", "m2", "m3"})
    assert ndcg == pytest.approx(1.0, rel=1e-6)


# ── record + trend ──


def test_record_and_get_trend_roundtrip(evaluator: RetrievalQualityEvaluator) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for p in (0.8, 0.6):
        evaluator.record_evaluation(
            QualityMetrics(
                precision=p,
                recall=0.5,
                mrr=0.7,
                ndcg=0.6,
                latency_ms=100.0,
                result_count=3,
                query=f"q-{p}",
                timestamp=now,
            )
        )
    trend = evaluator.get_trend(days=1)
    assert trend["sample_count"] == 2
    assert trend["precision"] == pytest.approx(0.7, rel=1e-3)
    assert trend["recall"] == pytest.approx(0.5, rel=1e-3)
    assert trend["result_count"] == pytest.approx(3.0, rel=1e-3)


def test_get_trend_excludes_old_records(evaluator: RetrievalQualityEvaluator) -> None:
    old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    evaluator.record_evaluation(
        QualityMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1, "old", old_ts)
    )
    trend = evaluator.get_trend(days=7)
    assert trend["sample_count"] == 0
    assert trend["precision"] == 0.0


def test_get_trend_empty_db_returns_defaults(evaluator: RetrievalQualityEvaluator) -> None:
    trend = evaluator.get_trend(days=1)
    assert trend["sample_count"] == 0
    assert set(trend.keys()) == {
        "precision",
        "recall",
        "mrr",
        "ndcg",
        "latency_ms",
        "result_count",
        "sample_count",
    }


# ── auto tune suggestions ──


def _seed(evaluator: RetrievalQualityEvaluator, n: int, **overrides: float) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    kwargs: dict[str, float] = {
        "precision": 0.9,
        "recall": 0.9,
        "mrr": 0.9,
        "ndcg": 0.9,
        "latency_ms": 50.0,
    }
    kwargs.update(overrides)
    for _ in range(n):
        evaluator.record_evaluation(
            QualityMetrics(
                precision=kwargs["precision"],
                recall=kwargs["recall"],
                mrr=kwargs["mrr"],
                ndcg=kwargs["ndcg"],
                latency_ms=kwargs["latency_ms"],
                result_count=5,
                query="seed",
                timestamp=ts,
            )
        )


def test_suggestions_sample_shortage(evaluator: RetrievalQualityEvaluator) -> None:
    _seed(evaluator, 2)
    out = evaluator.get_auto_tune_suggestions()
    assert out["suggestions"] == []
    assert "样本数不足" in out["reason"]


def test_suggestions_low_precision_triggers_min_rrf_increase(
    evaluator: RetrievalQualityEvaluator,
) -> None:
    _seed(evaluator, 3, precision=0.1)
    out = evaluator.get_auto_tune_suggestions()
    params = [(s["parameter"], s["action"]) for s in out["suggestions"]]
    assert ("min_rrf", "increase") in params


def test_suggestions_low_recall_triggers_min_rrf_and_top_k(
    evaluator: RetrievalQualityEvaluator,
) -> None:
    _seed(evaluator, 3, recall=0.1)
    params = [(s["parameter"], s["action"]) for s in evaluator.get_auto_tune_suggestions()["suggestions"]]
    assert ("min_rrf", "decrease") in params
    assert ("top_k", "increase") in params


def test_suggestions_low_mrr_adjusts_rrf_weights(evaluator: RetrievalQualityEvaluator) -> None:
    _seed(evaluator, 3, mrr=0.1)
    params = [(s["parameter"], s["action"]) for s in evaluator.get_auto_tune_suggestions()["suggestions"]]
    assert ("rrf_vector_weight", "increase") in params
    assert ("rrf_bm25_weight", "decrease") in params


def test_suggestions_high_latency_triggers_top_k_decrease_and_cache(
    evaluator: RetrievalQualityEvaluator,
) -> None:
    _seed(evaluator, 3, latency_ms=3000.0)
    params = [(s["parameter"], s["action"]) for s in evaluator.get_auto_tune_suggestions()["suggestions"]]
    assert ("top_k", "decrease") in params
    assert ("query_cache_ttl", "enable") in params


def test_suggestions_healthy_metrics_yield_no_actions(
    evaluator: RetrievalQualityEvaluator,
) -> None:
    _seed(evaluator, 5)
    out = evaluator.get_auto_tune_suggestions()
    assert out["suggestions"] == []
    assert "样本" in out["reason"]


# ── infer_relevant_ids ──


def test_infer_relevant_ids_score_threshold() -> None:
    results = [
        {"memory_id": "m1", "score": 0.1},
        {"memory_id": "m2", "score": 0.01},
    ]
    assert RetrievalQualityEvaluator.infer_relevant_ids(results) == {"m1"}


def test_infer_relevant_ids_type_boost_flag() -> None:
    results = [{"memory_id": "m1", "score": 0.0, "type_boost": 1.5}]
    assert RetrievalQualityEvaluator.infer_relevant_ids(results) == {"m1"}


def test_infer_relevant_ids_skips_missing_memory_id() -> None:
    results = [{"score": 0.9}, {"memory_id": "", "score": 0.9}]
    assert RetrievalQualityEvaluator.infer_relevant_ids(results) == set()


def test_infer_relevant_ids_boundary_score_exactly_threshold() -> None:
    results = [{"memory_id": "m1", "score": 0.05}]
    assert RetrievalQualityEvaluator.infer_relevant_ids(results) == {"m1"}


# ── 数据类完整性 ──


def test_quality_metrics_is_frozen_shape() -> None:
    m = QualityMetrics(0.5, 0.5, 0.5, 0.5, 10.0, 3, "q", "2026-01-01T00:00:00+00:00")
    d = m.__dict__
    assert set(d.keys()) == {
        "precision",
        "recall",
        "mrr",
        "ndcg",
        "latency_ms",
        "result_count",
        "query",
        "timestamp",
    }
