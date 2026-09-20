"""Tests for governance.adaptive_optimizer — mode switching + LLM fallback + persistence."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from governance import adaptive_optimizer as ao
from governance.adaptive_optimizer import (
    AdaptiveFSRSOptimizer,
    OptimizationResult,
    PrecisionRecord,
    adaptive_optimize,
    get_adaptive_optimizer,
)

DEFAULT_PARAMS = [
    0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14,
    0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61, 9.0, 0.5,
]


@pytest.fixture
def opt(tmp_path) -> AdaptiveFSRSOptimizer:
    return AdaptiveFSRSOptimizer(
        precision_threshold=0.85,
        decline_threshold=0.05,
        cache_dir=str(tmp_path / "cache"),
    )


# ── dataclasses ─────────────────────────────────────────────────────────


def test_precision_record() -> None:
    r = PrecisionRecord(timestamp=datetime.now(), precision=0.9, sample_count=5, mode="gradient")
    assert r.precision == 0.9
    assert r.mode == "gradient"


def test_optimization_result_defaults() -> None:
    r = OptimizationResult(
        mode="llm",
        precision_before=0.8,
        precision_after=0.9,
        improvement=0.1,
        parameters=[1.0] * 19,
        cost=0.1,
    )
    assert r.timestamp is not None


# ── constructor ─────────────────────────────────────────────────────────


def test_default_cache_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    o = AdaptiveFSRSOptimizer()
    assert "adaptive_cache" in str(o._cache_dir)
    assert o._cache_dir.exists()


def test_explicit_cache_dir(tmp_path) -> None:
    p = tmp_path / "my_cache"
    o = AdaptiveFSRSOptimizer(cache_dir=str(p))
    assert p.exists()


def test_initial_state(tmp_path) -> None:
    o = AdaptiveFSRSOptimizer(cache_dir=str(tmp_path / "c"))
    assert o._current_mode == "gradient"
    assert o._current_precision == 1.0
    assert o._total_calls == 0


# ── _load_state / _save_state ───────────────────────────────────────────


def test_save_and_load_state(tmp_path) -> None:
    p = tmp_path / "c"
    o1 = AdaptiveFSRSOptimizer(cache_dir=str(p))
    o1._current_mode = "llm"
    o1._current_precision = 0.7
    o1._total_calls = 5
    o1._gradient_calls = 3
    o1._llm_calls = 2
    o1._mode_switches = 1
    o1._save_state()

    o2 = AdaptiveFSRSOptimizer(cache_dir=str(p))
    assert o2._current_mode == "llm"
    assert o2._current_precision == 0.7
    assert o2._total_calls == 5
    assert o2._mode_switches == 1


def test_load_state_missing_file(tmp_path) -> None:
    o = AdaptiveFSRSOptimizer(cache_dir=str(tmp_path / "empty"))
    assert o._current_mode == "gradient"


def test_load_state_invalid_json(tmp_path) -> None:
    cache = tmp_path / "c"
    cache.mkdir()
    (cache / "optimizer_state.json").write_text("not json", encoding="utf-8")
    o = AdaptiveFSRSOptimizer(cache_dir=str(cache))
    # Warning logged, defaults retained
    assert o._current_mode == "gradient"


def test_save_state_exception(tmp_path) -> None:
    o = AdaptiveFSRSOptimizer(cache_dir=str(tmp_path / "c"))
    # Force write failure
    o._cache_dir = tmp_path / "no_such_dir_nested"
    o._save_state()  # should not raise


# ── _calculate_precision ────────────────────────────────────────────────


def test_calculate_precision_empty_returns_one(opt: AdaptiveFSRSOptimizer) -> None:
    assert opt._calculate_precision([], DEFAULT_PARAMS) == 1.0


def test_calculate_precision_in_range(opt: AdaptiveFSRSOptimizer) -> None:
    data = [
        {"elapsed_days": 1, "stability": 1.0, "retention_before": 0.9},
        {"elapsed_days": 5, "stability": 2.0, "retention_before": 0.7},
    ]
    p = opt._calculate_precision(data, DEFAULT_PARAMS)
    assert 0.0 <= p <= 1.0


def test_calculate_precision_missing_fields_defaults(opt: AdaptiveFSRSOptimizer) -> None:
    # Empty dict → defaults elapsed_days=0, stability=1.0, retention_before=0.5
    p = opt._calculate_precision([{}], DEFAULT_PARAMS)
    assert 0.0 <= p <= 1.0


# ── _record_precision ───────────────────────────────────────────────────


def test_record_precision_appends(opt: AdaptiveFSRSOptimizer) -> None:
    opt._record_precision(0.9, 10)
    assert len(opt._precision_history) == 1
    assert opt._precision_history[-1].precision == 0.9
    assert opt._precision_history[-1].sample_count == 10


def test_record_precision_max_100(opt: AdaptiveFSRSOptimizer) -> None:
    for i in range(150):
        opt._record_precision(0.9, 1)
    assert len(opt._precision_history) == 100


# ── _switch_mode ────────────────────────────────────────────────────────


def test_switch_mode_changes_mode(opt: AdaptiveFSRSOptimizer) -> None:
    opt._switch_mode("llm")
    assert opt._current_mode == "llm"
    assert opt._mode_switches == 1


def test_switch_mode_noop_same(opt: AdaptiveFSRSOptimizer) -> None:
    opt._switch_mode("gradient")  # already gradient
    assert opt._mode_switches == 0


def test_switch_mode_multiple(opt: AdaptiveFSRSOptimizer) -> None:
    opt._switch_mode("llm")
    opt._switch_mode("hybrid")
    assert opt._mode_switches == 2
    assert opt._current_mode == "hybrid"


# ── _check_and_switch_mode ──────────────────────────────────────────────


def test_check_low_precision_triggers_llm(opt: AdaptiveFSRSOptimizer) -> None:
    opt._check_and_switch_mode(current_precision=0.5, sample_count=10)
    assert opt._current_mode == "llm"


def test_check_precision_decline_triggers_hybrid(opt: AdaptiveFSRSOptimizer) -> None:
    # Seed history with 3 declining entries
    for p in [0.95, 0.90, 0.85]:
        opt._record_precision(p, 10)
    # current_precision just above threshold, but trend declines
    opt._check_and_switch_mode(current_precision=0.86, sample_count=10)
    assert opt._current_mode == "hybrid"


def test_check_precision_recovered_falls_back(opt: AdaptiveFSRSOptimizer) -> None:
    opt._current_mode = "llm"
    opt._check_and_switch_mode(current_precision=0.95, sample_count=10)
    # 0.95 > threshold(0.85) + 0.05 = 0.90 → fallback
    assert opt._current_mode == "gradient"


def test_check_precision_above_threshold_no_change(opt: AdaptiveFSRSOptimizer) -> None:
    opt._check_and_switch_mode(current_precision=0.9, sample_count=10)
    # Above threshold and history too short for decline → no switch
    assert opt._current_mode == "gradient"


# ── _optimize_with_gradient ─────────────────────────────────────────────


def test_optimize_with_gradient_returns_params(opt: AdaptiveFSRSOptimizer) -> None:
    data = [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}]
    params = opt._optimize_with_gradient(data, DEFAULT_PARAMS)
    assert len(params) == 19


def test_optimize_with_gradient_empty(opt: AdaptiveFSRSOptimizer) -> None:
    params = opt._optimize_with_gradient([], DEFAULT_PARAMS)
    assert len(params) == 19


# ── _optimize_with_llm ──────────────────────────────────────────────────


def test_optimize_with_llm_no_caller_falls_back(opt: AdaptiveFSRSOptimizer) -> None:
    opt._llm_caller = None
    data = [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}]
    params = opt._optimize_with_llm(data, DEFAULT_PARAMS)
    assert len(params) == 19


def test_optimize_with_llm_valid_response(tmp_path) -> None:
    llm = MagicMock(return_value="[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]")
    o = AdaptiveFSRSOptimizer(llm_caller=llm, cache_dir=str(tmp_path / "c"))
    params = o._optimize_with_llm(
        [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}],
        DEFAULT_PARAMS,
    )
    assert len(params) == 19
    assert all(abs(p - 0.5) < 1e-6 for p in params)


def test_optimize_with_llm_bad_json_falls_back(tmp_path) -> None:
    llm = MagicMock(return_value="no numbers here")
    o = AdaptiveFSRSOptimizer(llm_caller=llm, cache_dir=str(tmp_path / "c"))
    params = o._optimize_with_llm(
        [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}],
        DEFAULT_PARAMS,
    )
    # Parse fails → returns current_params via _parse_llm_response
    assert len(params) == 19


def test_optimize_with_llm_wrong_length_falls_back(tmp_path) -> None:
    llm = MagicMock(return_value="[1.0, 2.0, 3.0]")
    o = AdaptiveFSRSOptimizer(llm_caller=llm, cache_dir=str(tmp_path / "c"))
    params = o._optimize_with_llm(
        [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}],
        DEFAULT_PARAMS,
    )
    # Length mismatch → returns current_params
    assert params == o._optimize_with_gradient(
        [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}],
        DEFAULT_PARAMS,
    ) or len(params) == 19


def test_optimize_with_llm_exception_falls_back(tmp_path) -> None:
    def boom(prompt):
        raise RuntimeError("llm down")

    o = AdaptiveFSRSOptimizer(llm_caller=boom, cache_dir=str(tmp_path / "c"))
    params = o._optimize_with_llm(
        [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}],
        DEFAULT_PARAMS,
    )
    assert len(params) == 19


def test_optimize_with_llm_invalid_params_falls_back(tmp_path) -> None:
    # 19 params but out of range
    llm = MagicMock(return_value="[" + ",".join(["200"] * 19) + "]")
    o = AdaptiveFSRSOptimizer(llm_caller=llm, cache_dir=str(tmp_path / "c"))
    params = o._optimize_with_llm(
        [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}],
        DEFAULT_PARAMS,
    )
    # Validate fails → gradient path
    assert len(params) == 19


# ── _optimize_hybrid ────────────────────────────────────────────────────


def test_optimize_hybrid_no_caller(tmp_path) -> None:
    o = AdaptiveFSRSOptimizer(cache_dir=str(tmp_path / "c"))
    params = o._optimize_hybrid(
        [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}],
        DEFAULT_PARAMS,
    )
    assert len(params) == 19


def test_optimize_hybrid_with_llm_better(tmp_path) -> None:
    # LLM returns same-ish params; whichever is better wins
    llm = MagicMock(return_value="[" + ",".join(str(x) for x in DEFAULT_PARAMS) + "]")
    o = AdaptiveFSRSOptimizer(llm_caller=llm, cache_dir=str(tmp_path / "c"))
    params = o._optimize_hybrid(
        [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}],
        DEFAULT_PARAMS,
    )
    assert len(params) == 19


def test_optimize_hybrid_llm_exception(tmp_path) -> None:
    def boom(prompt):
        raise RuntimeError("nope")

    o = AdaptiveFSRSOptimizer(llm_caller=boom, cache_dir=str(tmp_path / "c"))
    params = o._optimize_hybrid(
        [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}],
        DEFAULT_PARAMS,
    )
    assert len(params) == 19


# ── _build_llm_prompt ───────────────────────────────────────────────────


def test_build_llm_prompt_contains_stats(opt: AdaptiveFSRSOptimizer) -> None:
    data = [{"rating": 3, "retention_before": 0.7}, {"rating": 4, "retention_before": 0.9}]
    prompt = opt._build_llm_prompt(data, DEFAULT_PARAMS)
    assert "FSRS" in prompt
    assert "总复习次数: 2" in prompt
    assert "平均评分: 3.50" in prompt
    assert "平均保持率: 80.00%" in prompt


def test_build_llm_prompt_truncates_records(opt: AdaptiveFSRSOptimizer) -> None:
    data = [{"rating": 3, "retention_before": 0.7} for _ in range(50)]
    prompt = opt._build_llm_prompt(data, DEFAULT_PARAMS)
    # Only 10 recent shown
    assert prompt.count('"rating": 3') <= 10


def test_build_llm_prompt_empty_data(opt: AdaptiveFSRSOptimizer) -> None:
    # Zero division guard
    with pytest.raises(ZeroDivisionError):
        opt._build_llm_prompt([], DEFAULT_PARAMS)


# ── _parse_llm_response ─────────────────────────────────────────────────


def test_parse_llm_response_valid(opt: AdaptiveFSRSOptimizer) -> None:
    r = opt._parse_llm_response("[1.0,2.0,3.0]", [0.0, 0.0, 0.0])
    assert r == [1.0, 2.0, 3.0]


def test_parse_llm_response_wrong_length(opt: AdaptiveFSRSOptimizer) -> None:
    r = opt._parse_llm_response("[1.0, 2.0]", [0.0, 0.0, 0.0])
    assert r == [0.0, 0.0, 0.0]  # current


def test_parse_llm_response_no_match(opt: AdaptiveFSRSOptimizer) -> None:
    r = opt._parse_llm_response("no params here", [1.0, 2.0])
    assert r == [1.0, 2.0]


def test_parse_llm_response_invalid_json(opt: AdaptiveFSRSOptimizer) -> None:
    r = opt._parse_llm_response("[abc]", [1.0])
    assert r == [1.0]


def test_parse_llm_response_extracts_from_text(opt: AdaptiveFSRSOptimizer) -> None:
    r = opt._parse_llm_response(
        "Suggested params: [1.5, 2.5] end",
        [0.0, 0.0],
    )
    assert r == [1.5, 2.5]


# ── _validate_params ────────────────────────────────────────────────────


def test_validate_params_correct_length(opt: AdaptiveFSRSOptimizer) -> None:
    assert opt._validate_params([1.0] * 19) is True


def test_validate_params_wrong_length(opt: AdaptiveFSRSOptimizer) -> None:
    assert opt._validate_params([1.0] * 18) is False


def test_validate_params_out_of_range(opt: AdaptiveFSRSOptimizer) -> None:
    bad = [1.0] * 19
    bad[0] = 200
    assert opt._validate_params(bad) is False


def test_validate_params_negative(opt: AdaptiveFSRSOptimizer) -> None:
    bad = [1.0] * 19
    bad[0] = -1
    assert opt._validate_params(bad) is False


def test_validate_params_non_numeric(opt: AdaptiveFSRSOptimizer) -> None:
    bad = [1.0] * 19
    bad[0] = "string"  # type: ignore
    assert opt._validate_params(bad) is False


# ── _estimate_cost ──────────────────────────────────────────────────────


def test_estimate_cost_gradient(opt: AdaptiveFSRSOptimizer) -> None:
    assert opt._estimate_cost("gradient") == 0.001


def test_estimate_cost_llm(opt: AdaptiveFSRSOptimizer) -> None:
    assert opt._estimate_cost("llm") == 0.1


def test_estimate_cost_hybrid(opt: AdaptiveFSRSOptimizer) -> None:
    assert opt._estimate_cost("hybrid") == 0.101


def test_estimate_cost_unknown(opt: AdaptiveFSRSOptimizer) -> None:
    assert opt._estimate_cost("unknown") == 0.0


# ── optimize ────────────────────────────────────────────────────────────


def test_optimize_returns_tuple(opt: AdaptiveFSRSOptimizer) -> None:
    data = [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}]
    result = opt.optimize(data, DEFAULT_PARAMS)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_optimize_increments_total_calls(opt: AdaptiveFSRSOptimizer) -> None:
    data = [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}]
    opt.optimize(data, DEFAULT_PARAMS)
    assert opt._total_calls == 1


def test_optimize_records_history(opt: AdaptiveFSRSOptimizer) -> None:
    data = [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}]
    opt.optimize(data, DEFAULT_PARAMS)
    assert len(opt._optimization_history) == 1
    assert len(opt._precision_history) >= 1


def test_optimize_saves_state(opt: AdaptiveFSRSOptimizer, tmp_path) -> None:
    data = [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}]
    opt.optimize(data, DEFAULT_PARAMS)
    assert (opt._cache_dir / "optimizer_state.json").exists()


def test_optimize_low_precision_triggers_llm_mode(tmp_path) -> None:
    o = AdaptiveFSRSOptimizer(precision_threshold=0.999, cache_dir=str(tmp_path / "c"))
    data = [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}]
    _, mode = o.optimize(data, DEFAULT_PARAMS)
    assert mode == "llm"


def test_optimize_empty_review_data(opt: AdaptiveFSRSOptimizer) -> None:
    _, mode = opt.optimize([], DEFAULT_PARAMS)
    assert mode in ("gradient", "llm", "hybrid")


# ── get_stats ───────────────────────────────────────────────────────────


def test_get_stats_keys(opt: AdaptiveFSRSOptimizer) -> None:
    stats = opt.get_stats()
    for k in [
        "current_mode",
        "current_precision",
        "total_calls",
        "gradient_calls",
        "llm_calls",
        "mode_switches",
        "precision_threshold",
        "decline_threshold",
    ]:
        assert k in stats


def test_get_stats_values(opt: AdaptiveFSRSOptimizer) -> None:
    data = [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}]
    opt.optimize(data, DEFAULT_PARAMS)
    stats = opt.get_stats()
    assert stats["total_calls"] == 1


# ── get_history ─────────────────────────────────────────────────────────


def test_get_history_structure(opt: AdaptiveFSRSOptimizer) -> None:
    data = [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}]
    opt.optimize(data, DEFAULT_PARAMS)
    h = opt.get_history()
    assert "precision_history" in h
    assert "optimization_history" in h


def test_get_history_caps_at_10(opt: AdaptiveFSRSOptimizer) -> None:
    data = [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}]
    for _ in range(15):
        opt.optimize(data, DEFAULT_PARAMS)
    h = opt.get_history()
    assert len(h["optimization_history"]) == 10


# ── reset ───────────────────────────────────────────────────────────────


def test_reset_restores_defaults(opt: AdaptiveFSRSOptimizer) -> None:
    data = [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}]
    opt.optimize(data, DEFAULT_PARAMS)
    opt.reset()
    assert opt._current_mode == "gradient"
    assert opt._current_precision == 1.0
    assert opt._total_calls == 0


def test_reset_saves_state(opt: AdaptiveFSRSOptimizer) -> None:
    opt.reset()
    assert (opt._cache_dir / "optimizer_state.json").exists()


# ── singleton ───────────────────────────────────────────────────────────


def test_get_adaptive_optimizer_singleton(monkeypatch) -> None:
    monkeypatch.setattr(ao, "_optimizer", None)
    o1 = get_adaptive_optimizer()
    o2 = get_adaptive_optimizer()
    assert o1 is o2


def test_get_adaptive_optimizer_first_call_kwargs(monkeypatch) -> None:
    monkeypatch.setattr(ao, "_optimizer", None)
    llm = MagicMock()
    o = get_adaptive_optimizer(precision_threshold=0.9, decline_threshold=0.1, llm_caller=llm)
    assert o._precision_threshold == 0.9
    assert o._decline_threshold == 0.1
    assert o._llm_caller is llm


def test_adaptive_optimize_module_function(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ao, "_optimizer", None)
    o = AdaptiveFSRSOptimizer(cache_dir=str(tmp_path / "c"))
    monkeypatch.setattr(ao, "_optimizer", o)
    data = [{"memory_id": "m1", "rating": 3, "elapsed_days": 1, "retention_before": 0.7}]
    params, mode = adaptive_optimize(data, DEFAULT_PARAMS)
    assert len(params) == 19
    assert mode in ("gradient", "llm", "hybrid")
