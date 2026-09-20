"""Tests for governance.personalized_fsrs — 个性化 FSRS 参数学习。"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pytest

from governance import personalized_fsrs as pf
from governance.personalized_fsrs import (
    ParameterHistory,
    PersonalizedFSRS,
    ReviewRecord,
    get_learner,
    learn_from_reviews,
)


@pytest.fixture
def param_file(tmp_path) -> str:
    return str(tmp_path / "personalized_params.json")


@pytest.fixture
def learner(param_file: str) -> PersonalizedFSRS:
    return PersonalizedFSRS(learning_rate=0.05, convergence_threshold=0.001, max_iterations=20, param_file=param_file)


# ── dataclasses ──────────────────────────────────────────────────────────


def test_review_record_defaults_timestamp() -> None:
    r = ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.6, retention_after=0.9)
    assert r.memory_id == "m1"
    assert isinstance(r.timestamp, datetime)


def test_parameter_history_construction() -> None:
    ts = datetime(2025, 1, 1)
    h = ParameterHistory(timestamp=ts, parameters=[1.0, 2.0], loss=0.05, sample_count=3)
    assert h.timestamp == ts
    assert h.parameters == [1.0, 2.0]
    assert h.loss == 0.05
    assert h.sample_count == 3


# ── constructor ──────────────────────────────────────────────────────────


def test_default_params_length_19(learner: PersonalizedFSRS) -> None:
    assert len(learner._default_params) == 19


def test_current_params_copy_of_default(learner: PersonalizedFSRS) -> None:
    assert learner._current_params == learner._default_params
    assert learner._current_params is not learner._default_params


def test_explicit_param_file_used(learner: PersonalizedFSRS, param_file: str) -> None:
    assert learner._param_file == param_file


def test_param_file_default_when_none() -> None:
    lrn = PersonalizedFSRS(param_file=None)
    assert "personalized_params.json" in lrn._param_file


# ── get_parameters / set_parameters ─────────────────────────────────────


def test_get_parameters_returns_copy(learner: PersonalizedFSRS) -> None:
    p = learner.get_parameters()
    assert p == learner._current_params
    assert p is not learner._current_params


def test_set_parameters_valid_length(learner: PersonalizedFSRS) -> None:
    new = [0.5] * len(learner._default_params)
    learner.set_parameters(new)
    assert learner._current_params == new


def test_set_parameters_wrong_length_warns(learner: PersonalizedFSRS, caplog) -> None:
    original = learner._current_params.copy()
    learner.set_parameters([0.5, 0.6])  # too short
    assert learner._current_params == original  # not replaced


# ── _load_params / _save_params ──────────────────────────────────────────


def test_save_and_load_roundtrip(tmp_path) -> None:
    path = str(tmp_path / "p.json")
    l1 = PersonalizedFSRS(param_file=path)
    l1.set_parameters([1.0] * 19)
    l1._save_params()

    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["parameters"] == [1.0] * 19
    assert "updated_at" in data
    assert data["history_count"] == 0

    l2 = PersonalizedFSRS(param_file=path)
    assert l2._current_params == [1.0] * 19


def test_load_missing_file_non_fatal(tmp_path) -> None:
    path = str(tmp_path / "nonexistent.json")
    lrn = PersonalizedFSRS(param_file=path)
    assert lrn._current_params == lrn._default_params  # fell back


def test_load_invalid_json_non_fatal(tmp_path, caplog) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json{", encoding="utf-8")
    lrn = PersonalizedFSRS(param_file=str(path))
    # Warning logged, defaults retained
    assert lrn._current_params == lrn._default_params


def test_load_no_parameters_key(tmp_path) -> None:
    path = tmp_path / "n.json"
    path.write_text('{"other": 1}', encoding="utf-8")
    lrn = PersonalizedFSRS(param_file=str(path))
    assert lrn._current_params == lrn._default_params


# ── calculate_loss ───────────────────────────────────────────────────────


def test_calculate_loss_empty_returns_zero(learner: PersonalizedFSRS) -> None:
    assert learner.calculate_loss([]) == 0.0


def test_calculate_loss_positive(learner: PersonalizedFSRS) -> None:
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.5, retention_after=0.9),
        ReviewRecord(memory_id="m2", rating=3, elapsed_days=10, retention_before=0.3, retention_after=0.9),
    ]
    loss = learner.calculate_loss(records)
    assert loss >= 0.0
    assert loss < 1.0


def test_calculate_loss_perfect_prediction_zero(learner: PersonalizedFSRS) -> None:
    predicted = learner._predict_retention(elapsed_days=5, rating=3)
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=predicted, retention_after=1.0),
    ]
    assert learner.calculate_loss(records) == pytest.approx(0.0, abs=1e-9)


# ── _predict_retention ──────────────────────────────────────────────────


def test_predict_retention_zero_days_is_one(learner: PersonalizedFSRS) -> None:
    r = learner._predict_retention(elapsed_days=0, rating=3)
    assert r == pytest.approx(1.0)


def test_predict_retention_clamped_0_to_1(learner: PersonalizedFSRS) -> None:
    r = learner._predict_retention(elapsed_days=10000, rating=1)
    assert 0.0 <= r <= 1.0


def test_predict_retention_invalid_rating_falls_to_default_stability(learner: PersonalizedFSRS) -> None:
    r = learner._predict_retention(elapsed_days=5, rating=99)
    assert 0.0 <= r <= 1.0


def test_predict_retention_zero_alpha_falls_back(learner: PersonalizedFSRS) -> None:
    params = learner._current_params.copy()
    params[17] = 0.0  # alpha
    learner.set_parameters(params)
    r = learner._predict_retention(elapsed_days=5, rating=3)
    assert 0.0 <= r <= 1.0


def test_predict_retention_monotone_decreasing(learner: PersonalizedFSRS) -> None:
    r1 = learner._predict_retention(elapsed_days=1, rating=3)
    r10 = learner._predict_retention(elapsed_days=10, rating=3)
    r100 = learner._predict_retention(elapsed_days=100, rating=3)
    assert r1 > r10 > r100


def test_predict_retention_zero_stability_returns_one(learner: PersonalizedFSRS) -> None:
    params = learner._current_params.copy()
    params[2] = 0.0  # stability for rating=3
    learner.set_parameters(params)
    r = learner._predict_retention(elapsed_days=5, rating=3)
    assert r == 1.0


# ── learn_from_reviews ──────────────────────────────────────────────────


def test_learn_from_reviews_empty_returns_no_data(learner: PersonalizedFSRS) -> None:
    result = learner.learn_from_reviews([])
    assert result == {"status": "no_data", "iterations": 0}


def test_learn_from_reviews_persists_history(learner: PersonalizedFSRS) -> None:
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.5, retention_after=0.9),
    ]
    result = learner.learn_from_reviews(records)
    assert result["status"] == "success"
    assert result["sample_count"] == 1
    assert len(learner._history) == 1


def test_learn_from_reviews_saves_file(learner: PersonalizedFSRS, param_file: str) -> None:
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.5, retention_after=0.9),
    ]
    learner.learn_from_reviews(records)
    assert os.path.exists(param_file)


def test_learn_from_reviews_improvement_nonneg(learner: PersonalizedFSRS) -> None:
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.5, retention_after=0.9),
        ReviewRecord(memory_id="m2", rating=4, elapsed_days=3, retention_before=0.7, retention_after=1.0),
    ]
    result = learner.learn_from_reviews(records)
    # final loss cannot exceed initial (we keep best_params)
    assert result["final_loss"] <= result["initial_loss"]


def test_learn_from_reviews_respects_max_iterations(learner: PersonalizedFSRS) -> None:
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.5, retention_after=0.9),
    ]
    result = learner.learn_from_reviews(records)
    assert result["iterations"] <= learner._max_iterations


# ── _calculate_gradients ────────────────────────────────────────────────


def test_calculate_gradients_same_length_as_params(learner: PersonalizedFSRS) -> None:
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.5, retention_after=0.9),
    ]
    grads = learner._calculate_gradients(records)
    assert len(grads) == len(learner._current_params)


def test_calculate_gradients_all_floats(learner: PersonalizedFSRS) -> None:
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.5, retention_after=0.9),
    ]
    grads = learner._calculate_gradients(records)
    assert all(isinstance(g, float) for g in grads)


def test_calculate_gradients_does_not_mutate_params(learner: PersonalizedFSRS) -> None:
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.5, retention_after=0.9),
    ]
    original = learner._current_params.copy()
    learner._calculate_gradients(records)
    assert learner._current_params == original


# ── get_history ─────────────────────────────────────────────────────────


def test_get_history_empty_by_default(learner: PersonalizedFSRS) -> None:
    assert learner.get_history() == []


def test_get_history_after_learn(learner: PersonalizedFSRS) -> None:
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.5, retention_after=0.9),
    ]
    learner.learn_from_reviews(records)
    h = learner.get_history()
    assert len(h) == 1
    assert "timestamp" in h[0]
    assert "loss" in h[0]
    assert "sample_count" in h[0]
    assert h[0]["sample_count"] == 1


# ── reset_to_default ────────────────────────────────────────────────────


def test_reset_to_default_restores_defaults(learner: PersonalizedFSRS) -> None:
    learner.set_parameters([2.0] * 19)
    learner.reset_to_default()
    assert learner._current_params == learner._default_params


def test_reset_to_default_saves_file(learner: PersonalizedFSRS, param_file: str) -> None:
    learner.reset_to_default()
    assert os.path.exists(param_file)
    with open(param_file) as f:
        data = json.load(f)
    assert data["parameters"] == learner._default_params


# ── compare_with_default ────────────────────────────────────────────────


def test_compare_with_default_keys(learner: PersonalizedFSRS) -> None:
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.5, retention_after=0.9),
    ]
    result = learner.compare_with_default(records)
    assert "current_loss" in result
    assert "default_loss" in result
    assert "improvement" in result
    assert "improvement_pct" in result


def test_compare_with_default_same_when_defaults(learner: PersonalizedFSRS) -> None:
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.5, retention_after=0.9),
    ]
    result = learner.compare_with_default(records)
    assert result["current_loss"] == pytest.approx(result["default_loss"])
    assert result["improvement"] == pytest.approx(0.0)
    assert result["improvement_pct"] == 0


def test_compare_with_default_zero_default_loss_pct(learner: PersonalizedFSRS) -> None:
    # Force default loss to 0 via perfect prediction
    predicted_default = learner._predict_retention(elapsed_days=5, rating=3)
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=predicted_default, retention_after=1.0),
    ]
    result = learner.compare_with_default(records)
    assert result["default_loss"] == pytest.approx(0.0)
    assert result["improvement_pct"] == 0  # guard: 0 default_loss → pct 0


def test_compare_with_default_preserves_current(learner: PersonalizedFSRS) -> None:
    learner.set_parameters([3.0] * 19)
    original = learner._current_params.copy()
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.5, retention_after=0.9),
    ]
    learner.compare_with_default(records)
    assert learner._current_params == original


# ── global singleton ────────────────────────────────────────────────────


def test_get_learner_singleton(monkeypatch) -> None:
    monkeypatch.setattr(pf, "_learner", None)
    l1 = get_learner()
    l2 = get_learner()
    assert l1 is l2
    assert isinstance(l1, PersonalizedFSRS)


def test_get_learner_first_call_constructs_with_args(monkeypatch) -> None:
    monkeypatch.setattr(pf, "_learner", None)
    lrn = get_learner(learning_rate=0.2, convergence_threshold=0.01)
    assert lrn._learning_rate == 0.2
    assert lrn._convergence_threshold == 0.01


def test_module_level_learn_from_reviews(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pf, "_learner", None)
    monkeypatch.setattr(pf.PersonalizedFSRS, "_load_params", lambda self: None)

    def _save_noop(self) -> None:
        return None

    monkeypatch.setattr(pf.PersonalizedFSRS, "_save_params", _save_noop)
    records = [
        ReviewRecord(memory_id="m1", rating=3, elapsed_days=5, retention_before=0.5, retention_after=0.9),
    ]
    result = learn_from_reviews(records)
    assert result["status"] == "success"
