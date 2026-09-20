"""Tests for governance.visualizer — MemoryVisualizer HTML chart generation."""

from __future__ import annotations

import os

from governance import visualizer as vz
from governance.visualizer import (
    MemoryVisualizer,
    generate_dashboard,
    get_visualizer,
)

# ── constructor ─────────────────────────────────────────────────────────


def test_default_output_dir() -> None:
    v = MemoryVisualizer()
    assert v._output_dir == "/tmp/omnimem_charts"


def test_explicit_output_dir(tmp_path) -> None:
    v = MemoryVisualizer(output_dir=str(tmp_path))
    assert v._output_dir == str(tmp_path)


# ── generate_heat_distribution_chart ────────────────────────────────────


def test_heat_chart_contains_labels() -> None:
    v = MemoryVisualizer()
    html = v.generate_heat_distribution_chart({"hot": 5, "warm": 3, "neutral": 2, "cold": 1})
    assert "热度分布" in html
    assert "Hot" in html
    assert "Warm" in html
    assert "Neutral" in html
    assert "Cold" in html


def test_heat_chart_contains_values() -> None:
    v = MemoryVisualizer()
    html = v.generate_heat_distribution_chart({"hot": 5, "warm": 3, "neutral": 2, "cold": 1})
    assert "5" in html
    assert "3" in html


def test_heat_chart_missing_keys_default_zero() -> None:
    v = MemoryVisualizer()
    html = v.generate_heat_distribution_chart({})
    assert "Hot: 0" in html


def test_heat_chart_percentages() -> None:
    v = MemoryVisualizer()
    html = v.generate_heat_distribution_chart({"hot": 50, "warm": 50})
    assert "50.0%" in html


# ── generate_strength_distribution_chart ────────────────────────────────


def test_strength_chart_contains_grades() -> None:
    v = MemoryVisualizer()
    html = v.generate_strength_distribution_chart({"S": 1, "A": 2, "B": 3, "C": 4, "D": 5})
    assert "记忆强度等级分布" in html
    for g in ["S", "A", "B", "C", "D"]:
        assert g in html


def test_strength_chart_bar_width_relative() -> None:
    v = MemoryVisualizer()
    html = v.generate_strength_distribution_chart({"S": 10, "A": 5, "B": 0, "C": 0, "D": 0})
    # Max is 10 → S=100%, A=50%
    assert "width: 100.0%" in html
    assert "width: 50.0%" in html


def test_strength_chart_all_zero() -> None:
    v = MemoryVisualizer()
    html = v.generate_strength_distribution_chart({})
    assert "width: 0%" in html or "width: 0.0%" in html


# ── generate_retention_distribution_chart ───────────────────────────────


def test_retention_chart_contains_labels() -> None:
    v = MemoryVisualizer()
    html = v.generate_retention_distribution_chart({"high": 10, "medium": 5, "low": 2})
    assert "保持率分布" in html
    assert "高 (>80%)" in html
    assert "中 (50-80%)" in html
    assert "低 (<50%)" in html


def test_retention_chart_values() -> None:
    v = MemoryVisualizer()
    html = v.generate_retention_distribution_chart({"high": 10, "medium": 5, "low": 2})
    assert "10" in html
    assert "5" in html
    assert "2" in html


# ── _generate_stage_bars ────────────────────────────────────────────────


def test_stage_bars_all_stages() -> None:
    v = MemoryVisualizer()
    html = v._generate_stage_bars({"active": 10, "consolidating": 5, "archived": 3, "forgotten": 2})
    for stage in ["active", "consolidating", "archived", "forgotten"]:
        assert stage in html


def test_stage_bars_empty_defaults_one() -> None:
    v = MemoryVisualizer()
    html = v._generate_stage_bars({})
    # total = sum([]) or 1 = 1 → all 0%
    assert "active: 0" in html


def test_stage_bars_percentages() -> None:
    v = MemoryVisualizer()
    html = v._generate_stage_bars({"active": 50, "consolidating": 50})
    assert "50.0%" in html


def test_stage_bars_unknown_stage_color() -> None:
    v = MemoryVisualizer()
    html = v._generate_stage_bars({"weird": 5})
    # Unknown stage not in iteration list → not rendered
    assert "weird" not in html


# ── _generate_pie_chart ─────────────────────────────────────────────────


def test_pie_chart_title() -> None:
    v = MemoryVisualizer()
    html = v._generate_pie_chart("Test Title", ["A", "B"], [1, 2], ["#fff", "#000"])
    assert "Test Title" in html


def test_pie_chart_total_zero_guard() -> None:
    v = MemoryVisualizer()
    html = v._generate_pie_chart("T", ["A"], [0], ["#fff"])
    # total = sum([0]) or 1 = 1 → 0/1 = 0%
    assert "0.0%" in html


def test_pie_chart_percentages() -> None:
    v = MemoryVisualizer()
    html = v._generate_pie_chart("T", ["A", "B"], [25, 75], ["#fff", "#000"])
    assert "25.0%" in html
    assert "75.0%" in html


# ── _generate_bar_chart ─────────────────────────────────────────────────


def test_bar_chart_title() -> None:
    v = MemoryVisualizer()
    html = v._generate_bar_chart("Bar Title", ["X", "Y"], [5, 10], ["#f00", "#0f0"])
    assert "Bar Title" in html


def test_bar_chart_max_val_scaling() -> None:
    v = MemoryVisualizer()
    html = v._generate_bar_chart("T", ["A", "B"], [5, 10], ["#fff", "#000"])
    # max=10 → A=50%, B=100%
    assert "width: 50.0%" in html
    assert "width: 100.0%" in html


def test_bar_chart_empty_values() -> None:
    v = MemoryVisualizer()
    html = v._generate_bar_chart("T", [], [], [])
    assert "T" in html


def test_bar_chart_all_zero_max_guard() -> None:
    v = MemoryVisualizer()
    html = v._generate_bar_chart("T", ["A"], [0], ["#fff"])
    # max_val = 0 → width = 0
    assert "width: 0%" in html or "width: 0.0%" in html


# ── generate_dashboard ──────────────────────────────────────────────────


def test_dashboard_html_structure() -> None:
    v = MemoryVisualizer()
    stats = {
        "total_memories": 100,
        "heat": {"hot": 10, "warm": 20, "neutral": 30, "cold": 40},
        "grades": {"S": 5, "A": 10, "B": 20, "C": 30, "D": 35},
        "retention": {"high": 50, "medium": 30, "low": 20},
        "fsrs": {"avg_retention": 0.75, "avg_stability": 12.5},
        "upgrade_candidates": 3,
        "recent_24h": 10,
        "recent_7d": 50,
        "recent_30d": 80,
        "stages": {"active": 40, "consolidating": 30, "archived": 20, "forgotten": 10},
    }
    html = v.generate_dashboard(stats)
    assert "<!DOCTYPE html>" in html
    assert "OmniMem 记忆系统仪表盘" in html
    assert "100" in html
    assert "75.0%" in html  # avg_retention formatted
    assert "12.5 天" in html


def test_dashboard_empty_stats() -> None:
    v = MemoryVisualizer()
    html = v.generate_dashboard({})
    assert "<!DOCTYPE html>" in html
    assert "0" in html


def test_dashboard_missing_fsrs_defaults() -> None:
    v = MemoryVisualizer()
    html = v.generate_dashboard({"total_memories": 5})
    # avg_retention defaults to 0 → "0.0%"
    assert "0.0%" in html


def test_dashboard_contains_all_cards() -> None:
    v = MemoryVisualizer()
    html = v.generate_dashboard({})
    assert "📊 总览" in html
    assert "🔥 热度分布" in html
    assert "💪 记忆强度" in html
    assert "📈 保持率分布" in html
    assert "⏰ 最近活动" in html
    assert "🎯 阶段分布" in html


# ── save_dashboard ──────────────────────────────────────────────────────


def test_save_dashboard_creates_file(tmp_path) -> None:
    v = MemoryVisualizer(output_dir=str(tmp_path))
    path = v.save_dashboard({"total_memories": 10})
    assert os.path.exists(path)
    assert path.endswith("dashboard.html")


def test_save_dashboard_custom_filename(tmp_path) -> None:
    v = MemoryVisualizer(output_dir=str(tmp_path))
    path = v.save_dashboard({}, filename="custom.html")
    assert path.endswith("custom.html")
    assert os.path.exists(path)


def test_save_dashboard_custom_output_dir(tmp_path) -> None:
    v = MemoryVisualizer()
    custom = tmp_path / "custom"
    path = v.save_dashboard({}, output_dir=str(custom))
    assert str(custom) in path
    assert os.path.exists(path)


def test_save_dashboard_creates_dir(tmp_path) -> None:
    v = MemoryVisualizer()
    nested = tmp_path / "a" / "b" / "c"
    path = v.save_dashboard({}, output_dir=str(nested))
    assert os.path.exists(path)


def test_save_dashboard_content_matches_generate(tmp_path) -> None:
    v = MemoryVisualizer(output_dir=str(tmp_path))
    stats = {"total_memories": 42}
    path = v.save_dashboard(stats)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    assert "42" in content
    assert "<!DOCTYPE html>" in content


# ── get_visualizer singleton ────────────────────────────────────────────


def test_get_visualizer_singleton(monkeypatch) -> None:
    monkeypatch.setattr(vz, "_visualizer", None)
    v1 = get_visualizer()
    v2 = get_visualizer()
    assert v1 is v2
    assert isinstance(v1, MemoryVisualizer)


def test_get_visualizer_first_call_with_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(vz, "_visualizer", None)
    v = get_visualizer(output_dir=str(tmp_path))
    assert v._output_dir == str(tmp_path)


# ── module-level generate_dashboard ─────────────────────────────────────


def test_module_generate_dashboard(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(vz, "_visualizer", None)
    path = generate_dashboard({"total_memories": 7}, output_dir=str(tmp_path))
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        assert "7" in f.read()


def test_module_generate_dashboard_uses_singleton(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(vz, "_visualizer", None)
    p1 = generate_dashboard({}, output_dir=str(tmp_path))
    p2 = generate_dashboard({}, output_dir=str(tmp_path))
    # Same singleton → same default filename → same path
    assert p1 == p2
