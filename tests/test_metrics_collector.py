"""utils/metrics.py 单元测试：Counter/Histogram/Gauge/Collector/AlertManager。"""

from __future__ import annotations

import pytest
from omnimem.utils import metrics as metrics_mod
from omnimem.utils.metrics import (
    AlertManager,
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    _format_labels,
    _format_le,
    _merge_labels,
)

# ─── 标签辅助 ────────────────────────────────────────────


def test_format_labels_none_and_empty():
    assert _format_labels(None) == ""
    assert _format_labels({}) == ""


def test_format_labels_sorted_and_escaped():
    out = _format_labels({"zeta": "1", "alpha": 'q"uote'})
    assert out == '{alpha="q\\"uote",zeta="1"}'


def test_format_labels_escapes_backslash_and_newline():
    out = _format_labels({"k": "a\\b\nc"})
    assert out == '{k="a\\\\b\\nc"}'


def test_merge_labels_extra_wins():
    merged = _merge_labels({"a": 1, "b": 2}, {"b": 3, "c": 4})
    assert merged == {"a": 1, "b": 3, "c": 4}


def test_merge_labels_none_base():
    assert _merge_labels(None, {"x": 1}) == {"x": 1}


# ─── Counter ─────────────────────────────────────────────


def test_counter_inc_and_get():
    c = Counter("test_counter", "desc")
    c.inc()
    c.inc(2)
    assert c.get() == 3


def test_counter_rejects_negative_inc():
    c = Counter("test_counter_neg", "desc")
    with pytest.raises(ValueError):
        c.inc(-1)


def test_counter_label_scoping():
    c = Counter("test_labeled", "desc", labels=["type"])
    c.inc(type="a")
    c.inc(2, type="b")
    assert c.get(type="a") == 1
    assert c.get(type="b") == 2
    assert c.get() == 0  # 未指定标签 → 空签名 0


def test_counter_collect_empty_and_data():
    c = Counter("c_empty", "desc")
    text = c.collect()
    assert "# HELP c_empty desc" in text
    assert "# TYPE c_empty counter" in text
    assert "c_empty 0" in text

    c.inc(5)
    text = c.collect()
    assert "c_empty 5.0" in text


def test_counter_collect_labeled_sorted():
    c = Counter("c_l", "desc", labels=["k"])
    c.inc(1, k="beta")
    c.inc(2, k="alpha")
    text = c.collect()
    # 按签名排序：alpha 应在 beta 之前
    alpha_pos = text.index('c_l{k="alpha"}')
    beta_pos = text.index('c_l{k="beta"}')
    assert alpha_pos < beta_pos


# ─── Histogram ──────────────────────────────────────────


def test_format_le_int_and_float():
    assert _format_le(1.0) == "1"
    assert _format_le(0.25) == "0.25"


def test_histogram_observe_updates_cumulative_buckets():
    h = Histogram("h1", "desc", buckets=[0.1, 0.5, 1.0])
    h.observe(0.05)  # 落在 <=0.1, <=0.5, <=1.0
    h.observe(0.7)   # 落在 <=1.0
    # 累计计数：<=0.1 只有 0.05 一次；<=0.5 也只 0.05；<=1.0 有 2 次
    text = h.collect()
    assert 'h1_bucket{le="0.1"} 1' in text
    assert 'h1_bucket{le="0.5"} 1' in text
    assert 'h1_bucket{le="1"} 2' in text
    assert 'h1_bucket{le="+Inf"} 2' in text
    assert "h1_sum 0.75" in text
    assert "h1_count 2" in text


def test_histogram_collect_empty_emits_zero_buckets():
    h = Histogram("h_empty", "desc", buckets=[1, 2])
    text = h.collect()
    assert "# TYPE h_empty histogram" in text
    assert 'h_empty_bucket{le="1"} 0' in text
    assert 'h_empty_bucket{le="2"} 0' in text
    assert 'h_empty_bucket{le="+Inf"} 0' in text
    assert "h_empty_sum 0" in text
    assert "h_empty_count 0" in text


def test_histogram_labeled_default_sorting():
    h = Histogram("h_lab", "desc", buckets=[1, 10], labels=["op"])
    h.observe(0.5, op="write")
    h.observe(5.0, op="read")
    text = h.collect()
    # read 应在 write 之前（sorted by key）
    assert text.index('op="read"') < text.index('op="write"')


# ─── Gauge ───────────────────────────────────────────────


def test_gauge_set_inc_dec_get():
    g = Gauge("g1", "desc")
    g.set(10)
    assert g.get() == 10
    g.inc(3)
    assert g.get() == 13
    g.dec(5)
    assert g.get() == 8


def test_gauge_label_scoping_and_collect_sorted():
    g = Gauge("g_lab", "desc", labels=["host"])
    g.set(1, host="zeta")
    g.set(2, host="alpha")
    text = g.collect()
    assert text.index('g_lab{host="alpha"} 2.0') < text.index('g_lab{host="zeta"} 1.0')


def test_gauge_collect_empty_emits_zero():
    g = Gauge("g_empty", "desc")
    text = g.collect()
    assert "# TYPE g_empty gauge" in text
    assert "g_empty 0" in text


# ─── MetricsCollector ────────────────────────────────────


def test_collector_register_idempotent_and_get():
    c = MetricsCollector()
    cnt = Counter("m_a", "desc")
    c.register(cnt)
    c.register(Counter("m_a", "duplicate"))  # 同名应被忽略
    assert c.get_metric("m_a") is cnt
    assert c.get_metric("unknown") is None


def test_collector_collect_all_concatenates():
    c = MetricsCollector()
    c.register(Counter("m_x", "x"))
    c.register(Gauge("m_y", "y"))
    text = c.collect_all()
    assert "# TYPE m_x counter" in text
    assert "# TYPE m_y gauge" in text


def test_collector_collect_all_empty_returns_blank():
    assert MetricsCollector().collect_all() == ""


def test_collector_format_labels_static():
    assert MetricsCollector.format_labels({"k": "v"}) == '{k="v"}'
    assert MetricsCollector.format_labels(None) == ""


# ─── AlertManager ────────────────────────────────────────


def test_alert_manager_fire_broadcasts_to_handlers():
    mgr = AlertManager()
    seen: list[dict] = []
    mgr.register_handler(seen.append)
    mgr.register_handler(seen.append)
    mgr.fire("alarm", "warning", "hello", extra=1)
    assert len(seen) == 2
    assert seen[0]["name"] == "alarm"
    assert seen[0]["severity"] == "warning"
    assert seen[0]["context"] == {"extra": 1}
    assert isinstance(seen[0]["timestamp"], float)


def test_alert_manager_swallows_handler_exception():
    mgr = AlertManager()

    def _bad(_alert):
        raise RuntimeError("boom")

    mgr.register_handler(_bad)
    mgr.fire("x", "info", "y")  # 不应抛
    assert len(mgr.get_active_alerts()) == 1


def test_alert_manager_clear():
    mgr = AlertManager()
    mgr.fire("a", "info", "m")
    mgr.clear()
    assert mgr.get_active_alerts() == []


# ─── 模块级便捷函数（使用 monkeypatch 避免污染全局） ──────


def test_record_cache_hit_updates_ratio_and_counters(monkeypatch):
    hits = Counter("h_hits", "")
    misses = Counter("h_misses", "")
    ratio = Gauge("h_ratio", "")
    monkeypatch.setattr(metrics_mod, "cache_hits_total", hits)
    monkeypatch.setattr(metrics_mod, "cache_misses_total", misses)
    monkeypatch.setattr(metrics_mod, "cache_hit_ratio", ratio)

    metrics_mod.record_cache_hit()
    metrics_mod.record_cache_hit()
    metrics_mod.record_cache_miss()
    # 2 hits / 3 total = 0.666...
    assert hits.get() == 2
    assert misses.get() == 1
    assert ratio.get() == pytest.approx(2 / 3)


def test_update_cache_hit_ratio_zero_total_sets_zero(monkeypatch):
    hits = Counter("r0h", "")
    misses = Counter("r0m", "")
    ratio = Gauge("r0g", "")
    monkeypatch.setattr(metrics_mod, "cache_hits_total", hits)
    monkeypatch.setattr(metrics_mod, "cache_misses_total", misses)
    monkeypatch.setattr(metrics_mod, "cache_hit_ratio", ratio)

    metrics_mod.update_cache_hit_ratio()
    assert ratio.get() == 0.0


@pytest.mark.parametrize(
    "raw,expected",
    [("closed", 0.0), ("half_open", 1.0), ("open", 2.0), ("CLOSED", 0.0), ("OPEN", 2.0), ("unknown", 0.0)],
)
def test_set_circuit_breaker_state_maps_strings(monkeypatch, raw, expected):
    gauge = Gauge("cb_state", "")
    monkeypatch.setattr(metrics_mod, "circuit_breaker_state", gauge)
    metrics_mod.set_circuit_breaker_state(raw)
    if raw == "unknown":
        # 未识别字符串 → float("unknown") 抛异常被静默吞掉，gauge 保持默认 0
        assert gauge.get() == 0.0
    else:
        assert gauge.get() == expected


def test_record_llm_call_and_error_uses_labels(monkeypatch):
    calls = Counter("llm_c", "", labels=["type"])
    errors = Counter("llm_e", "", labels=["type"])
    monkeypatch.setattr(metrics_mod, "llm_calls_total", calls)
    monkeypatch.setattr(metrics_mod, "llm_errors_total", errors)
    metrics_mod.record_llm_call("reflect")
    metrics_mod.record_llm_error("reflect")
    assert calls.get(type="reflect") == 1
    assert errors.get(type="reflect") == 1


def test_active_connections_inc_dec(monkeypatch):
    g = Gauge("ac", "")
    monkeypatch.setattr(metrics_mod, "active_connections", g)
    metrics_mod.inc_active_connections()
    metrics_mod.inc_active_connections()
    metrics_mod.dec_active_connections()
    assert g.get() == 1


def test_record_duration_helpers_populate_histogram(monkeypatch):
    h = Histogram("dur", "", buckets=[0.01, 0.1, 1.0])
    monkeypatch.setattr(metrics_mod, "recall_duration_seconds", h)
    metrics_mod.record_recall_duration(0.05)
    text = h.collect()
    assert 'dur_bucket{le="0.1"} 1' in text

    h2 = Histogram("mdur", "", buckets=[0.01, 0.1, 1.0])
    monkeypatch.setattr(metrics_mod, "memorize_duration_seconds", h2)
    metrics_mod.record_memorize_duration(0.5)
    assert "mdur_count 1" in h2.collect()

    h3 = Histogram("rdur", "", buckets=[0.01, 0.1, 1.0])
    monkeypatch.setattr(metrics_mod, "reflect_duration_seconds", h3)
    metrics_mod.record_reflect_duration(2.0)
    assert "rdur_count 1" in h3.collect()


def test_saga_helpers(monkeypatch):
    pending = Gauge("sp", "")
    dead = Counter("sd", "")
    monkeypatch.setattr(metrics_mod, "saga_pending_count", pending)
    monkeypatch.setattr(metrics_mod, "saga_dead_letters_total", dead)
    metrics_mod.set_saga_pending_count(7)
    metrics_mod.record_saga_dead_letter()
    assert pending.get() == 7
    assert dead.get() == 1


def test_get_collector_and_alert_manager_return_singletons():
    assert metrics_mod.get_metrics_collector() is metrics_mod._collector
    assert metrics_mod.get_alert_manager() is metrics_mod._alert_manager
