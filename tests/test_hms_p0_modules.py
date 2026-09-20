"""Session-local 算子 + Verifier 验证器单元测试（HMS P0 补充）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnimem.retrieval.session_local import SessionLocalOperator, _gaussian_weight, _parse_time
from omnimem.retrieval.verifier import EvidenceVerifier

# ── Session-local 算子 ──────────────────────────────────

class TestGaussianWeight:
    def test_zero_delta_max_weight(self):
        assert _gaussian_weight(0.0) == pytest.approx(1.0)

    def test_large_delta_low_weight(self):
        # 100 小时 >> sigma 24h → 权重趋近 0
        assert _gaussian_weight(100.0) < 0.05

    def test_one_sigma_medium(self):
        # Δt = sigma → exp(-0.5) ≈ 0.6065
        assert _gaussian_weight(24.0) == pytest.approx(0.6065, abs=0.01)

    def test_negative_delta_abs(self):
        assert _gaussian_weight(-12.0) == pytest.approx(_gaussian_weight(12.0))


class TestParseTime:
    def test_iso_format(self):
        t = _parse_time("2026-05-06T16:37:19.569827+00:00")
        assert t is not None and t > 1e9

    def test_none_returns_none(self):
        assert _parse_time(None) is None

    def test_invalid_string(self):
        assert _parse_time("not-a-time") is None


class TestSessionLocalOperator:
    def _mk(self, score, stored_at):
        return {"score": score, "content": "x", "stored_at": stored_at}

    def test_disabled_returns_copy(self):
        op = SessionLocalOperator(enabled=False)
        results = [self._mk(0.9, "2026-05-06T16:00:00+00:00")]
        out = op.enhance(results)
        assert out == results

    def test_single_result_no_change(self):
        op = SessionLocalOperator()
        results = [self._mk(0.9, "2026-05-06T16:00:00+00:00")]
        out = op.enhance(results)
        assert len(out) == 1

    def test_proximity_boost_higher_for_nearby(self):
        """时间邻近的记忆应获得更高提升。"""
        op = SessionLocalOperator(sigma_hours=24.0)
        base_t = "2026-05-06T12:00:00+00:00"
        results = [
            self._mk(0.8, base_t),          # 入口点（高分）
            self._mk(0.5, "2026-05-06T13:00:00+00:00"),  # 1 小时后 → 高提升
            self._mk(0.5, "2026-05-01T13:00:00+00:00"),  # 5 天前 → 低提升
        ]
        out = op.enhance(results)
        # 近邻提升量 > 远邻提升量
        boost_near = out[1]["_session_local_boost"] if out[1]["content"] == "x" else 1.0
        # 找两条候选的提升值
        boosts = {r["stored_at"]: r.get("_session_local_boost", 1.0) for r in out if "_session_local_boost" in r}
        near = boosts["2026-05-06T13:00:00+00:00"]
        far = boosts["2026-05-01T13:00:00+00:00"]
        assert near > far

    def test_no_entry_point_no_boost(self):
        """无高分入口点时跳过增强。"""
        op = SessionLocalOperator()
        results = [
            self._mk(0.2, "2026-05-06T12:00:00+00:00"),
            self._mk(0.1, "2026-05-06T13:00:00+00:00"),
        ]
        out = op.enhance(results)
        # 无入口点（都低于 0.6 阈值）→ 不增强
        for r in out:
            assert "_session_local_boost" not in r


# ── Verifier 验证器 ────────────────────────────────────

class TestEvidenceVerifier:
    def test_empty_results_insufficient(self):
        v = EvidenceVerifier()
        report = v.verify("上周做了什么", [], intent="temporal")
        assert not report.adequate
        assert report.verdict == "insufficient"
        assert report.missing_slots

    def test_general_query_with_results_adequate(self):
        v = EvidenceVerifier()
        results = [
            {"content": "记忆系统架构包括 Planner 和 Verifier"},
            {"content": "检索通道有向量、BM25、图谱"},
            {"content": "融合使用 RRF 算法"},
        ]
        report = v.verify("记忆系统架构", results, intent="general")
        assert report.adequate
        assert report.verdict == "adequate"

    def test_temporal_query_missing_anchor(self):
        v = EvidenceVerifier()
        results = [{"content": "处理了三个问题"}, {"content": "讨论了方案"}, {"content": "review 了代码"}]
        # 查询有时间词但结果无时间锚点 → 时间槽未填充，但内容槽已填
        report = v.verify("上周处理了几个问题", results, intent="temporal")
        # 期望：内容匹配（几个/处理）+ 至少一条 → coverage 应 >= 0.4
        assert report.coverage >= 0.4

    def test_disabled_always_adequate(self):
        v = EvidenceVerifier(enabled=False)
        report = v.verify("任意查询", [], intent="general")
        assert report.adequate

    def test_report_to_dict_shape(self):
        v = EvidenceVerifier()
        report = v.verify("完全无关的问题xyzabc", [], intent="general")
        d = report.to_dict()
        assert "filled_slots" in d and "coverage" in d and "verdict" in d


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
