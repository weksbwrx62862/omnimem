"""P2 模块单元测试：self_healing（HMS Self-Evolution 整合）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnimem.retrieval.self_healing import (
    MODE_AMOUNT_CALIBRATION,
    MODE_COUNT_DEDUP,
    MODE_DUAL_SOURCE_GROUNDING,
    MODE_RELATIVE_DATE_GROUNDING,
    MODE_STATE_ARBITRATION,
    SelfHealingMonitor,
)


class TestSelfHealingMonitor:
    def _mk(self, content, score=0.8, **kw):
        r = {"content": content, "score": score}
        r.update(kw)
        return r

    def test_disabled_returns_unchanged(self):
        m = SelfHealingMonitor(enabled=False)
        results = [self._mk("x")]
        assert m.heal("查询", results) == results

    def test_relative_date_grounding(self):
        m = SelfHealingMonitor()
        results = [self._mk("处理了部署问题")]
        out = m.heal("十天前做了什么", results)
        healing = out[0].get("_healing", [])
        modes = [h["mode"] for h in healing]
        assert MODE_RELATIVE_DATE_GROUNDING in modes

    def test_count_dedup(self):
        m = SelfHealingMonitor()
        results = [self._mk("处理了三个问题", _merged_count=2)]
        out = m.heal("上周处理了几个问题", results)
        healing = out[0].get("_healing", [])
        modes = [h["mode"] for h in healing]
        assert MODE_COUNT_DEDUP in modes

    def test_state_arbitration(self):
        m = SelfHealingMonitor()
        results = [self._mk("功能已完成上线")]
        out = m.heal("这个功能计划还是已完成", results)
        healing = out[0].get("_healing", [])
        modes = [h["mode"] for h in healing]
        assert MODE_STATE_ARBITRATION in modes

    def test_amount_calibration_sufficient(self):
        m = SelfHealingMonitor()
        results = [self._mk("A 售价 100 万"), self._mk("B 售价 200 万")]
        out = m.heal("A 和 B 差多少", results)
        healing = out[0].get("_healing", [])
        amount = [h for h in healing if h["mode"] == MODE_AMOUNT_CALIBRATION]
        assert amount and "满足" in amount[0]["action"]

    def test_amount_calibration_insufficient(self):
        m = SelfHealingMonitor()
        results = [self._mk("A 售价 100 万")]
        out = m.heal("A 和 B 差多少", results)
        healing = out[0].get("_healing", [])
        amount = [h for h in healing if h["mode"] == MODE_AMOUNT_CALIBRATION]
        assert amount and "不满足" in amount[0]["action"]

    def test_dual_source_grounding(self):
        m = SelfHealingMonitor()
        results = [self._mk("部署完成", _source_label="project/langfuse")]
        out = m.heal("部署情况", results)
        healing = out[0].get("_healing", [])
        modes = [h["mode"] for h in healing]
        assert MODE_DUAL_SOURCE_GROUNDING in modes

    def test_stats(self):
        m = SelfHealingMonitor()
        m.heal("昨天做了什么", [self._mk("x")])
        stats = m.stats()
        assert MODE_RELATIVE_DATE_GROUNDING in stats
        assert stats[MODE_RELATIVE_DATE_GROUNDING] >= 1

    def test_no_healing_applied_when_no_signal(self):
        m = SelfHealingMonitor()
        results = [self._mk("普通内容")]
        out = m.heal("普通查询", results)
        assert "_healing" not in out[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
