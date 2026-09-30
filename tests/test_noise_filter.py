"""写入侧噪声过滤测试（2026-09-29 噪声审计）。

审计数字：`index.db` 694 条里 112 条是框架噪声——
  - 74 条 `确认: [IMPORTANT: You are running as a scheduled cron job…`（cron 前言被 `REINFORCED:` 改名后落库，type=preference）
  - 21 条标点/超短载荷残句（`偏好: ）]\"`）
  - 17 条标签套娃链（`纠正: 偏好: ）]\"`）
本文件钉住两道闸：框架前言不入库、无实义载荷不入库，并确认短小但真实的事实不被误杀。

只用 mock store，绝不写 `~/.hermes`。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from omnimem.core.store_service import MemoryStoreService
from omnimem.perception.engine import (
    PerceptionEngine,
    PerceptionSignals,
    has_rememberable_payload,
    is_framework_boilerplate,
)

CRON_PREAMBLE = (
    "[IMPORTANT: You are running as a scheduled cron job. "
    "Your task is to review the conversation history and improve skills.]"
)


@pytest.fixture
def service():
    store = MagicMock()
    store.add.return_value = "mem-001"
    perception = MagicMock()
    perception._extract_core_fact.return_value = "核心事实内容足够长"
    provenance = MagicMock()
    provenance.track.return_value = {"source": "test", "method": "test"}
    return MemoryStoreService(
        store=store, perception=perception, provenance=provenance, session_id="s-test"
    ), store


class TestPayloadPredicate:
    @pytest.mark.parametrize(
        "content",
        [
            '偏好: ）]"\']',
            "确认: ）]",
            "纠正: 偏好: ）]",
            "姓名: ",
            "",
            "   ",
        ],
    )
    def test_rejects_noise(self, content):
        assert has_rememberable_payload(content) is False

    @pytest.mark.parametrize(
        "content",
        [
            "称呼偏好: 老板",
            "姓名: 徐",
            "偏好: 深色主题",
            "记住周报每周五发",
            "查 nan",  # 短但是真实内容
        ],
    )
    def test_accepts_real_facts(self, content):
        assert has_rememberable_payload(content) is True

    def test_boilerplate_detected(self):
        assert is_framework_boilerplate(CRON_PREAMBLE) is True
        assert is_framework_boilerplate("帮我看看 cron 脚本的报错") is False


class TestExtractionKeepsPayloadSubstantive:
    def test_punctuation_only_payload_is_not_returned_as_fact(self):
        engine = PerceptionEngine()
        fact = engine._extract_core_fact('我喜欢 ）]"\']')
        assert "偏好: ）]" not in fact
        assert has_rememberable_payload(fact) is True

    def test_normal_preference_still_extracts(self):
        engine = PerceptionEngine()
        assert engine._extract_core_fact("我喜欢深色主题，别用亮色") == "偏好: 深色主题"


class TestStoreServiceGate:
    def test_reinforcement_of_cron_preamble_is_dropped(self, service):
        svc, store = service
        signals = PerceptionSignals(
            has_reinforcement=True, reinforcement_target=CRON_PREAMBLE
        )
        assert svc.store_reinforcement(signals, f"REINFORCED: {CRON_PREAMBLE}") is None
        store.add.assert_not_called()

    def test_reinforcement_with_real_content_is_stored(self, service):
        svc, store = service
        signals = PerceptionSignals(
            has_reinforcement=True, reinforcement_target="以后都用中文回复"
        )
        assert svc.store_reinforcement(signals, "对，以后都用中文回复") == "mem-001"
        store.add.assert_called_once()

    def test_correction_with_punctuation_payload_is_dropped(self, service):
        svc, store = service
        signals = PerceptionSignals(has_correction=True, correction_target='）]"\'')
        assert svc.store_correction(signals, '不对，应该是 ）]"\'') is None
        store.add.assert_not_called()

    def test_fact_with_punctuation_payload_is_dropped(self, service):
        svc, store = service
        signals = PerceptionSignals(should_memorize=True, fact_content='偏好: ）]"')
        assert svc.store_fact(signals, '我喜欢 ）]"') is None
        store.add.assert_not_called()

    def test_fact_with_real_content_is_stored(self, service):
        svc, store = service
        signals = PerceptionSignals(should_memorize=True, fact_content="用户住在上海")
        assert svc.store_fact(signals, "我住在上海") == "mem-001"
        store.add.assert_called_once()

    def test_fallback_extraction_result_is_also_gated(self, service):
        """target 为空时走 perception 兜底，兜底产物同样要过闸。"""
        svc, store = service
        svc._perception._extract_core_fact.return_value = '确认: ）]"'
        signals = PerceptionSignals(has_reinforcement=True, reinforcement_target="")
        assert svc.store_reinforcement(signals, "很好") is None
        store.add.assert_not_called()
