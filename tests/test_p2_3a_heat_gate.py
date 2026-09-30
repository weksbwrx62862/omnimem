"""★ P2-3a：召回热度门控（无关查询不得给捞到的记忆加热）。

阈值 0.12 不是拍的：在生产数据副本上探针实测，相关/无关两类查询的融合分数
分布几乎完全重叠（relevant p25/50/75 = 0.102/0.112/0.130，
irrelevant = 0.077/0.100/0.120），纯分数阈值分不开，必须与词法重叠联用。
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import omnimem.provider as provider_module
from omnimem.provider import OmniMemProvider
from omnimem.services.recall_service import RecallService


def _payload(*items: tuple[str, float, str]) -> str:
    return json.dumps(
        {
            "status": "found",
            "memories": [
                {"memory_id": mid, "score": score, "content": content}
                for mid, score, content in items
            ],
        },
        ensure_ascii=False,
    )


class _Stub:
    """只提供 _handle_recall 用到的属性，绕开完整 initialize。"""

    def __init__(self, payload: str, config: dict | None = None):
        self._recall_payload = payload
        self._config = config or {}
        self._forgetting = MagicMock()
        self._feedback = None
        forgetting_conn = MagicMock()
        self._forgetting._conn = forgetting_conn
        self._forgetting._pending_writes = 3

    def _fake_recall_impl(self, _provider, _args):
        return self._recall_payload


def _heat(payload: str, query: str, config: dict | None = None) -> list[str]:
    """驱动热度门控的唯一负责点 —— ``RecallService._heat_hits``。

    ★ P2-3a 去重后 provider 不再自己 record_access，门控断言必须打在服务层，
      否则测的是「provider 不加热」而不是「门控挑对了行」。
    """
    memories = json.loads(payload).get("memories", [])
    deps = SimpleNamespace(config=config or {}, forgetting=MagicMock())
    service = RecallService(deps=deps)
    heated = service._heat_hits(memories, query)
    called = [c.args[0] for c in service.deps.forgetting.record_access.call_args_list]
    assert called == heated, "返回的加热列表必须与遗忘曲线实际收到的 record_access 一致"
    return heated


class TestHeatGate:
    def test_high_score_hit_is_heated(self):
        heated = _heat(_payload(("m1", 0.20, "完全无关的内容")), "机票")
        assert heated == ["m1"]

    def test_low_score_without_overlap_is_not_heated(self):
        """这就是报告里的缺陷：无关查询把捞到的每条记忆都加热了。"""
        heated = _heat(_payload(("m1", 0.08, "A股进出场规则回测")), "帮我订一张机票")
        assert heated == []

    def test_low_score_with_keyword_overlap_is_heated(self):
        heated = _heat(_payload(("m1", 0.05, "缩量涨停的样本复核结论")), "涨停样本 缩量 复核")
        assert heated == ["m1"]

    def test_english_keyword_overlap_is_heated(self):
        heated = _heat(_payload(("m1", 0.03, "kubectl rollout restart deployment")),
                       "kubectl restart")

        assert heated == ["m1"]

    def test_mixed_payload_heats_only_qualifying_rows(self):
        payload = _payload(
            ("good", 0.19, "无重叠内容"),
            ("overlap", 0.04, "SQLite WAL 并发写失败已修复"),
            ("noise", 0.06, "涨停样本复核"),
        )
        heated = _heat(payload, "SQLite WAL")
        assert heated == ["good", "overlap"]

    def test_threshold_is_configurable_down_to_legacy_behavior(self):
        """heat_min_score=0 → 分数门永远通过，退回旧的全量加热语义（可回滚）。"""
        heated = _heat(_payload(("m1", 0.0, "内容")), "机票", config={"heat_min_score": 0.0})
        assert heated == ["m1"]

    def test_threshold_is_configurable_upward(self):
        heated = _heat(_payload(("m1", 0.5, "无重叠")), "机票", config={"heat_min_score": 0.6})
        assert heated == []

    def test_missing_score_field_is_treated_as_zero(self):
        payload = json.dumps({"status": "found", "memories": [{"memory_id": "m1", "content": "xyz"}]})
        assert _heat(payload, "一个不会命中的查询词") == []

    def test_no_memories_id_skipped(self):
        payload = json.dumps({"status": "found", "memories": [{"score": 1.0, "content": "x"}]})
        assert _heat(payload, "查询") == []

    def test_provider_recall_path_no_longer_heats(self, monkeypatch):
        """★ 去重本身：provider._handle_recall 不得再对命中调 record_access。"""
        stub = _Stub(_payload(("m1", 0.30, "内容")))
        monkeypatch.setattr(provider_module, "_handle_recall_impl", stub._fake_recall_impl)
        OmniMemProvider._handle_recall(stub, {"query": "查询"})

        stub._forgetting.record_access.assert_not_called()

    def test_forgetting_commit_still_runs_after_gated_heat(self, monkeypatch):
        stub = _Stub(_payload(("m1", 0.30, "内容")))
        monkeypatch.setattr(provider_module, "_handle_recall_impl", stub._fake_recall_impl)
        OmniMemProvider._handle_recall(stub, {"query": "查询"})

        assert stub._forgetting._conn.commit.called
        assert stub._forgetting._pending_writes == 0

    def test_status_not_found_records_nothing(self, monkeypatch):
        stub = _Stub(json.dumps({"status": "no_results", "memories": []}))
        monkeypatch.setattr(provider_module, "_handle_recall_impl", stub._fake_recall_impl)
        OmniMemProvider._handle_recall(stub, {"query": "查询"})

        stub._forgetting.record_access.assert_not_called()


class TestConfigKey:
    def test_heat_min_score_declared_above_rrf_floor(self):
        from omnimem.config._config import _CONFIG_SCHEMA

        heat = _CONFIG_SCHEMA["heat_min_score"]
        rrf = _CONFIG_SCHEMA["rrf_min_score"]
        assert heat["default"] > rrf["default"], "热度门槛必须严格高于融合召回下限，否则等于没门控"

    def test_default_value_matches_calibration(self):
        from omnimem.config._config import _CONFIG_SCHEMA

        assert _CONFIG_SCHEMA["heat_min_score"]["default"] == pytest.approx(0.12)


class TestRecursionLimitPainkillerRemoved:
    """★ P2-1：provider.py 里的 sys.setrecursionlimit(5000) 是止痛贴，必须移除。"""

    def test_no_active_call_left_in_source(self):
        source = open(provider_module.__file__, encoding="utf-8").read()
        code_lines = [
            line.strip() for line in source.splitlines() if not line.strip().startswith("#")
        ]
        assert not any("setrecursionlimit(" in line for line in code_lines)

    def test_default_recursion_limit_unchanged_after_import(self):
        assert sys.getrecursionlimit() == 1000
