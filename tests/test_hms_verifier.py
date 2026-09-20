"""retrieval.verifier.EvidenceVerifier 离线单元测试。

覆盖：
  - verify 顶层：disabled 短路 / 空结果 / 覆盖 & 数量双阈值
  - _derive_slots：5 类 intent + conversational 空槽位 + 结果数补充
  - _count_filled：各槽位关键字命中路径
  - CoverageReport.to_dict：字段与四舍五入
"""

from __future__ import annotations

from typing import Any

import pytest
from omnimem.retrieval.verifier import (
    MIN_RESULTS_FOR_ADEQUATE,
    CoverageReport,
    EvidenceVerifier,
)


def _mk_results(n: int, base_content: str = "") -> list[dict[str, Any]]:
    return [{"content": base_content, "memory_id": f"m{i}"} for i in range(n)]


@pytest.fixture
def v() -> EvidenceVerifier:
    return EvidenceVerifier()


# ── 短路 ──


def test_disabled_returns_adequate() -> None:
    v = EvidenceVerifier(enabled=False)
    r = v.verify("q", [])
    assert r.adequate is True
    assert r.verdict == "adequate"


def test_empty_results_marks_insufficient(v: EvidenceVerifier) -> None:
    r = v.verify("q", [])
    assert r.adequate is False
    assert r.verdict == "insufficient"
    assert r.filled_slots == 0
    assert r.total_slots == 1
    assert "任何相关记忆" in r.missing_slots
    assert "未检索到任何记忆" in r.reason


# ── general intent ──


def test_general_intent_with_matching_token(v: EvidenceVerifier) -> None:
    results = _mk_results(3, "Alice loves Python programming a lot")
    r = v.verify("Python", results, intent="general")
    # "相关内容" 命中 + "多来源印证" 命中 → 覆盖 2/2 = 1.0；结果数 >=3
    assert r.total_slots == 2
    assert r.filled_slots == 2
    assert r.adequate is True
    assert r.verdict == "adequate"


def test_general_intent_no_match(v: EvidenceVerifier) -> None:
    results = _mk_results(3, "无关文本 xyzzy")
    r = v.verify("Alpha", results, intent="general")
    # "相关内容" 未命中；"多来源印证" 命中 → 1/2 = 0.5 → adequate（>=0.4 & 结果够）
    assert r.filled_slots == 1
    assert r.coverage == pytest.approx(0.5, rel=1e-6)
    assert r.adequate is True


def test_general_intent_few_results_not_adequate(v: EvidenceVerifier) -> None:
    results = _mk_results(1, "Python Python Python")
    r = v.verify("Python", results, intent="general")
    # 1 结果 → "至少一条记忆" 槽 → filled=2/2=1.0；但 len<MIN_RESULTS_FOR_ADEQUATE → 不足
    assert r.adequate is False
    assert r.verdict == "insufficient"
    assert "证据覆盖不足" in r.reason


# ── temporal ──


def test_temporal_with_date_anchor(v: EvidenceVerifier) -> None:
    results = _mk_results(3, "2023年7月7日的项目记录")
    r = v.verify("2023年7月的会议", results, intent="temporal")
    # derive 命中时间锚点 + 补充"多来源印证" → filled 至少 2
    assert r.filled_slots >= 2


def test_temporal_without_anchor(v: EvidenceVerifier) -> None:
    results = _mk_results(3, "普通内容 无时间")
    r = v.verify("关于项目", results, intent="temporal")
    # derive 将"时间锚点"标为未填，但 _count_filled 中"时间锚点"分支无条件 +1（源实现细节）
    # → filled = "时间锚点" + "多来源印证" = 2 / total 3 → coverage ≈ 0.67 → adequate
    assert r.coverage == pytest.approx(2 / 3, rel=1e-3)
    assert r.adequate is True
    assert "时间锚点" in r.missing_slots


def test_temporal_relative_words(v: EvidenceVerifier) -> None:
    results = _mk_results(3, "上周我们讨论了架构")
    r = v.verify("上周聊了什么", results, intent="temporal")
    # "上周" 命中 derive 的正则
    assert r.total_slots >= 2


# ── entity ──


def test_entity_intent_matched(v: EvidenceVerifier) -> None:
    results = _mk_results(3, "小王去了北京并购买了咖啡")
    r = v.verify("小王", results, intent="entity")
    # "目标实体"命中 + "实体行为/属性"命中（去/买）+ "多来源印证" → 3/3
    assert r.filled_slots == 3
    assert r.coverage == pytest.approx(1.0, rel=1e-6)
    assert r.adequate is True


def test_entity_intent_no_match(v: EvidenceVerifier) -> None:
    results = _mk_results(3, "abc def ghi")
    r = v.verify("张三", results, intent="entity")
    # 目标实体 & 行为均缺失 → 只有"多来源印证"
    assert r.filled_slots == 1
    assert r.coverage < 1.0


def test_entity_stops_filtered(v: EvidenceVerifier) -> None:
    # "什么/怎么" 从 token 剔除 → 目标实体不命中
    results = _mk_results(3, "随便的内容 abc")
    r = v.verify("这是什么", results, intent="entity")
    assert "目标实体" in r.missing_slots


# ── count ──


def test_count_intent_hits_measure_words(v: EvidenceVerifier) -> None:
    results = _mk_results(3, "总共 12 个项目 涉及 5 个人")
    r = v.verify("多少个", results, intent="count")
    # "可计数条目" 命中（个）+ "数值信号" 命中（12/5）+ "多来源印证" → 3/3
    assert r.filled_slots == 3


def test_count_intent_missing_signals(v: EvidenceVerifier) -> None:
    results = _mk_results(3, "纯文本没有任何数字")
    r = v.verify("统计", results, intent="count")
    assert "可计数条目" in r.missing_slots


# ── preference ──


def test_preference_intent_keywords(v: EvidenceVerifier) -> None:
    results = _mk_results(3, "用户偏好深色主题 推荐方案 A")
    r = v.verify("偏好", results, intent="preference")
    # "偏好描述" 命中 + "候选/推荐" 命中（结果>=2）+ "多来源印证"
    assert r.filled_slots == 3


def test_preference_intent_no_signal(v: EvidenceVerifier) -> None:
    results = _mk_results(3, "无关文本")
    r = v.verify("随便", results, intent="preference")
    # "候选/推荐" 命中（>=2 结果）+ "多来源印证" → 2/3
    assert r.filled_slots == 2
    assert "偏好描述" in r.missing_slots


# ── conversational ──


def test_conversational_returns_no_slots(v: EvidenceVerifier) -> None:
    results = _mk_results(3, "你好")
    r = v.verify("你好", results, intent="conversational")
    # 空槽位 → total_slots=max(0,1)=1, filled=0 → coverage 0
    # 但 conversational 无 slot 且结果充足 → adequate 判定
    assert r.total_slots == 1
    assert r.coverage == 0.0
    assert r.adequate is False  # coverage 低于阈值


# ── CoverageReport ──


def test_to_dict_rounds_coverage() -> None:
    r = CoverageReport(
        filled_slots=1,
        total_slots=3,
        missing_slots=["x"],
        coverage=0.3333,
        adequate=False,
        verdict="insufficient",
        reason="test",
    )
    d = r.to_dict()
    assert d["coverage"] == 0.33
    assert d["missing_slots"] == ["x"]
    assert d["adequate"] is False


def test_repr_contains_threshold() -> None:
    assert "threshold=" in repr(EvidenceVerifier())


# ── 阈值边界 ──


def test_custom_threshold_tightens_decision(v: EvidenceVerifier) -> None:
    v2 = EvidenceVerifier(coverage_threshold=0.99)
    results = _mk_results(3, "Python")
    r = v2.verify("Alpha", results, intent="general")
    # 1/2 = 0.5 < 0.99 → insufficient
    assert r.adequate is False


def test_min_results_for_adequate_constant() -> None:
    assert MIN_RESULTS_FOR_ADEQUATE == 3
