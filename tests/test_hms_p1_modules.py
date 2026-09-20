"""P1 模块单元测试：canonical_entity / organizer / contradiction / temporal_separation。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnimem.retrieval.canonical_entity import CanonicalEntityResolver, normalize_entity
from omnimem.retrieval.contradiction import ContradictionDetector
from omnimem.retrieval.organizer import EvidenceOrganizer
from omnimem.retrieval.temporal_separation import (
    annotate_results,
    parse_occurrence_time,
    sort_by_occurrence,
)

# ── Canonical Entity Resolver ───────────────────────────

class TestCanonicalEntity:
    def test_register_and_resolve(self):
        r = CanonicalEntityResolver()
        r.register_group("王明", ["小王", "王总", "王经理"])
        assert r.resolve("小王") == "王明"
        assert r.resolve("王总") == "王明"
        assert r.resolve("王经理") == "王明"
        assert r.resolve("王明") == "王明"

    def test_unregistered_returns_rule_normalized(self):
        r = CanonicalEntityResolver()
        assert r.resolve("李经理") == "李"  # 称谓剥离
        assert r.resolve("普通词") == "普通词"  # 无称谓不变

    def test_normalize_entities_dedup(self):
        r = CanonicalEntityResolver()
        r.register_group("王明", ["小王", "王总"])
        out = r.normalize_entities(["小王", "王总", "王明", "王明"])
        assert out == ["王明"]

    def test_save_load_roundtrip(self, tmp_path):
        p = str(tmp_path / "canonical.json")
        r = CanonicalEntityResolver()
        r.register_group("王明", ["小王"])
        r.save(p)
        r2 = CanonicalEntityResolver(p)
        assert r2.resolve("小王") == "王明"

    def test_normalize_entity_global(self):
        # 全局入口不崩溃
        assert isinstance(normalize_entity("测试"), str)


# ── Evidence Organizer ──────────────────────────────────

class TestEvidenceOrganizer:
    def test_gate_signals(self):
        o = EvidenceOrganizer()
        assert o.should_organize("上周处理了几个问题")     # 计数
        assert o.should_organize("按时间排序看进展")       # 排序
        assert o.should_organize("A 和 B 有什么区别")      # 比较
        assert not o.should_organize("Langfuse 是什么")    # 普通查询

    def test_organize_dedup(self):
        o = EvidenceOrganizer()
        results = [
            {"content": "部署了 Langfuse 自托管服务，使用 Docker Compose 六个容器", "score": 0.9, "stored_at": "2026-08-07T10:00:00+00:00", "wing": "project"},
            {"content": "部署了 Langfuse 自托管服务，使用 Docker Compose 六个容器", "score": 0.6, "stored_at": "2026-08-07T10:05:00+00:00", "wing": "project"},
            {"content": "看门狗 v2 配置为静默模式", "score": 0.5, "stored_at": "2026-08-06T09:00:00+00:00", "wing": "ops"},
        ]
        out = o.organize(results, force=True)
        assert len(out) == 2  # 重复被合并
        merged = [r for r in out if r.get("_merged_count", 1) > 1]
        assert len(merged) == 1
        assert merged[0]["_merged_count"] == 2

    def test_organize_source_label(self):
        o = EvidenceOrganizer()
        results = [
            {"content": "x", "score": 0.8, "wing": "project", "hall": "langfuse", "room": "部署"},
        ]
        out = o.organize(results, force=True)
        assert out[0]["_source_label"] == "project/langfuse/部署"


# ── Contradiction Detector ──────────────────────────────

class TestContradictionDetector:
    def test_numeric_conflict(self):
        c = ContradictionDetector()
        results = [
            {"content": "项目预算是 100 万", "score": 0.9, "stored_at": "2026-08-01T00:00:00+00:00"},
            {"content": "项目预算是 200 万", "score": 0.8, "stored_at": "2026-08-05T00:00:00+00:00"},
        ]
        conflicts = c.detect(results)
        assert len(conflicts) >= 1
        assert conflicts[0]["conflict_type"] == "numeric"

    def test_state_conflict(self):
        c = ContradictionDetector()
        results = [
            {"content": "功能已经完成", "score": 0.9},
            {"content": "功能尚未完成，还在开发", "score": 0.8},
        ]
        conflicts = c.detect(results)
        assert any(conf["conflict_type"] == "state" for conf in conflicts)

    def test_no_conflict(self):
        c = ContradictionDetector()
        results = [
            {"content": "部署了六个容器", "score": 0.9},
            {"content": "web 端口 3000", "score": 0.8},
        ]
        conflicts = c.detect(results)
        assert conflicts == []

    def test_too_few_results(self):
        c = ContradictionDetector(min_evidence=3)
        assert c.detect([{"content": "x"}]) == []


# ── Temporal Separation ─────────────────────────────────

class TestTemporalSeparation:
    def test_full_date(self):
        t = parse_occurrence_time("计划在2023年7月7日晚上进行")
        assert t is not None and t.startswith("2023-07-07")

    def test_month_day(self):
        t = parse_occurrence_time("7月7日晚上", "2023-12-01T00:00:00+00:00")
        assert t is not None and t.startswith("2023-07-07")

    def test_relative_yesterday(self):
        t = parse_occurrence_time("昨天去了游乐场", "2026-08-07T10:00:00+00:00")
        assert t is not None and t.startswith("2026-08-06")

    def test_no_time(self):
        assert parse_occurrence_time("随便聊聊") is None

    def test_annotate_and_sort(self):
        results = [
            {"content": "计划在2023年7月7日晚上", "stored_at": "2026-08-07T10:00:00+00:00"},
            {"content": "昨天处理了问题", "stored_at": "2026-08-07T11:00:00+00:00"},
            {"content": "无时间内容", "stored_at": "2026-08-07T12:00:00+00:00"},
        ]
        annotated = annotate_results(results)
        # 昨天 → 2026-08-06，早于 2023-07-07？不对——2023 更早
        sorted_r = sort_by_occurrence(annotated)
        assert sorted_r[0]["_occurrence_time"].startswith("2023-07-07")
        assert sorted_r[-1]["_occurrence_time"] is None  # 无时间的最后


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
