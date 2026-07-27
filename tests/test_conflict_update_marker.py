"""Task 2 单元测试：ConflictResolver 知识更新标记。

测试覆盖：
  1. ConflictResult 默认字段值
  2. 知识更新冲突时 is_updated=True
  3. is_updated 记忆在检索排序中获得提升
  4. 非更新冲突不影响 is_updated 字段
"""

from __future__ import annotations

from omnimem.governance.conflict import ConflictResolver, ConflictResult
from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator


class TestConflictResultDefaults:
    """测试 1: ConflictResult 默认字段值。"""

    def test_default_has_conflict(self) -> None:
        result = ConflictResult()
        assert result.has_conflict is False

    def test_default_is_updated(self) -> None:
        result = ConflictResult()
        assert result.is_updated is False

    def test_default_is_superseded(self) -> None:
        result = ConflictResult()
        assert result.is_superseded is False

    def test_default_superseded_id(self) -> None:
        result = ConflictResult()
        assert result.superseded_id == ""


class TestUpdateConflictMarker:
    """测试 2: 知识更新冲突时 is_updated=True。"""

    def test_update_conflict_sets_markers(self) -> None:
        """conflict_type="update" 时 resolve 应设置 is_updated/is_superseded/superseded_id。"""
        resolver = ConflictResolver(strategy="latest")
        conflict = ConflictResult(
            has_conflict=True,
            existing_id="mem-old-001",
            conflict_type="update",
            action="accept",
        )
        result = resolver.resolve("新内容", conflict)
        assert result.is_updated is True
        assert result.is_superseded is True
        assert result.superseded_id == "mem-old-001"

    def test_update_conflict_with_different_strategies(self) -> None:
        """不同策略下 update 冲突都应设置标记。"""
        for strategy in ("latest", "confidence", "manual"):
            resolver = ConflictResolver(strategy=strategy)
            conflict = ConflictResult(
                has_conflict=True,
                existing_id="mem-old-002",
                conflict_type="update",
            )
            result = resolver.resolve("新内容", conflict)
            assert result.is_updated is True
            assert result.superseded_id == "mem-old-002"


class TestUpdatedBoostInRetrieval:
    """测试 3: is_updated 记忆在检索排序中获得提升。"""

    def test_is_updated_metadata_gets_boost(self) -> None:
        """metadata 中 is_updated=True 的记忆分数应被提升。"""
        results = [
            {
                "memory_id": "mem-a",
                "type": "fact",
                "score": 0.5,
                "metadata": {"is_updated": True},
            },
            {
                "memory_id": "mem-b",
                "type": "fact",
                "score": 0.5,
                "metadata": {},
            },
        ]
        boosted = HybridOrchestrator.apply_type_boost(results, updated_boost=0.3)
        # mem-a 的分数应为 0.5 * (1 + 0.3) = 0.65
        mem_a = next(r for r in boosted if r["memory_id"] == "mem-a")
        mem_b = next(r for r in boosted if r["memory_id"] == "mem-b")
        assert mem_a["score"] == 0.65
        assert mem_b["score"] == 0.5
        # mem-a 应排在 mem-b 前面
        assert boosted[0]["memory_id"] == "mem-a"

    def test_is_updated_top_level_gets_boost(self) -> None:
        """结果中直接带 is_updated=True（非 metadata 内）也应被提升。"""
        results = [
            {"memory_id": "mem-c", "type": "fact", "score": 0.4, "is_updated": True},
            {"memory_id": "mem-d", "type": "fact", "score": 0.4, "metadata": {}},
        ]
        boosted = HybridOrchestrator.apply_type_boost(results, updated_boost=0.3)
        mem_c = next(r for r in boosted if r["memory_id"] == "mem-c")
        assert mem_c["score"] == 0.52  # 0.4 * 1.3

    def test_updated_boost_configurable(self) -> None:
        """updated_boost 可配置。"""
        results = [
            {
                "memory_id": "mem-e",
                "type": "fact",
                "score": 0.5,
                "metadata": {"is_updated": True},
            },
        ]
        boosted = HybridOrchestrator.apply_type_boost(results, updated_boost=0.5)
        mem_e = next(r for r in boosted if r["memory_id"] == "mem-e")
        assert mem_e["score"] == 0.75  # 0.5 * 1.5

    def test_non_updated_no_boost(self) -> None:
        """没有 is_updated 标记的记忆不应获得提升。"""
        results = [
            {"memory_id": "mem-f", "type": "fact", "score": 0.5, "metadata": {}},
        ]
        boosted = HybridOrchestrator.apply_type_boost(results, updated_boost=0.3)
        mem_f = next(r for r in boosted if r["memory_id"] == "mem-f")
        assert mem_f["score"] == 0.5
        assert "updated_boost" not in mem_f


class TestNonUpdateConflictNoMarker:
    """测试 4: 非更新冲突不影响 is_updated 字段。"""

    def test_semantic_contradiction_no_update_marker(self) -> None:
        """conflict_type="semantic_contradiction" 时不设置 is_updated。"""
        resolver = ConflictResolver(strategy="latest")
        conflict = ConflictResult(
            has_conflict=True,
            existing_id="mem-old-003",
            conflict_type="semantic_contradiction",
        )
        result = resolver.resolve("新内容", conflict)
        assert result.is_updated is False
        assert result.is_superseded is False
        assert result.superseded_id == ""

    def test_negation_no_update_marker(self) -> None:
        """conflict_type="negation" 时不设置 is_updated。"""
        resolver = ConflictResolver(strategy="latest")
        conflict = ConflictResult(
            has_conflict=True,
            existing_id="mem-old-004",
            conflict_type="negation",
        )
        result = resolver.resolve("新内容", conflict)
        assert result.is_updated is False

    def test_duplicate_no_update_marker(self) -> None:
        """conflict_type="duplicate" 时不设置 is_updated。"""
        resolver = ConflictResolver(strategy="latest")
        conflict = ConflictResult(
            has_conflict=True,
            existing_id="mem-old-005",
            conflict_type="duplicate",
        )
        result = resolver.resolve("新内容", conflict)
        assert result.is_updated is False
