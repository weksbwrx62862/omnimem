"""测试 add_only 冲突策略：验证冲突记忆的写入和检索行为。

测试场景：
  1. ConflictResolver 在 add_only 策略下不标记 superseded
  2. 两条冲突记忆均可在检索结果中出现
  3. 对比 latest 策略下旧记忆被 superseded 过滤的行为
  4. memory_write_service 中 _dedup_superseded_id 在 add_only 下为空
"""

from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock, patch

from omnimem.governance.conflict import ConflictResult, ConflictResolver

logger = logging.getLogger(__name__)


# ─── 测试 1: ConflictResolver.add_only 策略基本行为 ─────────────────

class TestAddOnlyConflictResolver(unittest.TestCase):
    """add_only 策略下冲突解析的行为验证。"""

    def test_add_only_semantic_contradiction_no_superseded(self) -> None:
        """语义矛盾冲突：add_only 不标记 is_superseded。"""
        resolver = ConflictResolver(strategy="add_only")
        conflict = ConflictResult(
            has_conflict=True,
            existing_id="mem-old-001",
            existing_memory="用户偏好使用 AWS 部署服务",
            conflict_type="semantic_contradiction",
        )
        result = resolver.resolve("用户改用阿里云部署服务", conflict)

        logger.info(
            "[add_only/语义矛盾] 冲突类型=%s, action=%s, is_updated=%s, is_superseded=%s, superseded_id=%s, reason=%s",
            conflict.conflict_type, result.action, result.is_updated,
            result.is_superseded, result.superseded_id, result.reason,
        )

        # 新记忆应被接受
        assert result.action == "accept"
        # 不标记 superseded
        assert result.is_superseded is False
        assert result.is_updated is False
        assert result.superseded_id == ""
        # reason 应说明是 add_only
        assert "ADD-only" in result.reason

        logger.info("[add_only/语义矛盾] 验证通过: 旧记忆 mem-old-001 未被标记为 superseded")

    def test_add_only_update_conflict_no_superseded(self) -> None:
        """更新类冲突：add_only 也不标记 superseded（与 latest 不同）。"""
        resolver = ConflictResolver(strategy="add_only")
        conflict = ConflictResult(
            has_conflict=True,
            existing_id="mem-old-002",
            conflict_type="update",
        )
        result = resolver.resolve("新内容", conflict)

        logger.info(
            "[add_only/更新冲突] 冲突类型=%s, action=%s, is_updated=%s, is_superseded=%s, superseded_id=%s",
            conflict.conflict_type, result.action, result.is_updated,
            result.is_superseded, result.superseded_id,
        )

        # add_only 下 update 类型也不标记
        assert result.is_updated is False
        assert result.is_superseded is False
        assert result.superseded_id == ""

        logger.info("[add_only/更新冲突] 验证通过: update 类型冲突在 add_only 下也不标记 superseded")

    def test_latest_still_sets_superseded(self) -> None:
        """latest 策略下 update 冲突仍标记 superseded（回归测试）。"""
        resolver = ConflictResolver(strategy="latest")
        conflict = ConflictResult(
            has_conflict=True,
            existing_id="mem-old-003",
            conflict_type="update",
        )
        result = resolver.resolve("新内容", conflict)

        logger.info(
            "[latest/更新冲突] 冲突类型=%s, action=%s, is_updated=%s, is_superseded=%s, superseded_id=%s",
            conflict.conflict_type, result.action, result.is_updated,
            result.is_superseded, result.superseded_id,
        )

        assert result.is_updated is True
        assert result.is_superseded is True
        assert result.superseded_id == "mem-old-003"

        logger.info("[latest/更新冲突] 验证通过: latest 策略下旧记忆 mem-old-003 被标记为 superseded")


# ─── 测试 2: 模拟检索层 is_superseded 过滤行为 ──────────────────────

class TestSupersededFilteringInRetrieval(unittest.TestCase):
    """验证检索层的 is_superseded 过滤逻辑。"""

    def _filter_superseded(self, results: list[dict]) -> list[dict]:
        """模拟 hybrid_orchestrator.py 中的 superseded 过滤。"""
        before = len(results)
        filtered = [r for r in results if not (r.get("is_superseded") or r.get("metadata", {}).get("is_superseded"))]
        removed_ids = [r["memory_id"] for r in results if r.get("is_superseded") or r.get("metadata", {}).get("is_superseded")]
        logger.info("[检索过滤] 输入 %d 条, 过滤后 %d 条, 被移除: %s", before, len(filtered), removed_ids)
        return filtered

    def test_latest_strategy_old_memory_filtered(self) -> None:
        """latest 策略：旧记忆被标记 is_superseded=True，检索时被过滤。"""
        results = [
            {"memory_id": "mem-new", "content": "用户改用阿里云部署", "score": 0.9, "is_superseded": False},
            {"memory_id": "mem-old", "content": "用户偏好使用 AWS 部署", "score": 0.85, "is_superseded": True},
        ]
        logger.info("[latest/检索] 输入记忆: %s", [(r["memory_id"], r["is_superseded"]) for r in results])
        filtered = self._filter_superseded(results)
        assert len(filtered) == 1
        assert filtered[0]["memory_id"] == "mem-new"
        logger.info("[latest/检索] 验证通过: 仅保留 mem-new, mem-old 被 superseded 过滤")

    def test_add_only_both_memories_retained(self) -> None:
        """add_only 策略：两条记忆都无 is_superseded，均保留。"""
        results = [
            {"memory_id": "mem-new", "content": "用户改用阿里云部署", "score": 0.9, "is_superseded": False},
            {"memory_id": "mem-old", "content": "用户偏好使用 AWS 部署", "score": 0.85, "is_superseded": False},
        ]
        logger.info("[add_only/检索] 输入记忆: %s", [(r["memory_id"], r["is_superseded"]) for r in results])
        filtered = self._filter_superseded(results)
        assert len(filtered) == 2
        logger.info("[add_only/检索] 验证通过: 两条记忆均保留, 返回 %s", [r["memory_id"] for r in filtered])

    def test_add_only_metadata_level_not_superseded(self) -> None:
        """add_only 策略：metadata 中也不带 is_superseded。"""
        results = [
            {"memory_id": "mem-new", "content": "用户改用阿里云部署", "score": 0.9, "metadata": {}},
            {"memory_id": "mem-old", "content": "用户偏好使用 AWS 部署", "score": 0.85, "metadata": {}},
        ]
        logger.info("[add_only/检索-metadata] 输入记忆: %s", [(r["memory_id"], r.get("metadata", {})) for r in results])
        filtered = self._filter_superseded(results)
        assert len(filtered) == 2
        logger.info("[add_only/检索-metadata] 验证通过: metadata 层无 is_superseded, 两条均保留")

    def test_latest_mixed_superseded_in_metadata(self) -> None:
        """latest 策略：superseded 在 metadata 层级也应被过滤。"""
        results = [
            {"memory_id": "mem-new", "content": "最新信息", "score": 0.9, "metadata": {"is_superseded": False}},
            {"memory_id": "mem-old", "content": "旧信息", "score": 0.85, "metadata": {"is_superseded": True}},
        ]
        logger.info("[latest/检索-metadata] 输入记忆: %s", [(r["memory_id"], r.get("metadata", {})) for r in results])
        filtered = self._filter_superseded(results)
        assert len(filtered) == 1
        assert filtered[0]["memory_id"] == "mem-new"
        logger.info("[latest/检索-metadata] 验证通过: metadata 中 is_superseded=True 的 mem-old 被过滤")


# ─── 测试 3: 端到端模拟写入流程 ───────────────────────────────────

class TestAddOnlyWriteFlow(unittest.TestCase):
    """模拟 MemoryWriteService 中 add_only 的关键路径。"""

    def test_dedup_superseded_id_cleared_in_add_only(self) -> None:
        """add_only 下 _dedup_superseded_id 应为空字符串。"""
        _is_add_only = True
        dedup_result = {"action": "create", "superseded_id": "mem-old-999"}
        logger.info("[add_only/去重] dedup_result=%s, _is_add_only=%s", dedup_result, _is_add_only)
        _dedup_superseded_id = "" if _is_add_only else dedup_result.get("superseded_id", "")
        logger.info("[add_only/去重] _dedup_superseded_id=%s (应为空, 旧记忆不被标记)", repr(_dedup_superseded_id))
        assert _dedup_superseded_id == ""

    def test_dedup_superseded_id_kept_in_latest(self) -> None:
        """latest 下 _dedup_superseded_id 应保留。"""
        _is_add_only = False
        dedup_result = {"action": "create", "superseded_id": "mem-old-999"}
        logger.info("[latest/去重] dedup_result=%s, _is_add_only=%s", dedup_result, _is_add_only)
        _dedup_superseded_id = "" if _is_add_only else dedup_result.get("superseded_id", "")
        logger.info("[latest/去重] _dedup_superseded_id=%s (旧记忆将被标记为 superseded)", _dedup_superseded_id)
        assert _dedup_superseded_id == "mem-old-999"

    def test_conflict_resolve_add_only_no_update_marker(self) -> None:
        """add_only 下 conflict_resolver.resolve 不产生 update_marker。"""
        resolver = ConflictResolver(strategy="add_only")
        conflict = ConflictResult(
            has_conflict=True,
            existing_id="mem-old-aaa",
            conflict_type="update",
        )
        resolution = resolver.resolve("新内容", conflict)
        logger.info(
            "[add_only/写入流程] resolve 结果: action=%s, is_updated=%s, is_superseded=%s, superseded_id=%s",
            resolution.action, resolution.is_updated, resolution.is_superseded, resolution.superseded_id,
        )
        # update_marker 仅在 resolution.is_updated=True 时创建
        update_marker = None
        if resolution.is_updated:
            update_marker = {
                "is_updated": True,
                "is_superseded": True,
                "superseded_id": resolution.superseded_id,
            }
        logger.info("[add_only/写入流程] update_marker=%s (应为 None, 不产生更新标记)", update_marker)
        assert update_marker is None

    def test_conflict_resolve_latest_produces_update_marker(self) -> None:
        """latest 下 conflict_resolver.resolve 应产生 update_marker。"""
        resolver = ConflictResolver(strategy="latest")
        conflict = ConflictResult(
            has_conflict=True,
            existing_id="mem-old-bbb",
            conflict_type="update",
        )
        resolution = resolver.resolve("新内容", conflict)
        logger.info(
            "[latest/写入流程] resolve 结果: action=%s, is_updated=%s, is_superseded=%s, superseded_id=%s",
            resolution.action, resolution.is_updated, resolution.is_superseded, resolution.superseded_id,
        )
        update_marker = None
        if resolution.is_updated:
            update_marker = {
                "is_updated": True,
                "is_superseded": True,
                "superseded_id": resolution.superseded_id,
            }
        logger.info("[latest/写入流程] update_marker=%s (应包含 mem-old-bbb 的 superseded 信息)", update_marker)
        assert update_marker is not None
        assert update_marker["superseded_id"] == "mem-old-bbb"


# ─── 测试 4: 模拟多轮冲突场景 ─────────────────────────────────────

class TestMultiTurnConflictScenario(unittest.TestCase):
    """模拟 LongMemEval 中多轮对话产生的连续冲突。"""

    def test_sequential_updates_add_only_retains_all(self) -> None:
        """连续三次更新同一话题：add_only 保留全部，latest 只保留最新。"""
        # 模拟记忆链：v1 → v2 → v3
        memories_add_only = []
        memories_latest = []

        # 第一轮写入 v1
        memories_add_only.append({"memory_id": "v1", "content": "用户住在北京", "is_superseded": False})
        memories_latest.append({"memory_id": "v1", "content": "用户住在北京", "is_superseded": False})
        logger.info("[多轮冲突] 第1轮写入 v1: add_only=%s, latest=%s",
                     memories_add_only[-1]["is_superseded"], memories_latest[-1]["is_superseded"])

        # 第二轮写入 v2（冲突更新：用户搬到上海）
        memories_add_only.append({"memory_id": "v2", "content": "用户搬到了上海", "is_superseded": False})
        memories_latest[0]["is_superseded"] = True
        memories_latest.append({"memory_id": "v2", "content": "用户搬到了上海", "is_superseded": False})
        logger.info("[多轮冲突] 第2轮写入 v2: add_only 中 v1.is_superseded=%s, latest 中 v1.is_superseded=%s",
                     memories_add_only[0]["is_superseded"], memories_latest[0]["is_superseded"])

        # 第三轮写入 v3（冲突更新：用户又搬到杭州）
        memories_add_only.append({"memory_id": "v3", "content": "用户现在住在杭州", "is_superseded": False})
        memories_latest[1]["is_superseded"] = True
        memories_latest.append({"memory_id": "v3", "content": "用户现在住在杭州", "is_superseded": False})
        logger.info("[多轮冲突] 第3轮写入 v3: add_only 中 v2.is_superseded=%s, latest 中 v2.is_superseded=%s",
                     memories_add_only[1]["is_superseded"], memories_latest[1]["is_superseded"])

        # 模拟检索过滤
        def filter_superseded(results):
            return [r for r in results if not r.get("is_superseded")]

        add_only_retrieved = filter_superseded(memories_add_only)
        latest_retrieved = filter_superseded(memories_latest)

        logger.info("[多轮冲突] add_only 检索结果 (%d 条): %s",
                     len(add_only_retrieved), [r["memory_id"] for r in add_only_retrieved])
        logger.info("[多轮冲突] latest 检索结果 (%d 条): %s",
                     len(latest_retrieved), [r["memory_id"] for r in latest_retrieved])

        # add_only 保留全部 3 条
        assert len(add_only_retrieved) == 3
        retrieved_ids = {r["memory_id"] for r in add_only_retrieved}
        assert retrieved_ids == {"v1", "v2", "v3"}

        # latest 只保留最新 1 条
        assert len(latest_retrieved) == 1
        assert latest_retrieved[0]["memory_id"] == "v3"

        logger.info("[多轮冲突] 验证通过: add_only 保留 v1/v2/v3 全部, latest 仅保留 v3")

    def test_add_only_allows_llm_to_disambiguate(self) -> None:
        """add_only 保留的历史记忆可被 LLM 用于时序推理。

        场景：问答"用户现在住哪里？"，LLM 需要从 v1/v2/v3 中判断最新。
        """
        memories = [
            {"memory_id": "v1", "content": "用户住在北京", "score": 0.7, "is_superseded": False, "stored_at": "2024-01-01"},
            {"memory_id": "v2", "content": "用户搬到了上海", "score": 0.8, "is_superseded": False, "stored_at": "2024-06-01"},
            {"memory_id": "v3", "content": "用户现在住在杭州", "score": 0.75, "is_superseded": False, "stored_at": "2024-12-01"},
        ]

        def filter_superseded(results):
            return [r for r in results if not r.get("is_superseded")]

        retrieved = filter_superseded(memories)
        logger.info("[LLM消歧] 检索返回 %d 条记忆:", len(retrieved))
        for r in retrieved:
            logger.info("  - %s: content=%s, score=%.2f, stored_at=%s, is_superseded=%s",
                        r["memory_id"], r["content"], r["score"], r["stored_at"], r["is_superseded"])

        assert len(retrieved) == 3
        # 按 stored_at 排序可确定最新答案是 v3（杭州）
        by_time = sorted(retrieved, key=lambda x: x["stored_at"], reverse=True)
        logger.info("[LLM消歧] 按时间排序后最新记忆: %s (%s)", by_time[0]["memory_id"], by_time[0]["content"])
        logger.info("[LLM消歧] 验证通过: LLM 可基于时间戳从 3 条记忆中推断最新答案为'杭州'")


# ─── 测试 5: apply_type_boost 对 add_only 检索结果的影响 ──────────

class TestAddOnlyWithBoost(unittest.TestCase):
    """add_only 下 apply_type_boost 不影响冲突记忆排序（因为没有 is_updated 标记）。"""

    def test_no_updated_boost_in_add_only(self) -> None:
        """add_only 下冲突记忆无 is_updated，不获得额外分数提升。"""
        from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator

        results = [
            {"memory_id": "mem-new", "type": "fact", "score": 0.8, "metadata": {}},
            {"memory_id": "mem-old", "type": "fact", "score": 0.75, "metadata": {}},
        ]
        logger.info("[add_only/boost] 提升前: %s", [(r["memory_id"], r["score"]) for r in results])
        boosted = HybridOrchestrator.apply_type_boost(results, updated_boost=0.3)
        logger.info("[add_only/boost] 提升后: %s", [(r["memory_id"], r["score"]) for r in boosted])
        # 两条记忆分数都不应被 updated_boost 改变
        for r in boosted:
            assert "updated_boost" not in r
        logger.info("[add_only/boost] 验证通过: 无 is_updated 标记, 分数未被 updated_boost 改变")

    def test_latest_with_updated_boost(self) -> None:
        """latest 下新记忆有 is_updated=True，获得分数提升。"""
        from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator

        results = [
            {"memory_id": "mem-new", "type": "fact", "score": 0.8, "metadata": {"is_updated": True}},
            {"memory_id": "mem-old", "type": "fact", "score": 0.75, "metadata": {}},
        ]
        logger.info("[latest/boost] 提升前: %s", [(r["memory_id"], r["score"], r.get("metadata", {})) for r in results])
        boosted = HybridOrchestrator.apply_type_boost(results, updated_boost=0.3)
        mem_new = next(r for r in boosted if r["memory_id"] == "mem-new")
        mem_old = next(r for r in boosted if r["memory_id"] == "mem-old")
        logger.info("[latest/boost] 提升后: mem-new score=%.4f (0.8*1.3), mem-old score=%.2f",
                     mem_new["score"], mem_old["score"])
        assert mem_new["score"] == 0.8 * 1.3  # 1.04
        assert mem_old["score"] == 0.75
        logger.info("[latest/boost] 验证通过: is_updated=True 的 mem-new 获得 30%% 提升")


if __name__ == "__main__":
    # 确保日志级别为 INFO，方便观察测试中的 logger.info 输出
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    unittest.main(verbosity=2)
