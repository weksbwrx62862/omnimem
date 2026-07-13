"""TemporalRetriever 时序重排序单元测试。

覆盖:
  1. 时序关键词检测
  2. 时序查询返回按时间排序的结果
  3. 普通查询不受影响
  4. 时间衰减权重计算正确
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any

from omnimem.retrieval.registry import _TemporalRetriever


class TestTemporalKeywordDetection(unittest.TestCase):
    """时序关键词检测测试。"""

    def test_chinese_keywords(self) -> None:
        """中文时序关键词应被检测到。"""
        for kw in ["最近", "上次", "什么时候", "第一次", "最后", "之前", "之后", "刚才", "以前", "后来"]:
            with self.subTest(keyword=kw):
                self.assertTrue(
                    _TemporalRetriever.is_temporal_query(f"你{kw}做了什么"),
                    f"应检测到中文时序关键词: {kw}",
                )

    def test_english_keywords(self) -> None:
        """英文时序关键词应被检测到。"""
        for kw in ["when", "last", "first", "recently", "before", "after", "earlier", "later", "previous", "latest"]:
            with self.subTest(keyword=kw):
                self.assertTrue(
                    _TemporalRetriever.is_temporal_query(f"What did you do {kw}?"),
                    f"应检测到英文时序关键词: {kw}",
                )

    def test_english_case_insensitive(self) -> None:
        """英文关键词匹配应忽略大小写。"""
        self.assertTrue(_TemporalRetriever.is_temporal_query("When did this happen?"))
        self.assertTrue(_TemporalRetriever.is_temporal_query("LAST time I checked"))
        self.assertTrue(_TemporalRetriever.is_temporal_query("RECENTLY updated"))

    def test_non_temporal_query(self) -> None:
        """不包含时序关键词的查询应返回 False。"""
        self.assertFalse(_TemporalRetriever.is_temporal_query("Python 编程入门"))
        self.assertFalse(_TemporalRetriever.is_temporal_query("机器学习算法原理"))
        self.assertFalse(_TemporalRetriever.is_temporal_query("how to install docker"))

    def test_empty_query(self) -> None:
        """空查询不应被检测为时序查询。"""
        self.assertFalse(_TemporalRetriever.is_temporal_query(""))

    def test_search_returns_empty(self) -> None:
        """TemporalRetriever.search() 应始终返回空结果。"""
        retriever = _TemporalRetriever()
        result = retriever.search("最近的会议记录")
        self.assertEqual(len(result.results), 0)
        self.assertEqual(len(result.scores), 0)
        self.assertEqual(result.channel, "temporal")


class TestTemporalRerank(unittest.TestCase):
    """时序查询返回按时间排序的结果测试。"""

    def test_temporal_rerank_reorders_by_recency(self) -> None:
        """时序查询应使最近的记忆排名提升。"""
        now = datetime.now(tz=timezone.utc)
        results = [
            {
                "memory_id": "old",
                "content": "旧的记忆",
                "score": 0.10,
                "metadata": {"created_at": (now - timedelta(days=30)).isoformat()},
            },
            {
                "memory_id": "recent",
                "content": "最近的记忆",
                "score": 0.10,
                "metadata": {"created_at": (now - timedelta(days=1)).isoformat()},
            },
        ]
        reranked = _TemporalRetriever.apply_temporal_rerank(
            "最近做了什么", results, alpha=0.5, decay_lambda=0.1,
        )
        # 最近的记忆应排在前面
        self.assertEqual(reranked[0]["memory_id"], "recent")
        self.assertTrue(reranked[0]["_temporal_reranked"])
        self.assertGreater(reranked[0]["_temporal_weight"], reranked[1]["_temporal_weight"])

    def test_temporal_rerank_preserves_high_score(self) -> None:
        """时序重排序不应完全覆盖原始分数差异。"""
        now = datetime.now(tz=timezone.utc)
        results = [
            {
                "memory_id": "high_score_old",
                "content": "高分旧记忆",
                "score": 0.20,
                "metadata": {"created_at": (now - timedelta(days=30)).isoformat()},
            },
            {
                "memory_id": "low_score_recent",
                "content": "低分新记忆",
                "score": 0.05,
                "metadata": {"created_at": (now - timedelta(days=1)).isoformat()},
            },
        ]
        reranked = _TemporalRetriever.apply_temporal_rerank(
            "最近做了什么", results, alpha=0.5, decay_lambda=0.1,
        )
        # 两条结果都有分数变化
        for r in reranked:
            self.assertIn("_temporal_reranked", r)


class TestNonTemporalQuery(unittest.TestCase):
    """普通查询不受时序重排序影响。"""

    def test_non_temporal_query_unchanged(self) -> None:
        """非时序查询的结果应原样返回，不被修改。"""
        results = [
            {"memory_id": "a", "content": "Python 入门", "score": 0.5},
            {"memory_id": "b", "content": "Java 基础", "score": 0.3},
        ]
        reranked = _TemporalRetriever.apply_temporal_rerank(
            "Python 编程入门", results,
        )
        self.assertEqual(len(reranked), 2)
        # 不应有时序重排序标记
        for r in reranked:
            self.assertNotIn("_temporal_reranked", r)

    def test_empty_results_unchanged(self) -> None:
        """空结果应直接返回。"""
        reranked = _TemporalRetriever.apply_temporal_rerank(
            "最近的记录", [],
        )
        self.assertEqual(len(reranked), 0)


class TestTemporalDecayWeight(unittest.TestCase):
    """时间衰减权重计算正确性测试。"""

    def test_recent_memory_high_weight(self) -> None:
        """最近的记忆应有高时间权重（接近 1.0）。"""
        now = datetime.now(tz=timezone.utc)
        result = {
            "memory_id": "test",
            "metadata": {"created_at": (now - timedelta(hours=1)).isoformat()},
        }
        weight = _TemporalRetriever.compute_temporal_weight(result, now=now, decay_lambda=0.1)
        self.assertGreater(weight, 0.9)
        self.assertLessEqual(weight, 1.0)

    def test_old_memory_low_weight(self) -> None:
        """很旧的记忆应有低时间权重。"""
        now = datetime.now(tz=timezone.utc)
        result = {
            "memory_id": "test",
            "metadata": {"created_at": (now - timedelta(days=100)).isoformat()},
        }
        weight = _TemporalRetriever.compute_temporal_weight(result, now=now, decay_lambda=0.1)
        # exp(-0.1 * 100) ≈ 0.0000454
        self.assertLess(weight, 0.001)

    def test_seven_day_half_life(self) -> None:
        """lambda=0.1 时约 7 天半衰期。"""
        now = datetime.now(tz=timezone.utc)
        result_7d = {
            "memory_id": "test",
            "metadata": {"created_at": (now - timedelta(days=7)).isoformat()},
        }
        weight = _TemporalRetriever.compute_temporal_weight(result_7d, now=now, decay_lambda=0.1)
        # exp(-0.1 * 7) ≈ 0.4966，接近 0.5
        self.assertAlmostEqual(weight, 0.5, places=1)

    def test_no_timestamp_returns_zero(self) -> None:
        """没有时间戳的结果权重应为 0。"""
        result = {"memory_id": "test", "metadata": {}}
        weight = _TemporalRetriever.compute_temporal_weight(result)
        self.assertEqual(weight, 0.0)

    def test_invalid_timestamp_returns_zero(self) -> None:
        """无效时间戳字符串权重应为 0。"""
        result = {"memory_id": "test", "metadata": {"created_at": "not-a-date"}}
        weight = _TemporalRetriever.compute_temporal_weight(result)
        self.assertEqual(weight, 0.0)

    def test_top_level_timestamp(self) -> None:
        """顶层 timestamp 字段也应被识别。"""
        now = datetime.now(tz=timezone.utc)
        result = {
            "memory_id": "test",
            "timestamp": (now - timedelta(days=1)).isoformat(),
        }
        weight = _TemporalRetriever.compute_temporal_weight(result, now=now, decay_lambda=0.1)
        self.assertGreater(weight, 0.8)

    def test_top_level_created_at(self) -> None:
        """顶层 created_at 字段也应被识别。"""
        now = datetime.now(tz=timezone.utc)
        result = {
            "memory_id": "test",
            "created_at": (now - timedelta(days=1)).isoformat(),
        }
        weight = _TemporalRetriever.compute_temporal_weight(result, now=now, decay_lambda=0.1)
        self.assertGreater(weight, 0.8)

    def test_score_fusion_formula(self) -> None:
        """验证 final_score = original_score * (1 + alpha * time_weight)。"""
        now = datetime.now(tz=timezone.utc)
        original_score = 0.10
        results = [
            {
                "memory_id": "test",
                "score": original_score,
                "metadata": {"created_at": (now - timedelta(days=7)).isoformat()},
            },
        ]
        alpha = 0.5
        reranked = _TemporalRetriever.apply_temporal_rerank(
            "最近的记录", results, alpha=alpha, decay_lambda=0.1,
        )
        time_weight = reranked[0]["_temporal_weight"]
        expected = original_score * (1.0 + alpha * time_weight)
        self.assertAlmostEqual(reranked[0]["score"], expected, places=5)


if __name__ == "__main__":
    unittest.main()
