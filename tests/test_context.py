"""上下文管理模块测试。"""

from __future__ import annotations

import unittest

from omnimem.context.manager import ContextManager, ContextBudget


class TestContextManager(unittest.TestCase):
    """ContextManager 测试。"""

    def setUp(self):
        self.cm = ContextManager(budget=ContextBudget(
            max_prefetch_tokens=300,
            max_summary_chars=60,
            max_prefetch_items=8,
        ))

    def test_refine_content_short(self):
        result = ContextManager.refine_content("短内容", max_chars=60)
        self.assertEqual(result, "短内容")

    def test_refine_content_long_truncated(self):
        long_text = "这是一段很长的内容" * 20
        result = ContextManager.refine_content(long_text, max_chars=60)
        self.assertLessEqual(len(result), 60)

    def test_refine_content_strip_prefix(self):
        result = ContextManager.refine_content("CORRECTION: 错误内容", max_chars=60)
        self.assertTrue(result.startswith("纠正:") or "错误内容" in result)

    def test_refine_content_turn_prefix(self):
        result = ContextManager.refine_content("[Turn 5] 重要事实", max_chars=60)
        self.assertNotIn("[Turn 5]", result)

    def test_content_fingerprint(self):
        fp1 = ContextManager._content_fingerprint("我喜欢Python")
        fp2 = ContextManager._content_fingerprint("我喜欢Python")
        self.assertEqual(fp1, fp2)

    def test_fingerprint_similarity_identical(self):
        fp = ContextManager._content_fingerprint("用户姓名: 徐信豪")
        sim = ContextManager._fingerprint_similarity(fp, fp)
        self.assertEqual(sim, 1.0)

    def test_fingerprint_similarity_different(self):
        fp1 = ContextManager._content_fingerprint("Python编程语言")
        fp2 = ContextManager._content_fingerprint("量子计算技术")
        sim = ContextManager._fingerprint_similarity(fp1, fp2)
        self.assertLess(sim, 0.5)

    def test_fingerprint_similarity_synonyms(self):
        fp1 = ContextManager._content_fingerprint("我喜欢暗色主题")
        fp2 = ContextManager._content_fingerprint("偏好深色模式")
        sim = ContextManager._fingerprint_similarity(fp1, fp2)
        self.assertGreater(sim, 0.5)

    def test_refine_prefetch_results(self):
        raw = [
            {"content": "用户喜欢Python", "type": "preference", "memory_id": "m1", "confidence": 4},
            {"content": "项目使用Docker部署", "type": "fact", "memory_id": "m2", "confidence": 3},
        ]
        result = self.cm.refine_prefetch_results(raw)
        self.assertIn("### Relevant Memories", result)
        self.assertIn("preference", result)
        self.assertIn("fact", result)

    def test_refine_prefetch_dedup(self):
        raw = [
            {"content": "用户偏好暗色主题", "type": "preference", "memory_id": "m1", "confidence": 4},
            {"content": "我喜欢深色模式", "type": "fact", "memory_id": "m2", "confidence": 3},
        ]
        result = self.cm.refine_prefetch_results(raw)
        lines = [l for l in result.split("\n") if l.startswith("- [")]
        self.assertLessEqual(len(lines), 2)

    def test_refine_prefetch_empty(self):
        result = self.cm.refine_prefetch_results([])
        self.assertEqual(result, "")

    def test_reset_for_new_turn(self):
        raw = [{"content": "测试", "type": "fact", "memory_id": "m1", "confidence": 3}]
        self.cm.refine_prefetch_results(raw)
        self.assertTrue(len(self.cm._injected_items) > 0)
        self.cm.reset_for_new_turn()
        self.assertEqual(len(self.cm._injected_items), 0)

    def test_persistent_fingerprints_preserved(self):
        self.cm.add_persistent_fingerprint("fp-test")
        self.assertIn("fp-test", self.cm._persistent_fingerprints)
        self.cm.reset_for_new_turn()
        self.assertIn("fp-test", self.cm.get_injected_fingerprints())

    def test_get_injected_items(self):
        raw = [{"content": "测试项", "type": "fact", "memory_id": "m1", "confidence": 3}]
        self.cm.refine_prefetch_results(raw)
        items = self.cm.get_injected_items()
        self.assertTrue(len(items) > 0)
        self.assertEqual(items[0]["memory_id"], "m1")

    def test_refine_recall_results(self):
        raw = [
            {"content": "关于量子计算的事实：量子比特可以叠加", "type": "fact", "memory_id": "m1", "confidence": 3},
            {"content": "关于深度学习的发现：Transformer架构", "type": "skill", "memory_id": "m2", "confidence": 3},
        ]
        result = self.cm.refine_recall_results(raw)
        self.assertGreaterEqual(len(result), 1)
        self.assertIn("original_content", result[0])
