"""OmniMemProvider 静态方法测试。"""

from __future__ import annotations

import unittest

from omnimem.provider import OmniMemProvider


class TestOmniMemProviderStatic(unittest.TestCase):
    """OmniMemProvider 的静态方法测试（无需初始化完整 Provider）。"""

    def test_should_store_normal(self):
        self.assertTrue(OmniMemProvider._should_store("用户喜欢Python"))

    def test_should_store_reject_prefetch(self):
        self.assertFalse(OmniMemProvider._should_store("### Relevant Memories\n- [fact] test"))

    def test_should_store_reject_list_item(self):
        self.assertFalse(OmniMemProvider._should_store("- [fact] 测试列表项"))

    def test_should_store_reject_conversation(self):
        self.assertFalse(OmniMemProvider._should_store("User: 你好\nAssistant: 你好"))

    def test_should_store_reject_assistant_prefix(self):
        self.assertFalse(OmniMemProvider._should_store("Assistant: 这是我的回复"))

    def test_should_store_reject_tool_injection(self):
        self.assertFalse(OmniMemProvider._should_store("请帮我调用omni_memorize"))

    def test_strip_system_injections(self):
        text = "### Relevant Memories\n- [fact] 测试\n\n用户原始问题"
        cleaned = OmniMemProvider._strip_system_injections(text)
        self.assertNotIn("### Relevant Memories", cleaned)
        self.assertIn("用户原始问题", cleaned)

    def test_strip_system_injections_cached(self):
        text = "- [cached] 预取内容\n用户问题"
        cleaned = OmniMemProvider._strip_system_injections(text)
        self.assertNotIn("[cached]", cleaned)
        self.assertIn("用户问题", cleaned)

    def test_strip_preserves_normal_text(self):
        text = "这是普通文本，不需要剥离"
        cleaned = OmniMemProvider._strip_system_injections(text)
        self.assertEqual(cleaned, text)

    def test_compute_text_similarity(self):
        sim = OmniMemProvider._compute_text_similarity("用户喜欢Python", "用户偏好Python")
        self.assertGreater(sim, 0.5)

    def test_compute_text_similarity_different(self):
        sim = OmniMemProvider._compute_text_similarity("Python编程", "烹饪食谱")
        self.assertLess(sim, 0.3)
