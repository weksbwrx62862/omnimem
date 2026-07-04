"""memorize 处理器后台任务日志测试。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from omnimem.services.memory_write_service import (
    MemoryWriteService,
    _log_bg_error,
    get_background_executor,
    shutdown_background_executor,
)


class TestBackgroundTaskLogging(unittest.TestCase):
    """验证后台任务异常被记录为 warning 且不会抛出。"""

    def setUp(self) -> None:
        """每个用例前 patch logger.warning。"""
        self.warning_patcher = patch("omnimem.services.memory_write_service.logger.warning")
        self.mock_warning = self.warning_patcher.start()

    def tearDown(self) -> None:
        """恢复 patch。"""
        self.warning_patcher.stop()

    def _assert_bg_error_logged(self, action: str, memory_id: str) -> None:
        """辅助断言：_log_bg_error 按要求格式调用了 logger.warning。"""
        self.mock_warning.assert_called_once()
        args = self.mock_warning.call_args[0]
        self.assertIn("%s", args[0])
        self.assertEqual(args[1], action)
        self.assertEqual(args[2], memory_id)

    def test_log_bg_error_format(self) -> None:
        """_log_bg_error 按统一格式输出 warning。"""
        exc = RuntimeError("测试异常")
        _log_bg_error("测试动作", "mem-001", exc)
        self._assert_bg_error_logged("测试动作", "mem-001")
        self.assertIn("测试异常", str(self.mock_warning.call_args[0][3]))

    def test_bg_llm_decision_exception_logs_and_does_not_raise(self) -> None:
        """llm_memory_manager.decide 异常时应记录 warning 且不抛出。"""
        llm_memory_manager = MagicMock()
        llm_memory_manager.decide.side_effect = RuntimeError("LLM decide failed")
        deps = MagicMock()
        memory_id = "mem-llm-001"
        service = MemoryWriteService(deps=deps, llm_memory_manager=llm_memory_manager)

        # 函数不应抛出任何异常
        service._bg_llm_decision("content", "fact", [], memory_id)

        self._assert_bg_error_logged("LLM决策", memory_id)
        self.assertIn("LLM decide failed", str(self.mock_warning.call_args[0][3]))

    def test_bg_kg_extract_exception_logs_and_does_not_raise(self) -> None:
        """KG 提取异常时应记录 warning 且不抛出。"""
        knowledge_graph = MagicMock()
        knowledge_graph.extract_and_store.side_effect = RuntimeError("KG failed")
        deps = MagicMock()
        deps.knowledge_graph = knowledge_graph
        memory_id = "mem-kg-001"
        service = MemoryWriteService(deps=deps)

        service._bg_kg_extract("content", memory_id, 0.8)

        self._assert_bg_error_logged("KG提取", memory_id)

    def test_bg_consolidation_submit_exception_logs_and_does_not_raise(self) -> None:
        """Consolidation 提交异常时应记录 warning 且不抛出。"""
        consolidation = MagicMock()
        consolidation.submit.side_effect = RuntimeError("consolidation failed")
        deps = MagicMock()
        deps.consolidation = consolidation
        memory_id = "mem-con-001"
        service = MemoryWriteService(deps=deps)

        service._bg_consolidation_submit(memory_id, "content", "fact")

        self._assert_bg_error_logged("Consolidation", memory_id)

    def test_bg_provenance_record_exception_logs_and_does_not_raise(self) -> None:
        """Provenance 记录异常时应记录 warning 且不抛出。"""
        provenance = MagicMock()
        provenance.record.side_effect = RuntimeError("provenance failed")
        deps = MagicMock()
        deps.provenance = provenance
        memory_id = "mem-prov-001"
        service = MemoryWriteService(deps=deps)

        service._bg_provenance_record(memory_id, {"source": "test"})

        self._assert_bg_error_logged("Provenance", memory_id)

    def test_bg_forgetting_record_exception_logs_and_does_not_raise(self) -> None:
        """Forgetting 记录异常时应记录 warning 且不抛出。"""
        forgetting = MagicMock()
        forgetting.record_access.side_effect = RuntimeError("forgetting failed")
        deps = MagicMock()
        deps.forgetting = forgetting
        memory_id = "mem-forget-001"
        service = MemoryWriteService(deps=deps)

        service._bg_forgetting_record(memory_id)

        self._assert_bg_error_logged("Forgetting", memory_id)


class TestBackgroundExecutorLifecycle(unittest.TestCase):
    """验证后台线程池的显式关闭与自动重建。"""

    def setUp(self) -> None:
        """保存当前模块级 executor 以便恢复。"""
        import omnimem.services.memory_write_service as _mem_mod

        self._mem_mod = _mem_mod
        self._original_executor = _mem_mod._fallback_executor

    def tearDown(self) -> None:
        """恢复可用 executor，避免状态泄漏影响其他测试。"""
        if self._original_executor is not None and not self._original_executor._shutdown:
            self._mem_mod._fallback_executor = self._original_executor
        else:
            self._mem_mod._fallback_executor = None
            self._mem_mod.get_background_executor()

    def test_shutdown_background_executor_sets_none(self) -> None:
        """shutdown 后模块级 executor 应置为 None。"""
        shutdown_background_executor()
        self.assertIsNone(self._mem_mod._fallback_executor)

    def test_get_background_executor_recreates_after_shutdown(self) -> None:
        """关闭后再次获取应自动重建线程池。"""
        shutdown_background_executor()
        executor = get_background_executor()
        self.assertIsNotNone(executor)
        self.assertFalse(executor._shutdown)
        self.assertEqual(executor._max_workers, 2)

    def test_shutdown_background_executor_idempotent(self) -> None:
        """多次调用 shutdown 不应抛异常。"""
        shutdown_background_executor()
        shutdown_background_executor()
        shutdown_background_executor(wait=False)
        self.assertIsNone(self._mem_mod._fallback_executor)
