"""SessionManager 单元测试 — 验证 SessionDependencies 参数封装。"""

from __future__ import annotations

from unittest.mock import MagicMock

from omnimem.core.session_deps import SessionDependencies
from omnimem.core.session_manager import SessionManager


class _Signals:
    """感知信号存根。"""

    def __init__(self, *, correction: bool = False, reinforcement: bool = False, memorize: bool = False):
        self.has_correction = correction
        self.has_reinforcement = reinforcement
        self.should_memorize = memorize


def _make_config(**overrides):
    """构造具备 get 方法的配置存根。"""
    defaults = {
        "save_interval": 15,
        "distill_enabled": False,
        "distill_interval": 15,
        "audit_interval_turns": 50,
        "backup_interval_hours": 24,
        "backup_max_copies": 3,
        "reflection_trigger": None,
    }
    defaults.update(overrides)
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: defaults.get(key, default)
    return config


def _make_deps(**overrides):
    """构造完整的 SessionDependencies，所有字段默认使用 MagicMock。"""
    defaults = {
        "config": _make_config(),
        "perception": MagicMock(),
        "store_service": MagicMock(),
        "retriever": MagicMock(),
        "bg_executor": MagicMock(),
        "forgetting": MagicMock(),
        "consolidation": None,
        "kv_cache": None,
        "lora_trainer": None,
        "store": MagicMock(),
        "index": MagicMock(),
        "auditor": None,
        "saga": MagicMock(),
        "prefetch_executor": MagicMock(),
        "pipeline_scheduler": None,
        "distill_init_fn": None,
        "distillation_engine": None,
        "session_id": "test-session",
        "should_write": True,
        "strip_system_injections_fn": None,
        "should_store_fn": None,
        "handle_memorize_fn": None,
        "retry_index_add_fn": None,
        "retry_retriever_add_fn": None,
        "retry_kg_extract_fn": None,
        "create_backup_fn": None,
        "cleanup_old_backups_fn": None,
    }
    defaults.update(overrides)
    return SessionDependencies(**defaults)


class TestSessionManagerCreation:
    """验证通过 SessionDependencies 创建 SessionManager。"""

    def test_create_with_dependencies(self):
        deps = _make_deps()
        manager = SessionManager(deps)
        assert manager.turn_count == 0
        assert manager.last_save_turn == 0
        assert manager._session_id == "test-session"
        assert manager._should_write is True

    def test_internal_attributes_preserved(self):
        """实例变量名与旧签名保持一致，外部延迟依赖仍可赋值。"""
        deps = _make_deps()
        manager = SessionManager(deps)
        assert manager._config is deps.config
        assert manager._perception is deps.perception
        assert manager._store_service is deps.store_service
        assert manager._retriever is deps.retriever
        assert manager._bg_executor is deps.bg_executor

        # 模拟 provider 在 _init_reflect 后的延迟注入
        mock_consolidation = MagicMock()
        manager._consolidation = mock_consolidation
        assert manager._consolidation is mock_consolidation


class TestSessionManagerSyncTurn:
    """验证 sync_turn 行为。"""

    def test_sync_turn_memorize_signal(self):
        perception = MagicMock()
        perception.detect_signals.return_value = _Signals(memorize=True)
        store_service = MagicMock()
        store_service.last_save_turn = 0
        bg_executor = MagicMock()
        retriever = MagicMock()

        deps = _make_deps(
            perception=perception,
            store_service=store_service,
            bg_executor=bg_executor,
            retriever=retriever,
        )
        manager = SessionManager(deps)

        manager.sync_turn("用户说：我喜欢Python", "助手回复：收到")

        perception.detect_signals.assert_called_once()
        store_service.store_fact.assert_called_once()
        store_service.store_correction.assert_not_called()
        store_service.store_reinforcement.assert_not_called()
        bg_executor.submit.assert_called()
        # index_update 通过 bg_executor.submit 异步提交
        submitted_calls = [call for call in bg_executor.submit.call_args_list
                           if call.args and call.args[0] is retriever.index_update]
        assert len(submitted_calls) == 1
        assert submitted_calls[0].args[1:] == ("用户说：我喜欢Python", "助手回复：收到")
        assert manager.turn_count == 1

    def test_sync_turn_correction_signal(self):
        perception = MagicMock()
        perception.detect_signals.return_value = _Signals(correction=True)
        store_service = MagicMock()
        store_service.last_save_turn = 0
        bg_executor = MagicMock()

        deps = _make_deps(
            perception=perception,
            store_service=store_service,
            bg_executor=bg_executor,
        )
        manager = SessionManager(deps)
        manager.sync_turn("不对，应该是Go", "抱歉，已更正")

        store_service.store_correction.assert_called_once()
        store_service.store_fact.assert_not_called()
        store_service.store_reinforcement.assert_not_called()
        assert manager.turn_count == 1

    def test_sync_turn_reinforcement_signal(self):
        perception = MagicMock()
        perception.detect_signals.return_value = _Signals(reinforcement=True)
        store_service = MagicMock()
        store_service.last_save_turn = 0
        bg_executor = MagicMock()

        deps = _make_deps(
            perception=perception,
            store_service=store_service,
            bg_executor=bg_executor,
        )
        manager = SessionManager(deps)
        manager.sync_turn("确认，Python很好用", "是的")

        store_service.store_reinforcement.assert_called_once()
        store_service.store_fact.assert_not_called()
        store_service.store_correction.assert_not_called()

    def test_sync_turn_disabled_write(self):
        deps = _make_deps(should_write=False)
        manager = SessionManager(deps)
        manager.sync_turn("用户输入", "助手回复")
        assert manager.turn_count == 0
        deps.perception.detect_signals.assert_not_called()

    def test_sync_turn_checkpoint_interval(self):
        perception = MagicMock()
        perception.detect_signals.return_value = _Signals(memorize=True)
        store_service = MagicMock()
        store_service.last_save_turn = 0
        bg_executor = MagicMock()

        deps = _make_deps(
            config=_make_config(save_interval=2),
            perception=perception,
            store_service=store_service,
            bg_executor=bg_executor,
        )
        manager = SessionManager(deps)

        # save_interval=2，第二轮触发 checkpoint
        manager.sync_turn("a", "b")
        store_service.auto_checkpoint.assert_not_called()
        manager.sync_turn("c", "d")
        store_service.auto_checkpoint.assert_called_once()


class TestSessionManagerReflection:
    """验证反思相关方法行为稳定。"""

    def test_reflection_methods(self):
        deps = _make_deps()
        manager = SessionManager(deps)
        manager.record_tool_call("omni_memorize")
        manager.mark_reflected()
        manager.reset_reflection_session()
        stats = manager.reflection_stats
        assert isinstance(stats, dict)


class TestSessionManagerSessionEnd:
    """验证 on_session_end 行为。"""

    def test_on_session_end_basic(self):
        store_service = MagicMock()
        store_service.extract_session_memories.return_value = None
        forgetting = MagicMock()
        forgetting.run_archive_cycle.return_value = 0
        saga = MagicMock()
        saga.get_pending.return_value = []
        prefetch_executor = MagicMock()
        bg_executor = MagicMock()
        store = MagicMock()
        retriever = MagicMock()

        deps = _make_deps(
            store_service=store_service,
            forgetting=forgetting,
            saga=saga,
            prefetch_executor=prefetch_executor,
            bg_executor=bg_executor,
            store=store,
            retriever=retriever,
        )
        manager = SessionManager(deps)
        manager.turn_count = 5
        manager.on_session_end([{"role": "user", "content": "hello"}])

        prefetch_executor.shutdown.assert_called_once_with(wait=True)
        store_service.extract_session_memories.assert_called_once()
        store.flush.assert_called_once()
        retriever.flush.assert_called_once()
        bg_executor.shutdown.assert_called_once_with(wait=True)

    def test_on_session_end_disabled(self):
        deps = _make_deps(should_write=False)
        manager = SessionManager(deps)
        manager.on_session_end([])
        deps.prefetch_executor.shutdown.assert_not_called()
