"""Provider 生命周期方法单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from omnimem.core.provider_initializer import ProviderInitializerMixin
from omnimem.core.provider_lifecycle import ProviderLifecycleMixin


class _TestableLifecycle(ProviderLifecycleMixin, ProviderInitializerMixin):
    """仅用于测试 ProviderLifecycleMixin 的最小子类，同时混入 Initializer 以获取状态属性。"""

    def __init__(self) -> None:
        super().__init__()


def _patch_all_init_dependencies() -> list:
    """为 initialize 测试准备统一的依赖 mock patchers。"""
    deps = [
        "omnimem.core.provider_lifecycle.MemoryMonitor",
        "omnimem.core.provider_lifecycle.WarmupManager",
    ]
    return [patch(dep) for dep in deps]


def test_initialize_sets_data_dir_and_session(tmp_path: Path) -> None:
    """initialize 应设置 data_dir 和 session_id。"""
    provider = _TestableLifecycle()
    provider._degraded_mode = False
    provider._init_l1 = MagicMock()
    provider._init_store = MagicMock()
    provider._init_retrieval = MagicMock()
    provider._init_governance_sync_services = MagicMock()
    provider._index = MagicMock()
    provider._store = MagicMock()
    provider._retriever = MagicMock()
    provider._retrieval = MagicMock()
    provider._auditor = MagicMock()

    patchers = _patch_all_init_dependencies()
    for p in patchers:
        p.start()

    try:
        provider.initialize("test-session", hermes_home=str(tmp_path))
    finally:
        for p in patchers:
            p.stop()

    assert provider._session_id == "test-session"
    assert provider._data_dir == tmp_path / "omnimem"
    assert provider._data_dir.exists()


def test_initialize_degraded_mode_skips_vector(tmp_path: Path) -> None:
    """降级模式下 initialize 应跳过检索治理初始化。"""
    provider = _TestableLifecycle()
    provider._degraded_mode = True
    provider._init_l1 = MagicMock()
    provider._init_store = MagicMock()
    provider._init_retrieval = MagicMock()
    provider._init_governance_sync_services = MagicMock()

    patchers = _patch_all_init_dependencies()
    for p in patchers:
        p.start()

    try:
        provider.initialize("test-session", hermes_home=str(tmp_path))
    finally:
        for p in patchers:
            p.stop()

    provider._init_l1.assert_called_once()
    provider._init_store.assert_not_called()
    provider._init_retrieval.assert_not_called()
    provider._init_governance_sync_services.assert_not_called()


def test_initialize_sets_should_write_by_agent_context(tmp_path: Path) -> None:
    """initialize 应根据 agent_context 设置 _should_write。"""
    provider = _TestableLifecycle()
    provider._degraded_mode = True
    provider._init_l1 = MagicMock()

    patchers = _patch_all_init_dependencies()
    for p in patchers:
        p.start()

    try:
        provider.initialize("s1", hermes_home=str(tmp_path), agent_context="secondary")
        assert provider._should_write is False

        provider.initialize("s2", hermes_home=str(tmp_path), agent_context="primary")
        assert provider._should_write is True
    finally:
        for p in patchers:
            p.stop()


def test_async_provider_lazily_created() -> None:
    """async_provider 应延迟创建异步包装器。"""
    provider = _TestableLifecycle()
    with patch("omnimem.core.async_provider.AsyncOmniMemProvider") as mock_async:
        ap = provider.async_provider
        assert ap is mock_async.return_value
        mock_async.assert_called_once_with(provider)


def test_system_prompt_block_delegates_to_builder() -> None:
    """system_prompt_block 应委托给 SystemPromptBuilder。"""
    provider = _TestableLifecycle()
    mock_builder = MagicMock()
    mock_builder.build.return_value = ("prompt", 5, "cached")
    provider._system_prompt_builder = mock_builder

    result = provider.system_prompt_block()
    assert result == "prompt"
    assert provider._system_prompt_cache_turn == 5
    assert provider._system_prompt_cache_value == "cached"


def test_system_prompt_block_returns_empty_without_builder() -> None:
    """无 SystemPromptBuilder 时应返回空字符串。"""
    provider = _TestableLifecycle()
    assert provider.system_prompt_block() == ""


def test_on_session_end_delegates_to_session_manager() -> None:
    """on_session_end 应委托给 SessionManager。"""
    provider = _TestableLifecycle()
    provider._should_write = True
    provider._turn_count = 10
    mock_session_manager = MagicMock()
    mock_session_manager.turn_count = 10
    provider._session_manager = mock_session_manager

    provider.on_session_end([{"role": "user", "content": "hi"}])
    mock_session_manager.on_session_end.assert_called_once()
    assert provider._turn_count == 10


def test_on_session_end_skips_when_not_should_write() -> None:
    """_should_write 为 False 时 on_session_end 应跳过。"""
    provider = _TestableLifecycle()
    provider._should_write = False
    provider._session_manager = MagicMock()
    provider.on_session_end()
    provider._session_manager.on_session_end.assert_not_called()


def test_on_delegation_delegates_to_store_service() -> None:
    """on_delegation 应委托给 StoreService。"""
    provider = _TestableLifecycle()
    provider._should_write = True
    provider._store_service = MagicMock()
    provider.on_delegation("task", "result", child_session_id="child-1")
    provider._store_service.store_delegation.assert_called_once_with(
        "task", "result", "child-1"
    )


def test_on_delegation_skips_when_not_should_write() -> None:
    """_should_write 为 False 时 on_delegation 应跳过。"""
    provider = _TestableLifecycle()
    provider._should_write = False
    provider._store_service = MagicMock()
    provider.on_delegation("task", "result")
    provider._store_service.store_delegation.assert_not_called()


def test_create_and_cleanup_backups_delegate_to_manager(tmp_path: Path) -> None:
    """备份创建/清理应委托给 BackupManager。"""
    _ = tmp_path
    provider = _TestableLifecycle()
    mock_manager = MagicMock()
    mock_manager.create_backup.return_value = ("path", 123)
    mock_manager.last_backup_time = 1000.0
    provider._backup_manager = mock_manager

    result = provider._create_backup()
    assert result == ("path", 123)
    assert provider._last_backup_time == 1000.0

    provider._cleanup_old_backups(max_copies=5)
    mock_manager.cleanup_old_backups.assert_called_once_with(5)


def test_shutdown_closes_resources() -> None:
    """shutdown 应关闭已初始化的资源。"""
    provider = _TestableLifecycle()
    provider._memory_monitor = MagicMock()
    provider._feedback = MagicMock()
    provider._prefetch_executor = MagicMock()
    provider._bg_executor = MagicMock()
    provider._store = MagicMock()
    provider._retriever = MagicMock()
    provider._md_store = MagicMock()
    provider._index = MagicMock()
    provider._perception = MagicMock()
    provider._knowledge_graph = MagicMock()
    provider._consolidation = MagicMock()
    provider._reflect_engine = MagicMock()
    provider._kv_cache = MagicMock()
    provider._lora_trainer = MagicMock()
    provider._provenance = MagicMock()
    provider._sync_engine = MagicMock()
    provider._governance = MagicMock()
    provider._llm_client_manager = MagicMock()
    provider._distillation_engine = MagicMock()
    provider._forgetting = MagicMock()
    provider._quality_evaluator = MagicMock()

    import gc

    # 先回收之前测试遗留的 provider 实例，避免其 __del__ 在当前 patch 块内触发 shutdown
    gc.collect()

    with patch("omnimem.core.provider_lifecycle.shutdown_background_executor") as mock_shutdown_bg:
        provider.shutdown()
        mock_shutdown_bg.assert_called_once_with(wait=True)
        provider._memory_monitor.stop.assert_called_once()
        provider._store.close.assert_called_once()
        provider._retriever.flush.assert_called_once()
        provider._forgetting.close.assert_called_once()
        # 当前 provider 已设置 _shutdown_done，删除并 gc 不会导致重复调用
        del provider
        gc.collect()
