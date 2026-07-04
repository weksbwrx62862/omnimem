"""Provider 初始化流程单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from omnimem.provider import OmniMemProvider


class _TestableInitializer(OmniMemProvider):
    """用于测试 Provider 初始化流程的最小子类，复用真实 Provider 的方法桩。"""

    pass


def test_initializer_sets_default_state() -> None:
    """ProviderInitializerMixin 初始化后应设置默认状态。"""
    provider = _TestableInitializer()
    assert provider._degraded_mode is False
    assert provider._turn_count == 0
    assert provider._system_prompt_cache_turn == -1
    assert provider._system_prompt_cache_value == ""
    assert provider._skill_index_built is False


def test_is_available_checks_core_deps() -> None:
    """is_available 应在核心依赖缺失时返回 False。"""
    provider = _TestableInitializer()
    with patch("builtins.__import__", side_effect=ModuleNotFoundError("no module")):
        assert provider.is_available() is False


def test_is_available_returns_true_when_deps_present() -> None:
    """is_available 在核心依赖存在时应返回 True。"""
    provider = _TestableInitializer()
    assert provider.is_available() is True


def test_init_l1_creates_storage_facade(tmp_path: Path) -> None:
    """_init_l1 应创建 StorageFacade 实例。"""
    provider = _TestableInitializer()
    provider._data_dir = tmp_path
    provider._config = MagicMock()
    provider._config.get.side_effect = lambda _key, default=None: default

    with patch("omnimem.core.provider_initializer.StorageFacade") as mock_facade:
        provider._init_l1()
        assert provider._storage is mock_facade.return_value
        mock_facade.assert_called_once_with(tmp_path, provider._config)


def test_init_store_delegates_to_storage(tmp_path: Path) -> None:
    """_init_store 应委托给 StorageFacade 的 init_l2。"""
    _ = tmp_path
    provider = _TestableInitializer()
    provider._storage = MagicMock()
    provider._init_store()
    provider._storage.init_l2.assert_called_once()


def test_init_retrieval_sets_recover_callback(tmp_path: Path) -> None:
    """_init_retrieval 应为熔断器设置恢复回调。"""
    provider = _TestableInitializer()
    provider._data_dir = tmp_path
    provider._config = MagicMock()
    provider._config.get.side_effect = lambda _key, default=None: default
    provider._storage = MagicMock()

    mock_retriever = MagicMock()
    mock_retriever._vector_breaker = MagicMock()
    mock_retrieval = MagicMock()
    mock_retrieval.retriever = mock_retriever

    with patch("omnimem.core.provider_initializer.RetrievalFacade", return_value=mock_retrieval):
        provider._init_retrieval()
        assert provider._retrieval is mock_retrieval
        assert callable(mock_retriever._vector_breaker._on_recover)


def test_init_governance_sync_services_assigns_facade_attrs(tmp_path: Path) -> None:
    """_init_governance_sync_services 应为 provider 赋值 facade 子属性。"""
    mock_gov = MagicMock()
    mock_gov.instance_id = "instance-1"
    mock_sync = MagicMock()

    # provider_initializer 模块级导入的类
    module_mocks = {
        "GovernanceFacade": mock_gov,
        "SyncFacade": mock_sync,
        "LLMClientManager": MagicMock(),
        "LLMMemoryManager": MagicMock(),
        "CompressionPipeline": MagicMock(),
        "CompatHandler": MagicMock(),
        "SemanticDedupService": MagicMock(),
        "ActionMemoryService": MagicMock(),
        "ToolRouter": MagicMock(),
        "BackupManager": MagicMock(),
        "SystemPromptBuilder": MagicMock(),
        "SessionManager": MagicMock(),
        "RetrievalQualityEvaluator": MagicMock(),
    }
    # _init_governance_sync_services 方法内部动态导入的类
    dynamic_mocks = {
        "omnimem.compression.mermaid_canvas.MermaidCanvas": MagicMock(),
        "omnimem.core.trace_chain.TraceChain": MagicMock(),
        "omnimem.core.pipeline_scheduler.PipelineScheduler": MagicMock(),
    }

    patchers = [
        patch(f"omnimem.core.provider_initializer.{name}", return_value=mock)
        for name, mock in module_mocks.items()
    ]
    patchers.extend(
        patch(target, return_value=mock) for target, mock in dynamic_mocks.items()
    )
    for p in patchers:
        p.start()

    try:
        provider = _TestableInitializer()
        provider._data_dir = tmp_path
        provider._config = MagicMock()
        provider._config.get.side_effect = lambda _key, default=None: default
        provider._session_id = "test-session"
        provider._should_write = True
        provider._storage = MagicMock()
        provider._retrieval = MagicMock()
        provider._retrieval.retriever = MagicMock()
        provider._retrieval.prefetch_lock = MagicMock()
        provider._retrieval._reflect_cache = {}
        provider._retrieval.prefetch_executor = MagicMock()
        provider._reflect_cache = {}
        # CompatHandler / ToolRouter 需要的方法桩
        provider._handle_memorize = MagicMock()
        provider._handle_recall = MagicMock()
        provider._handle_govern = MagicMock()
        provider._handle_reflect = MagicMock()
        provider._handle_compact = MagicMock()
        provider._handle_detail = MagicMock()
        provider._handle_builtin_memory_compat = MagicMock()
        provider._handle_record_action = MagicMock()
        provider._extract_core_fact = MagicMock()
        provider._wing_room = MagicMock()

        provider._init_governance_sync_services()
    finally:
        for p in patchers:
            p.stop()

    assert provider._governance is mock_gov
    assert provider._sync is mock_sync
    assert provider._instance_id == "instance-1"
    assert provider._store is provider._storage.store
