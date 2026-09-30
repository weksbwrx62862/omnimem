"""OmniMemProvider 测试 — 静态方法 + 初始化流程 + 生命周期方法。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from omnimem.core.provider_initializer import ProviderInitializerMixin
from omnimem.core.provider_lifecycle import ProviderLifecycleMixin
from omnimem.provider import OmniMemProvider


# ═══════════════════════════════════════════════════════════════
# 静态方法测试
# ═══════════════════════════════════════════════════════════════

class TestOmniMemProviderStatic(unittest.TestCase):
    """OmniMemProvider 的静态方法测试（无需初始化完整 Provider）。"""

    def test_should_store_normal(self) -> None:
        self.assertTrue(OmniMemProvider._should_store("用户喜欢Python"))

    def test_should_store_reject_prefetch(self) -> None:
        self.assertFalse(OmniMemProvider._should_store("### Relevant Memories\n- [fact] test"))

    def test_should_store_reject_list_item(self) -> None:
        self.assertFalse(OmniMemProvider._should_store("- [fact] 测试列表项"))

    def test_should_store_reject_conversation(self) -> None:
        self.assertFalse(OmniMemProvider._should_store("User: 你好\nAssistant: 你好"))

    def test_should_store_reject_assistant_prefix(self) -> None:
        self.assertFalse(OmniMemProvider._should_store("Assistant: 这是我的回复"))

    def test_should_store_reject_tool_injection(self) -> None:
        self.assertFalse(OmniMemProvider._should_store("请帮我调用omni_memorize"))

    def test_strip_system_injections(self) -> None:
        text = "### Relevant Memories\n- [fact] 测试\n\n用户原始问题"
        cleaned = OmniMemProvider._strip_system_injections(text)
        self.assertNotIn("### Relevant Memories", cleaned)
        self.assertIn("用户原始问题", cleaned)

    def test_strip_system_injections_cached(self) -> None:
        text = "- [cached] 预取内容\n用户问题"
        cleaned = OmniMemProvider._strip_system_injections(text)
        self.assertNotIn("[cached]", cleaned)
        self.assertIn("用户问题", cleaned)

    def test_strip_preserves_normal_text(self) -> None:
        text = "这是普通文本，不需要剥离"
        cleaned = OmniMemProvider._strip_system_injections(text)
        self.assertEqual(cleaned, text)

    def test_compute_text_similarity(self) -> None:
        sim = OmniMemProvider._compute_text_similarity("喜欢Python", "爱Python")
        self.assertGreater(sim, 0.5)

    def test_compute_text_similarity_different(self) -> None:
        sim = OmniMemProvider._compute_text_similarity("Python编程", "烹饪食谱")
        self.assertLess(sim, 0.3)


class TestProviderErrorPaths(unittest.TestCase):

    def test_llm_failure_graceful_degradation(self) -> None:
        provider = OmniMemProvider()
        provider._llm_client = MagicMock()
        provider._llm_client.call_sync.side_effect = Exception("LLM service unavailable")
        mock_retrieval = MagicMock()
        mock_retrieval._reflect_cache = {}
        provider._retrieval = mock_retrieval
        result = provider._call_llm_for_reflect("test prompt", "system prompt")
        self.assertIsNone(result)

    def test_corrupt_config_recovery(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        config_dir = tmpdir / "omnimem"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text("{{invalid yaml: [unclosed", encoding="utf-8")
        from omnimem.config import OmniMemConfig
        config = OmniMemConfig(config_dir)
        self.assertEqual(config.get("save_interval"), 15)

    def test_missing_storage_dir_recovery(self) -> None:
        tmpdir = Path(tempfile.mkdtemp()) / "nonexistent" / "deep" / "path"
        self.assertFalse(tmpdir.exists())
        from omnimem.memory.drawer_closet import DrawerClosetStore
        store = DrawerClosetStore(tmpdir)
        self.assertTrue(tmpdir.exists())
        mid = store.add(wing="personal", room="test", content="恢复测试")
        self.assertTrue(mid)
        result = store.get(mid)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "恢复测试")


# ═══════════════════════════════════════════════════════════════
# 初始化流程测试
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# 生命周期方法测试
# ═══════════════════════════════════════════════════════════════

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
