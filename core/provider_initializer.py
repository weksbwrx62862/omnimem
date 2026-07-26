"""Provider 初始化与资源构造：__init__、配置构建、L1-L4 资源初始化。"""

from __future__ import annotations

import logging
from typing import Any

from omnimem.compression.pipeline import CompressionPipeline
from omnimem.core.action_memory import ActionMemoryService
from omnimem.core.backup_manager import BackupManager
from omnimem.core.dedup import SemanticDedupService
from omnimem.core.distillation import DistillationEngine
from omnimem.core.llm_client_manager import LLMClientManager
from omnimem.core.llm_memory_manager import LLMMemoryManager
from omnimem.core.session_deps import SessionDependencies
from omnimem.core.session_manager import SessionManager
from omnimem.core.system_prompt_builder import SystemPromptBuilder
from omnimem.core.tool_router import ToolRouter
from omnimem.core.warmup_manager import WarmupManager
from omnimem.facades import (
    DeepMemoryFacade,
    GovernanceFacade,
    RetrievalFacade,
    StorageFacade,
    SyncFacade,
)
from omnimem.handlers.compat_handler import CompatHandler
from omnimem.retrieval.quality_eval import RetrievalQualityEvaluator

logger = logging.getLogger(__name__)


class ProviderInitializerMixin:
    """负责 Provider 的构造、配置以及 L1-L4 资源初始化。"""

    _warmup_manager: WarmupManager
    _system_prompt_builder: SystemPromptBuilder
    _session_manager: SessionManager
    _backup_manager: BackupManager
    _llm_client_manager: LLMClientManager

    def __init__(self) -> None:
        self._degraded_mode = False
        self._turn_count = 0
        self._system_prompt_cache_turn = -1
        self._system_prompt_cache_value = ""
        # wrap_model_call 状态（每个实例独立，不共享）
        self._skill_index_cache: list[dict[str, Any]] = []
        self._skill_index_built: bool = False
        self._wrap_call_cache: dict[str, str] = {}
        self._last_backup_time = 0.0
        # 确保 MemoryProvider 基类初始化也被执行
        super().__init__()

    def is_available(self) -> bool:
        core_deps = {"rank_bm25": "rank-bm25", "tiktoken": "tiktoken", "yaml": "pyyaml"}
        for module, pip_name in core_deps.items():
            try:
                __import__(module)
            except ImportError:
                logger.error(
                    "OmniMem 核心依赖 %s 缺失，请安装: pip install %s", module, pip_name
                )
                return False

        optional_deps = {
            "chromadb": "chromadb",
            "sentence_transformers": "sentence-transformers",
        }
        for module, pip_name in optional_deps.items():
            try:
                __import__(module)
            except ImportError:
                logger.warning(
                    "OmniMem 可选依赖 %s 缺失，将降级到 BM25-only 模式。安装: pip install %s",
                    module,
                    pip_name,
                )

        return True

    def _init_l1(self) -> None:
        self._storage = StorageFacade(self._data_dir, self._config)

    def _init_store(self) -> None:
        self._storage.init_l2()

    def _init_retrieval(self) -> None:
        self._retrieval = RetrievalFacade(self._data_dir, self._config, self._storage)
        if hasattr(self._retrieval.retriever, "_vector_breaker"):

            def _on_circuit_recover() -> None:
                try:
                    health = self._retrieval.retriever._check_vector_health()
                    if health.get("vector_count", -1) <= 0:
                        logger.warning(
                            "OmniMem: CircuitBreaker recovered but vector still empty, triggering rebuild"
                        )
                        if hasattr(self, "_index") and self._index:
                            indexed_entries = self._index.search_l1(limit=5000)
                            if indexed_entries:
                                result = self._retrieval.retriever.rebuild_all_from_entries(
                                    indexed_entries
                                )
                                logger.info(
                                    "OmniMem: vector rebuild after circuit recovery: %s", result
                                )
                except Exception as e:
                    logger.warning("OmniMem: circuit recovery rebuild failed: %s", e)

            self._retrieval.retriever._vector_breaker._on_recover = _on_circuit_recover

    def _init_governance_sync_services(self) -> None:
        self._governance = GovernanceFacade(
            self._data_dir,
            self._config,
            self._session_id,
            self._storage,
            self._retrieval.retriever,
        )
        self._sync = SyncFacade(
            self._data_dir,
            self._config,
            self._session_id,
            self._storage,
            self._retrieval,
        )
        self._sync.bind_provenance(self._governance.provenance)

        self._instance_id = self._governance.instance_id
        self._turn_count = 0
        self._last_save_turn = 0
        self._save_interval = self._config.get("save_interval", 15)
        self._system_prompt_cache_turn = -1
        self._system_prompt_cache_value = ""
        self._init_llm_client()
        # LLM-as-Memory-Manager：初始化 LLM 记忆决策管理器
        self._llm_memory_manager = LLMMemoryManager(
            llm_client=self._llm_client_manager.llm_client,
            config=self._config,
        )
        # ★ P1: LLM 驱动的事实抽取（hybrid 模式），LLM 不可用时自动保持规则模式
        try:
            self._retrieval.perception.configure_llm_extraction(
                mode=self._config.get("extraction_mode", "hybrid"),
                llm_client=self._llm_client_manager.llm_client,
            )
        except Exception as e:
            logger.warning("OmniMem: LLM extraction wiring failed (rule mode kept): %s", e)
        # OPT: 初始化 MermaidCanvas + CompressionPipeline
        from omnimem.compression.mermaid_canvas import MermaidCanvas

        self._mermaid_canvas = MermaidCanvas(self._data_dir, config=self._config)
        self._compression_pipeline = CompressionPipeline(
            llm_call_fn=self._llm_client_manager.make_llm_call_fn(),
            config=self._config,
            mermaid_canvas=self._mermaid_canvas,
            session_key=self._session_id,
        )
        self._compat_handler = CompatHandler(
            memorize_fn=self._handle_memorize,
            store=self._store,
            forgetting=self._forgetting,
            extract_core_fact_fn=self._extract_core_fact,
        )
        self._dedup_service = SemanticDedupService(self._store, self._retriever)
        self._action_memory = ActionMemoryService(
            self._store,
            self._index,
            self._retriever,
            self._wing_room,
            self._provenance,
            self._forgetting,
        )
        self._tool_router = ToolRouter(
            memorize_fn=self._handle_memorize,
            recall_fn=self._handle_recall,
            govern_fn=self._handle_govern,
            reflect_fn=self._handle_reflect,
            compact_fn=self._handle_compact,
            detail_fn=self._handle_detail,
            memory_compat_fn=self._handle_builtin_memory_compat,
            record_action_fn=self._handle_record_action,
        )
        # OPT: 初始化 TraceChain 全链路溯源
        from omnimem.core.trace_chain import TraceChain

        self._trace_chain = TraceChain(self._data_dir)
        # OPT: 初始化 PipelineScheduler（L2/L3 自动调度）
        from omnimem.core.pipeline_scheduler import PipelineScheduler

        self._pipeline_scheduler = PipelineScheduler(
            config=self._config,
            logger=logger,
            bg_executor=self._bg_executor if hasattr(self, "_bg_executor") else None,
            reflect_fn=self._handle_reflect if hasattr(self, "_handle_reflect") else None,
        )

        # 显式属性赋值（替代 __getattr__ 动态代理）
        self._soul = self._storage.soul
        self._core_block = self._storage.core_block
        self._budget = self._storage.budget
        self._wing_room = self._storage.wing_room
        self._store = self._storage.store
        self._index = self._storage.index
        self._md_store = self._storage.md_store
        self._retriever = self._retrieval.retriever
        self._context_manager = self._retrieval.context_manager
        self._perception = self._retrieval.perception
        self._feedback = self._retrieval.feedback
        self._prefetch_lock = self._retrieval.prefetch_lock
        self._reflect_cache = self._retrieval._reflect_cache
        self._prefetch_executor = self._retrieval.prefetch_executor
        self._conflict_resolver = self._governance.conflict_resolver
        self._temporal_decay = self._governance.temporal_decay
        self._forgetting = self._governance.forgetting
        self._privacy = self._governance.privacy
        self._provenance = self._governance.provenance
        self._sync_engine = self._governance.sync_engine
        self._vector_clock = self._governance.vector_clock
        self._auditor = self._governance.auditor
        self._audit_logger = self._governance.audit_logger
        self._rbac = self._governance.rbac
        self._temporal_kg = self._governance.temporal_kg
        self._saga = self._sync.saga
        self._bg_executor = self._sync.bg_executor
        self._store_service = self._sync.store_service
        self._kv_cache = self._sync.kv_cache
        self._lora_trainer = self._sync.lora_trainer
        # 质量评估器：检索质量指标计算 + 持久化 + 自动调优建议
        self._quality_evaluator = RetrievalQualityEvaluator(
            self._data_dir,
            config=self._config,
        )

        # 组合 Manager 初始化
        self._backup_manager = BackupManager(self._data_dir)
        self._system_prompt_builder = SystemPromptBuilder(
            data_dir=self._data_dir,
            store=self._store,
            core_block=self._core_block,
            context_manager=self._context_manager,
            config=self._config,
            retrieval=self._retrieval,
        )
        self._session_manager = SessionManager(
            SessionDependencies(
                config=self._config,
                perception=self._perception,
                store_service=self._store_service,
                retriever=self._retriever,
                bg_executor=self._bg_executor,
                forgetting=self._forgetting,
                consolidation=None,  # 延迟设置（_init_reflect 后才有）
                kv_cache=self._kv_cache,
                lora_trainer=self._lora_trainer,
                store=self._store,
                index=self._index,
                auditor=self._auditor,
                saga=self._saga,
                prefetch_executor=self._prefetch_executor,
                pipeline_scheduler=None,  # 延迟设置
                distill_init_fn=self._init_distill,
                distillation_engine=None,  # 延迟设置
                session_id=self._session_id,
                should_write=self._should_write,
                strip_system_injections_fn=self._strip_system_injections,
                should_store_fn=self._should_store,
                handle_memorize_fn=self._handle_memorize,
                retry_index_add_fn=self._retry_index_add,
                retry_retriever_add_fn=self._retry_retriever_add,
                retry_kg_extract_fn=self._retry_kg_extract,
                create_backup_fn=self._create_backup,
                cleanup_old_backups_fn=self._cleanup_old_backups,
            )
        )

    def _init_reflect(self) -> None:
        self._deep = DeepMemoryFacade(
            self._data_dir,
            self._config,
            recall_fn=self._l3_recall,
            llm_fn=self._call_llm_for_reflect,
            llm_client=self._llm_client_manager.llm_client,
        )
        # 显式属性赋值（_deep facade，在 _init_reflect 中初始化后赋值）
        self._consolidation = self._deep.consolidation
        self._knowledge_graph = self._deep.knowledge_graph
        self._reflect_engine = self._deep.reflect_engine
        # 更新 SessionManager 的延迟依赖
        if hasattr(self, "_session_manager"):
            self._session_manager._consolidation = self._consolidation

    def _init_distill(self) -> None:
        """初始化 LLM 蒸馏引擎（OPT-1）。"""
        self._distillation_engine = DistillationEngine(
            llm_fn=self._call_llm_for_distill,
            store=self._store,
            memorize_fn=self._handle_memorize,
            config=self._config,
        )
        # 更新 SessionManager 的延迟依赖
        if hasattr(self, "_session_manager"):
            self._session_manager._distillation_engine = self._distillation_engine

    def _init_lora(self) -> None:
        """初始化 L4 内化记忆。"""
        self._sync.init_l4()

    def _init_llm_client(self) -> None:
        """初始化 LLM 客户端 — 委托给 LLMClientManager。"""
        self._llm_client_manager = LLMClientManager(self._config, self._reflect_cache)
        self._llm_client_manager.init_llm_client()
        # 兼容属性：其他代码通过 self._llm_client 访问
        self._llm_client = self._llm_client_manager.llm_client
        # M7-11: 注入 LLM 客户端到 KG 三元组抽取模块，避免绕过 LLMBackend 直连
        if self._llm_client is not None:
            try:
                from omnimem.deep.kg.extraction import set_llm_client as set_kg_llm_client
                set_kg_llm_client(self._llm_client)
            except ImportError:
                pass
            try:
                from omnimem.governance.triple_extractor import get_triple_extractor
                model = getattr(self._llm_client, '_model', '') or self._config.get('llm_model', '')
                get_triple_extractor().inject_llm_client(self._llm_client, model)
            except ImportError:
                pass

    def _make_llm_call_fn(self):
        """生成 LLM 调用函数 — 委托给 LLMClientManager。"""
        return self._llm_client_manager.make_llm_call_fn()

    def _call_llm_for_reflect(
        self, prompt: str, system: str, max_tokens: int = 800
    ) -> str | None:
        """Reflect LLM 调用 — 委托给 LLMClientManager。"""
        return (
            self._llm_client_manager.call_llm_for_reflect(prompt, system, max_tokens)
            if self._llm_client_manager
            else None
        )

    def _call_llm_for_distill(
        self, prompt: str, system: str, max_tokens: int = 600, model: str | None = None
    ) -> str | None:
        """蒸馏 LLM 调用 — 委托给 LLMClientManager。"""
        return self._llm_client_manager.call_llm_for_distill(
            prompt, system, max_tokens, model
        )
