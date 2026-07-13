"""Handler 依赖注入层 — 打包 provider 子组件，减少直接穿透访问。

使用方式:
    在 handler 函数入口处:
        deps = extract_deps(provider)
    后续代码使用 deps.store, deps.retriever 等替代 provider._store, provider._retriever
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable

# ─────────────────────────────────────────────
# 服务层返回结构（Handler 与 Service 之间）
# ─────────────────────────────────────────────


class MemoryWriteResult(TypedDict, total=False):
    """memorize 处理结果（序列化为 JSON 前的结构化字典）。"""

    status: str
    memory_id: str
    wing: str
    room: str
    type: str
    privacy: str
    confidence: int
    kv_cached: bool
    encryption_status: str
    encryption_warning: str
    conflict_warning: dict[str, Any]
    # 早期返回/去重/冲突等场景携带的额外字段
    reason: str
    message: str
    existing_id: str
    existing: str
    dedup_result: dict[str, Any]
    content_preview: str


class RecallMemory(TypedDict, total=False):
    """recall 返回单条记忆摘要。"""

    memory_id: str
    content: str
    summary: str
    type: str
    confidence: float
    score: float
    privacy: str
    scope: str
    wing: str
    room: str
    stored_at: str
    entities: list[str]
    provenance: Any
    sealed: bool
    _source: str
    _group_start: bool
    _group_size: int
    _conflict: dict[str, str]
    _evidence_enriched: bool


class RecallResult(TypedDict, total=False):
    """recall 处理结果（序列化为 JSON 前的结构化字典）。"""

    status: str
    query: str
    count: int
    memories: list[RecallMemory]
    hint: str
    message: str
    trace: Any
    _quality: Any


class ConflictInfo(TypedDict, total=False):
    """冲突标记信息。"""

    conflict_type: str
    conflicting_with: str
    reason: str


# ─────────────────────────────────────────────
# Protocol 定义（避免循环导入，运行时轻量）
# ─────────────────────────────────────────────


class _SagaResultLike(Protocol):
    """Saga 执行结果的最小接口。"""

    success: bool
    failed_step: str
    error: str
    step_results: dict[str, Any]


class _VectorClockLike(Protocol):
    """向量时钟最小接口。"""

    def to_json(self) -> str: ...


@runtime_checkable
class _HasSearch(Protocol):
    def search(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class _HasGet(Protocol):
    def get(self, *args: Any, **kwargs: Any) -> Any: ...


class StoreProtocol(Protocol):
    """DrawerClosetStore 在 Handler 中使用的最小接口。"""

    def add(
        self,
        wing: str,
        room: str,
        content: str,
        memory_type: str = ...,
        confidence: int = ...,
        privacy: str = ...,
        provenance: dict[str, Any] | None = ...,
        vc: str = ...,
        memory_id: str = ...,
        **kwargs: Any,
    ) -> str: ...

    def get(self, memory_id: str) -> dict[str, Any] | None: ...
    def delete(self, memory_id: str) -> bool: ...
    def search(
        self,
        *,
        wing: str = ...,
        memory_type: str = ...,
        limit: int = ...,
    ) -> list[dict[str, Any]]: ...

    def search_by_content(self, query: str, limit: int = ...) -> list[dict[str, Any]]: ...
    def update_privacy(
        self, memory_id: str, privacy: str, new_wing: str | None = ...
    ) -> bool: ...

    def update_field(self, memory_id: str, **fields: Any) -> bool: ...
    def flush(self) -> None: ...

    @property
    def meta_store(self) -> Any: ...


class IndexProtocol(Protocol):
    """ThreeLevelIndex 在 Handler 中使用的最小接口。"""

    def add(self, **kwargs: Any) -> None: ...
    def get(self, memory_id: str) -> dict[str, Any] | None: ...
    def delete(self, memory_id: str) -> bool: ...
    def update_privacy(self, memory_id: str, privacy: str) -> bool: ...
    def update_field(
        self, memory_id: str, immediate: bool = ..., **fields: Any
    ) -> bool: ...

    def search_all_for_retrieval(self, limit: int = ...) -> list[dict[str, Any]]: ...
    def flush(self) -> None: ...


class RetrieverProtocol(Protocol):
    """HybridRetriever 在 Handler 中使用的最小接口。"""

    def add(self, content: str, memory_id: str, metadata: dict[str, Any]) -> None: ...
    def delete(self, memory_id: str) -> None: ...
    def search(
        self, query: str, *, max_tokens: int = ..., mode: str = ..., **kwargs: Any
    ) -> list[dict[str, Any]]: ...

    async def async_search(
        self, query: str, *, max_tokens: int = ..., mode: str = ..., **kwargs: Any
    ) -> list[dict[str, Any]]: ...

    def update_metadata(self, memory_id: str, metadata: dict[str, Any]) -> None: ...
    def flush(self) -> None: ...
    def persist_embedding_cache(self) -> None: ...
    def rebuild_all_from_entries(
        self, entries: list[dict[str, Any]]
    ) -> dict[str, int]: ...


class ContextManagerProtocol(Protocol):
    """ContextManager 在 Handler 中使用的最小接口。"""

    def refine_recall_results(
        self, results: list[dict[str, Any]], *, max_tokens: int = ...
    ) -> list[dict[str, Any]]: ...


class PrivacyManagerProtocol(Protocol):
    """PrivacyManager 在 Handler 中使用的最小接口。"""

    def filter(
        self,
        results: list[dict[str, Any]],
        session_id: str = ...,
        max_privacy: str = ...,
    ) -> list[dict[str, Any]]: ...

    def set(self, memory_id: str, level: str, new_wing: str = ...) -> None: ...
    def encrypt_content_with_status(self, content: str) -> dict[str, str]: ...


class ConflictResultLike(Protocol):
    """ConflictResolver 返回结果的最小接口。"""

    has_conflict: bool
    existing_memory: str
    existing_id: str
    conflict_type: str
    action: str
    reason: str


class ConflictResolverProtocol(Protocol):
    """ConflictResolver 在 Handler 中使用的最小接口。"""

    def check(
        self, content: str, existing_memories: list[dict[str, Any]] | None = ...
    ) -> ConflictResultLike: ...

    def resolve(self, content: str, conflict: ConflictResultLike) -> ConflictResultLike: ...


class ForgettingCurveProtocol(Protocol):
    """ForgettingCurve 在 Handler 中使用的最小接口。"""

    def archive(self, memory_id: str) -> None: ...
    def reactivate(self, memory_id: str) -> None: ...
    def record_access(self, memory_id: str, memory_type: str = ...) -> None: ...
    def get_status(self) -> dict[str, Any]: ...
    def get_upgrade_candidates(self, min_recall: int = ...) -> list[dict[str, Any]]: ...
    def mark_upgraded_to_wiki(self, memory_id: str, wiki_path: str) -> None: ...


class AuditLoggerProtocol(Protocol):
    """AuditLogger 在 Handler 中使用的最小接口。"""

    def log(
        self,
        operation: str,
        *,
        memory_id: str | None = ...,
        details: dict[str, Any] | None = ...,
        result: str = ...,
        instance_id: str | None = ...,
    ) -> None: ...

    def query(
        self,
        *,
        operation: str | None = ...,
        memory_id: str | None = ...,
        from_time: str | None = ...,
        to_time: str | None = ...,
        limit: int = ...,
    ) -> list[dict[str, Any]]: ...


class AuditorProtocol(Protocol):
    """GovernanceAuditor 在 Handler 中使用的最小接口。"""

    def run_full_audit(self, limit: int = ...) -> dict[str, Any]: ...


class KnowledgeGraphProtocol(Protocol):
    """KnowledgeGraph 在 Handler 中使用的最小接口。"""

    def extract_and_store(
        self, content: str, *, memory_id: str = ..., confidence: float = ...
    ) -> dict[str, Any]: ...

    def graph_search(
        self, query: str, *, max_depth: int = ..., limit: int = ...
    ) -> list[dict[str, Any]]: ...

    def graph_rag_search(
        self, query: str, *, max_depth: int = ..., limit: int = ...
    ) -> str: ...

    def _get_all_triples(self, limit: int = ...) -> list[dict[str, Any]]: ...


class TemporalKnowledgeGraphProtocol(Protocol):
    """TemporalKnowledgeGraph 在 Handler 中使用的最小接口。"""

    def add_triple_from_kg(
        self,
        *,
        subject: str = ...,
        predicate: str = ...,
        obj: str = ...,
        valid_at: str = ...,
        source_memory_id: str = ...,
        confidence: int = ...,
    ) -> None: ...

    def temporal_rag_context(self, query_entities: list[str]) -> str: ...


class VectorClockProtocol(Protocol):
    """VectorClock 在 Handler 中使用的最小接口。"""

    def increment(self, node_id: str) -> VectorClockProtocol: ...
    def to_json(self) -> str: ...


class SyncEngineProtocol(Protocol):
    """SyncEngine 在 Handler 中使用的最小接口。"""

    def get_instance_info(self) -> dict[str, Any]: ...
    def get_active_instances(self) -> list[dict[str, Any]]: ...


class RBACManagerProtocol(Protocol):
    """RBACManager 在 Handler 中使用的最小接口。"""

    def check_permission(self, user_id: str, permission: str) -> bool: ...
    def assign_role(self, user_id: str, role_name: str) -> None: ...
    def revoke_role(self, user_id: str, role_name: str) -> None: ...
    def add_role(self, role_name: str, permissions: list[str]) -> None: ...
    def get_user_permissions(self, user_id: str) -> list[str]: ...


class ProvenanceTrackerProtocol(Protocol):
    """ProvenanceTracker 在 Handler 中使用的最小接口。"""

    def track(
        self, content: str, *, source: str = ..., method: str = ...
    ) -> dict[str, Any]: ...

    def record(self, memory_id: str, provenance_data: Any) -> None: ...
    def lookup(self, memory_id: str) -> Any: ...


class TemporalDecayProtocol(Protocol):
    """TemporalDecay 在 Handler 中使用的最小接口。"""

    def apply(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class WingRoomManagerProtocol(Protocol):
    """WingRoomManager 在 Handler 中使用的最小接口。"""

    def resolve_wing(self, scope: str) -> str: ...
    def resolve_wing_from_privacy(self, privacy: str, memory_type: str = ...) -> str: ...
    def resolve_hall(self, memory_type: str) -> str: ...
    def resolve_room(self, content: str, wing: str, memory_type: str = ...) -> str: ...
    def tree(self, wing: str = ..., hall: str = ...) -> dict[str, Any]: ...
    def grep_rooms(self, pattern: str) -> list[dict[str, str]]: ...
    def count_memories(
        self, wing: str = ..., hall: str = ..., room: str = ...
    ) -> dict[str, int]: ...


class KVCacheProtocol(Protocol):
    """KVCache 在 Handler 中使用的最小接口。"""

    def check_and_auto_preload(
        self,
        *,
        key: str = ...,
        content: str = ...,
        metadata: dict[str, Any] = ...,
        source_memory_ids: list[str] = ...,
    ) -> bool: ...

    def get_stats(self) -> dict[str, Any]: ...


class LoRATrainerProtocol(Protocol):
    """LoRATrainer 在 Handler 中使用的最小接口。"""

    @property
    def is_available(self) -> bool: ...

    def train(self, *, shade: str = ..., epochs: int = ...) -> dict[str, Any]: ...
    def get_shades(self) -> list[dict[str, Any]]: ...
    def register_shade(self, shade_name: str, description: str) -> None: ...
    def switch_shade(self, shade_name: str) -> dict[str, Any] | None: ...

    @property
    def active_shade(self) -> str: ...


class ConsolidationEngineProtocol(Protocol):
    """ConsolidationEngine 在 Handler 中使用的最小接口。"""

    def submit(
        self, memory_id: str, content: str, *, memory_type: str = ...
    ) -> None: ...

    def get_stats(self) -> dict[str, Any]: ...


class SagaManagerProtocol(Protocol):
    """SagaManager / SagaCoordinator 在 Handler 中使用的最小接口。"""

    def execute(self, memory_id: str, steps: list[Any]) -> _SagaResultLike: ...


class DeduplicationEngineProtocol(Protocol):
    """DeduplicationEngine 在 Handler 中使用的最小接口。"""

    def semantic_dedup(
        self, content: str, memory_type: str, candidates: list[dict[str, Any]]
    ) -> dict[str, Any]: ...


class KMSManagerProtocol(Protocol):
    """KMSManager 在 Handler 中使用的最小接口。"""

    def configure_provider(self, provider_name: str, **kwargs: Any) -> None: ...
    def rotate_key(self, key_id: str) -> None: ...

    @property
    def provider(self) -> str: ...

    @property
    def _config(self) -> dict[str, Any]: ...


class TraceChainProtocol(Protocol):
    """TraceChain 在 Handler 中使用的最小接口。"""

    def record_derivation(
        self,
        *,
        parent_node_ids: list[str] = ...,
        child_node_id: str = ...,
        child_layer: str = ...,
        ref_path: str = ...,
    ) -> None: ...


class PipelineSchedulerProtocol(Protocol):
    """PipelineScheduler 在 Handler 中使用的最小接口。"""

    def on_new_memory(self, *, session_key: str = ...) -> None: ...


class BackgroundTaskExecutorProtocol(Protocol):
    """后台任务执行器在 Handler 中使用的最小接口。"""

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...


class QualityEvaluatorProtocol(Protocol):
    """RetrievalQualityEvaluator 在 Handler 中使用的最小接口。"""

    def evaluate(
        self,
        *,
        query: str = ...,
        results: list[dict[str, Any]] = ...,
        relevant_ids: set[str] = ...,
        latency_ms: float = ...,
    ) -> Any: ...

    def record_evaluation(self, metrics: Any) -> None: ...


class SecurityValidatorProtocol(Protocol):
    """SecurityValidator 在 Handler 中使用的最小接口。"""

    @staticmethod
    def scan_threats(content: str) -> str | None: ...


@dataclass(frozen=True)
class HandlerDependencies:
    """Handler 所需依赖的不可变快照。

    所有字段均为 Optional，handler 按需访问。
    frozen=True 防止 handler 内意外修改依赖引用。
    """

    # ── 核心存储 ──
    store: StoreProtocol | None = None               # DrawerClosetStore
    index: IndexProtocol | None = None               # ThreeLevelIndex
    retriever: RetrieverProtocol | None = None       # HybridRetriever

    # ── 治理引擎 ──
    forgetting: ForgettingCurveProtocol | None = None          # ForgettingCurve
    conflict_resolver: ConflictResolverProtocol | None = None  # ConflictResolver
    privacy: PrivacyManagerProtocol | None = None              # PrivacyManager
    provenance: ProvenanceTrackerProtocol | None = None        # ProvenanceTracker
    temporal_decay: TemporalDecayProtocol | None = None        # TemporalDecay
    temporal_kg: TemporalKnowledgeGraphProtocol | None = None  # TemporalKnowledgeGraph
    rbac: RBACManagerProtocol | None = None                    # RBACManager (optional)
    security: SecurityValidatorProtocol | None = None          # SecurityValidator

    # ── 深层记忆 ──
    knowledge_graph: KnowledgeGraphProtocol | None = None      # KnowledgeGraph
    kg: KnowledgeGraphProtocol | None = None                   # KnowledgeGraph 别名
    consolidation: ConsolidationEngineProtocol | None = None   # ConsolidationEngine
    kv_cache: KVCacheProtocol | None = None                    # KVCache
    lora_trainer: LoRATrainerProtocol | None = None            # LoRATrainer
    sync_engine: SyncEngineProtocol | None = None              # SyncEngine

    # ── 工作记忆 ──
    context_manager: ContextManagerProtocol | None = None      # ContextManager
    quality_evaluator: QualityEvaluatorProtocol | None = None  # QualityEvaluator
    wing_room: WingRoomManagerProtocol | None = None           # WingRoomManager
    audit_logger: AuditLoggerProtocol | None = None            # AuditLogger
    auditor: AuditorProtocol | None = None                     # GovernanceAuditor
    l3_consolidation: ConsolidationEngineProtocol | None = None  # L3Consolidation (alias)

    # ── 服务方法 ──
    should_store: Callable[[str], bool] | None = None
    unified_candidate_search: Callable[[str], list[dict[str, Any]]] | None = None
    semantic_dedup: Callable[..., dict[str, Any]] | None = None

    dedup: DeduplicationEngineProtocol | None = None           # DeduplicationEngine
    kms: KMSManagerProtocol | None = None                      # KMSManager
    create_backup: Callable[..., tuple[str | None, int]] | None = None
    cleanup_old_backups: Callable[[int], None] | None = None
    get_next_vc: Callable[[], _VectorClockLike | None] | None = None
    trace_chain: TraceChainProtocol | None = None              # TraceChain
    pipeline_scheduler: PipelineSchedulerProtocol | None = None  # PipelineScheduler

    # ── 配置 ──
    config: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    instance_id: str | None = None
    saga: SagaManagerProtocol | None = None                    # SagaManager
    data_dir: Any = None                                       # Path
    l3_engine: Any = None                                      # L3Engine
    memory_monitor: Any = None                                 # MemoryMonitor
    bg_executor: BackgroundTaskExecutorProtocol | None = None  # BackgroundTaskExecutor
    vector_clock: VectorClockProtocol | None = None            # VectorClock


def extract_deps(provider: Any) -> HandlerDependencies:
    """从 provider 提取 Handler 所需的全部依赖。

    安全访问：对 optional 组件使用 getattr(provider, '_xxx', None)。
    """
    # 配置：优先 _config，兼容 config 属性名；OmniMemConfig 实例转为其 values 字典
    config = getattr(provider, "_config", None)
    if config is not None and hasattr(config, "values"):
        config = config.values
    elif not isinstance(config, dict):
        config = getattr(provider, "config", None)
        if config is not None and hasattr(config, "values"):
            config = config.values
    if not isinstance(config, dict):
        config = {}

    return HandlerDependencies(
        # 核心存储
        store=getattr(provider, "_store", None),
        index=getattr(provider, "_index", None),
        retriever=getattr(provider, "_retriever", None),
        # 治理引擎
        forgetting=getattr(provider, "_forgetting", None),
        conflict_resolver=getattr(provider, "_conflict_resolver", None),
        privacy=getattr(provider, "_privacy", None),
        provenance=getattr(provider, "_provenance", None),
        temporal_decay=getattr(provider, "_temporal_decay", None),
        temporal_kg=getattr(provider, "_temporal_kg", None),
        rbac=getattr(provider, "_rbac", None),
        security=getattr(provider, "_security", None),
        # 深层记忆
        knowledge_graph=getattr(provider, "_knowledge_graph", None),
        kg=getattr(provider, "_knowledge_graph", None),
        consolidation=getattr(provider, "_consolidation", None),
        kv_cache=getattr(provider, "_kv_cache", None),
        lora_trainer=getattr(provider, "_lora_trainer", None),
        sync_engine=getattr(provider, "_sync_engine", None),
        # 工作记忆
        context_manager=getattr(provider, "_context_manager", None),
        quality_evaluator=getattr(provider, "_quality_evaluator", None),
        wing_room=getattr(provider, "_wing_room", None),
        audit_logger=getattr(provider, "_audit_logger", None),
        auditor=getattr(provider, "_auditor", None),
        l3_consolidation=getattr(provider, "_consolidation", None),
        # 服务方法
        should_store=getattr(provider, "_should_store", None),
        unified_candidate_search=getattr(provider, "_unified_candidate_search", None),
        semantic_dedup=getattr(provider, "_semantic_dedup", None),
        dedup=getattr(provider, "_dedup", None),
        kms=getattr(provider, "_kms", None),
        create_backup=getattr(provider, "_create_backup", None),
        cleanup_old_backups=getattr(provider, "_cleanup_old_backups", None),
        get_next_vc=getattr(provider, "get_next_vc", None),
        trace_chain=getattr(provider, "_trace_chain", None),
        pipeline_scheduler=getattr(provider, "_pipeline_scheduler", None),
        config=config,

        session_id=getattr(provider, "_session_id", ""),
        instance_id=getattr(provider, "_instance_id", None),
        saga=getattr(provider, "_saga", None),
        data_dir=getattr(provider, "_data_dir", None),
        l3_engine=getattr(provider, "_l3_engine", None),
        memory_monitor=getattr(provider, "_memory_monitor", None),
        bg_executor=getattr(provider, "_bg_executor", None),
        vector_clock=getattr(provider, "_vector_clock", None),
    )
