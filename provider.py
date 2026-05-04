"""OmniMem Provider — 五层混合记忆系统，实现 Hermes MemoryProvider ABC。

安装: 将本目录放入 plugins/memory/omnimem/
配置: config.yaml → memory.provider: omnimem

五层架构:
  L0 感知层  — 主动监控 + 信号检测 + 意图预测
  L1 工作记忆 — CoreBlock(常驻上下文) + Attachment(压缩后状态)
  L2 结构化记忆 — Wing/Room 宫殿导航 + Drawer/Closet 双存储
  L3 深层记忆 — Consolidation(事实→观察→心智模型) + 知识图谱
  L4 内化记忆 — KV Cache(高频) + LoRA(深层) [可选]

治理引擎(横切面):
  冲突仲裁 + 时间衰减 + 遗忘曲线 + 隐私分级 + 溯源追踪
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider

from omnimem.compression.pipeline import CompressionPipeline
from omnimem.config import OmniMemConfig
from omnimem.core.attachment import build_attachments
from omnimem.core.dedup import SemanticDedupService
from omnimem.core.tool_router import (
    ToolRouter,
    handle_compact,
    handle_reflect,
    handle_detail,
    build_system_prompt,
    run_prefetch,
    run_queue_prefetch,
    l3_recall,
    init_llm_client,
    make_llm_call_fn,
    call_llm_for_reflect,
    retry_index_add,
    retry_retriever_add,
    retry_kg_extract,
    apply_sync_change,
    get_config_schema as _get_config_schema_impl,
    save_config as _save_config_impl,
)
from omnimem.governance.vector_clock import VectorClock
from omnimem.handlers.compat_handler import CompatHandler
from omnimem.handlers.govern import _scan_memory_conflicts as _scan_memory_conflicts_impl
from omnimem.handlers.govern import handle_govern as _handle_govern_impl
from omnimem.handlers.memorize import handle_memorize as _handle_memorize_impl
from omnimem.handlers.recall import handle_recall as _handle_recall_impl
from omnimem.handlers.schemas import get_tool_schemas as _get_tool_schemas
from omnimem.utils.security import SecurityValidator
from omnimem.core.memory_monitor import MemoryMonitor
from omnimem.facades import (
    StorageFacade,
    RetrievalFacade,
    GovernanceFacade,
    DeepMemoryFacade,
    SyncFacade,
)

logger = logging.getLogger(__name__)


class OmniMemProvider(MemoryProvider):  # type: ignore[misc]
    """OmniMem: 五层混合记忆系统，实现 Hermes MemoryProvider ABC。

    五层架构:
      L0 感知层  — 主动监控 + 信号检测 + 意图预测
      L1 工作记忆 — CoreBlock(常驻上下文) + Attachment(压缩后状态)
      L2 结构化记忆 — Wing/Room 宫殿导航 + Drawer/Closet 双存储
      L3 深层记忆 — Consolidation(事实→观察→心智模型) + 知识图谱
      L4 内化记忆 — KV Cache(高频) + LoRA(深层) [可选]

    治理引擎(横切面):
      冲突仲裁 + 时间衰减 + 遗忘曲线 + 隐私分级 + 溯源追踪

    暴露给 Agent 的 7 个工具接口:
      omni_memorize — 主动存储记忆
      omni_recall   — 主动检索记忆（RAG/LLM/关键词三种模式）
      omni_compact  — 压缩前准备
      omni_reflect  — L3 深层反思（四步循环 + Disposition 性格修饰）
      omni_govern   — 治理操作（shade/conflict/kv_cache/stats等）
      omni_detail   — 按需拉取记忆细节（lazy provisioning）
      memory        — 兼容内置 memory 工具（add/replace/remove）
    """

    # ─── MemoryProvider ABC 必须实现 ────────────────────────────

    @property
    def name(self) -> str:
        return "omnimem"

    def is_available(self) -> bool:
        """检查核心依赖是否就绪。无需 API Key，本地零依赖模式始终可用。"""
        try:
            import chromadb  # noqa: F401
            import rank_bm25  # noqa: F401

            return True
        except ImportError:
            logger.warning("OmniMem: 缺少依赖。运行: pip install chromadb rank-bm25")
            return False

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        hermes_home = kwargs.get("hermes_home", str(Path.home() / ".hermes"))
        platform = kwargs.get("platform", "cli")
        agent_context = kwargs.get("agent_context", "primary")

        self._should_write = agent_context == "primary"

        self._data_dir = Path(hermes_home) / "omnimem"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id

        self._config = OmniMemConfig(self._data_dir)

        self._init_l1()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._init_store): "store",
                executor.submit(self._init_retrieval): "retrieval",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning("Init %s failed: %s", name, e)

        try:
            self._init_governance_sync_services()
        except Exception as e:
            logger.warning("Init governance_sync_services failed: %s", e)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self._init_reflect): "reflect",
                executor.submit(self._init_lora): "lora",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning("Init %s failed: %s", name, e)

        logger.info(
            "OmniMem initialized: session=%s, platform=%s, data_dir=%s, L3=enabled, L4=enabled",
            session_id,
            platform,
            self._data_dir,
        )

        self._memory_monitor = MemoryMonitor(
            interval=self._config.get("memory_monitor_interval", 60.0),
            warning_mb=self._config.get("memory_warning_mb", 500.0),
        )
        self._memory_monitor.start()

        try:
            indexed_entries = self._index.search_l1(limit=2000)
            if indexed_entries:
                self._store.warm_up(indexed_entries[:500])
                rebuilt = self._retriever.rebuild_bm25_from_entries(indexed_entries)
                if rebuilt > 0:
                    logger.debug(
                        "OmniMem: warmed up %d entries, rebuilt BM25 with %d entries",
                        min(len(indexed_entries), 500),
                        rebuilt,
                    )
        except Exception as e:
            logger.debug("OmniMem warm-up/BM25 rebuild failed (non-fatal): %s", e)

    def _init_l1(self) -> None:
        self._storage = StorageFacade(self._data_dir, self._config)

    def _init_store(self) -> None:
        self._storage.init_l2()

    def _init_retrieval(self) -> None:
        self._retrieval = RetrievalFacade(self._data_dir, self._config, self._storage)

    def _init_governance_sync_services(self) -> None:
        self._governance = GovernanceFacade(
            self._data_dir, self._config, self._session_id,
            self._storage, self._retrieval.retriever,
        )
        self._sync = SyncFacade(
            self._data_dir, self._config, self._session_id,
            self._storage, self._retrieval,
        )
        self._sync.bind_provenance(self._governance.provenance)

        self._instance_id = self._governance.instance_id
        self._turn_count = 0
        self._last_save_turn = 0
        self._save_interval = self._config.get("save_interval", 15)
        self._system_prompt_cache_turn = -1
        self._system_prompt_cache_value = ""
        self._init_llm_client()
        self._compression_pipeline = CompressionPipeline(
            llm_call_fn=self._make_llm_call_fn(),
            config=self._config,
        )
        self._compat_handler = CompatHandler(
            memorize_fn=self._handle_memorize,
            store=self._store,
            forgetting=self._forgetting,
            extract_core_fact_fn=self._extract_core_fact,
        )
        self._dedup_service = SemanticDedupService(self._store, self._retriever)
        self._tool_router = ToolRouter(
            memorize_fn=self._handle_memorize,
            recall_fn=self._handle_recall,
            govern_fn=self._handle_govern,
            reflect_fn=self._handle_reflect,
            compact_fn=self._handle_compact,
            detail_fn=self._handle_detail,
            memory_compat_fn=self._handle_builtin_memory_compat,
        )

    def _init_reflect(self) -> None:
        self._deep = DeepMemoryFacade(
            self._data_dir, self._config,
            recall_fn=self._l3_recall,
            llm_fn=self._call_llm_for_reflect,
            llm_client=self._llm_client,
        )

    def _init_lora(self) -> None:
        """初始化 L4 内化记忆。"""
        self._sync.init_l4()

    # ─── Facade 兼容属性（旧代码通过 _X 访问新 Facade） ────────

    # Storage
    @property
    def _soul(self) -> Any: return self._storage.soul
    @property
    def _core_block(self) -> Any: return self._storage.core_block
    @property
    def _budget(self) -> Any: return self._storage.budget
    @property
    def _attachments(self) -> Any: return self._storage.attachments
    @_attachments.setter
    def _attachments(self, val: Any) -> None: self._storage.attachments = val
    @property
    def _wing_room(self) -> Any: return self._storage.wing_room
    @property
    def _store(self) -> Any: return self._storage.store
    @property
    def _index(self) -> Any: return self._storage.index
    @property
    def _md_store(self) -> Any: return self._storage.md_store

    # Retrieval
    @property
    def _retriever(self) -> Any: return self._retrieval.retriever
    @property
    def _context_manager(self) -> Any: return self._retrieval.context_manager
    @property
    def _perception(self) -> Any: return self._retrieval.perception
    @property
    def _feedback(self) -> Any: return self._retrieval.feedback
    @property
    def _prefetch_cache(self) -> Any: return self._retrieval.prefetch_cache
    @_prefetch_cache.setter
    def _prefetch_cache(self, val: Any) -> None: self._retrieval.prefetch_cache = val
    @property
    def _prefetch_lock(self) -> Any: return self._retrieval.prefetch_lock
    @property
    def _reflect_cache(self) -> Any: return self._retrieval._reflect_cache
    @property
    def _prefetch_executor(self) -> Any: return self._retrieval._prefetch_executor

    # Dedup
    @property
    def _dedup(self) -> Any: return self._dedup_service

    # Governance
    @property
    def _conflict_resolver(self) -> Any: return self._governance.conflict_resolver
    @property
    def _temporal_decay(self) -> Any: return self._governance.temporal_decay
    @property
    def _forgetting(self) -> Any: return self._governance.forgetting
    @property
    def _privacy(self) -> Any: return self._governance.privacy
    @property
    def _provenance(self) -> Any: return self._governance.provenance
    @property
    def _sync_engine(self) -> Any: return self._governance.sync_engine
    @property
    def _vector_clock(self) -> Any: return self._governance.vector_clock
    @property
    def _auditor(self) -> Any: return self._governance.auditor
    @property
    def _audit_logger(self) -> Any: return self._governance.audit_logger
    @property
    def _rbac(self) -> Any: return self._governance.rbac

    # Sync
    @property
    def _saga(self) -> Any: return self._sync.saga
    @property
    def _bg_executor(self) -> Any: return self._sync.bg_executor
    @property
    def _store_service(self) -> Any: return self._sync.store_service
    @property
    def _kv_cache(self) -> Any: return self._sync.kv_cache
    @property
    def _lora_trainer(self) -> Any: return self._sync.lora_trainer

    # Deep
    @property
    def _consolidation(self) -> Any: return self._deep.consolidation
    @property
    def _knowledge_graph(self) -> Any: return self._deep.knowledge_graph
    @property
    def _reflect_engine(self) -> Any: return self._deep.reflect_engine

    # ─── 异步接口（P2方案五） ───────────────────────────────────

    @property
    def async_provider(self) -> Any:
        """获取异步包装器（延迟初始化）。"""
        if not hasattr(self, "_async_provider"):
            from omnimem.core.async_provider import AsyncOmniMemProvider

            self._async_provider = AsyncOmniMemProvider(self)
        return self._async_provider

    # ─── 上下文注入 ─────────────────────────────────────────────

    def system_prompt_block(self) -> str:
        result, cache_turn, cache_value = build_system_prompt(
            data_dir=str(self._data_dir),
            store=self._store,
            core_block=self._core_block,
            context_manager=self._context_manager,
            config=self._config,
            turn_count=self._turn_count,
            system_prompt_cache_turn=self._system_prompt_cache_turn,
            system_prompt_cache_value=self._system_prompt_cache_value,
            last_query=getattr(self, "_last_query", ""),
        )
        self._system_prompt_cache_turn = cache_turn
        self._system_prompt_cache_value = cache_value
        return result

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_query = query
        result, new_cache = run_prefetch(
            query=query,
            session_id=session_id,
            config=self._config,
            retriever=self._retriever,
            context_manager=self._context_manager,
            kv_cache=self._kv_cache,
            knowledge_graph=self._knowledge_graph,
            temporal_decay=self._temporal_decay,
            privacy=self._privacy,
            prefetch_cache=self._prefetch_cache,
            prefetch_lock=self._prefetch_lock,
        )
        with self._prefetch_lock:
            self._prefetch_cache = new_cache
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        def _bg_prefetch() -> None:
            serialized = run_queue_prefetch(
                query=query,
                session_id=session_id,
                config=self._config,
                retriever=self._retriever,
                temporal_decay=self._temporal_decay,
                privacy=self._privacy,
                prefetch_lock=self._prefetch_lock,
            )
            with self._prefetch_lock:
                self._prefetch_cache = serialized

        self._prefetch_executor.submit(_bg_prefetch)

    # ─── 对话同步 ─────────────────────────────────────────────

    # ─── 输入净化：剥离系统注入内容，防止递归膨胀 ───

    @staticmethod
    def _strip_system_injections(text: str) -> str:
        """剥离 prefetch 注入的记忆区块，只保留用户原始输入。

        委托 SecurityValidator 实现，支持 Unicode 归一化和编码绕过检测。
        """
        return SecurityValidator.strip_system_injections(text)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """每轮对话后：感知 → 写入 → 治理。"""
        if not self._should_write:
            return

        # ★ 输入净化：剥离系统注入内容，只对原始用户输入做感知
        clean_user = self._strip_system_injections(user_content)

        # L0 感知
        signals = self._perception.detect_signals(clean_user, assistant_content)

        # 信号驱动的记忆写入
        # ★ 信号互斥：correction > reinforcement > fact
        # 避免同一条信息被多个信号触发重复写入
        if signals.has_correction:
            self._store_service.store_correction(signals, user_content)
        elif signals.has_reinforcement:
            self._store_service.store_reinforcement(signals, user_content)
        elif signals.should_memorize:
            self._store_service.store_fact(signals, user_content)

        # 定期自动存档
        self._turn_count += 1
        self._store_service.turn_count = self._turn_count
        if self._turn_count - self._store_service.last_save_turn >= self._save_interval:
            self._store_service.auto_checkpoint(user_content, self._save_interval)
            self._last_save_turn = self._store_service.last_save_turn

        # ★ P0方案二：统一后台任务执行器替代每轮新建 threading.Thread
        self._bg_executor.submit(
            self._retriever.index_update,
            user_content,
            assistant_content,
        )

    # ─── 工具暴露 ─────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """OmniMem 暴露 7 个工具给 Agent — 委托到 handlers/schemas.py。"""
        return _get_tool_schemas()

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        try:
            return self._tool_router.route(tool_name, args)
        except Exception as e:
            logger.error("OmniMem tool %s failed: %s", tool_name, e)
            return json.dumps({"error": str(e)})

    # ─── 扩展 Hooks ─────────────────────────────────────────────

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        """每轮开始：配置热重载 + 意图预测 + 预加载 + 分布式同步。"""
        if not self._should_write:
            return
        # ★ 周期性热重载配置（每 10 轮检查一次文件变更）
        if turn_number % 10 == 0:
            self._config.reload()
        # ★ P1方案四：每 20 轮更新检索来源权重（基于反馈统计）
        if turn_number % 20 == 0 and hasattr(self, "_feedback") and self._feedback:
            try:
                weights = self._feedback.get_source_weights(window=100)
                if weights:
                    self._retriever.set_source_weights(weights)
                    logger.debug("Updated source weights from feedback: %s", weights)
            except Exception as e:
                logger.warning("Feedback weight update failed: %s", e)
        # ★ 分布式同步：每 5 轮从其他实例拉取变更（changelog 模式）
        if (
            turn_number % 5 == 0
            and hasattr(self, "_sync_engine")
            and self._sync_engine
            and self._sync_engine._config.mode == "changelog"
        ):
            try:
                applied = self._sync_engine.sync_from_others(
                    apply_fn=self._apply_sync_change,
                    get_local_fn=lambda mid: self._store.get(mid),
                )
                if applied > 0:
                    logger.info("OmniMem sync: applied %d changes from other instances", applied)
            except Exception as e:
                logger.warning("OmniMem sync failed: %s", e)
        predicted = self._perception.predict_intent(message)
        if predicted:
            self.queue_prefetch(predicted)

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """会话结束：Consolidation + 治理归档。"""
        if not self._should_write:
            return

        # 关闭后台预取线程池（优雅释放资源）
        if hasattr(self, "_prefetch_executor") and self._prefetch_executor:
            self._prefetch_executor.shutdown(wait=False)

        # 1. 从完整对话中提取遗漏的记忆
        self._store_service.extract_session_memories(
            messages,
            self._strip_system_injections,
            self._should_store,
            lambda args: self._handle_memorize(args),
        )

        # 2. 治理：遗忘曲线归档
        self._forgetting.run_archive_cycle()

        # 3. L3 Consolidation: 处理待升华的记忆
        if self._consolidation:
            processed = self._consolidation.process_pending()
            if processed > 0:
                logger.info("OmniMem consolidation: processed %d memories", processed)

        # 4. L4: 将心智模型提交到 LoRA 训练队列
        if self._consolidation and self._lora_trainer:
            try:
                models = self._consolidation.get_mental_models(limit=20)
                if models:
                    self._lora_trainer.submit_training_data(models, shade="default")
                    logger.info(
                        "OmniMem L4: submitted %d mental models for LoRA training", len(models)
                    )
            except Exception as e:
                logger.warning("OmniMem L4 submit failed: %s", e)

        # 5. 刷新存储缓冲与索引
        self._store.flush()
        self._retriever.flush()

        # ★ P0方案六：治理巡检（每 10 轮执行一次一致性审计）
        if self._turn_count % 10 == 0 and hasattr(self, "_auditor") and self._auditor:
            try:
                health = self._auditor.quick_health_check()
                if not health["healthy"]:
                    audit = self._auditor.run_full_audit(limit=1000)
                    if audit["total_issues"] > 0:
                        fixed = self._auditor.repair(audit)
                        logger.info(
                            "OmniMem governance audit: %d issues found, %d fixed",
                            audit["total_issues"],
                            fixed,
                        )
            except Exception as e:
                logger.warning("Governance audit failed: %s", e)

        # ★ P0方案二：Saga pending 重试（会话结束前补偿未完成的索引写入）
        if self._saga.get_pending():
            fixed = self._saga.retry_pending(
                {
                    "three_level_index": lambda mid: self._retry_index_add(mid),
                    "retriever": lambda mid: self._retry_retriever_add(mid),
                    "knowledge_graph": lambda mid: self._retry_kg_extract(mid),
                }
            )
            if fixed > 0:
                logger.info("OmniMem saga retry: fixed %d pending records", fixed)

        # ★ P0方案二：关闭统一后台任务执行器
        if hasattr(self, "_bg_executor") and self._bg_executor:
            self._bg_executor.shutdown(wait=True)

        logger.info("OmniMem session end: processed %d messages", len(messages))

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """压缩前：构建 Attachment + 紧急保存。"""
        self._store.flush()
        saved_context = self._store_service.emergency_save(messages)

        attachments = build_attachments(messages)

        parts = []
        if saved_context:
            parts.append(saved_context)
        if attachments:
            att_text = "\n".join(f"[{a.kind}] {a.title}: {a.body[:200]}" for a in attachments)
            parts.append(f"### Pre-Compression Attachments\n{att_text}")

        result = "\n\n".join(parts)

        if self._config.get("enable_compression", False) and result:
            result = self._compression_pipeline.compress(result)

        return result

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """内置记忆写入时：冲突检测。"""
        if action == "add":
            conflict = self._conflict_resolver.check(content)
            if conflict.has_conflict:
                logger.warning(
                    "OmniMem: conflict detected with existing memory: %s",
                    conflict.existing_memory,
                )

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any
    ) -> None:
        """子 Agent 完成时：记录过程记忆。"""
        if not self._should_write:
            return
        self._store_service.store_delegation(task, result, child_session_id)

    # ─── 配置 ─────────────────────────────────────────────────

    def get_config_schema(self) -> list[dict[str, Any]]:
        return _get_config_schema_impl()

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        _save_config_impl(values, hermes_home)

    def shutdown(self) -> None:
        """清理：刷新所有缓冲到磁盘。

        关闭顺序: retriever → md_store → index → knowledge_graph →
                  consolidation → reflect_engine → kv_cache → lora_trainer →
                  provenance → sync_engine → forgetting → executors
        """
        if hasattr(self, "_memory_monitor") and self._memory_monitor:
            self._memory_monitor.stop()
        # 0. 关闭线程池（先于存储关闭，避免后台任务写入已关闭的存储）
        if hasattr(self, "_prefetch_executor") and self._prefetch_executor:
            self._prefetch_executor.shutdown(wait=False)
        if hasattr(self, "_bg_executor") and self._bg_executor:
            self._bg_executor.shutdown(wait=True)

        # 1. 存储层
        self._store.flush()
        self._retriever.flush()
        self._md_store.flush()
        self._index.close()
        if hasattr(self, "_perception") and self._perception:
            self._perception.close()
        if self._knowledge_graph:
            self._knowledge_graph.close()
        if self._consolidation:
            self._consolidation.close()
        if hasattr(self, "_reflect_engine") and self._reflect_engine:
            self._reflect_engine.close()
        if hasattr(self, "_kv_cache") and self._kv_cache:
            self._kv_cache.close()
        if hasattr(self, "_lora_trainer") and self._lora_trainer:
            self._lora_trainer.close()
        if hasattr(self, "_provenance") and self._provenance:
            self._provenance.close()
        if hasattr(self, "_sync_engine") and self._sync_engine:
            self._sync_engine.close()
        # OPT-2: 关闭 LLM 客户端
        if hasattr(self, "_llm_client") and self._llm_client:
            self._llm_client.close()
        self._forgetting.close()
        logger.info("OmniMem shutdown complete")

    # ─── 工具实现（委托到 handlers 子模块） ─────────────────────

    def get_next_vc(self) -> VectorClock:
        """获取下一个向量时钟值（递增当前实例计数器）。"""
        self._vector_clock.increment(self._instance_id)
        return self._vector_clock  # type: ignore[no-any-return]

    def _apply_sync_change(self, change: dict[str, Any]) -> bool:
        return apply_sync_change(change, self._store, self._index, self._retriever, self._forgetting)

    def _handle_memorize(self, args: dict[str, Any]) -> str:
        """委托到 handlers/memorize.py。"""
        return _handle_memorize_impl(self, args)

    # ─── Saga 重试辅助方法 ────────────────────────────────────

    def _retry_index_add(self, memory_id: str) -> None:
        retry_index_add(memory_id, self._store, self._index)

    def _retry_retriever_add(self, memory_id: str) -> None:
        retry_retriever_add(memory_id, self._store, self._retriever)

    def _retry_kg_extract(self, memory_id: str) -> None:
        retry_kg_extract(memory_id, self._store, self._knowledge_graph)

    def _handle_recall(self, args: dict[str, Any]) -> str:
        """委托到 handlers/recall.py。"""
        result = _handle_recall_impl(self, args)
        # ★ 记录反馈：recall 返回的候选列表
        if hasattr(self, "_feedback") and self._feedback:
            try:
                data = json.loads(result)
                if data.get("status") == "found":
                    self._feedback.record_shown(
                        query=args.get("query", ""),
                        candidates=data.get("memories", []),
                    )
            except Exception as e:
                logger.warning("Feedback recording failed: %s", e)
        return result

    def _handle_govern(self, args: dict[str, Any]) -> str:
        """委托到 handlers/govern.py。"""
        return _handle_govern_impl(self, args)

    def _scan_memory_conflicts(self) -> list[dict[str, Any]]:
        """委托到 handlers/govern.py。"""
        return _scan_memory_conflicts_impl(self)

    def _handle_compact(self, args: dict[str, Any]) -> str:
        return handle_compact(args)

    def _handle_reflect(self, args: dict[str, Any]) -> str:
        return handle_reflect(args, self._consolidation, self._reflect_engine)

    # ─── omni_detail：按需拉取记忆细节 ─────────────────────────

    def _handle_detail(self, args: dict[str, Any]) -> str:
        return handle_detail(
            args,
            context_manager=self._context_manager,
            store=self._store,
            forgetting=self._forgetting,
            feedback=self._feedback if hasattr(self, "_feedback") else None,
            turn_count=self._turn_count,
            last_query=getattr(self, "_last_query", ""),
        )

    def _handle_builtin_memory_compat(self, args: dict[str, Any]) -> str:
        return self._compat_handler.handle(args)

    # ─── 反递归防护 ─────────────────────────────────────────────

    @staticmethod
    def _should_store(content: str) -> bool:
        """判断内容是否值得存储，过滤系统注入和递归内容。

        委托 SecurityValidator 实现，支持 Unicode 归一化和编码绕过检测。
        所有存储路径都应经过此检查。
        """
        should_store, reason = SecurityValidator.should_store(content)
        if not should_store and reason:
            logger.debug("SecurityValidator._should_store blocked: %s", reason)
        return should_store

    # ─── 语义去重 ─────────────────────────────────────────────

    def _extract_core_fact(self, text: str) -> str:
        """从原文中提取精简核心事实（委托给感知引擎）。"""
        return str(self._perception._extract_core_fact(text))

    @staticmethod
    def _compute_text_similarity(text_a: str, text_b: str) -> float:
        return SemanticDedupService.compute_text_similarity(text_a, text_b)

    def _semantic_dedup(
        self, content: str, memory_type: str, candidates: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        return self._dedup.semantic_dedup(content, memory_type, candidates)

    # ─── 内部辅助方法 ─────────────────────────────────────────────

    def _unified_candidate_search(self, content: str) -> list[dict[str, Any]]:
        return self._dedup.unified_candidate_search(content)

    def _search_candidates(self, content: str) -> list[dict[str, Any]]:
        return self._dedup.search_candidates(content)

    # ─── 内部辅助方法 ─────────────────────────────────────────

    def _l3_recall(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return l3_recall(query, self._retriever, self._store, limit)

    _REFLECT_CACHE_TTL = 60.0

    def _init_llm_client(self) -> None:
        self._llm_client = init_llm_client(self._config)
        # ★ R25修复ARCH-1：若 LLM 客户端凭证为空，尝试从 Hermes 主配置获取
        if self._llm_client and not getattr(self._llm_client, "_api_key", "").strip():
            try:
                from omnimem.utils.llm_client import AsyncLLMClient
                hermes_creds = AsyncLLMClient.load_credentials_from_hermes_config()
                if hermes_creds.get("api_key") and hermes_creds.get("base_url"):
                    logger.info("OmniMem: using Hermes main config LLM credentials for Reflect")
                    self._llm_client = AsyncLLMClient(
                        api_key=hermes_creds["api_key"],
                        base_url=hermes_creds["base_url"],
                        model=hermes_creds.get("model", "glm-5.1"),
                        max_concurrent=3,
                        timeout=30.0,
                        cache_ttl=self._REFLECT_CACHE_TTL,
                    )
            except Exception as e:
                logger.debug("OmniMem: failed to load Hermes main config LLM credentials: %s", e)

    def _make_llm_call_fn(self):
        return make_llm_call_fn(self._llm_client)

    def _call_llm_for_reflect(self, prompt: str, system: str, max_tokens: int = 800) -> str | None:
        return call_llm_for_reflect(
            prompt, system,
            llm_client=self._llm_client if hasattr(self, "_llm_client") else None,
            reflect_cache=self._reflect_cache,
            max_tokens=max_tokens,
        )
