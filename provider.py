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
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider
from omnimem.config import OmniMemConfig
from omnimem.context.manager import ContextBudget, ContextManager
from omnimem.core.attachment import CompactAttachment, build_attachments
from omnimem.core.background import BackgroundTaskExecutor
from omnimem.core.block import CoreBlock
from omnimem.core.budget import BudgetManager
from omnimem.core.saga import SagaCoordinator
from omnimem.core.soul import SoulSystem
from omnimem.core.store_service import MemoryStoreService
from omnimem.governance.auditor import GovernanceAuditor
from omnimem.governance.conflict import ConflictResolver
from omnimem.governance.decay import TemporalDecay
from omnimem.governance.feedback import FeedbackCollector
from omnimem.governance.forgetting import ForgettingCurve
from omnimem.governance.privacy import PrivacyManager
from omnimem.governance.provenance import ProvenanceTracker
from omnimem.governance.sync import SyncConfig, SyncEngine
from omnimem.governance.vector_clock import VectorClock
from omnimem.handlers._compat import compat_scan_memory_content as _compat_scan_memory_content
from omnimem.handlers.govern import _scan_memory_conflicts as _scan_memory_conflicts_impl
from omnimem.handlers.govern import handle_govern as _handle_govern_impl
from omnimem.handlers.memorize import handle_memorize as _handle_memorize_impl
from omnimem.handlers.recall import handle_recall as _handle_recall_impl

# ★ 委托调用：从 handlers 子模块导入拆分后的处理器
from omnimem.handlers.schemas import get_tool_schemas as _get_tool_schemas
from omnimem.memory.drawer_closet import DrawerClosetStore
from omnimem.memory.index import ThreeLevelIndex
from omnimem.memory.markdown_store import MarkdownStore
from omnimem.memory.wing_room import WingRoomManager
from omnimem.perception.engine import PerceptionEngine
from omnimem.retrieval.engine import HybridRetriever
from omnimem.utils.llm_client import AsyncLLMClient
from omnimem.utils.security import SecurityValidator

logger = logging.getLogger(__name__)


class OmniMemProvider(MemoryProvider):
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

    def initialize(self, session_id: str, **kwargs) -> None:
        """初始化所有 OmniMem 子系统。

        初始化流程:
          1. 解析 hermes_home / platform / agent_context 参数
          2. 非主要上下文(cron/flush)跳过写入
          3. 加载 OmniMemConfig 配置
          4. 分层初始化: _init_store(L1+L2) → _init_retriever(检索+治理+同步+感知) → _init_reflect(L3) → _init_lora(L4)
          5. 从 SQLite 索引预热内存索引，避免首次查询 rglob

        Args:
            session_id: 会话标识符
            **kwargs: hermes_home(存储根目录), platform(cli/web/api), agent_context(primary/cron/flush)
        """
        hermes_home = kwargs.get("hermes_home", str(Path.home() / ".hermes"))
        platform = kwargs.get("platform", "cli")
        agent_context = kwargs.get("agent_context", "primary")

        # 跳过非主要上下文（cron/flush 不写入记忆）
        self._should_write = agent_context == "primary"

        # 存储根目录
        self._data_dir = Path(hermes_home) / "omnimem"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id

        # ─── 配置 ───
        self._config = OmniMemConfig(self._data_dir)

        # ─── 分层初始化 ───
        self._init_store()
        self._init_retriever()
        self._init_reflect()
        self._init_lora()

        logger.info(
            "OmniMem initialized: session=%s, platform=%s, data_dir=%s, L3=enabled, L4=enabled",
            session_id,
            platform,
            self._data_dir,
        )

        # ★ 启动预热 + BM25 重建：合并为一次查询，减少 SQLite IO
        # 向量检索(ChromaDB)是持久化的，但 BM25 是内存的，跨会话后为空
        try:
            indexed_entries = self._index.search_l1(limit=2000)
            if indexed_entries:
                # 前 500 条用于 warm_up，全部 2000 条用于 BM25 重建
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

    def _init_store(self) -> None:
        """初始化 L1 工作记忆 + L2 结构化记忆。"""
        # L1 工作记忆
        self._soul = SoulSystem(self._data_dir / "soul")
        self._core_block = CoreBlock(
            identity_block=self._soul.load_identity(),
            context_block="",
            plan_block="",
        )
        self._budget = BudgetManager(max_tokens=self._config.get("budget_tokens", 4000))
        self._attachments: list[CompactAttachment] = []

        # L2 结构化记忆
        self._wing_room = WingRoomManager(self._data_dir / "palace")
        self._store = DrawerClosetStore(self._data_dir / "palace")
        self._index = ThreeLevelIndex(self._data_dir / "index")
        self._md_store = MarkdownStore(self._data_dir / "palace")

    def _init_retriever(self) -> None:
        """初始化检索引擎 + 治理引擎 + 同步引擎 + 感知 + 上下文管理。"""
        # 检索引擎
        self._retriever = HybridRetriever(
            vector_backend=self._config.get("vector_backend", "chromadb"),
            data_dir=self._data_dir / "retrieval",
            enable_reranker=self._config.get("enable_reranker", False),
        )

        # 治理引擎
        self._conflict_resolver = ConflictResolver(
            strategy=self._config.get("conflict_strategy", "latest")
        )
        self._temporal_decay = TemporalDecay()
        self._forgetting = ForgettingCurve(self._data_dir / "governance")
        self._privacy = PrivacyManager(
            default_level=self._config.get("default_privacy", "personal"),
            session_id=self._session_id,
        )
        self._privacy.bind_store(self._store)  # ★ 绑定存储层，支持持久化回填
        self._store.bind_privacy_manager(self._privacy)  # OPT-1: 存储层绑定加密器
        self._provenance = ProvenanceTracker(data_dir=self._data_dir / "governance")

        # OPT-2: 初始化异步 LLM 客户端
        self._init_llm_client()

        # 同步引擎
        sync_mode = self._config.get("sync_mode", "none")
        self._sync_engine = SyncEngine(
            self._data_dir,
            SyncConfig(
                mode=sync_mode,
                instance_name=f"omnimem-{self._session_id[:8]}",
                sync_interval=self._config.get("sync_interval", 30),
                conflict_resolution=self._config.get("sync_conflict_resolution", "latest_wins"),
            ),
        )

        # ★ 分布式向量时钟：每个实例独立计数器
        self._instance_id = self._sync_engine._config.instance_id
        self._vector_clock = VectorClock()

        # L0 感知
        self._perception = PerceptionEngine()
        self._turn_count = 0
        self._last_save_turn = 0
        self._save_interval = self._config.get("save_interval", 15)

        # ★ R27优化：system_prompt_block 每轮缓存，避免同一轮内重复查询 store
        self._system_prompt_cache_turn = -1
        self._system_prompt_cache_value = ""

        # ★ P0方案二：Saga 事务协调器 + 统一后台任务执行器
        self._saga = SagaCoordinator(
            pending_path=self._data_dir / "governance" / "saga_pending.json"
        )
        self._bg_executor = BackgroundTaskExecutor(max_workers=2)

        # OPT-4: 初始化存储服务层
        self._store_service = MemoryStoreService(
            store=self._store,
            perception=self._perception,
            provenance=self._provenance,
            session_id=self._session_id,
            turn_count=0,
        )

        # 后台线程
        self._prefetch_cache: str = ""
        self._prefetch_lock = threading.Lock()
        # ★ R26优化：在初始化阶段创建缓存，而非运行时 hasattr 延迟初始化
        self._reflect_cache: dict[str, tuple] = {}
        # ★ 后台预取线程池（复用线程，避免频繁创建/销毁）
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="omnimem_prefetch"
        )

        # 上下文管理层（三层分离的核心）
        context_budget = ContextBudget(
            max_prefetch_tokens=self._config.get("max_prefetch_tokens", 300),
            max_summary_chars=self._config.get("max_summary_chars", 60),
            max_prefetch_items=self._config.get("max_prefetch_items", 8),
        )
        # ★ 传递 embedding 函数供语义去重使用（延迟调用，VectorRetriever 初始化后才可用）
        self._context_manager = ContextManager(
            budget=context_budget,
            embedding_fn=lambda text: self._retriever.embed_text(text),
        )

        # ★ P0方案六：治理巡检器（长期一致性保障）
        self._auditor = GovernanceAuditor(
            store=self._store,
            index=self._index,
            retriever=self._retriever,
            forgetting=self._forgetting,
        )

        # ★ P1方案四：反馈收集器（Cross-Encoder 在线学习数据基础）
        self._feedback = FeedbackCollector(self._data_dir / "feedback")

    def _init_reflect(self) -> None:
        """初始化 L3 深层记忆（Consolidation + 知识图谱 + 反思引擎）。"""
        from omnimem.deep.consolidation import ConsolidationEngine
        from omnimem.deep.knowledge_graph import KnowledgeGraph
        from omnimem.deep.reflect import ReflectEngine

        deep_dir = self._data_dir / "deep"
        self._consolidation = ConsolidationEngine(
            deep_dir,
            fact_threshold=self._config.get("fact_threshold", 10),
        )
        self._knowledge_graph = KnowledgeGraph(deep_dir)
        self._reflect_engine = ReflectEngine(
            deep_dir,
            consolidation_engine=self._consolidation,
            recall_fn=self._l3_recall,
            llm_fn=self._call_llm_for_reflect,
        )

    def _init_lora(self) -> None:
        """初始化 L4 内化记忆（KV Cache + LoRA 训练器）。"""
        from omnimem.internalize.kv_cache import KVCacheManager
        from omnimem.internalize.lora_train import LoRATrainer

        internalize_dir = self._data_dir / "internalize"
        self._kv_cache = KVCacheManager(
            internalize_dir,
            auto_preload_threshold=self._config.get("kv_cache_threshold", 10),
            max_cache_size=self._config.get("kv_cache_max", 100),
        )
        self._lora_trainer = LoRATrainer(
            internalize_dir,
            base_model=self._config.get("lora_base_model", "Qwen2.5-7B"),
            lora_rank=self._config.get("lora_rank", 16),
            lora_alpha=self._config.get("lora_alpha", 32),
        )

    # ─── 异步接口（P2方案五） ───────────────────────────────────

    @property
    def async_provider(self):
        """获取异步包装器（延迟初始化）。"""
        if not hasattr(self, "_async_provider"):
            from omnimem.core.async_provider import AsyncOmniMemProvider

            self._async_provider = AsyncOmniMemProvider(self)
        return self._async_provider

    # ─── 上下文注入 ─────────────────────────────────────────────

    def system_prompt_block(self) -> str:
        """返回 OmniMem 身份说明 — 精炼版，经 ContextManager 精炼后注入。

        旧版问题：
        1. 直接从 store 读原文拼入，绕过 ContextManager 精炼 → token 膨胀
        2. 与 prefetch() 双路径重复注入同一条记忆
        3. 每条记忆可能 200-800 tokens，30+20 条 = 可能 6000+ tokens

        改进：
        1. 所有记忆经 ContextManager.refine_content() 精炼为 ≤60 字摘要
        2. 与 prefetch 本轮已注入的记忆去重，避免双份
        3. 总预算硬限制 500 字符（约 125 tokens）
        """
        # ★ R27优化：同一轮内缓存结果，避免重复 store 查询
        if self._system_prompt_cache_turn == self._turn_count:
            return self._system_prompt_cache_value

        parts = [
            "## OmniMem Memory System (Unified)",
            f"Memory directory: {self._data_dir}",
            "",
        ]

        # ★ 分层注入：BIOS 模式
        # L1 恒定注入：preference + correction（身份/偏好/纠正，永远在）
        # L2 按需注入：fact（用剩余预算）
        boot_entries = []
        fact_entries = []
        for mtype in ("preference", "correction"):
            entries = self._store.search(memory_type=mtype, limit=10)
            for e in entries:
                e["_mtype"] = mtype
                boot_entries.append(e)
        for e in self._store.search(memory_type="fact", limit=15):
            e["_mtype"] = "fact"
            fact_entries.append(e)

        if not boot_entries and not fact_entries:
            parts.append("### Identity")
            parts.append(self._core_block.identity_block)
            result = "\n".join(parts)
            self._system_prompt_cache_turn = self._turn_count
            self._system_prompt_cache_value = result
            return result

        # ★ 经 ContextManager 精炼 + 与 prefetch 去重
        # 分两层：boot（恒定）+ facts（剩余预算）
        refined_lines = []
        total_chars = 0
        base_budget = self._config.get("system_prompt_char_limit", 500)
        # ★ P1方案三：自适应预算 — 根据查询复杂度动态调整
        last_query = getattr(self, "_last_query", "")
        query_kw_count = (
            len(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", last_query.lower()))
            if last_query
            else 0
        )
        char_budget = base_budget + min(query_kw_count * 40, 300)
        max_summary = self._context_manager.max_summary_chars  # 60
        seen_fps = set(self._context_manager.get_injected_fingerprints())  # 与已有注入去重

        def _refine_and_add(entries, budget_remaining):
            """精炼一批条目，返回使用的字符数和添加的行。"""
            lines = []
            used = 0
            for entry in entries:
                raw = entry.get("content", "")
                if not raw:
                    continue
                summary = ContextManager.refine_content(raw, max_summary)
                if len(summary) < 3:
                    continue
                fp = ContextManager._content_fingerprint(summary)
                if fp:
                    is_dup = any(
                        ContextManager._fingerprint_similarity(fp, existing) > 0.7
                        for existing in seen_fps
                    )
                    if is_dup:
                        continue
                    seen_fps.add(fp)
                    # ★ 标记为持久指纹，prefetch 不会再注入同一条
                    self._context_manager.add_persistent_fingerprint(fp)
                line = f"- [{entry.get('_mtype', 'fact')}] {summary}"
                if used + len(line) + 1 > budget_remaining:
                    break
                lines.append(line)
                used += len(line) + 1
            return lines, used

        # L1: 恒定注入（preference + correction 优先占预算）
        boot_lines, boot_used = _refine_and_add(boot_entries, char_budget)
        refined_lines.extend(boot_lines)
        total_chars += boot_used

        # L2: 事实注入（用剩余预算）
        remaining = char_budget - total_chars
        if remaining > 50 and fact_entries:
            fact_lines, fact_used = _refine_and_add(fact_entries, remaining)
            refined_lines.extend(fact_lines)
            total_chars += fact_used

        if refined_lines:
            parts.append("### Core Memories (summaries — use omni_detail for full content)")
            parts.extend(refined_lines)
            parts.append("")

        # Identity Block
        parts.append("### Identity")
        parts.append(self._core_block.identity_block)

        result = "\n".join(parts)
        self._system_prompt_cache_turn = self._turn_count
        self._system_prompt_cache_value = result
        return result

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """混合检索：KV Cache + 向量 + BM25 + 图谱 → 经 ContextManager 精炼后注入。

        三层分离设计（受 Anthropic managed-agents 启发）:
          存储层: 全量检索，不丢弃
          上下文管理层: ContextManager 精炼/去重/预算控制
          上下文窗口: 只注入精炼摘要，细节通过 omni_detail 按需拉取
        """
        # 记录最近查询，供 system_prompt_block 自适应预算使用
        self._last_query = query

        # 每轮重置注入状态
        self._context_manager.reset_for_new_turn()

        # L4: 先查 KV Cache
        kv_results = []
        if self._kv_cache:
            kv_results = self._kv_cache.search_cache(query, limit=5)
            if kv_results:
                # KV Cache 结果也需要精炼
                for cr in kv_results:
                    cr["source_type"] = "kv_cache"

        # 检查异步预取缓存
        async_results = []
        with self._prefetch_lock:
            cached = self._prefetch_cache
            self._prefetch_cache = ""
        # 注意: 异步缓存已经是格式化文本，如果存在则直接用
        # 但为了统一经过 ContextManager，我们改为存原始结果
        if cached and cached.startswith("___RAW_RESULTS___"):
            # 新格式: 存的是 JSON 化的原始结果
            try:
                async_results = json.loads(cached[len("___RAW_RESULTS___") :])
            except Exception:
                async_results = []

        # 实时检索（如果异步缓存没命中）
        live_results = []
        if not kv_results and not async_results:
            max_tokens = self._config.get("max_prefetch_tokens", 300)
            live_results = self._retriever.search(query, max_tokens=max_tokens)

            # 图谱检索通道（第6通道）
            if self._knowledge_graph:
                try:
                    graph_results = self._knowledge_graph.graph_search(query, max_depth=2, limit=10)
                    if graph_results:
                        for gr in graph_results[:5]:
                            gr[
                                "content"
                            ] = f"{gr.get('subject','')} {gr.get('predicate','')} {gr.get('object','')}"
                            gr["type"] = "graph_triple"
                            gr["confidence"] = gr.get("confidence", 0.5)
                        live_results.extend(graph_results[:5])
                except Exception as e:
                    logger.debug("OmniMem graph prefetch failed: %s", e)

            # 应用时间衰减
            live_results = self._temporal_decay.apply(live_results)

            # 应用隐私过滤
            live_results = self._privacy.filter(live_results, session_id=session_id)

            # ★ 预热 KV Cache：高相关性结果增加访问计数，加速后续相同查询
            if self._kv_cache and live_results:
                for r in live_results[:3]:
                    if r.get("score", 0) > 0.6:
                        self._kv_cache.check_and_auto_preload(
                            key=r.get("memory_id", ""),
                            content=r.get("content", ""),
                            metadata={"source": "prefetch", "query": query},
                            source_memory_ids=[r.get("memory_id", "")],
                        )

        # 合并所有来源
        all_results = kv_results + async_results + live_results

        if not all_results:
            return ""

        # ★ 核心：经 ContextManager 精炼后再注入
        return self._context_manager.refine_prefetch_results(all_results)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """后台异步预检索下一轮 — 存原始结果，由 prefetch() 经 ContextManager 精炼。"""

        def _bg_prefetch():
            try:
                max_tokens = self._config.get("max_prefetch_tokens", 300)
                result = self._retriever.search(query, max_tokens=max_tokens)
                result = self._temporal_decay.apply(result)
                result = self._privacy.filter(result, session_id=session_id)
                # ★ 存原始结果而非格式化文本，让 prefetch() 统一精炼
                if result:
                    serialized = "___RAW_RESULTS___" + json.dumps(result, ensure_ascii=False)
                else:
                    serialized = ""
                with self._prefetch_lock:
                    self._prefetch_cache = serialized
            except Exception as e:
                logger.debug("OmniMem background prefetch failed: %s", e)

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

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        """路由工具调用到对应功能。

        路由逻辑:
          omni_memorize → _handle_memorize (存储记忆)
          omni_recall   → _handle_recall (检索记忆)
          omni_compact  → _handle_compact (压缩准备)
          omni_reflect  → _handle_reflect (深层反思)
          omni_govern   → _handle_govern (治理操作)
          omni_detail   → _handle_detail (细节拉取)
          memory        → _handle_builtin_memory_compat (兼容内置工具)

        Returns:
            JSON 格式字符串，包含 status/error 字段。
            异常时返回 {"error": "<异常信息>"}，保证不崩溃。
        """
        try:
            if tool_name == "omni_memorize":
                return self._handle_memorize(args)
            elif tool_name == "omni_recall":
                return self._handle_recall(args)
            elif tool_name == "omni_compact":
                return self._handle_compact(args)
            elif tool_name == "omni_reflect":
                return self._handle_reflect(args)
            elif tool_name == "omni_govern":
                return self._handle_govern(args)
            elif tool_name == "omni_detail":
                return self._handle_detail(args)
            # ★ 新增: 内置 memory 工具兼容路由
            elif tool_name == "memory":
                return self._handle_builtin_memory_compat(args)
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            logger.error("OmniMem tool %s failed: %s", tool_name, e)
            return json.dumps({"error": str(e)})

    # ─── 扩展 Hooks ─────────────────────────────────────────────

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
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
                logger.debug("Feedback weight update failed: %s", e)
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
                logger.debug("OmniMem sync failed: %s", e)
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
                logger.debug("OmniMem L4 submit failed: %s", e)

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
                logger.debug("Governance audit failed: %s", e)

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
        # 0. 确保存储缓冲落盘（压缩前数据必须持久化）
        self._store.flush()
        # 1. 紧急保存即将被压缩的消息
        saved_context = self._store_service.emergency_save(messages)

        # 2. 构建 Attachment 摘要
        attachments = build_attachments(messages)

        # 3. 返回给压缩引擎的上下文提示
        parts = []
        if saved_context:
            parts.append(saved_context)
        if attachments:
            att_text = "\n".join(f"[{a.kind}] {a.title}: {a.body[:200]}" for a in attachments)
            parts.append(f"### Pre-Compression Attachments\n{att_text}")

        return "\n\n".join(parts)

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
        self, task: str, result: str, *, child_session_id: str = "", **kwargs
    ) -> None:
        """子 Agent 完成时：记录过程记忆。"""
        if not self._should_write:
            return
        self._store_service.store_delegation(task, result, child_session_id)

    # ─── 配置 ─────────────────────────────────────────────────

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "save_interval",
                "description": "Auto-save every N turns (default: 15)",
                "default": 15,
            },
            {
                "key": "retrieval_mode",
                "description": "Default retrieval mode (rag/llm)",
                "default": "rag",
                "choices": ["rag", "llm"],
            },
            {
                "key": "vector_backend",
                "description": "Vector storage backend",
                "default": "chromadb",
                "choices": ["chromadb", "qdrant", "pgvector"],
            },
            {
                "key": "fact_threshold",
                "description": "Consolidation trigger threshold (default: 10)",
                "default": 10,
            },
            {
                "key": "enable_reranker",
                "description": "Enable Cross-Encoder reranking (needs sentence-transformers)",
                "default": False,
            },
            {
                "key": "conflict_strategy",
                "description": "Conflict resolution strategy",
                "default": "latest",
                "choices": ["latest", "confidence", "manual"],
            },
            {
                "key": "budget_tokens",
                "description": "Token budget for working memory (default: 4000)",
                "default": 4000,
            },
            {
                "key": "kv_cache_threshold",
                "description": "KV Cache auto-preload threshold (access count, default: 10)",
                "default": 10,
            },
            {
                "key": "kv_cache_max",
                "description": "KV Cache max entries (default: 100)",
                "default": 100,
            },
            {
                "key": "lora_base_model",
                "description": "LoRA base model name (default: Qwen2.5-7B)",
                "default": "Qwen2.5-7B",
            },
            {
                "key": "sync_mode",
                "description": "Multi-instance sync mode (none/file_lock/changelog)",
                "default": "none",
                "choices": ["none", "file_lock", "changelog"],
            },
            {
                "key": "sync_interval",
                "description": "Sync interval in seconds for changelog mode (default: 30)",
                "default": 30,
            },
            {
                "key": "sync_conflict_resolution",
                "description": "How to resolve sync conflicts",
                "default": "latest_wins",
                "choices": ["latest_wins", "manual"],
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "omnimem" / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(values, f, allow_unicode=True, default_flow_style=False)
        except ImportError:
            logger.warning("yaml not available — config not saved")

    def shutdown(self) -> None:
        """清理：刷新所有缓冲到磁盘。

        关闭顺序: retriever → md_store → index → knowledge_graph →
                  consolidation → reflect_engine → kv_cache → lora_trainer →
                  provenance → sync_engine → forgetting
        """
        self._retriever.flush()
        self._md_store.flush()
        self._index.close()
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
        return self._vector_clock

    def _apply_sync_change(self, change: dict[str, Any]) -> bool:
        """应用来自其他实例的同步变更。

        由 sync_from_others 调用，将远程变更写入本地存储。
        """
        data = change.get("data", {})
        op = change.get("operation", "INSERT")
        memory_id = data.get("memory_id", "")
        if not memory_id:
            return False

        if op == "DELETE":
            self._forgetting.archive(memory_id)
            return True

        # INSERT / UPDATE：写入本地存储（保留远程 memory_id 和 vc）
        try:
            self._store.add(
                memory_id=memory_id,
                wing=data.get("wing", "auto"),
                room=data.get("room", "sync"),
                content=data.get("content", ""),
                memory_type=data.get("type", "fact"),
                confidence=data.get("confidence", 3),
                privacy=data.get("privacy", "personal"),
                vc=data.get("vc", change.get("vc", "")),
            )
            # 同步更新索引和检索器
            self._index.add(
                memory_id=memory_id,
                wing=data.get("wing", "auto"),
                hall=data.get("type", "fact"),
                room=data.get("room", "sync"),
                content=data.get("content", ""),
                summary=data.get("content", "")[:200]
                .replace("\n", " ")
                .replace("\r", " ")
                .replace("\t", " "),
                type=data.get("type", "fact"),
                confidence=data.get("confidence", 3),
                privacy=data.get("privacy", "personal"),
                scope=data.get("privacy", "personal"),
                stored_at=data.get("stored_at", ""),
                provenance=json.dumps({"sync_from": change.get("instance_id", "unknown")}),
            )
            self._retriever.add(
                data.get("content", ""),
                memory_id=memory_id,
                metadata={
                    "memory_id": memory_id,
                    "type": data.get("type", "fact"),
                    "confidence": data.get("confidence", 3),
                    "scope": data.get("privacy", "personal"),
                    "privacy": data.get("privacy", "personal"),
                    "wing": data.get("wing", "auto"),
                    "room": data.get("room", "sync"),
                    "stored_at": data.get("stored_at", ""),
                },
            )
            return True
        except Exception as e:
            logger.debug("OmniMem apply_sync_change failed for %s: %s", memory_id, e)
            return False

    def _handle_memorize(self, args: dict[str, Any]) -> str:
        """委托到 handlers/memorize.py。"""
        return _handle_memorize_impl(self, args)

    # ─── Saga 重试辅助方法 ────────────────────────────────────

    def _retry_index_add(self, memory_id: str) -> None:
        """从 store 读取记忆，重新写入三层索引。"""
        entry = self._store.get(memory_id)
        if not entry:
            raise RuntimeError(f"Memory {memory_id} not found in store")
        self._index.add(
            memory_id=memory_id,
            wing=entry.get("wing", ""),
            hall=entry.get("hall", entry.get("type", "fact")),
            room=entry.get("room", ""),
            content=entry.get("content", ""),
            summary=entry.get("content", "")[:200]
            .replace("\n", " ")
            .replace("\r", " ")
            .replace("\t", " "),
            type=entry.get("type", "fact"),
            confidence=entry.get("confidence", 3),
            privacy=entry.get("privacy", "personal"),
            scope=entry.get("privacy", "personal"),
            stored_at=entry.get("stored_at", ""),
            provenance="",
        )

    def _retry_retriever_add(self, memory_id: str) -> None:
        """从 store 读取记忆，重新写入检索索引。"""
        entry = self._store.get(memory_id)
        if not entry:
            raise RuntimeError(f"Memory {memory_id} not found in store")
        self._retriever.add(
            entry.get("content", ""),
            memory_id=memory_id,
            metadata={
                "memory_id": memory_id,
                "type": entry.get("type", "fact"),
                "confidence": entry.get("confidence", 3),
                "scope": entry.get("privacy", "personal"),
                "privacy": entry.get("privacy", "personal"),
                "wing": entry.get("wing", ""),
                "room": entry.get("room", ""),
                "stored_at": entry.get("stored_at", ""),
            },
        )

    def _retry_kg_extract(self, memory_id: str) -> None:
        """从 store 读取记忆，重新提取知识图谱。"""
        entry = self._store.get(memory_id)
        if not entry:
            raise RuntimeError(f"Memory {memory_id} not found in store")
        if self._knowledge_graph:
            self._knowledge_graph.extract_and_store(
                entry.get("content", ""),
                memory_id=memory_id,
                confidence=entry.get("confidence", 3) / 5.0,
            )

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
            except Exception:
                pass
        return result

    def _handle_govern(self, args: dict[str, Any]) -> str:
        """委托到 handlers/govern.py。"""
        return _handle_govern_impl(self, args)

    def _scan_memory_conflicts(self) -> list[dict[str, Any]]:
        """委托到 handlers/govern.py。"""
        return _scan_memory_conflicts_impl(self)

    def _handle_compact(self, args: dict[str, Any]) -> str:
        budget = args.get("budget", 4000)
        return json.dumps(
            {
                "status": "ready",
                "budget": budget,
                "message": (
                    "OmniMem will save context before compaction via on_pre_compress. "
                    "Trigger compaction normally — OmniMem hooks handle the rest."
                ),
            }
        )

    def _handle_reflect(self, args: dict[str, Any]) -> str:
        query = args["query"]
        disposition = args.get("disposition")

        # 先确保 pending 数据已处理
        # ★ reflect 强制处理 pending，不受 fact_threshold 限制
        if self._consolidation and self._consolidation.pending_count > 0:
            self._consolidation.process_pending()

        # 使用 ReflectEngine 执行完整四步循环
        result = self._reflect_engine.reflect(
            query=query,
            disposition=disposition,
        )
        return json.dumps(
            {
                "status": "reflected",
                "query": query,
                "observation": result.observation,
                "mental_model": result.mental_model,
                "confidence": result.confidence,
                "reflection_depth": result.reflection_depth,
                "disposition_used": result.disposition_used,
            },
            ensure_ascii=False,
        )

    # ─── omni_detail：按需拉取记忆细节 ─────────────────────────

    def _handle_detail(self, args: dict[str, Any]) -> str:
        """按需拉取记忆细节 — lazy provisioning 模式。

        受 Anthropic managed-agents 的 getEvents() 启发:
        - prefetch 只注入摘要（节省 token）
        - 需要细节时通过此工具按需拉取
        - 支持 session event log 的切片查询

        三种 action:
          get: 用 memory_id 拉取单条记忆的完整内容
          list: 列出本轮注入的所有记忆（含 ID，供 get 使用）
          events: 按 turn 范围查询 session log（getEvents 模式）
        """
        action = args.get("action", "list")

        if action == "list":
            # 列出本轮注入的记忆
            items = self._context_manager.get_injected_items()
            # ★ 主存储验证：过滤掉主存储已不存在的幽灵条目（如被归档的 sync- 条目）
            if items:
                items = [
                    item
                    for item in items
                    if item.get("memory_id") and self._store.get(item["memory_id"])
                ]
            if not items:
                return json.dumps(
                    {
                        "status": "empty",
                        "message": "No memories injected this turn.",
                    }
                )
            return json.dumps(
                {
                    "status": "ok",
                    "count": len(items),
                    "memories": items,
                },
                ensure_ascii=False,
            )

        elif action == "get":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return json.dumps(
                    {
                        "status": "error",
                        "message": "memory_id is required for action='get'",
                    }
                )
            result = self._context_manager.get_detail_for(memory_id, self._store)
            # ★ R24修复EXT-3b/3d：补充 archived 字段（forgetting stage）
            if result.get("status") == "found" and self._forgetting:
                stage = self._forgetting.get_stage(memory_id)
                result["archived"] = stage in ("archived", "forgotten")
            # ★ 记录反馈点击
            if hasattr(self, "_feedback") and self._feedback and result.get("status") == "found":
                self._feedback.record_click(
                    query=getattr(self, "_last_query", ""),
                    memory_id=memory_id,
                    source_type=result.get("type", "unknown"),
                )
            return json.dumps(result, ensure_ascii=False)

        elif action == "events":
            # getEvents 模式：按 turn 范围查询 session log
            from_turn = args.get("from_turn", 0)
            to_turn = args.get("to_turn", self._turn_count)
            query = args.get("query", "")

            # ★ 用 type=event 索引查询，替代不可靠的内容子串搜索
            events = []
            try:
                all_events = self._store.search(memory_type="event", limit=100)
                for evt in all_events:
                    evt_content = evt.get("content", "")
                    # 关键词过滤（如果提供了 query）
                    if query and query.lower() not in evt_content.lower():
                        continue
                    # 解析 turn 编号
                    turn_match = re.search(
                        r"\[Turn (\d+)\]|\[Checkpoint at turn (\d+)\]|\[Emergency save\].*?turn[_ ](\d+)",
                        evt_content,
                    )
                    if turn_match:
                        turn_num = int(
                            turn_match.group(1) or turn_match.group(2) or turn_match.group(3)
                        )
                    else:
                        # 无 turn 标记的事件，用时间排序
                        turn_num = 0
                    if from_turn <= turn_num <= to_turn:
                        events.append(
                            {
                                "turn": turn_num,
                                "memory_id": evt.get("memory_id", ""),
                                "content": evt_content,
                                "type": evt.get("type", "event"),
                                "stored_at": evt.get("stored_at", ""),
                            }
                        )
            except Exception as e:
                logger.debug("OmniMem events query failed: %s", e)

            # 按 turn 排序
            events.sort(key=lambda x: x.get("turn", 0))

            return json.dumps(
                {
                    "status": "ok",
                    "from_turn": from_turn,
                    "to_turn": to_turn,
                    "count": len(events),
                    "events": events[:20],
                },
                ensure_ascii=False,
            )

        return json.dumps({"error": f"Unknown action: {action}"})

    def _handle_builtin_memory_compat(self, args: dict[str, Any]) -> str:
        """处理兼容内置 memory 工具的调用，映射到 OmniMem 存储。"""
        action = args.get("action", "")
        target = args.get("target", "memory")
        content = args.get("content", "").strip()
        old_text = args.get("old_text", "").strip()

        if action not in ("add", "replace", "remove"):
            return json.dumps({"error": f"Unknown action '{action}'. Use: add, replace, remove"})

        if action in ("add", "replace") and not content:
            return json.dumps({"error": "Content is required for 'add' and 'replace'."})

        if action in ("replace", "remove") and not old_text:
            return json.dumps({"error": "old_text is required for 'replace' and 'remove'."})

        # 安全扫描
        if content:
            scan_error = _compat_scan_memory_content(content)
            if scan_error:
                return json.dumps({"success": False, "error": scan_error})

        mem_type = "preference" if target == "user" else "fact"

        # ★ 精炼：通过内置 memory 工具存的内容也可能很长，走 _extract_core_fact
        if content and len(content) > 100:
            refined = self._extract_core_fact(content)
            if refined and len(refined) < len(content):
                content = refined

        if action == "add":
            return self._compat_set(content, mem_type)
        elif action == "replace":
            return self._compat_get(content, old_text, mem_type, target)
        elif action == "remove":
            return self._compat_delete(old_text, mem_type, target)

        return json.dumps({"error": "Unreachable"})

    def _compat_set(self, content: str, mem_type: str) -> str:
        """兼容层：add 操作 → 映射到 OmniMem memorize。"""
        result = self._handle_memorize(
            {
                "content": content,
                "memory_type": mem_type,
                "confidence": 4,
                "scope": "personal",
                "privacy": "personal",
            }
        )
        parsed = json.loads(result)
        parsed["compat_note"] = "Routed from builtin 'memory' tool to OmniMem"
        return json.dumps(parsed)

    def _compat_get(self, content: str, old_text: str, mem_type: str, target: str) -> str:
        """兼容层：replace 操作 → 查找旧条目并替换。"""
        matches = self._store.search_by_content(old_text, limit=10)
        filtered = [m for m in matches if m.get("type") == mem_type]

        if not filtered:
            return json.dumps(
                {
                    "success": False,
                    "error": f"No matching {target} entry found for '{old_text[:50]}'.",
                }
            )

        if len(filtered) > 1:
            previews = [m.get("content", "")[:60] for m in filtered[:5]]
            return json.dumps(
                {
                    "success": False,
                    "error": "Multiple entries matched. Be more specific.",
                    "matches": previews,
                }
            )

        old_id = filtered[0]["memory_id"]
        self._forgetting.archive(old_id)

        result = self._handle_memorize(
            {
                "content": content,
                "memory_type": mem_type,
                "confidence": 4,
                "scope": "personal",
                "privacy": "personal",
            }
        )
        parsed = json.loads(result)
        parsed["replaced_id"] = old_id
        parsed["compat_note"] = "Replaced via builtin compat layer"
        return json.dumps(parsed)

    def _compat_delete(self, old_text: str, mem_type: str, target: str) -> str:
        """兼容层：remove 操作 → 查找并归档旧条目。"""
        matches = self._store.search_by_content(old_text, limit=10)
        filtered = [m for m in matches if m.get("type") == mem_type]

        if not filtered:
            return json.dumps({"success": False, "error": f"No matching {target} entry found."})

        if len(filtered) > 1:
            previews = [m.get("content", "")[:60] for m in filtered[:5]]
            return json.dumps(
                {
                    "success": False,
                    "error": "Multiple entries matched. Be more specific.",
                    "matches": previews,
                }
            )

        old_id = filtered[0]["memory_id"]
        self._forgetting.archive(old_id)

        return json.dumps(
            {
                "success": True,
                "action": "archived",
                "memory_id": old_id,
                "message": f"{target} entry archived (soft delete).",
                "compat_note": "Removed via builtin compat layer (uses forgetting curve)",
            }
        )

    # ─── 反递归防护 ─────────────────────────────────────────────

    @staticmethod
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
        return self._perception._extract_core_fact(text)

    @staticmethod
    def _compute_text_similarity(text_a: str, text_b: str) -> float:
        """计算两段文本的词语重叠相似度。

        委托给 ContextManager 的指纹相似度算法，保持与注入端去重一致。
        """
        fp_a = ContextManager._content_fingerprint(text_a)
        fp_b = ContextManager._content_fingerprint(text_b)
        return ContextManager._fingerprint_similarity(fp_a, fp_b)

    def _semantic_dedup(
        self, content: str, memory_type: str, candidates: list = None
    ) -> dict[str, Any]:
        """写入前语义去重检查。

        Args:
            candidates: 预搜索的候选记忆列表。如果提供则复用，避免重复搜索。

        返回:
          {"action": "create"} — 无重复，新建
          {"action": "update", "existing_id": ...} — 有相似记忆，归档旧的后更新
          {"action": "skip", "existing_id": ..., "reason": ...} — 完全重复，跳过
        """
        # 短内容用精确匹配
        if len(content) <= 20:
            exact = candidates or self._store.search_by_content(content, limit=5)
            for m in exact:
                if m.get("content", "").strip() == content.strip():
                    return {
                        "action": "skip",
                        "existing_id": m.get("memory_id", ""),
                        "reason": "Exact duplicate",
                    }
            return {"action": "create"}

        # ★ EDGE修复：短内容(≤50字)提高 skip 阈值至 0.92，避免模板化编号内容被误判
        _SHORT_SKIP_THRESHOLD = 0.92 if len(content) <= 50 else 0.8

        # 长内容用语义相似度
        similar = candidates
        if similar is None:
            similar = self._search_candidates(content)

        for m in similar:
            existing_content = m.get("content", "")
            sim = self._compute_text_similarity(content, existing_content)

            # ★ EDGE修复：数值差异化检测——仅数字不同的模板内容不判为重复
            if sim > _SHORT_SKIP_THRESHOLD:
                nums_a = set(re.findall(r"\d+", content))
                nums_b = set(re.findall(r"\d+", existing_content))
                has_numeric_diff = bool(nums_a ^ nums_b) or (len(nums_a) >= 2 and len(nums_b) >= 2)
                if has_numeric_diff:
                    sim = max(sim - 0.18, 0.5)  # 降低相似度，避免误判

            if sim > 0.85:
                return {
                    "action": "skip",
                    "existing_id": m.get("memory_id", ""),
                    "reason": f"Near-duplicate (sim={sim:.2f})",
                }
            if sim > 0.6:
                return {
                    "action": "update",
                    "existing_id": m.get("memory_id", ""),
                    "reason": f"Similar (sim={sim:.2f}), archiving old",
                }

        return {"action": "create"}

    # ─── 内部辅助方法 ─────────────────────────────────────────────

    def _unified_candidate_search(self, content: str) -> list:
        """统一候选搜索：向量+store多片段，供去重和冲突检测共享。

        一次搜索，多处复用，避免 _handle_memorize 中重复调用
        _store.search_by_content 和 _retriever._vector.search。
        """
        return self._search_candidates(content)

    def _search_candidates(self, content: str) -> list:
        """搜索与 content 语义相似的已有记忆。"""
        similar = []
        # 先尝试向量检索（更准）
        try:
            if self._retriever and hasattr(self._retriever, "_vector") and self._retriever._vector:
                vector_results = self._retriever._vector.search(content, top_k=10)
                if vector_results:
                    similar = vector_results
        except Exception:
            pass
        # 回退到 store 搜索（多片段）
        if not similar:
            similar = self._store.search_by_content(content[:50], limit=10)
        # 补充：内容中间片段搜索
        if len(content) > 100:
            mid_results = self._store.search_by_content(content[50:100], limit=5)
            existing_ids = {m.get("memory_id", "") for m in similar}
            for m in mid_results:
                if m.get("memory_id", "") not in existing_ids:
                    similar.append(m)
        return similar

    # ─── 内部辅助方法 ─────────────────────────────────────────

    def _l3_recall(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """L3 检索辅助函数，供 ReflectEngine 回调使用。

        与 handle_recall 保持一致：retriever 搜索无结果时，
        fallback 到 store 关键词匹配，确保 reflect 不会因检索引擎
        阈值过滤而返回空结果。

        ★ 优化：使用 search_by_content 替代全量 scan(limit=200)，
        避免大数据量时的 O(n) 性能问题。
        """
        results = self._retriever.search(query, max_tokens=3000)
        if results:
            return results[:limit]

        import re as _re

        query_keywords = set(_re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", query.lower()))
        if query_keywords:
            # ★ 优化：按关键词搜索替代全量扫描，避免 O(n)
            seen_ids: set[str] = set()
            for kw in list(query_keywords)[:5]:
                for sf in self._store.search_by_content(kw, limit=20):
                    mid = sf.get("memory_id", "")
                    if mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                    sf_content = sf.get("content", "").lower()
                    keyword_hits = sum(1 for kw2 in query_keywords if kw2 in sf_content)
                    if keyword_hits >= 1:
                        sf["_source"] = "store_fallback"
                        sf["score"] = min(0.15 + keyword_hits * 0.05, 0.35)
                        results.append(sf)
                        if len(results) >= limit:
                            break
                if len(results) >= limit:
                    break
        return results[:limit]

    _REFLECT_CACHE_TTL = 60.0

    def _init_llm_client(self) -> None:
        """OPT-2: 初始化异步 LLM 客户端，集中管理凭证和配置。"""
        # 加载凭证（优先级：环境变量 > .env 文件 > config.yaml）
        creds = AsyncLLMClient.load_credentials_from_env()
        if not creds.get("api_key") or not creds.get("base_url"):
            creds.update(AsyncLLMClient.load_credentials_from_hermes_env())
        config_creds = AsyncLLMClient.load_credentials_from_hermes_config()
        if not creds.get("base_url"):
            creds["base_url"] = config_creds.get("base_url", "")
        model = config_creds.get("model") or self._config.get("default", "glm-5.1")

        self._llm_client = AsyncLLMClient(
            api_key=creds.get("api_key", ""),
            base_url=creds.get("base_url", ""),
            model=model,
            max_concurrent=3,
            timeout=30.0,
            cache_ttl=self._REFLECT_CACHE_TTL,
        )
        logger.debug("AsyncLLMClient initialized: model=%s", model)

    def _call_llm_for_reflect(self, prompt: str, system: str, max_tokens: int = 800) -> str:
        """供 ReflectEngine 调用 LLM 的包装函数。

        OPT-2 改进:
          - 使用 AsyncLLMClient 进行异步调用，支持并发控制
          - 保留 reflect 结果缓存（60s TTL）
          - 保留 fallback 到 auxiliary_client.call_llm

        Args:
            prompt: 用户 prompt
            system: 系统 prompt
            max_tokens: 最大输出 token 数

        Returns:
            LLM 生成的文本，失败时抛出异常
        """
        import time

        # ★ reflect 结果缓存：相同 query 60s 内复用
        # ★ R26优化：限制缓存大小，避免长期运行内存泄漏
        _MAX_REFLECT_CACHE = 64
        cache_key = prompt[:200]
        now = time.time()
        # 清理过期缓存
        if len(self._reflect_cache) > _MAX_REFLECT_CACHE:
            self._reflect_cache = {
                k: (v, t)
                for k, (v, t) in self._reflect_cache.items()
                if now - t < self._REFLECT_CACHE_TTL
            }
        if cache_key in self._reflect_cache:
            cached_result, cached_time = self._reflect_cache[cache_key]
            if now - cached_time < self._REFLECT_CACHE_TTL:
                logger.debug("ReflectEngine LLM cache hit")
                return cached_result

        # OPT-2: 优先使用 AsyncLLMClient（异步 + 并发控制）
        if hasattr(self, "_llm_client") and self._llm_client:
            try:
                result = self._llm_client.call_sync(
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=0.5,
                )
                if result.content:
                    self._reflect_cache[cache_key] = (result.content, now)
                    return result.content
            except Exception as e:
                logger.debug("ReflectEngine AsyncLLM failed: %s", e)

        # fallback: auxiliary_client.call_llm
        try:
            from agent.auxiliary_client import call_llm

            response = call_llm(
                task="reflect",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=max_tokens,
            )
            if response and hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content
                if content and content.strip():
                    self._reflect_cache[cache_key] = (content, now)
                    return content
        except Exception as e:
            logger.debug("ReflectEngine LLM (auxiliary_client) failed: %s", e)

        raise RuntimeError("所有LLM调用路径均失败")
