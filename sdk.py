"""OmniMemSDK — 独立 SDK 模式，直接初始化子组件，不依赖 Hermes MemoryProvider。

提供轻量级 API，可直接创建 OmniMem 实例并调用记忆操作，
无需 Hermes 框架注册机制，也无需 MagicMock 注入。

用法:
    from omnimem.sdk import OmniMemSDK

    sdk = OmniMemSDK()                       # 数据目录按 $HERMES_HOME 解析
    sdk = OmniMemSDK(storage_dir="~/omnimem")  # 显式路径须自行 expanduser，Path() 不会展开 ~
    sdk.memorize("用户喜欢Python", memory_type="preference")
    result = sdk.recall("用户喜欢什么")
    sdk.close()
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from omnimem.config import OmniMemConfig
from omnimem.config._config import resolve_default_data_dir
from omnimem.core.dedup import SemanticDedupService
from omnimem.core.saga import SagaCoordinator
from omnimem.facades.governance import GovernanceFacade
from omnimem.facades.retrieval import RetrievalFacade
from omnimem.facades.storage import StorageFacade
from omnimem.handlers.memorize import shutdown_background_executor

logger = logging.getLogger(__name__)

# 健康检查磁盘空间阈值（GB）— 低于此值标记为 degraded
_DISK_SPACE_WARNING_GB = 1.0
_DISK_SPACE_CRITICAL_GB = 0.1


def _join_text_segments(segments: Any) -> str:
    """从多模态内容片段列表中抽取并拼接纯文本部分。"""
    out: list[str] = []
    for seg in segments or []:
        if isinstance(seg, dict):
            t = seg.get("text")
            if isinstance(t, str) and t.strip():
                out.append(t.strip())
    return " ".join(out)


class OmniMemSDK:
    """OmniMem 独立 SDK — 直接初始化子组件，不依赖 agent.memory_provider。"""

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if storage_dir is None:
            # 与 doctor / provider 共用同一解析顺序（显式 → $HERMES_HOME → ~/.hermes → ~/.omnimem）。
            # 原先硬编码 ~/.omnimem，使 SDK 写入的实例与插件实际服务的实例不是同一个：
            # 主库 861 条记忆，而 ~/.omnimem 里只有 5 条测试垃圾。
            storage_dir = resolve_default_data_dir()
        self._data_dir = Path(storage_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = f"sdk-{uuid.uuid4().hex[:12]}"

        # 直接初始化子组件（不依赖 agent.memory_provider）
        self._config = OmniMemConfig(self._data_dir)
        if config:
            for key, value in config.items():
                self._config.set(key, value)

        # 初始化存储层
        self._storage = StorageFacade(self._data_dir, self._config)
        self._storage.init_l2()

        # 初始化检索层
        self._retrieval = RetrievalFacade(self._data_dir, self._config, self._storage)

        # 初始化治理层
        self._governance = GovernanceFacade(
            self._data_dir, self._config, self._session_id,
            self._storage, self._retrieval.retriever,
        )

        # 显式绑定子组件
        self._store = self._storage.store
        self._index = self._storage.index
        self._retriever = self._retrieval.retriever
        self._forgetting = self._governance.forgetting
        self._conflict_resolver = self._governance.conflict_resolver
        self._privacy = self._governance.privacy
        self._provenance = self._governance.provenance
        self._perception = self._retrieval.perception
        self._dedup = SemanticDedupService(self._store, self._retriever)
        self._saga = SagaCoordinator(pending_path=self._data_dir / "saga_pending.json")

        logger.info("OmniMemSDK initialized: session=%s, data_dir=%s", self._session_id, self._data_dir)

    def memorize(self, content: str, memory_type: str = "fact", **kwargs: Any) -> dict:
        """存储记忆。"""
        from omnimem.handlers.memorize import handle_memorize

        args = {"content": content, "memory_type": memory_type, **kwargs}
        raw = handle_memorize(self._build_provider_proxy(), args)
        return json.loads(raw)

    def memorize_media(
        self,
        content: Any,
        media: Any = None,
        *,
        memory_type: str = "fact",
        embed: bool = False,
        backend: str = "hash",
        **kwargs: Any,
    ) -> dict:
        """多模态摄入（改进项 #12）：内容寻址落盘媒体 + 记录可检索记忆。

        - `content` 可为文本说明，或 OpenAI/Anthropic 风格的多模态结构（自动解析媒体）；
        - `media` 可显式传入 MediaPart 列表（与 content 解析结果合并）；
        - 内联字节按 sha256 去重落盘到 <data_dir>/media，仅 URL 引用不落盘；
        - 生成的记忆正文 = 文本说明 + 各媒体引用行，从而复用既有文本检索通道；
        - embed=True 时按需计算媒体向量（默认 hash 后端，clip 需装依赖），失败非致命。
        返回 memorize 结果并附 `media` 引用列表。
        """
        from omnimem.multimodal import MediaStore, parse_content, parse_message

        parts = list(media or [])
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, dict):
            parts += parse_message(content)
            inner = content.get("content")
            if isinstance(inner, str):
                text = inner
            elif isinstance(inner, list):
                text = _join_text_segments(inner)
        elif isinstance(content, list):
            parts += parse_content(content)
            text = _join_text_segments(content)
        else:
            text = "" if content is None else str(content)

        store = MediaStore(self._data_dir)
        raw_refs = store.put_all(parts)
        seen: set[str] = set()
        refs = [r for r in raw_refs if not (r.sha256 in seen or seen.add(r.sha256))]

        lines = [text] if text else []
        for r in refs:
            lines.append(f"[media kind={r.kind} mime={r.mime} sha256={r.sha256} size={r.size}]")
        caption = "\n".join(lines) if lines else f"multimodal memory ({len(refs)} media)"

        result = self.memorize(caption, memory_type=memory_type, **kwargs)

        if embed and refs:
            try:
                from omnimem.multimodal import create_media_embedder

                embedder = create_media_embedder(backend=backend)
                vectors: dict[str, list[float]] = {}
                for r in refs:
                    blob = store.get(r)
                    if blob is not None:
                        vectors[r.sha256] = embedder.embed(blob)
                result["media_embeddings"] = {"backend": backend, "dimension": embedder.dimension, "vectors": vectors}
            except Exception as e:  # 非致命：如缺 clip 依赖
                logger.warning("媒体嵌入失败（非致命）: %s", e)
                result["media_embeddings"] = {"error": str(e)}

        result["media"] = [r.to_dict() for r in refs]
        return result

    def recall(self, query: str, mode: str = "rag", **kwargs: Any) -> dict:
        """检索记忆。"""
        from omnimem.handlers.recall import handle_recall

        args = {"query": query, "mode": mode, **kwargs}
        raw = handle_recall(self._build_provider_proxy(), args)
        return json.loads(raw)

    def agentic_search(
        self,
        query: str,
        *,
        max_steps: int = 3,
        top_k: int = 20,
        final_k: int = 10,
        mode: str = "rag",
        min_hits: int = 3,
        explain: bool = False,
    ) -> dict:
        """Agentic 迭代检索（改进项 #9）：检索→判据→再检索，逐轮补检后 RRF 融合。

        纯增量入口，复用 self._retriever（HybridRetriever）；默认 mode="rag" 与
        主检索一致。可后续通过注入 rewrite_fn/judge_fn 接 LLM（此处走启发式默认）。
        """
        from omnimem.retrieval.agentic import AgenticConfig, AgenticRetriever

        cfg = AgenticConfig(
            max_steps=max_steps, top_k=top_k, final_k=final_k, mode=mode, min_hits=min_hits,
        )
        engine = AgenticRetriever(self._retriever, config=cfg)
        return engine.run(query).to_recall_dict(explain=explain)

    def reflect(self, query: str, **kwargs: Any) -> dict:
        """深层反思。"""
        from omnimem.core.tool_router import handle_reflect

        args = {"query": query, **kwargs}
        raw = handle_reflect(args, None, None)
        return json.loads(raw)

    def govern(self, action: str, **kwargs: Any) -> dict:
        """治理操作。"""
        from omnimem.handlers.govern import handle_govern

        args = {"action": action, **kwargs}
        raw = handle_govern(self._build_provider_proxy(), args)
        return json.loads(raw)

    def compact(self, **kwargs: Any) -> dict:
        """压缩前准备。"""
        from omnimem.core.tool_router import handle_compact

        args = {**kwargs}
        raw = handle_compact(args)
        return json.loads(raw)

    def detail(self, memory_id: str, **kwargs: Any) -> dict:
        """按需拉取记忆细节。"""
        from omnimem.core.tool_router import handle_detail

        args = {"action": "get", "memory_id": memory_id, **kwargs}
        raw = handle_detail(
            args,
            context_manager=self._retrieval.context_manager,
            store=self._store,
            forgetting=self._forgetting,
            feedback=getattr(self, "_feedback", None),
            turn_count=kwargs.get("turn_count", 0),
            last_query=kwargs.get("last_query", ""),
            # ★ M8-17: secret 解密审计
            audit_logger=getattr(self._governance, "audit_logger", None),
        )
        return json.loads(raw)

    def detail_list(self, **kwargs: Any) -> dict:
        """列出记忆。"""
        from omnimem.core.tool_router import handle_detail

        args = {"action": "list", **kwargs}
        raw = handle_detail(
            args,
            context_manager=self._retrieval.context_manager,
            store=self._store,
            forgetting=self._forgetting,
            feedback=None,
            turn_count=0,
            last_query="",
        )
        return json.loads(raw)

    def detail_events(self, from_turn: int = 0, to_turn: int | None = None, **kwargs: Any) -> dict:
        """获取事件记录。"""
        from omnimem.core.tool_router import handle_detail

        args: dict[str, Any] = {"action": "events", "from_turn": from_turn, **kwargs}
        if to_turn is not None:
            args["to_turn"] = to_turn
        raw = handle_detail(
            args,
            context_manager=self._retrieval.context_manager,
            store=self._store,
            forgetting=self._forgetting,
            feedback=None,
            turn_count=0,
            last_query="",
        )
        return json.loads(raw)

    def health_check(self) -> dict:
        """健康检查 — 返回增强的健康信息字典。

        检查项：
          1. Store 可访问性（原有）
          2. ChromaDB 连通性（尝试 vector.count()）
          3. LLM 可达性（检查客户端初始化状态）
          4. 磁盘空间（shutil.disk_usage）
          5. 审计健康（原有）

        Returns:
            健康信息字典，status 取值 healthy/degraded/unhealthy
        """
        result: dict[str, Any] = {
            "status": "healthy",
            "session_id": self._session_id,
            "data_dir": str(self._data_dir),
        }

        # 1. Store 可访问性检查
        try:
            store_count = len(self._store.search(limit=1))
            result["store_accessible"] = True
            result["store_count"] = store_count
        except Exception as e:
            result["store_accessible"] = False
            result["store_error"] = str(e)
            result["status"] = "unhealthy"

        # 2. ChromaDB 连通性检查（尝试 vector.count()）
        try:
            vector = getattr(self._retriever, "_vector", None)
            if vector is not None:
                vec_count = vector.count()
                result["chromadb_accessible"] = True
                result["vector_count"] = vec_count
                # 顺便记录熔断器状态
                breaker = getattr(self._retriever, "_vector_breaker", None)
                if breaker is not None:
                    result["circuit_breaker_state"] = breaker.state
            else:
                result["chromadb_accessible"] = False
                result["chromadb_error"] = "vector retriever not initialized"
                result["status"] = "degraded"
        except Exception as e:
            result["chromadb_accessible"] = False
            result["chromadb_error"] = str(e)
            if result["status"] == "healthy":
                result["status"] = "degraded"

        # 3. LLM 可达性检查（检查客户端初始化状态，不发起实际调用）
        try:
            llm_status = self._check_llm_reachability()
            result["llm_reachable"] = llm_status["reachable"]
            if not llm_status["reachable"]:
                result["llm_detail"] = llm_status.get("detail", "")
                if result["status"] == "healthy":
                    result["status"] = "degraded"
            else:
                result["llm_model"] = llm_status.get("model", "")
        except Exception as e:
            result["llm_reachable"] = False
            result["llm_detail"] = str(e)
            if result["status"] == "healthy":
                result["status"] = "degraded"

        # 4. 磁盘空间检查
        try:
            disk_info = self._check_disk_space()
            result["disk"] = disk_info
            if disk_info["status"] == "critical":
                result["status"] = "unhealthy"
            elif disk_info["status"] == "warning" and result["status"] == "healthy":
                result["status"] = "degraded"
        except Exception as e:
            result["disk_error"] = str(e)

        # 5. 审计健康检查
        try:
            health = self._governance.auditor.quick_health_check()
            result["audit"] = health
            if not health.get("healthy", True):
                if result["status"] == "healthy":
                    result["status"] = "degraded"
        except Exception as e:
            result["audit_error"] = str(e)

        return result

    def _check_llm_reachability(self) -> dict[str, Any]:
        """检查 LLM 客户端可达性（不发起实际网络调用）。

        仅验证客户端是否已初始化、凭证是否非空。
        实际网络可达性需通过业务调用验证，此处避免引入延迟。
        """
        # 尝试从 tool_router 获取已初始化的 LLM 客户端
        try:
            from omnimem.core import tool_router

            client = getattr(tool_router, "_llm_client", None)
            if client is None:
                return {"reachable": False, "detail": "LLM client not initialized"}
            # 检查 API key 是否配置
            api_key = getattr(client, "_api_key", "") or getattr(client, "api_key", "")
            if not api_key or not str(api_key).strip():
                return {"reachable": False, "detail": "LLM API key not configured"}
            model = getattr(client, "_model", "") or getattr(client, "model", "")
            return {"reachable": True, "model": str(model)}
        except Exception as e:
            return {"reachable": False, "detail": str(e)}

    def _check_disk_space(self) -> dict[str, Any]:
        """检查数据目录所在磁盘的可用空间。

        Returns:
            包含 total/used/free（GB）和 status（ok/warning/critical）的字典
        """
        usage = shutil.disk_usage(self._data_dir)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < _DISK_SPACE_CRITICAL_GB:
            status = "critical"
        elif free_gb < _DISK_SPACE_WARNING_GB:
            status = "warning"
        else:
            status = "ok"
        return {
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "status": status,
        }

    def export_memories(
        self, output_path: str, format: str = "json", **kwargs: Any
    ) -> dict:
        """导出记忆。"""
        import os

        from omnimem.core.import_export import MemoryExporter

        exporter = MemoryExporter(
            self._store,
            self._index,
            self._store.meta_store,
        )
        encryption_key = kwargs.get("encryption_key") or self._config.get("export_key") or os.environ.get("OMNIMEM_EXPORT_KEY")
        if format == "markdown":
            count = exporter.export_markdown(output_path, wing=kwargs.get("wing"))
        else:
            count = exporter.export_json(
                output_path,
                wing=kwargs.get("wing"),
                memory_type=kwargs.get("memory_type"),
                encryption_key=encryption_key,
            )
        return {"status": "exported", "count": count, "path": str(output_path)}

    def import_memories(self, input_path: str, **kwargs: Any) -> dict:
        """导入记忆。"""
        import os

        from omnimem.core.import_export import MemoryImporter

        importer = MemoryImporter(
            self._store,
            self._index,
            self._retriever,
            self._dedup,
            self._conflict_resolver,
            self._forgetting,
        )
        encryption_key = kwargs.get("encryption_key") or self._config.get("export_key") or os.environ.get("OMNIMEM_EXPORT_KEY")
        result = importer.import_json(
            input_path,
            skip_duplicates=kwargs.get("skip_duplicates", True),
            resolve_conflicts=kwargs.get("resolve_conflicts", True),
            encryption_key=encryption_key,
        )
        return {"status": "imported", **result}

    def import_file(
        self,
        input_path: str | Path,
        *,
        source: str = "auto",
        skip_duplicates: bool = True,
        resolve_conflicts: bool = True,
        workdir: str | Path | None = None,
        **kwargs: Any,
    ) -> dict:
        """导入外部记忆（改进项 #8）：mem0/letta/zep/graphiti/cognee/auto -> 落库。

        先由来源适配器拆解为 OmniMem 原生信封，再复用 import_memories 管线。
        返回导入统计，并附带生成的原生信封路径（native_envelope）便于排查。
        """
        from omnimem.importers import import_file as _import_file

        return _import_file(
            self,
            input_path,
            source=source,
            skip_duplicates=skip_duplicates,
            resolve_conflicts=resolve_conflicts,
            workdir=workdir,
            **kwargs,
        )

    def _build_provider_proxy(self) -> Any:
        """构建轻量级 Provider 代理对象，供 handler 函数访问子组件。

        handler 函数（如 handle_memorize、handle_recall）需要通过 provider 对象
        访问 _store、_retriever 等属性。此代理避免创建完整的 OmniMemProvider 实例。
        """
        return _SDKProviderProxy(self)

    def _provider_style_memorize(self, args: dict[str, Any]) -> str:
        """Provider 风格记忆写入（接受 args 字典，返回 JSON 字符串）。

        供 AsyncOmniMemSDK 通过 asyncio.to_thread 调用。
        """
        from omnimem.handlers.memorize import handle_memorize

        return handle_memorize(self._build_provider_proxy(), args)

    def _provider_style_recall(self, args: dict[str, Any]) -> str:
        """Provider 风格检索（接受 args 字典，返回 JSON 字符串）。"""
        from omnimem.handlers.recall import handle_recall

        return handle_recall(self._build_provider_proxy(), args)

    def _provider_style_reflect(self, args: dict[str, Any]) -> str:
        """Provider 风格反思（接受 args 字典，返回 JSON 字符串）。"""
        from omnimem.core.tool_router import handle_reflect

        return handle_reflect(args, None, None)

    def _provider_style_govern(self, args: dict[str, Any]) -> str:
        """Provider 风格治理（接受 args 字典，返回 JSON 字符串）。"""
        from omnimem.handlers.govern import handle_govern

        return handle_govern(self._build_provider_proxy(), args)

    def close(self) -> None:
        """关闭 SDK，释放资源。"""
        try:
            self._storage.flush()
        except Exception as e:
            logger.warning("SDK close: storage flush failed: %s", e)
        try:
            self._retrieval.flush()
        except Exception as e:
            logger.warning("SDK close: retrieval flush failed: %s", e)
        try:
            self._governance.close()
        except Exception as e:
            logger.warning("SDK close: governance close failed: %s", e)
        try:
            self._storage.close()
        except Exception as e:
            logger.warning("SDK close: storage close failed: %s", e)
        try:
            # 关闭 memorize 模块级后台 fallback 线程池
            shutdown_background_executor(wait=True)
        except Exception as e:
            logger.warning("SDK close: background executor shutdown failed: %s", e)
        logger.info("OmniMemSDK closed: session=%s", self._session_id)

    def __enter__(self) -> OmniMemSDK:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class _MissingType:
    pass

_MISSING = _MissingType()

class _SDKProviderProxy:
    """轻量级 Provider 代理，将 handler 需要的属性委托到 SDK 子组件。

    handler 函数通过 provider._store、provider._retriever 等属性访问子组件，
    此代理将属性访问转发到 OmniMemSDK 实例，避免创建完整 OmniMemProvider。
    """

    def __init__(self, sdk: OmniMemSDK) -> None:
        self._sdk = sdk

    def __getattr__(self, name: str) -> Any:
        """回退：访问 SDK 子组件。不存在则抛 AttributeError（让 hasattr 正确返回 False）。"""
        if name.startswith('_'):
            val = getattr(self._sdk, name, _MISSING)
            if val is _MISSING:
                raise AttributeError(f"_SDKProviderProxy has no attribute {name!r}")
            return val
        raise AttributeError(name)

    @property
    def _store(self):
        return self._sdk._store

    @property
    def _index(self):
        return self._sdk._index

    @property
    def _retriever(self):
        return self._sdk._retriever

    @property
    def _forgetting(self):
        return self._sdk._forgetting

    @property
    def _conflict_resolver(self):
        return self._sdk._conflict_resolver

    @property
    def _privacy(self):
        return self._sdk._privacy

    @property
    def _provenance(self):
        return self._sdk._provenance

    @property
    def _perception(self):
        return self._sdk._perception

    @property
    def _dedup(self):
        return self._sdk._dedup

    @property
    def _saga(self):
        return self._sdk._saga

    @property
    def _config(self):
        return self._sdk._config

    @property
    def _data_dir(self):
        return self._sdk._data_dir

    @property
    def _session_id(self):
        return self._sdk._session_id

    @property
    def _wing_room(self):
        return self._sdk._storage.wing_room

    @property
    def _context_manager(self):
        return self._sdk._retrieval.context_manager

    @property
    def _auditor(self):
        return self._sdk._governance.auditor

    @property
    def _storage(self):
        return self._sdk._storage

    @property
    def _retrieval(self):
        return self._sdk._retrieval

    @property
    def _governance(self):
        return self._sdk._governance

    @property
    def _knowledge_graph(self):
        return getattr(self._sdk, '_knowledge_graph', None)

    @property
    def _temporal_decay(self):
        gov = getattr(self._sdk, '_governance', None)
        return gov.temporal_decay if gov else None

    @property
    def _temporal_kg(self):
        gov = getattr(self._sdk, '_governance', None)
        return gov.temporal_kg if gov else None

    @property
    def _consolidation(self):
        return getattr(self._sdk, '_consolidation', None)

    @property
    def _kv_cache(self):
        return getattr(self._sdk, '_kv_cache', None)

    @property
    def _lora_trainer(self):
        return getattr(self._sdk, '_lora_trainer', None)

    @property
    def _trace_chain(self):
        gov = getattr(self._sdk, '_governance', None)
        return getattr(gov, 'trace_chain', None) if gov else None
