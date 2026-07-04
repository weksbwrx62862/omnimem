"""Provider 向后兼容代理：集中 __getattr__ / __setattr__ 动态代理逻辑。"""

from __future__ import annotations

from typing import Any


class ProviderProxyMixin:
    """通过 __getattr__ / __setattr__ 在初始化早期提供容错属性访问。"""

    # ─── Facade 动态代理映射表 ─────────────────────────────────
    _STORAGE_ATTRS = {
        "_soul": "soul",
        "_core_block": "core_block",
        "_budget": "budget",
        "_wing_room": "wing_room",
        "_store": "store",
        "_index": "index",
        "_md_store": "md_store",
    }
    _RETRIEVAL_ATTRS = {
        "_retriever": "retriever",
        "_context_manager": "context_manager",
        "_perception": "perception",
        "_feedback": "feedback",
        "_prefetch_lock": "prefetch_lock",
        "_reflect_cache": "_reflect_cache",
        "_prefetch_executor": "_prefetch_executor",
    }
    _GOVERNANCE_ATTRS = {
        "_conflict_resolver": "conflict_resolver",
        "_temporal_decay": "temporal_decay",
        "_forgetting": "forgetting",
        "_privacy": "privacy",
        "_provenance": "provenance",
        "_sync_engine": "sync_engine",
        "_vector_clock": "vector_clock",
        "_auditor": "auditor",
        "_audit_logger": "audit_logger",
        "_rbac": "rbac",
        "_temporal_kg": "temporal_kg",
    }
    _SYNC_ATTRS = {
        "_saga": "saga",
        "_bg_executor": "bg_executor",
        "_store_service": "store_service",
        "_kv_cache": "kv_cache",
        "_lora_trainer": "lora_trainer",
    }
    _DEEP_ATTRS = {
        "_consolidation": "consolidation",
        "_knowledge_graph": "knowledge_graph",
        "_reflect_engine": "reflect_engine",
    }
    _DIRECT_MAP = {"_dedup": "_dedup_service"}
    _SETTER_MAP = {
        "_attachments": ("_storage", "attachments"),
        "_prefetch_cache": ("_retrieval", "prefetch_cache"),
    }

    def __getattr__(self, name: str) -> Any:
        """Fallback 动态代理：仅在显式赋值完成前生效。

        显式赋值完成后，__getattr__ 不再被调用。保留此机制的原因：
        初始化早期（治理/同步服务就绪前）Facade 子组件尚未就绪，
        此时属性访问需要容错。
        """
        if name.startswith("_"):
            for mapping, facade_attr in [
                (self._STORAGE_ATTRS, "_storage"),
                (self._RETRIEVAL_ATTRS, "_retrieval"),
                (self._GOVERNANCE_ATTRS, "_governance"),
                (self._SYNC_ATTRS, "_sync"),
                (self._DEEP_ATTRS, "_deep"),
            ]:
                if name in mapping:
                    facade = getattr(self, facade_attr, None)
                    return getattr(facade, mapping[name]) if facade else None
            if name in self._DIRECT_MAP:
                return getattr(self, self._DIRECT_MAP[name])
        # Manager 属性 — 未初始化时返回 None（测试兼容）
        if name in (
            "_session_manager",
            "_warmup_manager",
            "_system_prompt_builder",
            "_backup_manager",
            "_llm_client_manager",
            "_last_query",
            "_trace_chain",
        ):
            return None
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._SETTER_MAP:
            facade_name, attr_name = self._SETTER_MAP[name]
            facade = getattr(self, facade_name, None)
            if facade:
                setattr(facade, attr_name, value)
                return
        object.__setattr__(self, name, value)
