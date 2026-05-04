"""RetrievalFacade — 检索引擎 + 上下文管理 + 感知 + 反馈。

封装: HybridRetriever, ContextManager, PerceptionEngine,
      FeedbackCollector, AsyncLLMClient
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from omnimem.context.manager import ContextBudget, ContextManager
from omnimem.governance.feedback import FeedbackCollector
from omnimem.perception.engine import PerceptionEngine
from omnimem.retrieval.engine import HybridRetriever


class RetrievalFacade:
    """检索门面：向量/BM25 检索 + 上下文精炼 + 感知 + 反馈学习。"""

    def __init__(self, data_dir: Path, config: Any, storage_facade: Any):
        # 检索引擎
        self._retriever = HybridRetriever(
            vector_backend=config.get("vector_backend", "chromadb"),
            data_dir=data_dir / "retrieval",
            enable_reranker=config.get("enable_reranker", False),
        )

        # 上下文管理
        context_budget = ContextBudget(
            max_prefetch_tokens=config.get("max_prefetch_tokens", 300),
            max_summary_chars=config.get("max_summary_chars", 60),
            max_prefetch_items=config.get("max_prefetch_items", 8),
        )
        self._context_manager = ContextManager(
            budget=context_budget,
            embedding_fn=lambda text: self._retriever.embed_text(text),
        )

        # L0 感知
        self._perception = PerceptionEngine()

        # 反馈收集
        self._feedback = FeedbackCollector(data_dir / "feedback")

        # 缓存
        self.prefetch_cache: str = ""
        self.prefetch_lock = threading.Lock()
        self._reflect_cache: dict[str, tuple[Any, ...]] = {}
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="omnimem_prefetch"
        )

    @property
    def retriever(self) -> HybridRetriever:
        return self._retriever

    @property
    def context_manager(self) -> ContextManager:
        return self._context_manager

    @property
    def perception(self) -> PerceptionEngine:
        return self._perception

    @property
    def feedback(self) -> FeedbackCollector:
        return self._feedback

    def flush(self) -> None:
        """刷新检索缓存。"""
        self.retriever.flush()

    def close(self) -> None:
        """关闭资源。"""
        self._prefetch_executor.shutdown(wait=False)
