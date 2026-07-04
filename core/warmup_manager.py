"""后台预热管理器 — 负责 L3/L4 初始化、数据预热、检索引擎预热和启动审计。"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)


class WarmupManager:
    """管理 OmniMem 后台异步预热流程。

    职责:
      1. L3/L4 并行初始化（reflect + lora）
      2. 数据预热（索引 + BM25 重建 + index.db 同步）
      3. 检索引擎预热（SentenceTransformer + ChromaDB + 向量健康检查）
      4. 启动审计（一致性检查 + 自动修复）
    """

    def __init__(
        self,
        init_reflect_fn: Any,
        init_lora_fn: Any,
        index: Any,
        store: Any,
        retriever: Any,
        retrieval: Any,
        auditor: Any,
    ) -> None:
        self._init_reflect_fn = init_reflect_fn
        self._init_lora_fn = init_lora_fn
        self._index = index
        self._store = store
        self._retriever = retriever
        self._retrieval = retrieval
        self._auditor = auditor

    def run(self) -> None:
        """执行完整的后台预热流程（非阻塞调用，但自身是同步的）。"""
        logger.info("OmniMem background warmup: starting...")
        t0 = time.time()

        # L3/L4 并行初始化
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {
                executor.submit(self._init_reflect_fn): "reflect",
                executor.submit(self._init_lora_fn): "lora",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning("BG init %s failed: %s", name, e)

        # 数据预热（索引 + BM25）
        self._warmup_data()

        # 检索引擎预热（SentenceTransformer + ChromaDB）
        self._warmup_retrieval()

        # 启动审计
        self._startup_audit()

        elapsed = time.time() - t0
        logger.info("OmniMem background warmup: complete in %.1fs", elapsed)

    def _warmup_data(self) -> None:
        """数据预热：索引条目 → store 预热 + BM25 重建 + index.db 同步。"""
        try:
            indexed_entries = self._index.search_l1(limit=2000)
            if indexed_entries:
                self._store.warm_up(indexed_entries[:500])
                rebuilt = self._retriever.rebuild_bm25_from_entries(indexed_entries)
                if rebuilt > 0:
                    logger.info(
                        "OmniMem: warmed up %d entries, rebuilt BM25 with %d entries",
                        min(len(indexed_entries), 500),
                        rebuilt,
                    )
            # index.db 全量同步 — 通过 MetaStore 高层接口
            try:
                stale, missing = self._store.meta_store.sync_from_index(self._index.db_path)
                if stale or missing:
                    logger.info("OmniMem: index.db synced — cleaned %d stale, added %d missing", stale, missing)
            except Exception as _e:
                logger.warning("OmniMem index.db sync skipped (non-fatal): %s", _e)
        except Exception as e:
            logger.warning("OmniMem warm-up/BM25 rebuild failed (non-fatal): %s", e)

    def _warmup_retrieval(self) -> None:
        """检索引擎预热 + 向量健康检查。"""
        try:
            self._retrieval.warmup()
            logger.info("OmniMem: retrieval engine warmup complete")
            try:
                health = self._retriever._check_vector_health()
                vec_count = health.get("vector_count", -1)
                if vec_count == 0 and self._index:
                    indexed_entries = self._index.search_l1(limit=5000)
                    if indexed_entries:
                        logger.warning(
                            "OmniMem: vector index empty but %d entries in meta_store, triggering rebuild",
                            len(indexed_entries),
                        )
                        result = self._retriever.rebuild_all_from_entries(indexed_entries)
                        logger.info("OmniMem: vector rebuild complete: %s", result)
                        from omnimem.retrieval.vector_store import _emit
                        _emit("[OmniMem] 向量索引已自动重建")
            except Exception as e:
                logger.warning("OmniMem: vector health check failed: %s", e)
        except Exception as e:
            logger.warning("OmniMem retrieval warmup failed (non-fatal): %s", e)

    def _startup_audit(self) -> None:
        """启动时运行审计+修复，确保 store/index/retriever 一致。"""
        try:
            health = self._auditor.quick_health_check()
            if not health.get("healthy", True):
                audit = self._auditor.run_full_audit(limit=2000)
                if audit.get("total_issues", 0) > 0:
                    fixed = self._auditor.repair(audit)
                    logger.info(
                        "OmniMem startup audit: %d inconsistencies found, %d repaired",
                        audit["total_issues"], fixed,
                    )
        except Exception as e:
            logger.warning("OmniMem startup audit skipped (non-fatal): %s", e)
