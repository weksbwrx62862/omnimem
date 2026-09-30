"""VectorRetriever — 向量检索。

使用 VectorStore 抽象接口，支持多后端切换（chromadb/qdrant）。
默认使用 ChromaDBStore（向后兼容）。
OPT-3: 支持自定义 Embedding Function，实现嵌入结果缓存。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

from omnimem.embedding.base import EmbeddingProvider
from omnimem.embedding.sentence_transformers_provider import SentenceTransformersProvider
from omnimem.retrieval.base import RetrievalResult
from omnimem.retrieval.vector_factory import create_vector_store
from omnimem.retrieval.vector_store import (
    ChromaDBStore,
    _CachedEmbeddingFunction,
    _emit,
    read_vector_pending,
    vector_pending_path,
)

logger = logging.getLogger(__name__)

# 排空待回填队列的互斥锁：队列文件是进程级共享资源，warmup 线程与 doctor/手工调用
# 可能同时 drain。用非阻塞获取，抢不到就本轮跳过（积压仍在，下轮继续）。
_DRAIN_LOCK = threading.Lock()


class VectorRetriever:
    """向量检索，委托 VectorStore 抽象接口。"""

    def __init__(
        self,
        backend: str = "chromadb",
        data_dir: Path | None = None,
        embedding_model_path: str = "",
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: Any | None = None,
    ):
        self._backend = backend
        self._data_dir = data_dir or Path("/tmp/omnimem/retrieval")
        self._store: Any | None = None
        self._initialized = False
        self._embedding_fn: Any = None
        self._embedding_provider: EmbeddingProvider | None = None
        self._encoder = None
        self._embedding_model_path = embedding_model_path
        # 新 VectorStore 后端下记录分块 ID，用于删除
        self._chunk_ids: dict[str, list[str]] = {}

        if vector_store is not None:
            self._store = vector_store
            self._is_new_store = self._detect_new_store(vector_store)
            self._initialized = True
            if self._is_new_store:
                self._embedding_provider = embedding_provider or self._default_embedding_provider()
        else:
            self._is_new_store = False
            if embedding_provider is not None:
                # 显式传入了 provider 但未传入 store：保留旧式 store，provider 暂不使用
                logger.debug("embedding_provider ignored when vector_store is not provided")

        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass

    @staticmethod
    def _detect_new_store(store: Any) -> bool:
        """通过 duck typing 判断 store 是否为新 storage/base.py 接口。"""
        return hasattr(store, "search") and not hasattr(store, "query")

    def _default_embedding_provider(self) -> EmbeddingProvider:
        return SentenceTransformersProvider(model_path=self._embedding_model_path)

    @property
    def name(self) -> str:
        """检索通道名称（兼容 BaseRetriever 接口）。"""
        return "vector"

    def search_sync(self, query: str, **kwargs: Any) -> RetrievalResult:
        """同步检索，返回统一 RetrievalResult（兼容 BaseRetriever）。"""
        top_k = kwargs.get("top_k", 10)
        results = self.search(query, top_k=top_k)
        scores = [float(r.get("score", 0.0)) for r in results]
        return RetrievalResult(results=results, scores=scores, channel=self.name)

    async def asearch(self, query: str, **kwargs: Any) -> RetrievalResult:
        """异步检索（兼容 BaseRetriever）。

        优先使用 VectorStore 的 asearch 异步接口；否则在线程池中执行同步 search。
        """
        import asyncio

        self._ensure_initialized()
        if self._store is None:
            return RetrievalResult(results=[], scores=[], channel=self.name)

        top_k = kwargs.get("top_k", 10)

        # 新 VectorStore 接口且支持异步搜索时，使用异步路径
        if self._is_new_store and hasattr(self._store, "asearch"):
            if self._embedding_provider is None:
                return RetrievalResult(results=[], scores=[], channel=self.name)
            try:
                query_embeddings = await self._embedding_provider.aembed([query])
                results = await self._store.asearch(
                    query_embedding=query_embeddings[0], top_k=top_k
                )
                output = self._post_process_new_store_results(results)
                scores = [float(r.get("score", 0.0)) for r in output]
                return RetrievalResult(results=output, scores=scores, channel=self.name)
            except Exception as e:
                logger.warning("Vector async search (new store) failed, fallback to thread: %s", e)

        return await asyncio.to_thread(self.search_sync, query, **kwargs)

    def _ensure_initialized(self) -> None:
        if self._initialized and self._store is not None:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)
        if self._store is None:
            self._is_new_store = False
            if self._backend == "faiss":
                from omnimem.retrieval.faiss_store import FAISSStore

                try:
                    cache_path = self._data_dir / "embedding_cache.json"
                    self._embedding_fn = _CachedEmbeddingFunction(
                        cache_path=cache_path, model_path=self._embedding_model_path
                    )
                except Exception as e:
                    logger.warning("Failed to create CachedEmbeddingFunction for faiss: %s, using default", e)
                    self._embedding_fn = None
                self._store = FAISSStore(
                    persist_dir=self._data_dir / "faiss",
                    embedding_fn=self._embedding_fn,
                )
            elif self._backend == "chromadb":
                try:
                    cache_path = self._data_dir / "embedding_cache.json"
                    self._embedding_fn = _CachedEmbeddingFunction(
                        cache_path=cache_path, model_path=self._embedding_model_path
                    )
                except Exception as e:
                    logger.warning("Failed to create CachedEmbeddingFunction: %s, using default", e)
                    self._embedding_fn = None
                self._store = ChromaDBStore(
                    collection_name="omnimem",
                    persist_dir=self._data_dir / "chroma",
                    embedding_fn=self._embedding_fn,
                )
            else:
                self._store = create_vector_store(
                    backend=self._backend,
                    persist_dir=self._data_dir / "chroma",
                    data_dir=self._data_dir / "chroma",
                    model_path=self._embedding_model_path,
                )
        self._initialized = True

    _CHUNK_SIZE = 500
    _CHUNK_OVERLAP = 100

    def add(self, content: str, memory_id: str, metadata: dict[str, Any]) -> None:
        self._add_single(content, memory_id, metadata)

    def _build_batch_entries(
        self, documents: list[dict[str, Any]], include_content: bool = False
    ) -> tuple[list[str], list[str], list[dict[str, str]]]:
        """统一构建批量写入所需的 ids、documents 和 metadatas。

        处理逻辑：
        - 跳过 content 或 memory_id 为空的条目
        - 将除 content、memory_id 外的字段转为 str 类型的 metadata，None 跳过
        - metadata 为空时自动添加 `_default: "1"` 以满足 ChromaDB 非空要求
        - 长文本按 _CHUNK_SIZE / _CHUNK_OVERLAP 拆分，并为每个 chunk 生成独立 ID
        """
        all_ids: list[str] = []
        all_docs: list[str] = []
        all_metas: list[dict[str, str]] = []
        for doc in documents:
            content = doc.get("content", "")
            memory_id = doc.get("memory_id", "")
            if not content or not memory_id:
                continue
            meta: dict[str, str] = {
                k: str(v)
                for k, v in doc.items()
                if k not in ("content", "memory_id") and v is not None
            }
            # ChromaDB 要求 metadata 非空
            if not meta:
                meta["_default"] = "1"
            if len(content) > self._CHUNK_SIZE:
                chunks = self._split_chunks(content, self._CHUNK_SIZE, self._CHUNK_OVERLAP)
                for i, chunk in enumerate(chunks):
                    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                    chunk_id = f"{memory_id}_chunk{chunk_hash}"
                    all_ids.append(chunk_id)
                    all_docs.append(chunk)
                    chunk_meta = dict(meta, _parent_id=memory_id, _chunk_idx=str(i))
                    if include_content:
                        chunk_meta["content"] = chunk
                    all_metas.append(chunk_meta)
            else:
                all_ids.append(memory_id)
                all_docs.append(content)
                if include_content:
                    meta["content"] = content
                all_metas.append(meta)
        return all_ids, all_docs, all_metas

    def add_batch(self, documents: list[dict[str, Any]]) -> None:
        self._ensure_initialized()
        if self._store is None:
            return
        if self._is_new_store:
            self._add_batch_new(documents)
            return
        all_ids, all_docs, all_metas = self._build_batch_entries(documents)
        if not all_ids:
            return
        try:
            self._store.add(ids=all_ids, documents=all_docs, metadatas=all_metas)
        except Exception as e:
            logger.warning("Vector add_batch failed: %s", e)

    def _add_batch_new(self, documents: list[dict[str, Any]]) -> None:
        """新 VectorStore 接口批量写入路径：调用方预计算 embeddings。"""
        if self._embedding_provider is None or self._store is None:
            return
        all_ids, all_docs, all_metas = self._build_batch_entries(documents, include_content=True)
        if not all_ids:
            return
        try:
            embeddings = self._embedding_provider.embed(all_docs)
            self._store.add(ids=all_ids, embeddings=embeddings, metadatas=all_metas)
            self._record_chunk_ids(all_ids, all_metas)
        except Exception as e:
            logger.warning("Vector add_batch (new store) failed: %s", e)

    def _record_chunk_ids(self, ids: list[str], metadatas: list[dict[str, str]]) -> None:
        """记录分块 ID 与父文档的映射，便于删除时清理。"""
        for doc_id, meta in zip(ids, metadatas, strict=False):
            parent_id = meta.get("_parent_id", "")
            if parent_id:
                self._chunk_ids.setdefault(parent_id, []).append(doc_id)

    # ── ★ P1-2: 待回填向量队列的消费者 ──

    def _pending_path(self) -> Path | None:
        """当前向量后端的待回填队列路径；后端不支持时返回 None。

        ★ 必须先把 store 拉起来：``self._store`` 是 ``_ensure_initialized()`` 懒建的，
          而 warmup 在**任何检索发生之前**就调 count/drain_vector_pending。原先 store
          为 None 时这里直接返回 None，于是 count 报 0、drain 空转返回，积压永远不会
          被消费（线上 6001 条待回填就是这样一直躺着的）。
        """
        if self._store is None:
            try:
                self._ensure_initialized()
            except Exception as e:
                logger.warning("初始化向量后端失败，无法定位待回填队列: %s: %s", type(e).__name__, e)
                return None
        getter = getattr(self._store, "pending_path", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                logger.debug("store.pending_path() 调用失败", exc_info=True)
                return None
        persist_dir = getattr(self._store, "_persist_dir", None)
        return vector_pending_path(persist_dir) if persist_dir else None

    def _processing_path(self, path: Path) -> Path:
        return path.with_name(path.name + ".processing")

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        try:
            with path.open(encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())
        except Exception as e:
            logger.warning("统计待回填向量队列失败 %s: %s", path.name, e)
            return 0

    def count_vector_pending(self) -> int:
        """队列里还有多少条降级写入未被补回（损坏行也计数，便于告警）。

        ★ 计入 ``.processing``：排空过程会先把队列改名接管，若只数正牌文件，
          一次长排空期间监控会看到积压"降到 0"，实际条目正躺在改名后的文件里。
        """
        path = self._pending_path()
        if path is None:
            return 0
        return self._count_lines(path) + self._count_lines(self._processing_path(path))

    def _recover_stranded_processing(self, processing: Path, path: Path) -> None:
        """把上一轮被杀/崩溃留下的 ``.processing`` 并回队列。

        排空在嵌入途中被 kill（warmup 跑在 daemon 线程上，解释器退出即被强杀）会把
        整批积压遗留在 ``.processing``，而没有任何启动路径会再读它；下一次 drain 的
        ``os.replace(path, processing)`` 会把它直接覆盖——积压静默消失。
        """
        if not processing.exists():
            return
        stranded = self._count_lines(processing)
        try:
            if not path.exists():
                os.replace(processing, path)
            else:
                with processing.open(encoding="utf-8") as src, path.open("a", encoding="utf-8") as dst:
                    for line in src:
                        if line.strip():
                            dst.write(line)
                processing.unlink(missing_ok=True)
            logger.warning(
                "回收孤悬的待回填向量队列 %s（%d 条）→ %s",
                processing.name,
                stranded,
                path.name,
            )
        except OSError as e:
            logger.error("回收孤悬队列 %s 失败: %s", processing, e)

    def drain_vector_pending(self, limit: int = 2000) -> dict[str, int]:
        """把 ``vector_index_pending.jsonl`` 里的降级写入重新补进向量索引。

        流程（先整体改名再处理，避免"边读边清"把新写入的条目一起抹掉）：

        1. 回收上一轮被杀留下的 ``.processing`` 残骸；``path`` → ``path.processing``
           原子改名；处理期间新的降级写入自然落到新文件；
        2. 去重读入，取前 ``limit`` 条重放；``add_batch`` 若再次失败，写方自己会把它们
           重新记回队列，所以这里不重复回写；
        3. 超出 ``limit`` 的剩余条目原样追加回队列，下一轮继续。
        """
        stats = {"pending": 0, "replayed": 0, "deferred": 0, "failed": 0}
        path = self._pending_path()
        if path is None:
            return stats

        if not _DRAIN_LOCK.acquire(blocking=False):
            stats["pending"] = self.count_vector_pending()
            logger.info("待回填向量排空已在进行中，本轮跳过（积压 %d 条）", stats["pending"])
            return stats

        processing = self._processing_path(path)
        try:
            self._recover_stranded_processing(processing, path)
            if not path.exists():
                return stats
            try:
                os.replace(path, processing)
            except OSError as e:
                logger.warning("无法接管待回填向量队列 %s: %s", path, e)
                return stats

            try:
                records = read_vector_pending(processing)
                stats["pending"] = len(records)
                if not records:
                    return stats

                batch, deferred = records[:limit], records[limit:]
                docs = [
                    {
                        "memory_id": str(r.get("id", "")),
                        "content": str(r.get("document", "")),
                        **(r.get("metadata") or {}),
                    }
                    for r in batch
                ]
                try:
                    self.add_batch(docs)
                    stats["replayed"] = len(docs)
                except Exception as e:
                    # add_batch 在抵达 store 之前失败（例如 store 未初始化），
                    # 此时写方没有机会重新记账，必须自己回写，否则这批记忆永久丢失。
                    stats["failed"] = len(batch)
                    logger.warning("待回填向量重放失败，退回队列: %s", e)
                    deferred = batch + deferred

                if deferred:
                    self._requeue_vector_pending(deferred, path)
                    stats["deferred"] = len(deferred)

                if stats["replayed"]:
                    self.flush()
                    logger.info(
                        "待回填向量队列排空: 重放 %d 条，延后 %d 条（原积压 %d 条）",
                        stats["replayed"], stats["deferred"], stats["pending"],
                    )
                return stats
            finally:
                try:
                    processing.unlink(missing_ok=True)
                except OSError:
                    logger.debug("清理 %s 失败", processing, exc_info=True)
        finally:
            _DRAIN_LOCK.release()

    @staticmethod
    def _requeue_vector_pending(records: list[dict[str, Any]], path: Path) -> None:
        """把未处理的队列条目追加回队列（append-only，损坏不可逆）。"""
        import json

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("待回填向量队列回写失败，丢弃 %d 条", len(records), exc_info=True)

    def queue_vector_backfill(
        self,
        content: str,
        memory_id: str,
        metadata: dict[str, Any] | None = None,
        reason: str = "saga_gap",
    ) -> bool:
        """★ P1-4：把「主存储已写成功、检索层没写上」的记忆投递到待回填队列。

        写方（ChromaDBStore._record_unindexed）只在自己失败时记账；当 saga 因为
        更早的步骤失败而根本没走到向量写入时，没人记账 —— 这条记忆就永久不可语义召回。
        这里补上那只手，队列与消费者仍是 P1-2 那一份。
        """
        if not memory_id or not content:
            return False
        path = self._pending_path()
        if path is None:
            return False
        self._requeue_vector_pending(
            [{
                "ts": time.time(),
                "id": memory_id,
                "document": content,
                "metadata": metadata or {},
                "reason": reason,
            }],
            path,
        )
        return True

    def add_batch_optimized(self, entries: list[dict[str, Any]]) -> None:
        self._ensure_initialized()
        if self._store is None:
            return
        if self._is_new_store:
            # 新接口本身已要求预计算 embeddings，无需额外优化分支
            self._add_batch_new(entries)
            return
        all_ids, all_docs, all_metas = self._build_batch_entries(entries)
        if not all_ids:
            return
        # 尝试预计算 embeddings 并通过 ChromaDB 原生 upsert 写入，失败则回退
        if self._embedding_fn is not None:
            try:
                embeddings = self._embedding_fn(all_docs)
                if isinstance(self._store, ChromaDBStore) and self._store._collection is not None:
                    self._store._collection.upsert(
                        ids=all_ids,
                        embeddings=embeddings,
                        documents=all_docs,
                        metadatas=all_metas,
                    )
                    self._store._persist_client()
                    return
            except Exception as e:
                logger.warning("Vector add_batch_optimized embedding pre-compute failed: %s", e)
        try:
            self._store.add(ids=all_ids, documents=all_docs, metadatas=all_metas)
        except Exception as e:
            logger.warning("Vector add_batch_optimized fallback failed: %s", e)

    def _add_single(self, content: str, memory_id: str, metadata: dict[str, Any]) -> None:
        self._ensure_initialized()
        if self._store is None:
            return
        if self._is_new_store:
            self._add_single_new(content, memory_id, metadata)
            return
        try:
            meta = {k: str(v) for k, v in metadata.items() if v is not None}
            # ChromaDB 要求 metadata 非空，添加占位字段
            if not meta:
                meta["_default"] = "1"
            if len(content) > self._CHUNK_SIZE:
                chunks = self._split_chunks(content, self._CHUNK_SIZE, self._CHUNK_OVERLAP)
                ids = [
                    f"{memory_id}_chunk{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
                    for chunk in chunks
                ]
                metas = [
                    dict(meta, _parent_id=memory_id, _chunk_idx=str(i)) for i in range(len(chunks))
                ]
                self._store.add(ids=ids, documents=chunks, metadatas=metas)
            else:
                self._store.add(
                    ids=[memory_id],
                    documents=[content],
                    metadatas=[meta],
                )
            # ★ R25修复Minor-3：写入后立即 persist 确保向量索引可搜
            if isinstance(self._store, ChromaDBStore):
                self._store._persist_client()
        except Exception as e:
            logger.warning("Vector add failed for %s: %s", memory_id, e)

    def _add_single_new(
        self, content: str, memory_id: str, metadata: dict[str, Any]
    ) -> None:
        """新 VectorStore 接口单条写入路径。"""
        if self._embedding_provider is None or self._store is None:
            return
        try:
            meta = {k: str(v) for k, v in metadata.items() if v is not None}
            if not meta:
                meta["_default"] = "1"
            if len(content) > self._CHUNK_SIZE:
                chunks = self._split_chunks(content, self._CHUNK_SIZE, self._CHUNK_OVERLAP)
                ids = [
                    f"{memory_id}_chunk{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
                    for chunk in chunks
                ]
                metas = [
                    dict(meta, _parent_id=memory_id, _chunk_idx=str(i), content=chunk)
                    for i, chunk in enumerate(chunks)
                ]
                embeddings = self._embedding_provider.embed(chunks)
                self._store.add(ids=ids, embeddings=embeddings, metadatas=metas)
                self._record_chunk_ids(ids, metas)
            else:
                meta["content"] = content
                embeddings = self._embedding_provider.embed([content])
                self._store.add(ids=[memory_id], embeddings=embeddings, metadatas=[meta])
        except Exception as e:
            logger.warning("Vector add (new store) failed for %s: %s", memory_id, e)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        self._ensure_initialized()
        if self._store is None:
            return []
        if self._is_new_store:
            return self._search_new(query, top_k)
        return self._search_legacy(query, top_k)

    def _search_legacy(self, query: str, top_k: int) -> list[dict[str, Any]]:
        try:
            count = self._store.count()
            if count == 0:
                return []
            results = self._store.query(
                query_texts=[query],
                n_results=min(top_k, count),
            )
            output = []
            if results and results.get("documents"):
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
                ids = results.get("ids", [[]])[0] if results.get("ids") else [""] * len(docs)
                for doc_id, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
                    entry = dict(meta) if meta else {}
                    entry["content"] = doc
                    sim = 1.0 - dist
                    # ★ R29修复Minor-3：按类型动态调整相似度阈值
                    mem_type = entry.get("type", "fact")
                    type_thresholds = {"secret": 0.03, "skill": 0.05, "procedural": 0.05}
                    threshold = type_thresholds.get(mem_type, 0.05)
                    if sim < threshold:
                        continue
                    entry["score"] = sim
                    # ★ metadata 中 memory_id 可能为空串(而非缺失), 同样回填 doc_id
                    #   否则归档过滤/API 引用因空 id 失效
                    if not entry.get("memory_id") and doc_id:
                        entry["memory_id"] = doc_id
                    output.append(entry)

            output = self._merge_chunk_results(output)
            return output
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            return [{"degraded": True, "content": "", "score": 0.0, "memory_id": ""}]

    def _search_new(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self._embedding_provider is None:
            return []
        try:
            count = self._store.count()
            if count == 0:
                return []
            query_embeddings = self._embedding_provider.embed([query])
            results = self._store.search(
                query_embedding=query_embeddings[0],
                top_k=min(top_k, count),
            )
            output = []
            for entry in results:
                sim = float(entry.get("score", 0.0))
                mem_type = entry.get("type", "fact")
                type_thresholds = {"secret": 0.03, "skill": 0.05, "procedural": 0.05}
                threshold = type_thresholds.get(mem_type, 0.05)
                if sim < threshold:
                    continue
                entry["score"] = sim
                if "memory_id" not in entry:
                    entry["memory_id"] = entry.get("id", "")
                output.append(entry)
            output = self._merge_chunk_results(output)
            return output
        except Exception as e:
            logger.warning("Vector search (new store) failed: %s", e)
            return [{"degraded": True, "content": "", "score": 0.0, "memory_id": ""}]

    def count(self) -> int:
        self._ensure_initialized()
        if self._store is None:
            return 0
        try:
            return self._store.count()
        except Exception:
            logger.warning("VectorRetriever: count() failed", exc_info=True)
            return 0

    # ★ P1-11：count() 数的是**向量行**，长记忆会被切成多条 chunk
    #   （见 _prepare_batch 里 f"{memory_id}_chunk{hash}"），所以 chunk 数永远不等于
    #   「多少条记忆有向量」。覆盖率判据必须走下面这两个方法，不能用 count()。
    def stored_vector_ids(self) -> set[str] | None:
        """返回向量库里存的原始 id 集合（含 chunk id）；后端不支持枚举时返回 None。"""
        self._ensure_initialized()
        if self._store is None:
            return None
        try:
            all_ids = getattr(self._store, "all_ids", None)
            if callable(all_ids):
                return {str(i) for i in all_ids()}
            collection = getattr(self._store, "_collection", None)
            if collection is not None:
                return {str(i) for i in collection.get(include=[])["ids"]}
        except Exception:
            logger.warning("VectorRetriever: stored_vector_ids() 失败", exc_info=True)
            return None
        logger.warning("VectorRetriever: 后端 %s 不支持枚举 id，覆盖率只能退回标量口径",
                       type(self._store).__name__)
        return None

    def covered_memory_ids(self) -> set[str] | None:
        """按**记忆**去重后的已向量化 id 集合（剥掉 ``_chunk*`` 后缀）。"""
        stored = self.stored_vector_ids()
        if stored is None:
            return None
        return {self._strip_chunk_suffix(i) for i in stored}

    @staticmethod
    def _strip_chunk_suffix(vector_id: str) -> str:
        return re.sub(r"_chunk[0-9a-f]{4,}$", "", vector_id)

    def orphan_vector_ids(self, known_memory_ids: set[str]) -> list[str]:
        """向量库里记忆已不在 ``known_memory_ids`` 中的原始 id（chunk 级）。

        重建走的是 upsert，永远不回收这类残留，只能显式删。``known_memory_ids`` 必须是
        **全表** id（含 superseded），调用方给不出可信基线时应该传空集——这里对空集
        一律返回空列表，绝不因为「基线里查不到」就把整库判成孤儿。
        """
        if not known_memory_ids:
            return []
        stored = self.stored_vector_ids()
        if stored is None:
            return []
        return sorted(vid for vid in stored if self._strip_chunk_suffix(vid) not in known_memory_ids)

    def delete_stored_ids(self, vector_ids: list[str]) -> int:
        """按向量库里的原始 id（含 chunk id）删除。

        后端（Chroma/FAISS）的 delete 不回报实际删除数，所以返回的是**请求**删除的条数，
        只用于日志；失败返回 0。
        """
        self._ensure_initialized()
        if self._store is None or not vector_ids:
            return 0
        try:
            self._store.delete(list(vector_ids))
            return len(vector_ids)
        except Exception:
            logger.warning("VectorRetriever: delete_stored_ids 失败", exc_info=True)
            return 0

    def warmup(self) -> None:
        """预热：启动时预加载模型和初始化 ChromaDB，避免首次搜索延迟。"""
        logger.info("VectorRetriever warmup: initializing...")
        t0 = time.time()
        success = True
        try:
            self._ensure_initialized()
            if self._is_new_store:
                if self._embedding_provider is not None:
                    self._embedding_provider.embed(["warmup"])
                    elapsed = time.time() - t0
                    logger.info("Embedding provider warmed up in %.1fs", elapsed)
            elif self._embedding_fn is not None:
                self._embedding_fn(["warmup"])
                elapsed = time.time() - t0
                logger.info("SentenceTransformer model loaded in %.1fs", elapsed)
                _emit(f"[OmniMem] 嵌入模型就绪 ({elapsed:.1f}s), 内存缓存: {self._embedding_fn.cache_size} 条")
            if self._store is not None:
                doc_count = self._store.count()
                logger.info("Vector store initialized in %.1fs, docs=%d", time.time() - t0, doc_count)
                _emit(f"[OmniMem] {self._backend} 就绪: {doc_count} 条文档")
                logger.info("VectorRetriever [%s] initialized in %.1fs, docs=%d", self._backend, time.time() - t0, doc_count)
        except Exception as e:
            success = False
            logger.warning("VectorRetriever warmup failed (non-fatal): %s", e)
            _emit(f"[OmniMem] ⚠ 向量检索引擎预热失败: {e}")

        if success:
            logger.info("VectorRetriever warmup complete in %.1fs", time.time() - t0)

    def embed_text(self, text: str) -> list[float]:
        self._ensure_initialized()
        if self._is_new_store:
            if self._embedding_provider is None:
                return []
            try:
                vecs = self._embedding_provider.embed([text])
                return vecs[0] if vecs else []
            except Exception as e:
                logger.warning("VectorRetriever embed_text failed: %s", e)
                return []
        if self._embedding_fn is None:
            return []
        try:
            vecs = self._embedding_fn([text])
            return vecs[0] if vecs else []
        except Exception as e:
            logger.warning("VectorRetriever embed_text failed: %s", e)
            return []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文本（用于重建索引等批处理场景）。"""
        self._ensure_initialized()
        if not texts:
            return []
        if self._is_new_store:
            if self._embedding_provider is None:
                return []
            try:
                return self._embedding_provider.embed(texts)
            except Exception as e:
                logger.warning("VectorRetriever embed_texts failed: %s", e)
                return []
        if self._embedding_fn is None:
            return []
        try:
            return self._embedding_fn(texts)
        except Exception as e:
            logger.warning("VectorRetriever embed_texts failed: %s", e)
            return []

    def add_batch_with_embeddings(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, str]],
        embeddings: list[list[float]],
    ) -> None:
        """使用预计算 embeddings 批量写入向量索引。

        用于重建索引等需要并行计算 embeddings 后统一写入的场景。
        """
        self._ensure_initialized()
        if self._store is None or not ids:
            return
        if (
            len(ids) != len(documents)
            or len(ids) != len(metadatas)
            or len(ids) != len(embeddings)
        ):
            logger.warning("add_batch_with_embeddings: 列表长度不一致")
            return
        try:
            if self._is_new_store:
                self._store.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                self._record_chunk_ids(ids, metadatas)
            elif isinstance(self._store, ChromaDBStore) and self._store._collection is not None:
                self._store._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                self._store._persist_client()
            else:
                self._store.add(ids=ids, documents=documents, metadatas=metadatas)
        except Exception as e:
            logger.warning("Vector add_batch_with_embeddings failed: %s", e)

    def rebuild_vectors_parallel(
        self,
        entries: list[dict[str, Any]],
        batch_size: int = 32,
        max_workers: int = 4,
        budget_sec: float = 1800.0,
    ) -> int:
        """分批并行重建向量索引。

        使用 ThreadPoolExecutor 并行计算各 batch 的 embedding，
        最后统一批量写入，降低重建索引的总耗时。

        Args:
            entries: 待重建的记忆条目列表
            batch_size: 每批条目数，默认 32
            max_workers: 并行线程数，默认 4

        Returns:
            实际写入的文档（含 chunk）数量
        """
        self._ensure_initialized()
        if self._store is None or not entries:
            return 0

        # 统一构建批量写入所需的 ids / documents / metadatas（含长文本拆分）
        all_ids, all_docs, all_metas = self._build_batch_entries(entries, include_content=True)
        if not all_ids:
            return 0

        # ★ 先在调用线程同步跑一次嵌入，把模型加载的代价/失败判定收在扇出之前。
        #   P1-6 给模型加载锁加了 5s 有界等待：重建一上来就派 4 个 worker 抢同一把冷锁，
        #   超时的 batch 经 embed_texts 静默返回 []，数量校验判整轮重建失败（副本实测
        #   ids=1020 / embeddings=828）。这里提前失败，让调用方有机会保住原索引。
        if not self.embed_texts(["omnimem 向量重建预热"]):
            logger.error("向量重建中止：嵌入后端不可用（未写入任何条目）")
            return 0

        # 按 batch_size 切分
        batches: list[tuple[list[str], list[str], list[dict[str, str]]]] = []
        for i in range(0, len(all_ids), batch_size):
            batches.append(
                (all_ids[i : i + batch_size], all_docs[i : i + batch_size], all_metas[i : i + batch_size])
            )

        # 并行计算每个 batch 的 embeddings
        def _embed_batch(batch: tuple[list[str], list[str], list[dict[str, str]]]) -> list[list[float]]:
            _, docs, _ = batch
            return self.embed_texts(docs)

        all_embeddings: list[list[float]] = []
        # ★ P1-6: 原先 ``with ThreadPoolExecutor`` 在退出时隐式 shutdown(wait=True)，
        #   而调用方（rebuild_all_from_entries）正持有引擎写锁 —— 一个 batch 卡在
        #   embedding 上就会把写锁钉住，进而冻住每一轮对话的检索。改为总预算有界，
        #   超时即放弃剩余 batch（下面的数量校验会把这次重建判为失败并留日志）。
        budget_sec = max(60.0, float(budget_sec))
        started = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=min(max_workers, len(batches)))
        try:
            if max_workers <= 1 or len(batches) <= 1:
                for batch in batches:
                    all_embeddings.extend(_embed_batch(batch))
            else:
                remaining = budget_sec - (time.monotonic() - started)
                for emb_batch in executor.map(_embed_batch, batches, timeout=remaining):
                    all_embeddings.extend(emb_batch)
        except FuturesTimeoutError:
            logger.warning(
                "Vector parallel rebuild 超过 %.0fs 预算，丢弃剩余 batch（已得 %d/%d）",
                budget_sec, len(all_embeddings), len(all_ids),
            )
        except Exception as e:
            logger.warning("Vector parallel rebuild embedding failed: %s", e)
            return 0
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if len(all_embeddings) != len(all_ids):
            logger.warning(
                "Vector parallel rebuild size mismatch: ids=%d, embeddings=%d",
                len(all_ids),
                len(all_embeddings),
            )
            # ★ 串行补算一次再放弃。并行 worker 撞到的多是「模型首次加载」这一过桥成本：
            #   此刻模型已在主线程加载完，串行 embed 不再争锁，能把一轮注定作废的重建
            #   救回来（否则调用方只能保留旧索引，覆盖率缺口一直挂着）。
            if time.monotonic() - started < budget_sec:
                all_embeddings = []
                for batch in batches:
                    all_embeddings.extend(_embed_batch(batch))
            if len(all_embeddings) != len(all_ids):
                logger.error(
                    "向量重建 embedding 仍不完整（ids=%d, embeddings=%d），放弃写入并保持现有索引",
                    len(all_ids),
                    len(all_embeddings),
                )
                return 0

        # 批量写入
        try:
            self.add_batch_with_embeddings(all_ids, all_docs, all_metas, all_embeddings)
            self.flush()
        except Exception as e:
            logger.warning("Vector parallel rebuild write failed: %s", e)
            return 0

        return len(all_ids)

    def flush(self) -> None:
        if self._is_new_store:
            try:
                if hasattr(self._store, "close"):
                    self._store.close()
                elif hasattr(self._store, "_persist_client"):
                    self._store._persist_client()
            except Exception as e:
                logger.warning("Vector store persist failed: %s", e)
            return
        try:
            if isinstance(self._store, ChromaDBStore):
                self._store._persist_client()
        except Exception as e:
            logger.warning("ChromaDB persist failed: %s", e)
        if self._embedding_fn:
            try:
                self._embedding_fn.persist()
            except Exception as e:
                logger.warning("Embedding cache persist failed: %s", e)

    def delete(self, memory_id: str) -> None:
        """从向量索引中删除指定条目（包括分块）。"""
        self._ensure_initialized()
        if self._store is None:
            return
        try:
            if self._is_new_store:
                ids_to_delete = [memory_id]
                chunk_ids = self._chunk_ids.get(memory_id, [])
                if chunk_ids:
                    ids_to_delete.extend(chunk_ids)
                    self._chunk_ids.pop(memory_id, None)
                self._store.delete(ids_to_delete)
                return
            if isinstance(self._store, ChromaDBStore) and self._store._collection is not None:
                # 查询所有以 memory_id 开头的 ID（含分块）
                all_ids = self._store._collection.get(ids=None, include=[])["ids"]
                ids_to_delete = [
                    i for i in all_ids
                    if i == memory_id or i.startswith(f"{memory_id}_chunk")
                ]
                if ids_to_delete:
                    self._store.delete(ids_to_delete)
            else:
                self._store.delete([memory_id])
        except Exception as e:
            logger.warning("Vector delete failed for %s: %s", memory_id, e)

    def _split_chunks(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        if self._encoder is not None:
            max_tokens = chunk_size // 2
            overlap_tokens = overlap // 2
            return self._split_by_tokens(text, max_tokens, overlap_tokens)
        return self._split_by_chars(text, chunk_size, overlap)

    @staticmethod
    def _split_by_chars(text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                best_end = end
                para_pos = text.rfind("\n\n", start + chunk_size // 3, end)
                if para_pos > start:
                    best_end = para_pos + 2
                else:
                    for sep in ["。", "！", "？", ".", "!", "?"]:
                        sep_pos = text.rfind(sep, start + chunk_size // 2, end)
                        # ★ ASCII 句点仅在后跟非 token 字符时才算句末
                        #   (保护 deploy.yml / ./manage.py / ../path / 3.12)
                        if sep == "." and sep_pos > start:
                            nxt = text[sep_pos + 1] if sep_pos + 1 < len(text) else " "
                            prev = text[sep_pos - 1] if sep_pos > 0 else " "
                            # prev=='.' : 点序列(./... 或省略号)结尾不作句末
                            if nxt.isalnum() or nxt in "/\\." or prev == ".":
                                sep_pos = -1
                        if sep_pos > start:
                            best_end = sep_pos + 1
                            break
                    else:
                        for sep in ["\n", "；", ";"]:
                            sep_pos = text.rfind(sep, start + chunk_size * 2 // 3, end)
                            if sep_pos > start:
                                best_end = sep_pos + 1
                                break
                        else:
                            comma_pos = text.rfind("，", start + chunk_size * 3 // 4, end)
                            if comma_pos > start:
                                best_end = comma_pos + 1
                end = best_end
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap if end < len(text) else end
            if start <= end - chunk_size:
                start = end - overlap
        return chunks if chunks else [text[:chunk_size]]

    def _split_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
        if self._encoder is None:
            return [text]
        tokens = self._encoder.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self._encoder.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            start = end - overlap_tokens if end < len(tokens) else end
        return chunks if chunks else [text]

    @staticmethod
    def _merge_chunk_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parent_chunks: dict[str, list[dict[str, Any]]] = {}
        non_chunked: dict[str, dict[str, Any]] = {}

        for r in results:
            parent_id = r.get("_parent_id", "")
            if parent_id:
                parent_chunks.setdefault(parent_id, []).append(r)
            else:
                mid = r.get("memory_id", "")
                if mid not in non_chunked or r["score"] > non_chunked[mid]["score"]:
                    non_chunked[mid] = r

        merged_list = []

        for parent_id, chunks in parent_chunks.items():
            chunks.sort(key=lambda c: int(c.get("_chunk_idx", "0")))

            scores = [c["score"] for c in chunks]
            best_score = max(scores)
            avg_score = sum(scores) / len(scores) if scores else 0
            total_chunks = max(int(chunks[-1].get("_chunk_idx", "0")) + 1, len(chunks))
            hit_ratio = len(chunks) / total_chunks if total_chunks > 0 else 1.0

            score_std = (
                (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5
                if len(scores) > 1
                else 0
            )
            has_star_chunk = (
                len(chunks) >= 2
                and best_score > avg_score + score_std
                and best_score > 0.6
            )

            if has_star_chunk:
                for chunk in chunks:
                    chunk_entry = {
                        k: v for k, v in chunk.items() if k not in ("_parent_id", "_chunk_idx")
                    }
                    chunk_entry["memory_id"] = f"{parent_id}_chunk{chunk.get('_chunk_idx', '0')}"
                    if chunk["score"] == best_score:
                        chunk_entry["score"] = min(chunk["score"] * 2.0, 1.0)
                    else:
                        chunk_entry["score"] = chunk["score"] * 0.9
                    merged_list.append(chunk_entry)
            else:
                merged_score = best_score * (1.0 + 1.0 * hit_ratio)
                combined_content = "\n".join(c.get("content", "") for c in chunks)
                merged = {
                    k: v
                    for k, v in chunks[0].items()
                    if k not in ("_parent_id", "_chunk_idx", "content", "score")
                }
                merged["memory_id"] = parent_id
                merged["content"] = combined_content
                merged["score"] = merged_score
                merged_list.append(merged)

        merged_list.extend(non_chunked.values())

        return sorted(merged_list, key=lambda x: x["score"], reverse=True)
