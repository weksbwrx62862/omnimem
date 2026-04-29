"""VectorRetriever — ChromaDB 向量检索。

使用 ChromaDB 作为向量存储后端，支持语义相似度检索。
OPT-3: 支持自定义 Embedding Function，实现嵌入结果缓存。
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _CachedEmbeddingFunction:
    """OPT-3: 带 LRU 缓存的 Embedding Function。

    包装 sentence-transformers，对相同文本的嵌入结果进行缓存，
    避免重复计算。缓存上限 1000 条，超限时淘汰一半。
    ★ 新增：缓存持久化到磁盘，跨会话后避免重复 embedding 计算。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_path: Path | None = None):
        self._model_name = model_name
        self._model = None
        self._cache: dict[str, list[float]] = {}
        self._max_cache = 1000
        self._lock = threading.Lock()
        self._cache_path = cache_path
        self._load_cache()

    def _load_cache(self) -> None:
        """从磁盘加载 embedding 缓存。"""
        if not self._cache_path or not self._cache_path.exists():
            return
        try:
            import json

            with open(self._cache_path, encoding="utf-8") as f:
                data = json.load(f)
            self._cache = {k: [float(v) for v in vec] for k, vec in data.items()}
            logger.debug("Loaded %d entries from embedding cache", len(self._cache))
        except Exception as e:
            logger.debug("Embedding cache load failed: %s", e)
            self._cache = {}

    def persist(self) -> None:
        """将 embedding 缓存持久化到磁盘。"""
        if not self._cache_path:
            return
        try:
            import json

            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = dict(self._cache)
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            logger.debug("Saved %d entries to embedding cache", len(data))
        except Exception as e:
            logger.debug("Embedding cache persist failed: %s", e)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def __call__(self, input: list[str]) -> list[list[float]]:
        """ChromaDB embedding function interface."""
        results = []
        to_encode = []
        to_encode_idx = []

        with self._lock:
            for i, text in enumerate(input):
                text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                if text_hash in self._cache:
                    results.append((i, self._cache[text_hash]))
                else:
                    to_encode.append(text)
                    to_encode_idx.append((i, text_hash))

        if to_encode:
            model = self._get_model()
            embeddings = model.encode(to_encode, convert_to_numpy=True)
            with self._lock:
                for (orig_idx, text_hash), emb in zip(to_encode_idx, embeddings, strict=False):
                    vec = emb.tolist()
                    self._cache[text_hash] = vec
                    results.append((orig_idx, vec))

                # LRU eviction: clear half when over limit
                if len(self._cache) > self._max_cache:
                    items = list(self._cache.items())
                    self._cache = dict(items[self._max_cache // 2 :])

        # Reassemble in original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


class VectorRetriever:
    """ChromaDB 向量检索。"""

    def __init__(self, backend: str = "chromadb", data_dir: Path | None = None):
        self._backend = backend
        self._data_dir = data_dir or Path("/tmp/omnimem/retrieval")
        self._client: Any = None
        self._collection: Any = None
        self._initialized = False
        self._embedding_fn: Any = None
        # ★ 尝试加载 tiktoken，用于基于 token 预算的动态分块（避免模型截断）
        self._encoder = None
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass

    def _ensure_initialized(self) -> None:
        """延迟初始化 ChromaDB 客户端和集合。"""
        if self._initialized:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self._data_dir / "chroma"))
            # OPT-3: 尝试使用带缓存的 embedding function
            try:
                cache_path = self._data_dir / "embedding_cache.json"
                self._embedding_fn = _CachedEmbeddingFunction(cache_path=cache_path)
                self._collection = self._client.get_or_create_collection(
                    name="omnimem",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self._embedding_fn,
                )
                logger.debug("ChromaDB collection initialized with cached embeddings")
            except Exception:
                # Fallback to default embedding (no cache)
                self._collection = self._client.get_or_create_collection(
                    name="omnimem",
                    metadata={"hnsw:space": "cosine"},
                )
                logger.debug("ChromaDB collection initialized with default embeddings")
            logger.debug(
                "ChromaDB collection initialized: %d documents",
                self._collection.count() if self._collection else 0,
            )
        except ImportError:
            logger.warning("chromadb not installed — vector search unavailable")
        except Exception as e:
            logger.warning("ChromaDB init failed: %s", e)
        self._initialized = True

    # ★ 长文本分块阈值（超过此长度自动分块索引）
    # ★ QUAL-2修复：从300提升至500
    #   原因：300字符对量子计算等密集技术内容过小，导致完整概念被截断
    #   500字符可容纳1-2个完整的子主题段落（如"墨子号卫星"的完整描述）
    _CHUNK_SIZE = 500
    _CHUNK_OVERLAP = 100  # ★ 从50提升至100：确保专业术语不被切断

    def add(self, content: str, memory_id: str, metadata: dict[str, Any]) -> None:
        """添加文档到向量索引。超长文本自动分块。"""
        self._add_single(content, memory_id, metadata)

    def add_batch(self, documents: list[dict[str, Any]]) -> None:
        """批量添加文档到向量索引，减少 ChromaDB 交互次数。"""
        self._ensure_initialized()
        if self._collection is None:
            return
        all_ids: list[str] = []
        all_docs: list[str] = []
        all_metas: list[dict[str, str]] = []
        for doc in documents:
            content = doc.get("content", "")
            memory_id = doc.get("memory_id", "")
            if not content or not memory_id:
                continue
            meta = {
                k: str(v)
                for k, v in doc.items()
                if k not in ("content", "memory_id") and v is not None
            }
            if len(content) > self._CHUNK_SIZE:
                chunks = self._split_chunks(content, self._CHUNK_SIZE, self._CHUNK_OVERLAP)
                for i, chunk in enumerate(chunks):
                    # ★ 稳定 chunk ID：基于内容哈希，避免重复索引导致存储膨胀
                    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                    all_ids.append(f"{memory_id}_chunk{chunk_hash}")
                    all_docs.append(chunk)
                    all_metas.append(dict(meta, _parent_id=memory_id, _chunk_idx=str(i)))
            else:
                all_ids.append(memory_id)
                all_docs.append(content)
                all_metas.append(meta)
        if not all_ids:
            return
        try:
            self._collection.upsert(ids=all_ids, documents=all_docs, metadatas=all_metas)
            try:
                if self._client and hasattr(self._client, "persist"):
                    self._client.persist()
            except Exception:
                pass
        except Exception as e:
            logger.warning("Vector add_batch failed: %s", e)

    def _add_single(self, content: str, memory_id: str, metadata: dict[str, Any]) -> None:
        """内部单条添加实现。"""
        self._ensure_initialized()
        if self._collection is None:
            return
        try:
            meta = {k: str(v) for k, v in metadata.items() if v is not None}
            # ★ 超长文本分块索引：每 500 字一块，重叠 50 字
            if len(content) > self._CHUNK_SIZE:
                chunks = self._split_chunks(content, self._CHUNK_SIZE, self._CHUNK_OVERLAP)
                # ★ 稳定 chunk ID：基于内容哈希，避免重复索引导致存储膨胀
                ids = [
                    f"{memory_id}_chunk{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
                    for chunk in chunks
                ]
                metas = [
                    dict(meta, _parent_id=memory_id, _chunk_idx=str(i)) for i in range(len(chunks))
                ]
                self._collection.upsert(
                    ids=ids,
                    documents=chunks,
                    metadatas=metas,
                )
            else:
                self._collection.upsert(
                    ids=[memory_id],
                    documents=[content],
                    metadatas=[meta],
                )
            # ★ R22修复：写入后强制持久化，确保 HNSW 索引立即更新
            try:
                if self._client and hasattr(self._client, "persist"):
                    self._client.persist()
            except Exception:
                pass
        except Exception as e:
            logger.warning("Vector add failed for %s: %s", memory_id, e)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """语义相似度检索。"""
        self._ensure_initialized()
        if self._collection is None:
            return []
        try:
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
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
                    # cosine distance → similarity score (1 - distance)
                    sim = 1.0 - dist
                    # ★ 最低相关性过滤：cosine similarity < 0.25 的结果视为不相关
                    # ★ QUAL-2修复：从0.3降至0.25
                    #   0.3阈值对中文子话题查询过于严格（如"AlexNet深度学习突破" sim=0.276被误过滤）
                    #   0.25仍过滤纯噪声，但保留弱语义关联结果，由RRF融合做最终排序
                    if sim < 0.25:
                        continue
                    entry["score"] = sim
                    # ★ 确保有 memory_id：优先从 metadata 取，回退到 ChromaDB 文档 ID
                    if "memory_id" not in entry and doc_id:
                        entry["memory_id"] = doc_id
                    output.append(entry)

            # ★ 合并分块结果：同一 parent_id 的多个 chunk 取最高分
            output = self._merge_chunk_results(output)
            return output
        except Exception as e:
            logger.debug("Vector search failed: %s", e)
            return []

    def count(self) -> int:
        """返回索引中的文档数量。"""
        self._ensure_initialized()
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def embed_text(self, text: str) -> list[float]:
        """计算单条文本的 embedding 向量。

        供外部模块（如 ContextManager 语义去重）使用。
        若 embedding 函数不可用，返回空列表。
        """
        self._ensure_initialized()
        if self._embedding_fn is None:
            return []
        try:
            vecs = self._embedding_fn([text])
            return vecs[0] if vecs else []
        except Exception as e:
            logger.debug("VectorRetriever embed_text failed: %s", e)
            return []

    def flush(self) -> None:
        """刷新索引。强制持久化 ChromaDB 数据与 embedding 缓存。"""
        try:
            if self._client and hasattr(self._client, "persist"):
                self._client.persist()
        except Exception as e:
            logger.debug("ChromaDB persist failed: %s", e)
        if self._embedding_fn:
            try:
                self._embedding_fn.persist()
            except Exception as e:
                logger.debug("Embedding cache persist failed: %s", e)

    def _split_chunks(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """将长文本分块。优先使用 tiktoken 按 token 预算切割（若可用），否则按字符预算+语义边界切割。"""
        if self._encoder is not None:
            # 将字符预算保守转换为 token 预算（中文约 1-2 char/token，英文约 4 char/token）
            max_tokens = chunk_size // 2
            overlap_tokens = overlap // 2
            return self._split_by_tokens(text, max_tokens, overlap_tokens)
        return self._split_by_chars(text, chunk_size, overlap)

    @staticmethod
    def _split_by_chars(text: str, chunk_size: int, overlap: int) -> list[str]:
        """按字符预算 + 语义边界分块（回退策略）。

        优先级：
        1. 段落边界（双换行）
        2. 句子边界（句号、问号、感叹号）
        3. 分号/换行
        4. 逗号（最后手段）
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                # ★ 多级语义边界检测：从粗到细
                best_end = end
                # 1. 段落边界（最高优先）
                para_pos = text.rfind("\n\n", start + chunk_size // 3, end)
                if para_pos > start:
                    best_end = para_pos + 2
                else:
                    # 2. 句子边界
                    for sep in ["。", "！", "？", ".", "!", "?"]:
                        sep_pos = text.rfind(sep, start + chunk_size // 2, end)
                        if sep_pos > start:
                            best_end = sep_pos + 1
                            break
                    else:
                        # 3. 分号/换行
                        for sep in ["\n", "；", ";"]:
                            sep_pos = text.rfind(sep, start + chunk_size * 2 // 3, end)
                            if sep_pos > start:
                                best_end = sep_pos + 1
                                break
                        else:
                            # 4. 逗号（最后手段，避免切断专业术语）
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
        """基于 tiktoken token 预算分块，确保每块 token 数均匀，避免 embedding 模型截断。"""
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
        """合并同一 parent 的分块结果（QUAL-2增强版）。

        ★ 核心改进：子主题保护机制
        - 高相关性独立chunk不再被强制合并到父文档
        - 单chunk命中获得更高评分加成（最高2.0x）
        - 只在多chunk均高度相关时才合并，避免稀释效应
        """
        # 按 (parent_id, chunk_idx) 分组收集所有 chunk
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

        # 处理分块结果：智能合并策略
        for parent_id, chunks in parent_chunks.items():
            chunks.sort(key=lambda c: int(c.get("_chunk_idx", "0")))

            # ★ 计算各chunk的相关性分布
            scores = [c["score"] for c in chunks]
            best_score = max(scores)
            avg_score = sum(scores) / len(scores) if scores else 0
            total_chunks = max(int(chunks[-1].get("_chunk_idx", "0")) + 1, len(chunks))
            hit_ratio = len(chunks) / total_chunks if total_chunks > 0 else 1.0

            # ★ QUAL-2核心修复：子主题检测与分离策略
            # 检测是否存在"明星chunk"（分数显著高于平均）
            score_std = (
                (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5
                if len(scores) > 1
                else 0
            )
            has_star_chunk = (
                len(chunks) >= 2
                and best_score > avg_score + score_std  # 显著高于平均
                and best_score > 0.6  # 自身也是高相关
            )

            if has_star_chunk:
                # ★ 子主题模式：将高分chunk作为独立结果返回
                # 这样"墨子号"所在的chunk不会被整篇量子计算文章淹没
                for chunk in chunks:
                    chunk_entry = {
                        k: v for k, v in chunk.items() if k not in ("_parent_id", "_chunk_idx")
                    }
                    chunk_entry["memory_id"] = f"{parent_id}_chunk{chunk.get('_chunk_idx', '0')}"
                    # ★ 明星chunk额外加分（最高2.0x）
                    if chunk["score"] == best_score:
                        chunk_entry["score"] = min(chunk["score"] * 2.0, 1.0)
                    else:
                        # 其他chunk适度降权，避免噪声
                        chunk_entry["score"] = chunk["score"] * 0.9
                    merged_list.append(chunk_entry)
            else:
                # ★ 常规模式：合并所有chunk（原有逻辑增强版）
                # 匹配 chunk 比例越高 → 额外加分（最高2.0x，原为1.5x）
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

        # 加入非分块结果
        merged_list.extend(non_chunked.values())

        # 按 score 排序
        return sorted(merged_list, key=lambda x: x["score"], reverse=True)
