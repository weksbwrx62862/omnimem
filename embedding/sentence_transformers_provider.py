"""sentence-transformers EmbeddingProvider 实现。"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Any

from omnimem.embedding.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformersProvider(EmbeddingProvider):
    """基于 sentence-transformers 的本地嵌入服务。"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_path: str = "",
        device: str = "auto",
        max_cache: int = 1000,
    ):
        self._model_name = model_name
        self._model_path = model_path
        self._device = device
        self._max_cache = max_cache
        self._model: Any = None
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_path or self._model_name

    @property
    def dimension(self) -> int:
        """返回模型维度（延迟加载模型以确定）。"""
        try:
            model = self._get_model()
            return int(model.get_sentence_embedding_dimension())
        except Exception as e:
            logger.warning("Failed to get embedding dimension: %s", e)
            return 0

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                import torch.distributed as dist

                if not hasattr(dist, "is_initialized"):
                    dist.is_initialized = lambda: False
            except Exception:
                logger.debug("torch.distributed patch failed", exc_info=True)
            from sentence_transformers import SentenceTransformer

            # GFW 修复：使用中国镜像
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

            model_path = self._model_path or self._model_name
            device = self._device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            logger.info("Loading sentence-transformers model: %s (device=%s)", model_path, device)
            self._model = SentenceTransformer(model_path, device=device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """同步嵌入，带内存缓存。"""
        results: list[tuple[int, list[float]]] = []
        to_encode: list[str] = []
        to_encode_idx: list[tuple[int, str]] = []

        with self._lock:
            for i, text in enumerate(texts):
                text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                cached = self._cache.get(text_hash)
                if cached is not None:
                    results.append((i, cached))
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
                if len(self._cache) > self._max_cache:
                    items = list(self._cache.items())
                    self._cache = dict(items[self._max_cache // 2 :])

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """异步包装：在线程池中执行同步嵌入。"""
        import asyncio

        return await asyncio.to_thread(self.embed, texts)
