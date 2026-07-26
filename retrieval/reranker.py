"""CrossEncoderReranker — Cross-Encoder 重排。

使用 Cross-Encoder 模型对检索结果进行精排，提升精度。
可选功能，需要 sentence-transformers 库。
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-Encoder 重排。"""

    # ★ 全局模型缓存：避免每题重新加载（跨 OmniMemMemoryProvider 实例共享）
    _global_model: Any = None
    _global_model_name: str = ""
    # ★ 修复 C8：全局模型加载锁，避免多线程首次加载重复加载模型
    _global_model_lock = threading.Lock()

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", model_path: str = "", device: str = ""):
        import os

        self._model_name = model_name
        self._model_path = model_path
        # ★ M7-14: 设备解析优先级：显式参数 > OMNIMEM_RERANKER_DEVICE 环境变量 > cpu
        self._device = device or os.environ.get("OMNIMEM_RERANKER_DEVICE", "") or "cpu"
        self._model: Any = None

    def _ensure_model(self) -> bool:
        """延迟加载 Cross-Encoder 模型，优先使用全局缓存（线程安全）。"""
        if self._model is not None:
            return True

        # ★ 修复 C8：使用全局锁保护检查-加载-赋值，避免多线程重复加载
        with CrossEncoderReranker._global_model_lock:
            # double-check：进入锁后再次检查全局缓存
            model_key = self._model_path or self._model_name
            if (
                CrossEncoderReranker._global_model is not None
                and CrossEncoderReranker._global_model_name == model_key
            ):
                self._model = CrossEncoderReranker._global_model
                logger.info("Cross-Encoder 模型从全局缓存加载（跳过重复加载）")
                return True

            # 持锁期间执行加载（可能耗时数秒，但保证只加载一次）
            try:
                # ROCm PyTorch 兼容性
                import torch.distributed as dist
                if not hasattr(dist, 'is_initialized'):
                    dist.is_initialized = lambda: False

                # M7-14: 使用配置的 device，默认 cpu
                import os
                if self._device == "cpu":
                    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(model_key, device=self._device)

                # ★ 缓存到全局
                CrossEncoderReranker._global_model = self._model
                CrossEncoderReranker._global_model_name = model_key

                return True
            except ImportError:
                logger.warning("sentence_transformers not installed — reranking disabled")
                return False
            except Exception as e:
                logger.warning("Cross-Encoder model load failed: %s", e)
                return False

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """对检索结果进行 Cross-Encoder 重排。

        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回前 K 个结果

        Returns:
            重排后的结果列表
        """
        if not results or not self._ensure_model():
            return results[:top_k]

        try:
            pairs = [(query, r.get("content", "")) for r in results]
            if self._model is None:
                return results[:top_k]
            scores = self._model.predict(pairs)

            # 按 score 降序排列
            scored = list(zip(results, scores, strict=False))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for doc, score in scored[:top_k]:
                entry = dict(doc)
                entry["rerank_score"] = float(score)
                reranked.append(entry)
            return reranked
        except Exception as e:
            logger.warning("Reranking failed: %s", e)
            return results[:top_k]
