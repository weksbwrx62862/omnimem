"""Embedding Provider 模块。"""

import threading
from typing import Any

from omnimem.embedding.base import EmbeddingProvider
from omnimem.embedding.openai_provider import OpenAIEmbeddingProvider
from omnimem.embedding.sentence_transformers_provider import SentenceTransformersProvider

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformersProvider",
    "ONNXEmbeddingProvider",
    "create_embedding_provider",
]

# ★ 性能优化：嵌入模型单例缓存
# 原实现每次调用 create_embedding_provider 创建新实例，导致多次加载模型（耗时数秒）
# 改为按 (provider, model_name) 缓存单例，跨 RetrievalFacade / VectorRetriever 共享
_PROVIDER_CACHE: dict[tuple[str, str], EmbeddingProvider] = {}
_PROVIDER_CACHE_LOCK = threading.Lock()


def create_embedding_provider(config: Any | None = None) -> EmbeddingProvider:
    """根据配置构造 EmbeddingProvider（带单例缓存）。

    Args:
        config: OmniMemConfig 实例或 dict；为空时使用默认 sentence-transformers。

    Returns:
        配置对应的 EmbeddingProvider 实例（相同配置返回缓存单例）。
    """

    def _cfg(key: str, default: Any) -> Any:
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return config.get(key, default)

    provider = _cfg("embedding.provider", "sentence_transformers")
    model_name = _cfg("embedding.model_name", "all-MiniLM-L6-v2")

    cache_key = (provider, model_name)
    with _PROVIDER_CACHE_LOCK:
        cached = _PROVIDER_CACHE.get(cache_key)
        if cached is not None:
            return cached

    if provider == "openai":
        instance = OpenAIEmbeddingProvider(
            model_name=model_name,
            api_key=_cfg("embedding.api_key", None),
            base_url=_cfg("embedding.base_url", None),
        )
    elif provider == "sentence_transformers":
        instance = SentenceTransformersProvider(model_name=model_name)
    elif provider == "onnx":
        from omnimem.embedding.onnx_provider import ONNXEmbeddingProvider
        instance = ONNXEmbeddingProvider(model_name=model_name)
    else:
        raise ValueError(f"不支持的 embedding provider: {provider}")

    with _PROVIDER_CACHE_LOCK:
        # double-check：另一线程可能已创建
        existing = _PROVIDER_CACHE.get(cache_key)
        if existing is not None:
            return existing
        _PROVIDER_CACHE[cache_key] = instance
    return instance


def reset_embedding_provider_cache() -> None:
    """清空嵌入模型单例缓存（用于测试或配置热重载）。"""
    with _PROVIDER_CACHE_LOCK:
        _PROVIDER_CACHE.clear()
