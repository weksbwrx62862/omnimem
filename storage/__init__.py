"""VectorStore 存储模块。"""

from typing import Any

from omnimem.storage.base import VectorStore
from omnimem.storage.chroma_store import ChromaVectorStore
from omnimem.storage.milvus_store import MilvusVectorStore

__all__ = [
    "ChromaVectorStore",
    "MilvusVectorStore",
    "VectorStore",
    "create_vector_store",
]


def create_vector_store(config: Any | None = None) -> VectorStore:
    """根据配置构造 VectorStore。

    Args:
        config: OmniMemConfig 实例或 dict；为空时使用默认 ChromaVectorStore。

    Returns:
        配置对应的 VectorStore 实例。
    """

    def _cfg(key: str, default: Any) -> Any:
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return config.get(key, default)

    provider = _cfg("vector_store.provider", "chroma")

    if provider == "chroma":
        return ChromaVectorStore(
            collection_name=_cfg("vector_store.collection_name", "omnimem"),
            persist_dir=_cfg("vector_store.persist_dir", "/tmp/omnimem/storage/chroma"),
            embedding_dimension=_cfg("vector_store.embedding_dimension", None),
        )
    if provider == "milvus":
        return MilvusVectorStore(
            collection_name=_cfg("vector_store.collection_name", "omnimem"),
            uri=_cfg("vector_store.uri", "http://localhost:19530"),
            token=_cfg("vector_store.token", ""),
            embedding_dimension=_cfg("vector_store.embedding_dimension", 384),
            metric_type=_cfg("vector_store.metric_type", "COSINE"),
            consistency_level=_cfg("vector_store.consistency_level", "Bounded"),
        )
    raise ValueError(f"不支持的 vector_store provider: {provider}")
