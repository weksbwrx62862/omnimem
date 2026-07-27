"""Task 2 统一抽象接口契约测试。

覆盖 BaseRetriever / EmbeddingProvider / VectorStore 抽象接口、
sentence-transformers / OpenAI embedding provider、ChromaDB VectorStore 适配器、
配置驱动工厂函数以及检索通道注册表集成。
"""

from __future__ import annotations

from typing import Any

import pytest

from omnimem.embedding import create_embedding_provider
from omnimem.embedding.base import EmbeddingProvider
from omnimem.retrieval.base import BaseRetriever, RetrievalResult
from omnimem.retrieval.registry import RetrieverRegistry
from omnimem.storage import create_vector_store
from omnimem.storage.base import VectorStore
from omnimem.utils.lock import LockProvider, create_lock_provider

# ── 抽象接口契约 ──


def test_base_retriever_is_abstract() -> None:
    """BaseRetriever 不能直接实例化。"""
    with pytest.raises(TypeError):
        BaseRetriever()


def test_embedding_provider_is_abstract() -> None:
    """EmbeddingProvider 不能直接实例化。"""
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_vector_store_is_abstract() -> None:
    """VectorStore 不能直接实例化。"""
    with pytest.raises(TypeError):
        VectorStore()


def test_lock_provider_is_abstract() -> None:
    """LockProvider 不能直接实例化。"""
    with pytest.raises(TypeError):
        LockProvider()


# ── Fake 实现辅助 ──


class _FakeEmbeddingProvider(EmbeddingProvider):
    """用于契约测试的假 EmbeddingProvider。"""

    @property
    def dimension(self) -> int:
        return 128

    @property
    def model_name(self) -> str:
        return "fake-embedding"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 128 for _ in texts]

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)


class _FakeVectorStore(VectorStore):
    """用于契约测试的假 VectorStore。"""

    def __init__(self) -> None:
        self.data: list[str] = []

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.data.extend(ids)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return []

    def delete(self, ids: list[str]) -> None:
        pass

    def count(self) -> int:
        return len(self.data)


class _FakeRetriever(BaseRetriever):
    """用于注册表集成的假检索通道。"""

    def __init__(self, data_dir: Any = None, config: Any = None, **kwargs: Any) -> None:
        pass

    @property
    def name(self) -> str:
        return "fake"

    def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        return RetrievalResult(
            results=[{"memory_id": "fake-1", "content": query, "score": 1.0}],
            scores=[1.0],
            channel=self.name,
        )


# ── EmbeddingProvider 实现 ──


def test_fake_embedding_provider_contract() -> None:
    """Fake EmbeddingProvider 应满足接口契约。"""
    p = _FakeEmbeddingProvider()
    assert p.dimension == 128
    assert p.model_name == "fake-embedding"
    assert p.embed(["a"]) == [[0.1] * 128]


def test_sentence_transformers_provider_defaults() -> None:
    """sentence-transformers provider 默认参数正确。"""
    pytest.importorskip("sentence_transformers")
    from omnimem.embedding.sentence_transformers_provider import SentenceTransformersProvider

    p = SentenceTransformersProvider()
    assert p.model_name == "all-MiniLM-L6-v2"


def test_openai_provider_dimension_lookup() -> None:
    """OpenAI provider 应能根据模型名推断维度。"""
    pytest.importorskip("openai")
    from omnimem.embedding.openai_provider import OpenAIEmbeddingProvider

    p = OpenAIEmbeddingProvider(model_name="text-embedding-3-large")
    assert p.dimension == 3072


# ── VectorStore 实现 ──


def test_fake_vector_store_contract() -> None:
    """Fake VectorStore 应满足接口契约。"""
    s = _FakeVectorStore()
    s.add(["1"], [[0.1] * 128], [{}])
    assert s.count() == 1


def test_chroma_vector_store_defaults() -> None:
    """ChromaVectorStore 默认集合名正确。"""
    pytest.importorskip("chromadb")
    from omnimem.storage.chroma_store import ChromaVectorStore

    s = ChromaVectorStore()
    assert s._collection_name == "omnimem"


# ── 配置驱动工厂 ──


def test_create_embedding_provider_defaults() -> None:
    """工厂函数默认返回 sentence-transformers provider。"""
    pytest.importorskip("sentence_transformers")

    p = create_embedding_provider()
    assert p.model_name == "all-MiniLM-L6-v2"


def test_create_embedding_provider_openai() -> None:
    """工厂函数根据配置返回 OpenAI provider。"""
    pytest.importorskip("openai")

    p = create_embedding_provider(
        {"embedding.provider": "openai", "embedding.model_name": "text-embedding-3-small"}
    )
    assert p.model_name == "text-embedding-3-small"


def test_create_vector_store_defaults() -> None:
    """工厂函数默认返回 ChromaVectorStore。"""
    pytest.importorskip("chromadb")

    s = create_vector_store()
    from omnimem.storage.chroma_store import ChromaVectorStore

    assert isinstance(s, ChromaVectorStore)


def test_create_lock_provider_file(tmp_path: Any) -> None:
    """工厂函数根据后端返回 FileLockProvider 并支持上下文管理器。"""
    lock_path = tmp_path / "test.lock"
    provider = create_lock_provider(lock_path, backend="file")
    with provider:
        assert provider._lock_count >= 1


def test_create_lock_provider_unsupported_backend(tmp_path: Any) -> None:
    """工厂函数遇到不支持的后端时抛出 ValueError。"""
    with pytest.raises(ValueError, match="不支持的 lock backend"):
        create_lock_provider(tmp_path / "test.lock", backend="unknown")


# ── 检索通道注册表 ──


def test_registry_register_get_list() -> None:
    """注册表支持注册、获取与列举通道。"""
    reg = RetrieverRegistry()
    reg.register("fake", _FakeRetriever)
    assert reg.get("fake") is _FakeRetriever
    assert "fake" in reg.list_channels()


def test_registry_unregister() -> None:
    """注册表支持注销通道。"""
    reg = RetrieverRegistry()
    reg.register("fake", _FakeRetriever)
    reg.unregister("fake")
    assert reg.get("fake") is None


def test_default_registry_contains_vector_and_bm25() -> None:
    """默认注册表包含 vector 与 bm25 通道。"""
    from omnimem.retrieval.registry import DEFAULT_REGISTRY

    channels = DEFAULT_REGISTRY.list_channels()
    assert "vector" in channels
    assert "bm25" in channels


# ── 现有检索通道兼容 BaseRetriever ──


def test_vector_retriever_search_sync_returns_retrieval_result(tmp_path: Any) -> None:
    """VectorRetriever.search_sync 返回统一 RetrievalResult。"""
    from omnimem.retrieval.vector import VectorRetriever

    v = VectorRetriever(data_dir=tmp_path)
    v._store = _FakeVectorStore()
    v._initialized = True
    v._is_new_store = True
    v._embedding_provider = _FakeEmbeddingProvider()

    v.add("测试内容", "m1", {"type": "fact"})
    result = v.search_sync("测试", top_k=5)
    assert isinstance(result, RetrievalResult)
    assert result.channel == "vector"


def test_bm25_retriever_search_sync_returns_retrieval_result(tmp_path: Any) -> None:
    """BM25Retriever.search_sync 返回统一 RetrievalResult。"""
    from omnimem.retrieval.bm25 import BM25Retriever

    b = BM25Retriever(data_dir=tmp_path)
    b.add("测试内容", "m1", {"type": "fact"})
    b.flush()
    result = b.search_sync("测试", top_k=5)
    assert isinstance(result, RetrievalResult)
    assert result.channel == "bm25"


# ── 注册表与 HybridRetriever 集成 ──


def test_hybrid_retriever_loads_extra_channel_from_registry(tmp_path: Any) -> None:
    """新增 BaseRetriever 通道注册后，HybridRetriever 自动加载并使用，无需修改 engine.py。"""
    from omnimem.retrieval.engine import HybridRetriever

    reg = RetrieverRegistry()
    reg.register("fake", _FakeRetriever)
    hybrid = HybridRetriever(data_dir=tmp_path, registry=reg)

    assert "fake" in hybrid._channels
    results = hybrid.search("anything", top_k=5)
    assert any(r.get("memory_id") == "fake-1" for r in results)
