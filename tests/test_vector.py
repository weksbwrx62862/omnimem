"""VectorRetriever 单元测试。

通过 mock VectorStore 与 embedding function，避免依赖真实的
sentence-transformers 模型与 ChromaDB 服务，聚焦 batch 写入、
chunk 拆分、metadata 构建及 add_batch / add_batch_optimized 行为一致性。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from omnimem.retrieval.vector import VectorRetriever
from omnimem.retrieval.vector_store import ChromaDBStore, VectorStore


class _FakeVectorStore(VectorStore):
    """用于测试的内存 VectorStore，记录所有 add 调用参数。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], list[str], list[dict[str, str]] | None]] = []

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict] | None = None) -> None:
        self.calls.append(("add", ids, documents, metadatas))

    def query(self, query_texts: list[str], n_results: int = 10, where: dict | None = None) -> dict:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def delete(self, ids: list[str]) -> None:
        pass

    def count(self) -> int:
        return 0

    def reset(self) -> None:
        pass


class _FakeChromaDBStore(ChromaDBStore):
    """不初始化真实 ChromaDB 的测试替身，用于验证 embedding 预计算路径。"""

    def __init__(self) -> None:
        self._collection = MagicMock()
        self.upsert_calls: list[dict[str, Any]] = []
        self.persist_count = 0

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict] | None = None) -> None:
        raise AssertionError("optimized path should use upsert, not add")

    def _persist_client(self) -> None:
        self.persist_count += 1


@pytest.fixture
def retriever(tmp_path: Path) -> VectorRetriever:
    """提供已初始化并挂载 FakeStore 的 VectorRetriever。"""
    v = VectorRetriever(data_dir=tmp_path)
    v._store = _FakeVectorStore()
    v._initialized = True
    return v


def test_add_batch_normal_write(retriever: VectorRetriever) -> None:
    """add_batch 应正确拆分参数并调用 store.add。"""
    docs = [
        {"content": "喜欢吃川菜", "memory_id": "b1", "type": "preference"},
        {"content": "住在北京", "memory_id": "b2", "type": "fact"},
    ]
    retriever.add_batch(docs)

    assert len(retriever._store.calls) == 1
    method, ids, documents, metadatas = retriever._store.calls[0]
    assert method == "add"
    assert ids == ["b1", "b2"]
    assert documents == ["喜欢吃川菜", "住在北京"]
    assert metadatas == [
        {"type": "preference"},
        {"type": "fact"},
    ]


def test_add_batch_optimized_fallback_when_no_embedding_fn(retriever: VectorRetriever) -> None:
    """无 embedding_fn 时，add_batch_optimized 应回退到 store.add。"""
    docs = [
        {"content": "使用 macOS", "memory_id": "b3", "type": "fact"},
    ]
    retriever.add_batch_optimized(docs)

    assert len(retriever._store.calls) == 1
    method, ids, documents, metadatas = retriever._store.calls[0]
    assert method == "add"
    assert ids == ["b3"]
    assert documents == ["使用 macOS"]
    assert metadatas == [{"type": "fact"}]


def test_add_batch_long_text_chunked(retriever: VectorRetriever) -> None:
    """长文本应自动拆分为 chunk，并为每个 chunk 生成独立 ID 与 metadata。"""
    # 关闭 token encoder，强制按字符拆分以保证测试确定性
    retriever._encoder = None
    content = "关键词。" + "x" * (VectorRetriever._CHUNK_SIZE * 3)
    docs = [{"content": content, "memory_id": "long", "type": "fact"}]
    retriever.add_batch(docs)

    assert len(retriever._store.calls) == 1
    _, ids, documents, metadatas = retriever._store.calls[0]
    assert len(ids) > 1
    assert all(i.startswith("long_chunk") for i in ids)
    assert len(documents) == len(ids) == len(metadatas)
    for i, meta in enumerate(metadatas):
        assert meta["_parent_id"] == "long"
        assert meta["_chunk_idx"] == str(i)
        assert meta["type"] == "fact"


def test_build_batch_entries_skips_invalid_documents(retriever: VectorRetriever) -> None:
    """_build_batch_entries 应跳过 content 或 memory_id 为空的条目。"""
    docs = [
        {"content": "", "memory_id": "empty-content"},
        {"content": "有效内容", "memory_id": ""},
        {"content": "保留", "memory_id": "keep"},
    ]
    ids, documents, metadatas = retriever._build_batch_entries(docs)
    assert ids == ["keep"]
    assert documents == ["保留"]
    assert metadatas == [{"_default": "1"}]


def test_build_batch_entries_empty_metadata_adds_default(retriever: VectorRetriever) -> None:
    """metadata 为空时应自动添加 `_default: "1"`。"""
    ids, documents, metadatas = retriever._build_batch_entries(
        [{"content": "无 metadata", "memory_id": "m1"}]
    )
    assert ids == ["m1"]
    assert documents == ["无 metadata"]
    assert metadatas == [{"_default": "1"}]


def test_build_batch_entries_converts_metadata_values_to_str(retriever: VectorRetriever) -> None:
    """metadata 中非 None 值应全部转为字符串。"""
    ids, documents, metadatas = retriever._build_batch_entries(
        [{"content": "c", "memory_id": "m2", "count": 42, "flag": True, "empty": None}]
    )
    assert metadatas == [{"count": "42", "flag": "True"}]


def test_add_batch_and_optimized_consistent_without_embedding_fn(retriever: VectorRetriever) -> None:
    """无 embedding_fn 时，add_batch 与 add_batch_optimized 写入参数应一致。"""
    retriever._encoder = None  # 强制按字符拆分，保证两次调用结果一致
    docs = [
        {"content": "短文本", "memory_id": "s1", "type": "fact"},
        {"content": "关键词。" + "x" * (VectorRetriever._CHUNK_SIZE * 3), "memory_id": "s2", "type": "fact"},
    ]
    retriever.add_batch(docs)
    retriever.add_batch_optimized(docs)

    assert len(retriever._store.calls) == 2
    assert retriever._store.calls[0] == retriever._store.calls[1]


def test_add_batch_optimized_precomputed_embeddings(tmp_path: Path) -> None:
    """有 embedding_fn 且 store 为 ChromaDBStore 时，应走 upsert 预计算路径。"""
    v = VectorRetriever(data_dir=tmp_path)
    fake_store = _FakeChromaDBStore()
    v._store = fake_store
    v._initialized = True
    v._embedding_fn = lambda _docs: [[0.1] * 384 for _ in docs]

    docs = [{"content": "测试预计算 embedding", "memory_id": "opt1", "type": "fact"}]
    v.add_batch_optimized(docs)

    assert fake_store._collection.upsert.call_count == 1
    call_kwargs = fake_store._collection.upsert.call_args.kwargs
    assert call_kwargs["ids"] == ["opt1"]
    assert call_kwargs["documents"] == ["测试预计算 embedding"]
    assert call_kwargs["metadatas"] == [{"type": "fact"}]
    assert call_kwargs["embeddings"] == [[0.1] * 384]
    assert fake_store.persist_count == 1


def test_add_batch_optimized_embedding_exception_fallback(retriever: VectorRetriever) -> None:
    """embedding 预计算异常时应回退到 store.add。"""
    retriever._embedding_fn = lambda _docs: (_ for _ in ()).throw(RuntimeError("embed failed"))

    docs = [{"content": "fallback", "memory_id": "fb1"}]
    retriever.add_batch_optimized(docs)

    assert len(retriever._store.calls) == 1
    _, ids, documents, metadatas = retriever._store.calls[0]
    assert ids == ["fb1"]
    assert documents == ["fallback"]
    assert metadatas == [{"_default": "1"}]
