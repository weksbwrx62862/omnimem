"""retrieval.vector_factory.create_vector_store 离线单元测试。

覆盖：
  - 3 种后端分派（chromadb / qdrant / faiss）→ 对应类
  - persist_dir / data_dir 别名，str → Path 归一化
  - embedding_fn 缺省时走 _CachedEmbeddingFunction 构造
  - embedding_fn 构造失败降级为 None
  - qdrant 别名 url / qdrant_url
  - 未知后端 → ValueError
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest
from omnimem.retrieval import vector_factory as vf
from omnimem.retrieval.vector_factory import create_vector_store


@pytest.fixture(autouse=True)
def _stub_faiss_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """在导入 faiss_store 前占位 faiss 顶层模块（离线环境无 faiss）。"""
    if "faiss" in sys.modules:
        return
    stub = ModuleType("faiss")
    stub.IndexFlatL2 = MagicMock()  # type: ignore[attr-defined]
    stub.IndexFlatIP = MagicMock()  # type: ignore[attr-defined]
    stub.write_index = MagicMock()  # type: ignore[attr-defined]
    stub.read_index = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faiss", stub)


@pytest.fixture
def stub_stores(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """替换 ChromaDBStore / QdrantStore / FAISSStore / _CachedEmbeddingFunction 为 MagicMock。"""
    chroma = MagicMock()
    qdrant = MagicMock()
    faiss_cls = MagicMock()
    cached_fn = MagicMock()
    monkeypatch.setattr(vf, "ChromaDBStore", chroma)
    monkeypatch.setattr(vf, "QdrantStore", qdrant)
    monkeypatch.setattr(vf, "_CachedEmbeddingFunction", cached_fn)

    # faiss 走惰性 import，需要 patch 模块属性
    import omnimem.retrieval.faiss_store as faiss_mod

    monkeypatch.setattr(faiss_mod, "FAISSStore", faiss_cls)
    return {
        "chroma": chroma,
        "qdrant": qdrant,
        "faiss": faiss_cls,
        "cached_fn": cached_fn,
    }


# ── chromadb 后端 ──


def test_chromadb_default_args(stub_stores: dict[str, Any]) -> None:
    create_vector_store("chromadb")
    stub_stores["chroma"].assert_called_once()
    kwargs = stub_stores["chroma"].call_args.kwargs
    assert kwargs["collection_name"] == "omnimem"
    assert isinstance(kwargs["persist_dir"], Path)
    # 未传 embedding_fn → 自动构造 _CachedEmbeddingFunction
    stub_stores["cached_fn"].assert_called_once()
    assert kwargs["embedding_fn"] is stub_stores["cached_fn"].return_value


def test_chromadb_data_dir_alias(stub_stores: dict[str, Any], tmp_path: Path) -> None:
    target = tmp_path / "custom"
    create_vector_store("chromadb", data_dir=str(target))
    kwargs = stub_stores["chroma"].call_args.kwargs
    assert kwargs["persist_dir"] == target


def test_chromadb_persist_dir_priority(stub_stores: dict[str, Any], tmp_path: Path) -> None:
    """persist_dir 先于 data_dir 被 pop，因此 data_dir 不会覆盖 persist_dir。"""
    p = tmp_path / "p"
    d = tmp_path / "d"
    create_vector_store("chromadb", persist_dir=str(p), data_dir=str(d))
    kwargs = stub_stores["chroma"].call_args.kwargs
    assert kwargs["persist_dir"] == p


def test_chromadb_path_passthrough(stub_stores: dict[str, Any], tmp_path: Path) -> None:
    create_vector_store("chromadb", persist_dir=tmp_path)
    kwargs = stub_stores["chroma"].call_args.kwargs
    assert kwargs["persist_dir"] is tmp_path  # 已是 Path 不再包装


def test_chromadb_explicit_embedding_fn(stub_stores: dict[str, Any]) -> None:
    custom = MagicMock()
    create_vector_store("chromadb", embedding_fn=custom)
    kwargs = stub_stores["chroma"].call_args.kwargs
    assert kwargs["embedding_fn"] is custom
    stub_stores["cached_fn"].assert_not_called()


def test_chromadb_collection_name_override(stub_stores: dict[str, Any]) -> None:
    create_vector_store("chromadb", collection_name="my_col")
    kwargs = stub_stores["chroma"].call_args.kwargs
    assert kwargs["collection_name"] == "my_col"


def test_chromadb_cached_fn_failure_yields_none_embedding(
    monkeypatch: pytest.MonkeyPatch,
    stub_stores: dict[str, Any],
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("cannot create cache")

    monkeypatch.setattr(vf, "_CachedEmbeddingFunction", boom)
    create_vector_store("chromadb")
    kwargs = stub_stores["chroma"].call_args.kwargs
    assert kwargs["embedding_fn"] is None


# ── qdrant 后端 ──


def test_qdrant_default_args(stub_stores: dict[str, Any]) -> None:
    create_vector_store("qdrant")
    kwargs = stub_stores["qdrant"].call_args.kwargs
    assert kwargs["collection_name"] == "omnimem"
    assert kwargs["url"] == "localhost:6333"
    assert kwargs["api_key"] is None
    # 未传 embedding_fn → 自动构造
    assert kwargs["embedding_fn"] is stub_stores["cached_fn"].return_value


def test_qdrant_url_alias(stub_stores: dict[str, Any]) -> None:
    create_vector_store("qdrant", url="http://qdrant:6333")
    kwargs = stub_stores["qdrant"].call_args.kwargs
    assert kwargs["url"] == "http://qdrant:6333"


def test_qdrant_url_and_qdrant_url_both_set(stub_stores: dict[str, Any]) -> None:
    """qdrant_url 先 pop，url 未使用。"""
    create_vector_store("qdrant", qdrant_url="primary", url="secondary")
    kwargs = stub_stores["qdrant"].call_args.kwargs
    assert kwargs["url"] == "primary"


def test_qdrant_api_key_and_explicit_embedding(stub_stores: dict[str, Any]) -> None:
    fn = MagicMock()
    create_vector_store("qdrant", api_key="sk-x", embedding_fn=fn)
    kwargs = stub_stores["qdrant"].call_args.kwargs
    assert kwargs["api_key"] == "sk-x"
    assert kwargs["embedding_fn"] is fn
    stub_stores["cached_fn"].assert_not_called()


def test_qdrant_cached_fn_failure_yields_none_embedding(
    monkeypatch: pytest.MonkeyPatch,
    stub_stores: dict[str, Any],
) -> None:
    monkeypatch.setattr(vf, "_CachedEmbeddingFunction", lambda **_: (_ for _ in ()).throw(RuntimeError("x")))
    create_vector_store("qdrant")
    kwargs = stub_stores["qdrant"].call_args.kwargs
    assert kwargs["embedding_fn"] is None


# ── faiss 后端 ──


def test_faiss_default_args(stub_stores: dict[str, Any]) -> None:
    create_vector_store("faiss")
    kwargs = stub_stores["faiss"].call_args.kwargs
    assert isinstance(kwargs["persist_dir"], Path)
    assert kwargs["embedding_fn"] is stub_stores["cached_fn"].return_value


def test_faiss_persist_dir_custom(stub_stores: dict[str, Any], tmp_path: Path) -> None:
    target = tmp_path / "idx"
    create_vector_store("faiss", persist_dir=target)
    kwargs = stub_stores["faiss"].call_args.kwargs
    assert kwargs["persist_dir"] is target


def test_faiss_explicit_embedding_fn(stub_stores: dict[str, Any]) -> None:
    fn = MagicMock()
    create_vector_store("faiss", embedding_fn=fn)
    kwargs = stub_stores["faiss"].call_args.kwargs
    assert kwargs["embedding_fn"] is fn


def test_faiss_cached_fn_failure_yields_none_embedding(
    monkeypatch: pytest.MonkeyPatch,
    stub_stores: dict[str, Any],
) -> None:
    def boom(**_: Any) -> Any:
        raise RuntimeError("fail")

    monkeypatch.setattr(vf, "_CachedEmbeddingFunction", boom)
    create_vector_store("faiss")
    kwargs = stub_stores["faiss"].call_args.kwargs
    assert kwargs["embedding_fn"] is None


# ── 未知后端 ──


def test_unknown_backend_raises_value_error() -> None:
    with pytest.raises(ValueError) as ei:
        create_vector_store("elasticsearch")
    assert "Unknown vector store backend" in str(ei.value)


def test_default_backend_is_chromadb(stub_stores: dict[str, Any]) -> None:
    create_vector_store()
    stub_stores["chroma"].assert_called_once()
    stub_stores["qdrant"].assert_not_called()
    stub_stores["faiss"].assert_not_called()
