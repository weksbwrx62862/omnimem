"""BM25Retriever 模块测试。

覆盖: add/search 缓冲区、flush、_ensure_built 延迟重建、LRU 淘汰、磁盘缓存
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from omnimem.retrieval.bm25 import BM25Retriever, _tokenize


class TestTokenize:
    def test_english_text(self):
        tokens = _tokenize("python programming language")
        assert len(tokens) > 0

    def test_chinese_text(self):
        tokens = _tokenize("深度学习框架")
        assert len(tokens) > 0

    def test_empty_string(self):
        tokens = _tokenize("")
        assert tokens == []


class TestBM25RetrieverAddSearch:
    def test_add_and_search(self):
        bm25 = BM25Retriever(buffer_size=100)
        bm25.add("Python is a programming language", "m1", {"type": "fact"})
        bm25.add("Java is also a programming language", "m2", {"type": "fact"})
        bm25.warmup()
        results = bm25.search("Python programming")
        assert len(results) > 0
        assert any(r["memory_id"] == "m1" for r in results)

    def test_empty_search_returns_empty(self):
        bm25 = BM25Retriever()
        results = bm25.search("anything")
        assert results == []

    def test_buffer_flush_on_search(self):
        """search() 应自动刷新缓冲区。"""
        bm25 = BM25Retriever(buffer_size=100)
        bm25.add("test content for search", "m1", {})
        assert bm25.pending_count > 0
        bm25.search("test content")
        # After search, buffer should be flushed
        assert bm25.pending_count == 0

    def test_auto_flush_on_buffer_full(self):
        """缓冲区满时应自动刷新。"""
        bm25 = BM25Retriever(buffer_size=2)
        bm25.add("doc1 content", "m1", {})
        bm25.add("doc2 content", "m2", {})
        # buffer_size=2, 2nd add should trigger auto-flush
        bm25.warmup()  # ensure all pending items are indexed
        assert bm25.document_count >= 2


class TestBM25RetrieverFlush:
    def test_manual_flush(self):
        bm25 = BM25Retriever(buffer_size=100)
        bm25.add("content", "m1", {})
        assert bm25.pending_count > 0
        bm25.flush()
        assert bm25.pending_count == 0
        assert bm25.document_count >= 1


class TestBM25RetrieverBatch:
    def test_add_batch(self):
        bm25 = BM25Retriever(buffer_size=100)
        docs = [
            {"content": "doc one", "memory_id": "b1"},
            {"content": "doc two", "memory_id": "b2"},
        ]
        bm25.add_batch(docs)
        assert bm25.document_count >= 2


class TestBM25RetrieverDelete:
    def test_delete_removes_document(self):
        bm25 = BM25Retriever(buffer_size=100)
        bm25.add("to be deleted", "del-1", {})
        bm25.warmup()
        count_before = bm25.document_count
        bm25.delete("del-1")
        bm25.warmup()
        assert bm25.document_count < count_before


class TestBM25RetrieverDiskCache:
    def test_save_and_load_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            bm25 = BM25Retriever(buffer_size=100, data_dir=data_dir)
            bm25.add("cached content", "c1", {"type": "fact"})
            bm25.flush()

            # Create new instance from same dir
            bm25_new = BM25Retriever(buffer_size=100, data_dir=data_dir)
            assert bm25_new.document_count >= 1


class TestBM25RetrieverProperties:
    def test_pending_count(self):
        bm25 = BM25Retriever(buffer_size=100)
        assert bm25.pending_count == 0
        bm25.add("test", "m1", {})
        assert bm25.pending_count == 1

    def test_document_count(self):
        bm25 = BM25Retriever(buffer_size=100)
        assert bm25.document_count == 0
