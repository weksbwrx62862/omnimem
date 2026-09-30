"""MemoryService 契约测试。

验证：
  - Saga 各步骤按顺序执行
  - 某一步骤失败时，前面步骤被正确补偿
  - 成功时所有后端均写入
  - 补偿逻辑处理未落盘边界（先 flush 再 delete）
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from omnimem.core.saga import SagaCoordinator
from omnimem.services.memory_service import MemoryService


@pytest.fixture
def deps(tmp_path):
    """构造带真实 SagaCoordinator 的模拟依赖。"""
    d = MagicMock()
    d.store.add.return_value = "mem-abc123"
    d.saga = SagaCoordinator(pending_path=tmp_path / "saga_pending.json")
    d.knowledge_graph._get_all_triples.return_value = [
        {
            "subject": "Alice",
            "predicate": "knows",
            "object": "Bob",
            "source_memory_id": "mem-abc123",
            "confidence": 3,
        }
    ]
    d.knowledge_graph.extract_and_store.return_value = {
        "entities_extracted": 1,
        "triples_extracted": 1,
        "triples_stored": 1,
        "conflicts_found": 0,
        "inferred_triples": 0,
    }
    return d


def _call_add_memory(service: MemoryService) -> tuple[str, Any]:
    """统一调用参数。"""
    return service.add_memory(
        content="Alice knows Bob",
        memory_type="fact",
        confidence=3,
        privacy="personal",
        wing="personal",
        room="people",
        hall="facts",
        summary="Alice knows Bob",
        scope="personal",
        provenance={"source": "test"},
        vc="",
        entities=["Alice", "Bob"],
        stored_at="2026-07-03T00:00:00+00:00",
    )


class TestMemoryServiceSuccess:
    """全步骤成功场景。"""

    def test_steps_execute_in_order(self, deps):
        """Saga 各步骤按 store → index → retriever → kg → temporal 顺序执行。"""
        service = MemoryService(deps)
        memory_id, result = _call_add_memory(service)

        assert memory_id == "mem-abc123"
        assert result.success is True
        assert result.completed_steps == [
            "store_add",
            "index_add",
            "retriever_add",
            "kg_extract",
            "temporal_kg_extract",
        ]

    def test_all_backends_written(self, deps):
        """成功时所有后端均接收到写入调用。"""
        service = MemoryService(deps)
        _call_add_memory(service)

        deps.store.add.assert_called_once()
        deps.index.add.assert_called_once()
        deps.retriever.add.assert_called_once()
        deps.knowledge_graph.extract_and_store.assert_called_once()
        deps.temporal_kg.add_triple_from_kg.assert_called_once()

    def test_retriever_content_enriched_for_secret(self, deps):
        """privacy=secret 写入检索器时仅保留语义锚点，明文不得进入 BM25/向量语料。"""
        service = MemoryService(deps)
        service.add_memory(
            content="sk-abc123",
            memory_type="secret",
            privacy="secret",
            wing="personal",
            room="credentials",
            hall="facts",
            summary="API key",
        )

        call_args = deps.retriever.add.call_args
        content = call_args[0][0]
        assert "[加密信息/密钥/凭证]" in content
        # ★ 安全修复回归: secret 明文曾直接写入检索语料（磁盘明文缓存），现已剥离
        assert "sk-abc123" not in content

    def test_secret_plaintext_not_in_index_or_store_kwargs(self, deps):
        """privacy=secret 时 L2 索引 content 与 original_content 均不含明文。"""
        service = MemoryService(deps)
        service.add_memory(
            content="sk-abc123",
            memory_type="fact",
            privacy="secret",
            wing="personal",
            room="secret",
            hall="facts",
            summary="[加密记忆]",
        )

        index_kwargs = deps.index.add.call_args.kwargs
        assert index_kwargs["content"] == "[加密记忆]"
        store_kwargs = deps.store.add.call_args.kwargs
        assert store_kwargs["original_content"] == ""


class TestMemoryServiceCompensation:
    """失败补偿场景 —— ★ P1-4：降级不得删掉已写成功的主存储。"""

    def test_retriever_failure_keeps_store_and_index(self, deps):
        """retriever_add 失败：抽屉与索引行保留，检索缺口投递待回填队列。"""
        deps.retriever.add.side_effect = RuntimeError("vector store unavailable")
        service = MemoryService(deps)

        memory_id, result = _call_add_memory(service)

        assert result.success is False
        assert result.failed_step == "retriever_add"
        assert result.completed_steps == ["store_add", "index_add"]

        deps.store.delete.assert_not_called()
        deps.index.delete.assert_not_called()
        deps.retriever.delete.assert_not_called()
        deps.knowledge_graph.extract_and_store.assert_not_called()
        deps.temporal_kg.add_triple_from_kg.assert_not_called()

        deps.retriever.queue_vector_backfill.assert_called_once()
        kwargs = deps.retriever.queue_vector_backfill.call_args.kwargs
        assert kwargs["memory_id"] == memory_id

    def test_kg_failure_keeps_all_written_layers(self, deps):
        """kg_extract 失败：三件套都已落定，删掉只会让记忆不可召回。"""
        deps.knowledge_graph.extract_and_store.side_effect = RuntimeError("kg extraction failed")
        service = MemoryService(deps)

        memory_id, result = _call_add_memory(service)

        assert result.success is False
        assert result.failed_step == "kg_extract"
        assert result.completed_steps == ["store_add", "index_add", "retriever_add"]

        deps.store.delete.assert_not_called()
        deps.index.delete.assert_not_called()
        deps.retriever.delete.assert_not_called()
        deps.temporal_kg.add_triple_from_kg.assert_not_called()

    def test_compensation_flushes_without_deleting(self, deps):
        """补偿仍要 flush（未提交的 WAL 写事务会长时间持锁），但不再 delete。"""
        deps.retriever.add.side_effect = RuntimeError("vector store unavailable")
        service = MemoryService(deps)

        _call_add_memory(service)

        deps.store.flush.assert_called()
        deps.store.delete.assert_not_called()
        deps.index.flush.assert_called()
        deps.index.delete.assert_not_called()

    def test_backfill_unavailable_still_keeps_the_memory(self, deps):
        """投递待回填队列本身失败时，也不能退回去删主存储。"""
        deps.retriever.add.side_effect = RuntimeError("vector store unavailable")
        deps.retriever.queue_vector_backfill.side_effect = RuntimeError("disk full")
        service = MemoryService(deps)

        _call_add_memory(service)

        deps.store.delete.assert_not_called()

    def test_index_failure_is_not_silently_rolled_back(self, deps):
        """index_add 失败（P1-3 之后会冒泡）→ 保留抽屉 + 投递检索缺口。"""
        deps.index.add.side_effect = RuntimeError("database is locked")
        service = MemoryService(deps)

        memory_id, result = _call_add_memory(service)

        assert result.success is False
        assert result.failed_step == "index_add"
        deps.store.delete.assert_not_called()
        deps.retriever.queue_vector_backfill.assert_called_once()



class TestMemoryServiceOptionalComponents:
    """可选组件缺失场景。"""

    def test_skips_kg_and_temporal_when_not_configured(self, deps):
        """未配置 KG 时只执行前三个步骤。"""
        deps.knowledge_graph = None
        deps.temporal_kg = None
        service = MemoryService(deps)

        memory_id, result = _call_add_memory(service)

        assert memory_id == "mem-abc123"
        assert result.success is True
        assert result.completed_steps == ["store_add", "index_add", "retriever_add"]
