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
        """secret 类型写入检索器时附加语义锚点。"""
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
        assert "sk-abc123" in content


class TestMemoryServiceCompensation:
    """失败补偿场景。"""

    def test_retriever_failure_compensates_store_and_index(self, deps):
        """retriever_add 失败时，store 与 index 被补偿删除。"""
        deps.retriever.add.side_effect = RuntimeError("vector store unavailable")
        service = MemoryService(deps)

        memory_id, result = _call_add_memory(service)

        assert result.success is False
        assert result.failed_step == "retriever_add"
        assert result.completed_steps == ["store_add", "index_add"]

        deps.store.delete.assert_called_once_with(memory_id)
        deps.index.delete.assert_called_once_with(memory_id)
        deps.retriever.delete.assert_not_called()
        deps.knowledge_graph.extract_and_store.assert_not_called()
        deps.temporal_kg.add_triple_from_kg.assert_not_called()

    def test_kg_failure_compensates_all_prior_steps(self, deps):
        """kg_extract 失败时，前面所有步骤被补偿。"""
        deps.knowledge_graph.extract_and_store.side_effect = RuntimeError("kg extraction failed")
        service = MemoryService(deps)

        memory_id, result = _call_add_memory(service)

        assert result.success is False
        assert result.failed_step == "kg_extract"
        assert result.completed_steps == ["store_add", "index_add", "retriever_add"]

        deps.store.delete.assert_called_once_with(memory_id)
        deps.index.delete.assert_called_once_with(memory_id)
        deps.retriever.delete.assert_called_once_with(memory_id)
        deps.temporal_kg.add_triple_from_kg.assert_not_called()

    def test_compensation_flushes_before_delete(self, deps):
        """补偿 store/index 时先 flush 再 delete，处理未落盘边界。"""
        deps.retriever.add.side_effect = RuntimeError("vector store unavailable")
        service = MemoryService(deps)

        memory_id, _ = _call_add_memory(service)

        # store: flush 应在 delete 之前被调用
        store_calls = [str(c) for c in deps.store.method_calls]
        flush_idx = next((i for i, c in enumerate(store_calls) if "flush" in c), -1)
        delete_idx = next((i for i, c in enumerate(store_calls) if "delete" in c), -1)
        assert flush_idx >= 0
        assert delete_idx >= 0
        assert flush_idx < delete_idx

        # index: flush 应在 delete 之前被调用
        index_calls = [str(c) for c in deps.index.method_calls]
        flush_idx = next((i for i, c in enumerate(index_calls) if "flush" in c), -1)
        delete_idx = next((i for i, c in enumerate(index_calls) if "delete" in c), -1)
        assert flush_idx >= 0
        assert delete_idx >= 0
        assert flush_idx < delete_idx


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
