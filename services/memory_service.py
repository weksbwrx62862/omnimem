"""OmniMem 统一记忆写入编排层 —— MemoryService。

将记忆写入所涉及的 Store、Index、Retriever、KnowledgeGraph、TemporalKG
统一纳入 Saga 事务编排，任一关键步骤失败时按逆序补偿已写入的数据，
保证多后端一致性。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from omnimem.core.saga import SagaCoordinator, SagaResult, SagaStep
from omnimem.handlers.deps import HandlerDependencies

logger = logging.getLogger(__name__)


def _enrich_retriever_content(content: str, memory_type: str, room: str = "") -> str:
    """为 secret/skill/procedural 类型附加可搜索描述，弥合语义鸿沟。"""
    if memory_type == "secret":
        return f"[加密信息/密钥/凭证] {room} {content}"
    if memory_type == "skill":
        return f"[技能/步骤/教程] {room} {content}"
    if memory_type == "procedural":
        return f"[流程/操作/指南] {room} {content}"
    return content


class MemoryService:
    """统一记忆写入编排服务。

    职责：
      1. 将 Store/Index/Retriever/KG/TemporalKG 写入封装为 Saga 步骤
      2. 每个步骤提供补偿回调，失败时逆序回滚
      3. 补偿逻辑处理数据尚未落盘的边界情况（先 flush 再删除）
    """

    def __init__(self, deps: HandlerDependencies) -> None:
        self.deps = deps

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def add_memory(
        self,
        *,
        content: str,
        memory_type: str = "fact",
        confidence: int = 3,
        privacy: str = "personal",
        wing: str = "",
        room: str = "",
        hall: str = "",
        summary: str = "",
        scope: str = "personal",
        provenance: dict[str, Any] | None = None,
        vc: str = "",
        entities: list[str] | None = None,
        stored_at: str = "",
    ) -> tuple[str, SagaResult]:
        """编排写入一条记忆到所有后端。

        Saga 步骤顺序：
          1. store_add   → DrawerClosetStore
          2. index_add   → ThreeLevelIndex
          3. retriever_add → 向量/BM25 检索器
          4. kg_extract  → KnowledgeGraph 三元组抽取
          5. temporal_kg_extract → TemporalKG 时序三元组同步

        任一失败即触发前面步骤的补偿回调。

        Returns:
            (memory_id, saga_result)
        """
        if not stored_at:
            stored_at = datetime.now(timezone.utc).isoformat()

        memory_id: str = ""
        steps: list[SagaStep] = []

        # 1. Store 写入（主存储作为事实来源）
        if self.deps.store is not None:
            store = self.deps.store

            def _store_add() -> str:
                nonlocal memory_id
                memory_id = store.add(
                    wing=wing,
                    room=room,
                    content=content,
                    memory_type=memory_type,
                    confidence=confidence,
                    privacy=privacy,
                    provenance=provenance,
                    vc=vc,
                    original_content=content,
                    entities=entities or [],
                )
                return memory_id

            def _compensate_store() -> None:
                if not memory_id:
                    return
                # 先 flush 缓冲，避免 drawer/closet 文件尚未落盘导致删除失败
                try:
                    store.flush()
                except Exception as e:
                    logger.warning("MemoryService store 补偿 flush 失败: %s", e)
                try:
                    store.delete(memory_id)
                except Exception as e:
                    logger.warning("MemoryService store 补偿 delete 失败: %s", e)

            steps.append(
                SagaStep(name="store_add", action=_store_add, compensate=_compensate_store)
            )

        # 2. 三级索引写入
        if self.deps.index is not None:
            index = self.deps.index

            def _index_add() -> None:
                index.add(
                    memory_id=memory_id,
                    wing=wing,
                    hall=hall,
                    room=room,
                    content=content,
                    summary=summary,
                    type=memory_type,
                    confidence=confidence,
                    privacy=privacy,
                    scope=scope,
                    stored_at=stored_at,
                    provenance=json.dumps(provenance, ensure_ascii=False) if provenance else "",
                )

            def _compensate_index() -> None:
                if not memory_id:
                    return
                try:
                    index.flush()
                except Exception as e:
                    logger.warning("MemoryService index 补偿 flush 失败: %s", e)
                try:
                    index.delete(memory_id)
                except Exception as e:
                    logger.warning("MemoryService index 补偿 delete 失败: %s", e)

            steps.append(
                SagaStep(name="index_add", action=_index_add, compensate=_compensate_index)
            )

        # 3. 检索器写入（向量 + BM25）
        if self.deps.retriever is not None:
            retriever = self.deps.retriever
            retriever_content = _enrich_retriever_content(content, memory_type, room)
            retriever_metadata: dict[str, Any] = {
                "memory_id": memory_id,
                "type": memory_type,
                "confidence": confidence,
                "scope": scope,
                "privacy": privacy,
                "wing": wing,
                "room": room,
                "stored_at": stored_at,
                "entities": entities or [],
            }

            def _retriever_add() -> None:
                retriever.add(
                    retriever_content,
                    memory_id=memory_id,
                    metadata=retriever_metadata,
                )

            def _compensate_retriever() -> None:
                if not memory_id:
                    return
                try:
                    retriever.delete(memory_id)
                except Exception as e:
                    logger.warning("MemoryService retriever 补偿 delete 失败: %s", e)
                try:
                    retriever.flush()
                except Exception as e:
                    logger.warning("MemoryService retriever 补偿 flush 失败: %s", e)

            steps.append(
                SagaStep(
                    name="retriever_add",
                    action=_retriever_add,
                    compensate=_compensate_retriever,
                )
            )

        # 4. 知识图谱三元组抽取
        if self.deps.knowledge_graph is not None:
            kg = self.deps.knowledge_graph
            kg_confidence = confidence / 5.0

            def _kg_extract() -> Any:
                return kg.extract_and_store(
                    content,
                    memory_id=memory_id,
                    confidence=kg_confidence,
                )

            def _compensate_kg() -> None:
                if not memory_id:
                    return
                self._delete_kg_triples(memory_id)

            steps.append(
                SagaStep(name="kg_extract", action=_kg_extract, compensate=_compensate_kg)
            )

        # 5. 时序知识图谱同步
        if self.deps.knowledge_graph is not None and self.deps.temporal_kg is not None:
            kg = self.deps.knowledge_graph
            tkg = self.deps.temporal_kg

            def _temporal_kg_extract() -> int:
                triples = kg._get_all_triples(limit=5000)
                synced = 0
                for triple in triples:
                    if triple.get("source_memory_id") != memory_id:
                        continue
                    tkg.add_triple_from_kg(
                        subject=triple["subject"],
                        predicate=triple["predicate"],
                        obj=triple["object"],
                        valid_at=stored_at,
                        source_memory_id=memory_id,
                        confidence=int(triple.get("confidence", 3)),
                    )
                    synced += 1
                return synced

            def _compensate_temporal_kg() -> None:
                if not memory_id:
                    return
                self._delete_temporal_triples(memory_id)

            steps.append(
                SagaStep(
                    name="temporal_kg_extract",
                    action=_temporal_kg_extract,
                    compensate=_compensate_temporal_kg,
                )
            )

        coordinator = self.deps.saga
        if coordinator is None:
            coordinator = SagaCoordinator()

        saga_result = coordinator.execute(memory_id or "pending", steps)
        return memory_id, saga_result

    # ------------------------------------------------------------------
    # 补偿辅助
    # ------------------------------------------------------------------
    def _delete_kg_triples(self, memory_id: str) -> None:
        """回滚 KnowledgeGraph 中指定记忆来源的三元组。"""
        if self.deps.knowledge_graph is None or not memory_id:
            return
        kg = self.deps.knowledge_graph
        try:
            if hasattr(kg, "delete_by_memory_id"):
                kg.delete_by_memory_id(memory_id)
                return
            # 兼容回退：直接操作底层连接
            conn = getattr(kg, "_conn", None)
            if conn is not None:
                conn.execute(
                    "DELETE FROM triples WHERE source_memory_id = ? OR source_memory_id LIKE ?",
                    (memory_id, f"inferred-from:{memory_id}"),
                )
                conn.commit()
        except Exception as e:
            logger.warning("MemoryService KG 补偿失败: %s", e)

    def _delete_temporal_triples(self, memory_id: str) -> None:
        """回滚 TemporalKG 中指定记忆来源的时序三元组。"""
        if self.deps.temporal_kg is None or not memory_id:
            return
        tkg = self.deps.temporal_kg
        try:
            if hasattr(tkg, "delete_by_memory_id"):
                tkg.delete_by_memory_id(memory_id)
                return
            conn = getattr(tkg, "_conn", None)
            if conn is not None:
                conn.execute(
                    "DELETE FROM temporal_triples WHERE source_memory_id = ?",
                    (memory_id,),
                )
                conn.commit()
        except Exception as e:
            logger.warning("MemoryService TemporalKG 补偿失败: %s", e)
