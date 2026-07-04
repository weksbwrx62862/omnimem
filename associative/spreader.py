"""AssociativeSpreader — 联想扩散引擎。

从精确检索结果出发，沿 KG 关系和语义空间做多跳扩散，
返回"相关但不完全匹配"的联想内容，模拟人类记忆的联想检索机制。

工作原理：
  1. 从查询中提取实体
  2. KG 扩散：沿知识图谱三元组扩展 1-2 跳（如 "RLHF → PPO → reward_model"）
  3. 语义扩散：从实体 embedding 在向量空间找邻居
  4. 所有联想结果标注来源和置信度，与精确结果分离输出

使用（在 recall.py 中集成）:
  spreader = AssociativeSpreader(kg=kg, retriever=retriever)
  assocs = spreader.spread(query, existing_ids)
  results.extend(assocs)  # 联想结果自带 _source: "association"
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 联想结果的默认置信度与跳数衰减系数
_KG_CONFIDENCE_DECAY: float = 0.6  # 每跳置信度乘以此系数
_SEMANTIC_SCALE_FACTOR: float = 0.6  # 语义扩散结果的分数打折系数
_BASE_ASSOC_SCORE: float = 0.35  # 联想结果的基础分数（通过后续过滤线）
_MAX_SEMANTIC_PER_ENTITY: int = 3  # 每个实体最多返回的语义邻居数


class AssociativeSpreader:
    """联想扩散引擎。

    Args:
        knowledge_graph: KnowledgeGraph 实例（用于 KG 多跳扩散）
        retriever: HybridRetriever 实例（用于向量语义扩散）
        max_depth: KG 扩散的最大跳数（默认 2）
        max_branches: 每跳最多扩展的实体数（默认 5）
    """

    def __init__(
        self,
        knowledge_graph: Any = None,
        retriever: Any = None,
        max_depth: int = 2,
        max_branches: int = 5,
    ) -> None:
        self._kg = knowledge_graph
        self._retriever = retriever
        self._max_depth = max_depth
        self._max_branches = max_branches

    def spread(
        self,
        query: str,
        existing_ids: set[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """执行联想扩散。

        Args:
            query: 用户查询
            existing_ids: 已存在的 memory_id 集合（用于去重）
            top_k: 最大返回联想结果数

        Returns:
            联想结果列表，每项包含：
              - content: 联想结果内容
              - _source: "association"
              - _assoc_type: "kg_spread" | "semantic_spread"
              - confidence: [0, 1] 置信度
              - score: 搜索分数（用于排序和过滤）
              - _spread_depth: 扩散跳数（KG 结果）
        """
        results: list[dict[str, Any]] = []
        seen = set(existing_ids) if existing_ids else set()
        # 内容去重（避免 KG 和语义扩散返回相同内容）
        seen_content: set[str] = set()

        # 1. 提取查询实体
        entities = self._extract_entities(query)
        if not entities:
            logger.debug("AssociativeSpreader: no entities, skipping")
            return results

        # 2. KG 扩散
        if self._kg is not None:
            try:
                kg_results = self._spread_kg(entities, seen, seen_content, top_k)
                results.extend(kg_results)
            except Exception as e:
                logger.warning("KG spread failed (non-fatal): %s", e)

        # 3. 语义扩散
        if self._retriever is not None:
            try:
                sem_results = self._spread_semantic(
                    entities, seen, seen_content, top_k
                )
                results.extend(sem_results)
            except Exception as e:
                logger.warning("Semantic spread failed (non-fatal): %s", e)

        # 按置信度降序排列
        results.sort(key=lambda x: -x.get("confidence", 0.0))
        return results[:top_k]

    # ── 内部方法 ──────────────────────────────────────────────

    def _extract_entities(self, text: str) -> list[str]:
        """从文本中提取实体，复用 KG 的 extract_entities。"""
        from omnimem.deep.kg import extract_entities

        return extract_entities(text)

    def _spread_kg(
        self,
        entities: list[str],
        seen_ids: set[str],
        seen_content: set[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """KG 多跳扩散：从实体出发，沿三元组扩展。

        对每个种子实体，递归查询其邻居（get_neighbors），
        每跳深度 +1，置信度乘 _KG_CONFIDENCE_DECAY。
        返回可读的三元组文本：subj predicate obj。
        """
        results: list[dict[str, Any]] = []
        visited_entities: set[str] = set()

        # BFS 队列: (entity, depth)
        queue: list[tuple[str, int]] = [(e, 0) for e in entities]

        while queue:
            if len(results) >= top_k * 2:  # 提前截断
                break
            entity, depth = queue.pop(0)
            if entity in visited_entities or depth > self._max_depth:
                continue
            visited_entities.add(entity)

            try:
                neighbors = self._kg.get_neighbors(entity, depth=1)
            except Exception as e:
                logger.debug("get_neighbors('%s') failed: %s", entity, e)
                continue

            for triple in neighbors:
                subj = triple.get("subject", "")
                obj = triple.get("object", "")
                predicate = triple.get("predicate", "")

                if not subj and not obj:
                    continue

                # 构建可读三元组文本
                content = f"{subj} {predicate} {obj}"
                content_lower = content.lower()
                if content_lower in seen_content:
                    continue
                seen_content.add(content_lower)

                # 权重随跳数衰减
                base_conf = triple.get("confidence", 0.5)
                if isinstance(base_conf, (int, float)):
                    confidence = float(base_conf) * (_KG_CONFIDENCE_DECAY**depth)
                else:
                    confidence = 0.5 * (_KG_CONFIDENCE_DECAY**depth)

                # 添加联想结果（无 memory_id，后续不经过 store 验证）
                result: dict[str, Any] = {
                    "content": content,
                    "type": "association",
                    "_source": "association",
                    "_assoc_type": "kg_spread",
                    "confidence": round(confidence, 4),
                    "score": round(max(confidence * 0.8, _BASE_ASSOC_SCORE), 4),
                    "_spread_depth": depth,
                    "_spread_entity": entity,
                }
                results.append(result)

                # 把新实体加入扩散队列（继续下一跳）
                if depth + 1 <= self._max_depth:
                    next_entity = obj if subj == entity else subj
                    if next_entity and next_entity not in visited_entities:
                        queue.append((next_entity, depth + 1))

        return results

    def _spread_semantic(
        self,
        entities: list[str],
        seen_ids: set[str],
        seen_content: set[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """语义空间扩散：从实体 embedding 在向量空间找邻居。

        对每个提取到的实体，在其 embedding 空间中做 KNN（3个），
        找到语义相近的存储记忆，但降权返回。
        """
        results: list[dict[str, Any]] = []

        for entity in entities:
            if len(results) >= top_k:
                break
            try:
                semantic_results = self._retriever.vector_search(
                    entity, top_k=_MAX_SEMANTIC_PER_ENTITY
                )
            except Exception as e:
                logger.debug("vector_search for '%s' failed: %s", entity, e)
                continue

            for r in semantic_results:
                mid = r.get("memory_id", "")
                if mid and mid in seen_ids:
                    continue
                if mid:
                    seen_ids.add(mid)

                content = r.get("content", "")
                content_lower = content.lower()
                if content_lower in seen_content:
                    continue
                seen_content.add(content_lower)

                # 语义扩散结果降权
                original_score = r.get("score", 0.0) or 0.0
                r["_source"] = "association"
                r["_assoc_type"] = "semantic_spread"
                r["_spread_entity"] = entity
                r["score"] = round(
                    max(original_score * _SEMANTIC_SCALE_FACTOR, _BASE_ASSOC_SCORE), 4
                )
                r["confidence"] = round(
                    max(original_score * _SEMANTIC_SCALE_FACTOR, 0.3), 4
                )
                # 标记为联想类型，让下游能区分
                r.setdefault("type", "association")
                results.append(r)

        return results

    def __repr__(self) -> str:
        return (
            f"AssociativeSpreader(kg={'✓' if self._kg else '✗'}, "
            f"retriever={'✓' if self._retriever else '✗'}, "
            f"max_depth={self._max_depth})"
        )
