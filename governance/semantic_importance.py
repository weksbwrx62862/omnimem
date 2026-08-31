"""
SemanticImportance — 语义重要性评估模块。

基于向量聚类计算记忆的语义重要性：
1. 向量中心性：在语义空间中的中心位置
2. 关联密度：与其他记忆的连接数量
3. 图结构重要性：在知识图谱中的位置

核心算法:
- 向量聚类中心性: centrality = mean(cosine_similarity(memory, cluster_center))
- 关联密度: density = connections / max_possible_connections
- 综合重要性: importance = w1 * centrality + w2 * density + w3 * graph_score
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SemanticFeatures:
    """语义特征向量"""

    vector_centrality: float = 0.5  # 向量中心性 (0-1)
    connection_density: float = 0.5  # 关联密度 (0-1)
    graph_importance: float = 0.5  # 图结构重要性 (0-1)
    content_richness: float = 0.5  # 内容丰富度 (0-1)
    uniqueness: float = 0.5  # 独特性 (0-1)

    def to_dict(self) -> dict[str, float]:
        return {
            "vector_centrality": self.vector_centrality,
            "connection_density": self.connection_density,
            "graph_importance": self.graph_importance,
            "content_richness": self.content_richness,
            "uniqueness": self.uniqueness,
        }


@dataclass
class SemanticWeights:
    """语义重要性权重"""

    centrality: float = 0.30
    density: float = 0.25
    graph: float = 0.20
    richness: float = 0.15
    uniqueness: float = 0.10


class SemanticImportanceEvaluator:
    """语义重要性评估器

    提供:
    - 向量中心性计算
    - 关联密度分析
    - 图结构重要性评估
    - 综合语义重要性评分
    """

    def __init__(
        self,
        db_path: str | None = None,
        embedding_path: str | None = None,
        weights: SemanticWeights | None = None,
    ):
        self._db_path = db_path or os.path.expanduser("~/.hermes/omnimem/index/index.db")
        self._embedding_path = embedding_path or os.path.expanduser(
            "~/.hermes/omnimem/retrieval/embedding_cache.json"
        )
        self._weights = weights or SemanticWeights()
        self._embeddings: dict[str, list[float]] = {}
        self._connections: dict[str, set[str]] = {}
        self._loaded = False

    def _load_data(self) -> None:
        """加载嵌入向量和连接关系"""
        if self._loaded:
            return

        # 加载嵌入向量
        try:
            if os.path.exists(self._embedding_path):
                import json

                with open(self._embedding_path) as f:
                    self._embeddings = json.load(f)
                logger.info("Loaded %d embeddings", len(self._embeddings))
        except Exception as e:
            logger.warning("Failed to load embeddings: %s", e)

        # 加载连接关系（从知识图谱）
        try:
            kg_path = os.path.expanduser("~/.hermes/omnimem/deep/knowledge_graph.db")
            if os.path.exists(kg_path):
                conn = sqlite3.connect(kg_path)
                rows = conn.execute("SELECT source_id, target_id FROM relationships").fetchall()
                for src, tgt in rows:
                    if src not in self._connections:
                        self._connections[src] = set()
                    if tgt not in self._connections:
                        self._connections[tgt] = set()
                    self._connections[src].add(tgt)
                    self._connections[tgt].add(src)
                conn.close()
                logger.info("Loaded %d connections", len(self._connections))
        except Exception as e:
            logger.warning("Failed to load connections: %s", e)

        self._loaded = True

    def calculate_vector_centrality(self, memory_id: str) -> float:
        """计算向量中心性

        中心性 = mean(cosine_similarity(memory, all_other_memories))

        Args:
            memory_id: 记忆 ID

        Returns:
            中心性分数 (0-1)
        """
        self._load_data()

        if memory_id not in self._embeddings:
            return 0.5

        target_vec = self._embeddings[memory_id]
        if not target_vec:
            return 0.5

        similarities = []
        for other_id, other_vec in self._embeddings.items():
            if other_id != memory_id and other_vec:
                sim = self._cosine_similarity(target_vec, other_vec)
                similarities.append(sim)

        if not similarities:
            return 0.5

        # 中心性 = 平均相似度
        return sum(similarities) / len(similarities)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2) or not vec1:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def calculate_connection_density(self, memory_id: str) -> float:
        """计算关联密度

        density = connections / max_possible_connections

        Args:
            memory_id: 记忆 ID

        Returns:
            关联密度 (0-1)
        """
        self._load_data()

        connections = self._connections.get(memory_id, set())
        if not connections:
            return 0.0

        # 最大可能连接数 = 总记忆数 - 1
        max_connections = len(self._embeddings) - 1 if self._embeddings else 1

        return len(connections) / max_connections if max_connections > 0 else 0.0

    def calculate_graph_importance(self, memory_id: str) -> float:
        """计算图结构重要性

        使用 PageRank 简化版：
        - 连接数越多，重要性越高
        - 连接到重要节点，重要性更高

        Args:
            memory_id: 记忆 ID

        Returns:
            图结构重要性 (0-1)
        """
        self._load_data()

        connections = self._connections.get(memory_id, set())
        if not connections:
            return 0.0

        # 基础分数：连接数
        base_score = min(1.0, len(connections) / 10.0)

        # 连接到重要节点的加分
        important_connections = 0
        for conn_id in connections:
            conn_connections = len(self._connections.get(conn_id, set()))
            if conn_connections >= 3:  # 连接到有 3+ 连接的节点
                important_connections += 1

        importance_bonus = min(0.5, important_connections * 0.1)

        return min(1.0, base_score + importance_bonus)

    def calculate_content_richness(self, content: str | None = None) -> float:
        """计算内容丰富度

        基于:
        - 内容长度
        - 词汇多样性
        - 结构复杂度

        Args:
            content: 记忆内容

        Returns:
            内容丰富度 (0-1)
        """
        if not content:
            return 0.5

        # 长度分数 (对数压缩)
        length_score = min(1.0, math.log(len(content) + 1) / 10.0)

        # 词汇多样性
        words = content.split()
        unique_words = set(words)
        diversity = len(unique_words) / len(words) if words else 0.0

        # 结构复杂度 (标点符号和换行)
        structure_chars = sum(1 for c in content if c in ".,;:!?\\n")
        structure_score = min(1.0, structure_chars / 20.0)

        # 综合
        return length_score * 0.4 + diversity * 0.3 + structure_score * 0.3

    def calculate_uniqueness(self, memory_id: str, content: str | None = None) -> float:
        """计算独特性

        基于:
        - 与其他记忆的相似度 (越低越独特)
        - 内容特征的独特性

        Args:
            memory_id: 记忆 ID
            content: 记忆内容

        Returns:
            独特性 (0-1)
        """
        self._load_data()

        if memory_id not in self._embeddings:
            return 0.5

        target_vec = self._embeddings[memory_id]
        if not target_vec:
            return 0.5

        # 计算与其他记忆的相似度
        similarities = []
        for other_id, other_vec in self._embeddings.items():
            if other_id != memory_id and other_vec:
                sim = self._cosine_similarity(target_vec, other_vec)
                similarities.append(sim)

        if not similarities:
            return 0.5

        # 独特性 = 1 - 平均相似度
        avg_similarity = sum(similarities) / len(similarities)
        uniqueness = 1.0 - avg_similarity

        return max(0.0, min(1.0, uniqueness))

    def evaluate(
        self,
        memory_id: str,
        content: str | None = None,
    ) -> SemanticFeatures:
        """评估记忆的语义重要性

        Args:
            memory_id: 记忆 ID
            content: 记忆内容

        Returns:
            语义特征向量
        """
        return SemanticFeatures(
            vector_centrality=self.calculate_vector_centrality(memory_id),
            connection_density=self.calculate_connection_density(memory_id),
            graph_importance=self.calculate_graph_importance(memory_id),
            content_richness=self.calculate_content_richness(content),
            uniqueness=self.calculate_uniqueness(memory_id, content),
        )

    def calculate_importance(self, features: SemanticFeatures) -> float:
        """计算综合语义重要性

        Args:
            features: 语义特征向量

        Returns:
            综合重要性 (0-1)
        """
        w = self._weights

        importance = (
            w.centrality * features.vector_centrality
            + w.density * features.connection_density
            + w.graph * features.graph_importance
            + w.richness * features.content_richness
            + w.uniqueness * features.uniqueness
        )

        return max(0.0, min(1.0, importance))

    def evaluate_importance(
        self,
        memory_id: str,
        content: str | None = None,
    ) -> dict[str, Any]:
        """评估记忆的语义重要性

        Returns:
            包含特征向量和综合重要性的字典
        """
        features = self.evaluate(memory_id, content)
        importance = self.calculate_importance(features)

        return {
            "memory_id": memory_id,
            "features": features.to_dict(),
            "importance": importance,
        }


# 全局实例
_evaluator: SemanticImportanceEvaluator | None = None


def get_semantic_evaluator(
    db_path: str | None = None,
    embedding_path: str | None = None,
    weights: SemanticWeights | None = None,
) -> SemanticImportanceEvaluator:
    """获取全局语义评估器实例"""
    global _evaluator
    if _evaluator is None or weights is not None:
        _evaluator = SemanticImportanceEvaluator(db_path, embedding_path, weights)
    return _evaluator


def evaluate_semantic_importance(
    memory_id: str,
    content: str | None = None,
) -> dict[str, Any]:
    """便捷函数：评估语义重要性"""
    evaluator = get_semantic_evaluator()
    return evaluator.evaluate_importance(memory_id, content)
