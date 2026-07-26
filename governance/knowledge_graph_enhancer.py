"""
KnowledgeGraphEnhancer — 知识图谱增强模块。

自动发现记忆间的关联：
1. 基于语义相似度
2. 基于共同访问模式
3. 基于内容引用
4. 动态更新图结构

核心算法:
- 语义相似度: cosine_similarity(embedding_i, embedding_j)
- 共同访问: 共同被访问的记忆对
- 内容引用: 记忆内容中引用其他记忆
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Relationship:
    """记忆关系"""
    source_id: str
    target_id: str
    relation_type: str      # semantic, co_access, content_ref
    strength: float         # 关系强度 (0-1)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GraphStats:
    """图统计信息"""
    total_nodes: int
    total_edges: int
    avg_degree: float
    connected_components: int
    densest_cluster: str | None = None


class KnowledgeGraphEnhancer:
    """知识图谱增强器

    提供:
    - 关系发现
    - 图结构更新
    - 图统计分析
    - 可视化支持
    """

    def __init__(
        self,
        db_path: str | None = None,
        embedding_path: str | None = None,
    ):
        self._db_path = db_path or os.path.expanduser(
            "~/.hermes/omnimem/deep/knowledge_graph.db"
        )
        self._embedding_path = embedding_path or os.path.expanduser(
            "~/.hermes/omnimem/retrieval/embedding_cache.json"
        )

        self._embeddings: dict[str, list[float]] = {}
        self._relationships: list[Relationship] = []
        self._loaded = False

        # 初始化数据库
        self._init_db()

    def _init_db(self) -> None:
        """初始化知识图谱数据库"""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                strength REAL NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(source_id, target_id, relation_type)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id)
        """)
        conn.commit()
        conn.close()

    def _load_data(self) -> None:
        """加载数据"""
        if self._loaded:
            return

        # 加载嵌入向量
        try:
            # ★ P2: 经共享工具读取，兼容 SQLite 新格式与 JSON 旧格式
            from omnimem.retrieval.vector_store import load_embedding_cache_dict

            self._embeddings = load_embedding_cache_dict(self._embedding_path)
            if self._embeddings:
                logger.info("Loaded %d embeddings", len(self._embeddings))
        except Exception as e:
            logger.warning("Failed to load embeddings: %s", e)

        # 加载已有关系
        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                "SELECT source_id, target_id, relation_type, strength, created_at FROM relationships"
            ).fetchall()
            for src, tgt, rel_type, strength, created_at in rows:
                self._relationships.append(Relationship(
                    source_id=src,
                    target_id=tgt,
                    relation_type=rel_type,
                    strength=strength,
                    created_at=datetime.fromisoformat(created_at),
                ))
            conn.close()
            logger.info("Loaded %d relationships", len(self._relationships))
        except Exception as e:
            logger.warning("Failed to load relationships: %s", e)

        self._loaded = True

    def discover_semantic_relationships(
        self,
        threshold: float = 0.7,
        max_relationships: int = 1000,
    ) -> list[Relationship]:
        """发现语义相关的关系

        Args:
            threshold: 相似度阈值
            max_relationships: 最大关系数

        Returns:
            新发现的关系列表
        """
        self._load_data()

        if len(self._embeddings) < 2:
            return []

        new_relationships = []
        memory_ids = list(self._embeddings.keys())

        # 计算所有记忆对的相似度
        similarities = []
        for i in range(len(memory_ids)):
            for j in range(i + 1, len(memory_ids)):
                id_i = memory_ids[i]
                id_j = memory_ids[j]

                vec_i = self._embeddings[id_i]
                vec_j = self._embeddings[id_j]

                if vec_i and vec_j:
                    sim = self._cosine_similarity(vec_i, vec_j)
                    if sim >= threshold:
                        similarities.append((id_i, id_j, sim))

        # 按相似度排序，取 top N
        similarities.sort(key=lambda x: x[2], reverse=True)
        similarities = similarities[:max_relationships]

        # 创建关系
        for id_i, id_j, sim in similarities:
            rel = Relationship(
                source_id=id_i,
                target_id=id_j,
                relation_type="semantic",
                strength=sim,
            )
            new_relationships.append(rel)

        logger.info("Discovered %d semantic relationships", len(new_relationships))
        return new_relationships

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

    def discover_co_access_relationships(
        self,
        access_log_path: str | None = None,
        threshold: int = 3,
    ) -> list[Relationship]:
        """发现共同访问的关系

        Args:
            access_log_path: 访问日志路径
            threshold: 共同访问次数阈值

        Returns:
            新发现的关系列表
        """
        # 从遗忘曲线数据库获取访问日志
        forgetting_db = os.path.expanduser(
            "~/.hermes/omnimem/governance/forgetting.db"
        )

        if not os.path.exists(forgetting_db):
            return []

        try:
            conn = sqlite3.connect(forgetting_db)

            # 获取每个记忆的访问时间
            rows = conn.execute(
                "SELECT memory_id, accessed_at FROM access_log ORDER BY accessed_at"
            ).fetchall()

            # 按时间窗口分组（同一小时内访问的记忆）
            from collections import defaultdict
            time_windows: dict[str, list[str]] = defaultdict(list)

            for memory_id, accessed_at in rows:
                # 使用小时作为时间窗口
                hour = accessed_at[:13]  # YYYY-MM-DDTHH
                time_windows[hour].append(memory_id)

            # 统计共同访问次数
            co_access_count: dict[tuple[str, str], int] = defaultdict(int)
            for hour, memories in time_windows.items():
                unique_memories = list(set(memories))
                for i in range(len(unique_memories)):
                    for j in range(i + 1, len(unique_memories)):
                        pair = tuple(sorted([unique_memories[i], unique_memories[j]]))
                        co_access_count[pair] += 1

            conn.close()

            # 创建关系
            new_relationships = []
            for (id_i, id_j), count in co_access_count.items():
                if count >= threshold:
                    strength = min(1.0, count / 10.0)  # 归一化
                    rel = Relationship(
                        source_id=id_i,
                        target_id=id_j,
                        relation_type="co_access",
                        strength=strength,
                    )
                    new_relationships.append(rel)

            logger.info("Discovered %d co-access relationships", len(new_relationships))
            return new_relationships

        except Exception as e:
            logger.warning("Failed to discover co-access relationships: %s", e)
            return []

    def save_relationships(self, relationships: list[Relationship]) -> int:
        """保存关系到数据库

        Args:
            relationships: 关系列表

        Returns:
            保存的关系数量
        """
        conn = sqlite3.connect(self._db_path)
        saved = 0

        for rel in relationships:
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO relationships
                       (source_id, target_id, relation_type, strength, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (rel.source_id, rel.target_id, rel.relation_type,
                     rel.strength, rel.created_at.isoformat())
                )
                saved += 1
            except Exception as e:
                logger.warning("Failed to save relationship: %s", e)

        conn.commit()
        conn.close()

        logger.info("Saved %d relationships", saved)
        return saved

    def get_relationships(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆的关系

        Args:
            memory_id: 记忆 ID

        Returns:
            关系列表
        """
        conn = sqlite3.connect(self._db_path)

        # 作为源
        rows = conn.execute(
            """SELECT target_id, relation_type, strength
               FROM relationships WHERE source_id = ?""",
            (memory_id,)
        ).fetchall()

        # 作为目标
        rows += conn.execute(
            """SELECT source_id, relation_type, strength
               FROM relationships WHERE target_id = ?""",
            (memory_id,)
        ).fetchall()

        conn.close()

        return [
            {"related_id": r[0], "type": r[1], "strength": r[2]}
            for r in rows
        ]

    def get_graph_stats(self) -> GraphStats:
        """获取图统计信息"""
        conn = sqlite3.connect(self._db_path)

        # 节点数
        nodes = conn.execute(
            "SELECT DISTINCT source_id FROM relationships UNION SELECT DISTINCT target_id FROM relationships"
        ).fetchall()

        # 边数
        edges = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]

        conn.close()

        # 计算平均度
        avg_degree = (edges * 2) / len(nodes) if nodes else 0

        return GraphStats(
            total_nodes=len(nodes),
            total_edges=edges,
            avg_degree=avg_degree,
            connected_components=1,  # 简化实现
        )


# 全局实例
_enhancer: KnowledgeGraphEnhancer | None = None


def get_enhancer(
    db_path: str | None = None,
    embedding_path: str | None = None,
) -> KnowledgeGraphEnhancer:
    """获取全局增强器实例"""
    global _enhancer
    if _enhancer is None:
        _enhancer = KnowledgeGraphEnhancer(db_path, embedding_path)
    return _enhancer


def discover_and_save_relationships(threshold: float = 0.7) -> dict[str, Any]:
    """便捷函数：发现并保存关系"""
    enhancer = get_enhancer()

    # 发现语义关系
    semantic_rels = enhancer.discover_semantic_relationships(threshold)

    # 发现共同访问关系
    co_access_rels = enhancer.discover_co_access_relationships()

    # 保存
    all_rels = semantic_rels + co_access_rels
    saved = enhancer.save_relationships(all_rels)

    return {
        "semantic": len(semantic_rels),
        "co_access": len(co_access_rels),
        "total_saved": saved,
    }
