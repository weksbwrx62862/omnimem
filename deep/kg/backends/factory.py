"""图存储后端工厂（改进项 #6）。

对齐 retrieval.vector_factory 的风格：按 backend 名返回 GraphStoreProtocol 实现。
默认 sqlite（委托现有 TemporalKnowledgeGraph，零迁移），neo4j/falkordb 走 Cypher 后端。
"""
from __future__ import annotations

import logging
from typing import Any

from omnimem.deep.kg.backends.protocol import GraphStoreProtocol

logger = logging.getLogger(__name__)

SUPPORTED_GRAPH_BACKENDS = ("sqlite", "neo4j", "falkordb")


def create_graph_store(backend: str = "sqlite", **kwargs: Any) -> GraphStoreProtocol:
    """构造图存储后端。

    Args:
        backend: sqlite | neo4j | falkordb
        kwargs: 透传给具体后端（如 data_dir / uri / auth / driver / database）
    """
    if backend == "sqlite":
        from omnimem.deep.kg.backends.sqlite_backend import SQLiteGraphStore

        return SQLiteGraphStore(**kwargs)
    if backend in ("neo4j", "falkordb"):
        from omnimem.deep.kg.backends.neo4j_backend import Neo4jGraphStore

        # FalkorDB 兼容 Bolt/Cypher，复用同一实现；调用方传 uri/auth 指向其端点。
        return Neo4jGraphStore(**kwargs)
    raise ValueError(f"Unknown graph store backend: {backend} (可选: {', '.join(SUPPORTED_GRAPH_BACKENDS)})")
