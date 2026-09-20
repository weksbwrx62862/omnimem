"""可插拔图存储后端包（改进项 #6）。

对外暴露工厂与协议；具体后端按需再导出，避免默认导入拉入可选 neo4j 驱动。
"""
from __future__ import annotations

from omnimem.deep.kg.backends.factory import (
    SUPPORTED_GRAPH_BACKENDS,
    create_graph_store,
)
from omnimem.deep.kg.backends.protocol import (
    GraphStoreProtocol,
    GraphTriple,
    TemporalTriple,
)

__all__ = [
    "SUPPORTED_GRAPH_BACKENDS",
    "create_graph_store",
    "GraphStoreProtocol",
    "GraphTriple",
    "TemporalTriple",
    "SQLiteGraphStore",
    "Neo4jGraphStore",
]


def __getattr__(name: str):
    # 懒加载后端，保持包导入轻量、可选项不前置。
    if name == "SQLiteGraphStore":
        from omnimem.deep.kg.backends.sqlite_backend import SQLiteGraphStore

        return SQLiteGraphStore
    if name == "Neo4jGraphStore":
        from omnimem.deep.kg.backends.neo4j_backend import Neo4jGraphStore

        return Neo4jGraphStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
