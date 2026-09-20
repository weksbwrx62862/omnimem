"""可插拔图存储协议（改进项 #6）。

镜像 governance.temporal_kg.TemporalKnowledgeGraph 的公开方法面，作为时序知识
图谱后端的结构类型契约：SQLite（默认，委托现有实现）/ Neo4j / FalkorDB（Cypher）
实现同一协议，便于灰度切换而不改动调用方。

返回类型复用 TemporalTriple，保证跨后端语义一致。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from omnimem.governance.temporal_kg import TemporalTriple

__all__ = ["GraphStoreProtocol", "TemporalTriple", "GraphTriple"]

# 语义别名：外部（含 Neo4j）后端同样返回 TemporalTriple。
GraphTriple = TemporalTriple


@runtime_checkable
class GraphStoreProtocol(Protocol):
    """时序知识图谱后端契约（与 TemporalKnowledgeGraph 方法签名对齐）。"""

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_at: str,
        source_memory_id: str = "",
        confidence: int = 3,
    ) -> str: ...

    def invalidate_triple(
        self,
        triple_id: str,
        invalid_at: str,
        superseded_by: str | None = None,
    ) -> None: ...

    def delete_by_memory_id(self, memory_id: str) -> int: ...

    def query_current(self, subject: str, predicate: str) -> list[TemporalTriple]: ...

    def query_at_time(self, subject: str, predicate: str, at_time: str) -> list[TemporalTriple]: ...

    def get_timeline(self, subject: str, limit: int = 50) -> list[TemporalTriple]: ...

    def temporal_search(
        self,
        query_entities: list[str],
        at_time: str | None = None,
        limit: int = 20,
    ) -> list[TemporalTriple]: ...

    def get_stats(self) -> dict[str, Any]: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...
