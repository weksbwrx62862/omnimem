"""SQLite 图后端（改进项 #6，参考实现）。

不重造存储：直接委托现有 TemporalKnowledgeGraph，仅套上 GraphStoreProtocol 契约，
用于验证协议形状与默认零迁移切换。方法均为薄转发。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omnimem.governance.temporal_kg import TemporalKnowledgeGraph, TemporalTriple


class SQLiteGraphStore:
    """委托现有 SQLite 时序图谱的协议实现。"""

    def __init__(self, data_dir: str | Path, config: Any = None) -> None:
        self._inner = TemporalKnowledgeGraph(Path(data_dir), config=config)

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_at: str,
        source_memory_id: str = "",
        confidence: int = 3,
    ) -> str:
        return self._inner.add_triple(
            subject, predicate, obj, valid_at, source_memory_id=source_memory_id, confidence=confidence
        )

    def invalidate_triple(
        self, triple_id: str, invalid_at: str, superseded_by: str | None = None
    ) -> None:
        self._inner.invalidate_triple(triple_id, invalid_at, superseded_by=superseded_by)

    def delete_by_memory_id(self, memory_id: str) -> int:
        return self._inner.delete_by_memory_id(memory_id)

    def query_current(self, subject: str, predicate: str) -> list[TemporalTriple]:
        return self._inner.query_current(subject, predicate)

    def query_at_time(self, subject: str, predicate: str, at_time: str) -> list[TemporalTriple]:
        return self._inner.query_at_time(subject, predicate, at_time)

    def get_timeline(self, subject: str, limit: int = 50) -> list[TemporalTriple]:
        return self._inner.get_timeline(subject, limit=limit)

    def temporal_search(
        self, query_entities: list[str], at_time: str | None = None, limit: int = 20
    ) -> list[TemporalTriple]:
        return self._inner.temporal_search(query_entities, at_time=at_time, limit=limit)

    def get_stats(self) -> dict[str, Any]:
        return self._inner.get_stats()

    def flush(self) -> None:
        self._inner.flush()

    def close(self) -> None:
        self._inner.close()
