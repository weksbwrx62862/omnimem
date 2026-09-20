"""Neo4j / FalkorDB 图后端（改进项 #6）。

用 Cypher（Bolt）实现 GraphStoreProtocol，把时序三元组落成
`(:Entity)-[r:谓词 {id, valid_at, invalid_at, superseded_by, source_memory_id,
confidence, created_at}]->(:Entity)`，其中 valid_at 为世界有效时间、
created_at 为事务/断言时间，天然支持双时序（bi-temporal）时点查询。

Neo4j 与 FalkorDB 同为 Cypher/Bolt 协议，故共用本实现（FalkorDB 走其 Bolt 兼容端点）。

安全：谓词会拼进关系类型（Cypher 不支持关系类型参数化），因此强制做标识符白名单校验，
其余值一律参数化，杜绝注入。driver 可注入以便离线单测。
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from omnimem.governance.temporal_kg import TemporalTriple

logger = logging.getLogger(__name__)

# 合法 Cypher 关系类型：字母/下划线开头，其后字母数字下划线。
_PRED_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# 一次查询返回的稳定列序，映射到 TemporalTriple。
_PROJECTION = (
    "s.name AS subject, type(r) AS predicate, o.name AS object, "
    "r.id AS id, r.valid_at AS valid_at, r.invalid_at AS invalid_at, "
    "r.superseded_by AS superseded_by, r.source_memory_id AS source_memory_id, "
    "r.confidence AS confidence, r.created_at AS created_at"
)


def _new_id() -> str:
    return "t_" + uuid.uuid4().hex[:12]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Neo4jGraphStore:
    """基于官方 neo4j Python 驱动的时序图谱后端。"""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: tuple[str, str] | None = None,
        *,
        driver: Any = None,
        database: str = "neo4j",
    ) -> None:
        if driver is None:
            driver = self._connect(uri, auth)
        self._driver = driver
        self._database = database

    @staticmethod
    def _connect(uri: str, auth: tuple[str, str] | None) -> Any:
        try:
            from neo4j import GraphDatabase  # 可选依赖
        except ImportError as e:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "Neo4jGraphStore 需要 neo4j 驱动：pip install 'omnimem[graph]'（或 neo4j）"
            ) from e
        return GraphDatabase.driver(uri, auth=auth)

    # ── 会话/执行辅助 ────────────────────────────────
    def _session(self) -> Any:
        # driver.session(database=...) 在 neo4j 5.x；FalkorDB/旧驱动回退无参。
        try:
            return self._driver.session(database=self._database)
        except TypeError:  # pragma: no cover
            return self._driver.session()

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self._session() as sess:
            return [dict(rec) for rec in sess.run(query, **params)]

    @staticmethod
    def _check_predicate(predicate: str) -> str:
        if not _PRED_RE.match(predicate or ""):
            raise ValueError(f"非法关系谓词（仅允许标识符字符）: {predicate!r}")
        return predicate

    @staticmethod
    def _to_triple(rec: dict[str, Any]) -> TemporalTriple:
        return TemporalTriple(
            id=rec.get("id") or "",
            subject=rec.get("subject") or "",
            predicate=rec.get("predicate") or "",
            object=rec.get("object") or "",
            valid_at=rec.get("valid_at") or "",
            invalid_at=rec.get("invalid_at"),
            superseded_by=rec.get("superseded_by"),
            source_memory_id=rec.get("source_memory_id") or "",
            confidence=rec.get("confidence") or 3,
            created_at=rec.get("created_at") or "",
        )

    # ── GraphStoreProtocol ───────────────────────────
    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_at: str,
        source_memory_id: str = "",
        confidence: int = 3,
    ) -> str:
        pred = self._check_predicate(predicate)
        triple_id = _new_id()
        # 矛盾消解：同一 (subject, predicate) 当前有效且宾语不同者，标记过时。
        self._run(
            f"MATCH (s:Entity {{name: $subject}})-[old:{pred}]->(o:Entity) "
            "WHERE old.invalid_at IS NULL AND o.name <> $obj "
            "SET old.invalid_at = $valid_at, old.superseded_by = $id",
            subject=subject, obj=obj, valid_at=valid_at, id=triple_id,
        )
        self._run(
            f"MATCH (s:Entity {{name: $subject}}) MERGE (o:Entity {{name: $obj}}) "
            f"CREATE (s)-[r:{pred} {{id: $id, valid_at: $valid_at, invalid_at: null, "
            "superseded_by: null, source_memory_id: $src, confidence: $conf, "
            "created_at: $now}}]->(o)",
            subject=subject, obj=obj, id=triple_id, valid_at=valid_at,
            src=source_memory_id, conf=confidence, now=_utcnow(),
        )
        return triple_id

    def invalidate_triple(
        self, triple_id: str, invalid_at: str, superseded_by: str | None = None
    ) -> None:
        self._run(
            "MATCH ()-[r {id: $id}]->() SET r.invalid_at = $at, r.superseded_by = $sup",
            id=triple_id, at=invalid_at, sup=superseded_by,
        )

    def delete_by_memory_id(self, memory_id: str) -> int:
        rows = self._run(
            "MATCH ()-[r]->() WHERE r.source_memory_id = $id "
            "WITH collect(r) AS rs FOREACH (x IN rs | DELETE x) RETURN size(rs) AS c",
            id=memory_id,
        )
        return int(rows[0]["c"]) if rows else 0

    def query_current(self, subject: str, predicate: str) -> list[TemporalTriple]:
        clauses, params = ["r.invalid_at IS NULL"], {}
        if subject:
            clauses.append("s.name = $subject")
            params["subject"] = subject
        if predicate:
            pred = self._check_predicate(predicate)
            clauses.append(f"type(r) = '{pred}'")
        rows = self._run(
            f"MATCH (s:Entity)-[r]->(o:Entity) WHERE {' AND '.join(clauses)} "
            f"RETURN {_PROJECTION} LIMIT 500",
            **params,
        )
        return [self._to_triple(rec) for rec in rows]

    def query_at_time(self, subject: str, predicate: str, at_time: str) -> list[TemporalTriple]:
        clauses = [
            "r.valid_at <= $at",
            "(r.invalid_at IS NULL OR r.invalid_at > $at)",
        ]
        params: dict[str, Any] = {"at": at_time}
        if subject:
            clauses.append("s.name = $subject")
            params["subject"] = subject
        if predicate:
            clauses.append(f"type(r) = '{self._check_predicate(predicate)}'")
        rows = self._run(
            f"MATCH (s:Entity)-[r]->(o:Entity) WHERE {' AND '.join(clauses)} "
            f"RETURN {_PROJECTION} LIMIT 500",
            **params,
        )
        return [self._to_triple(rec) for rec in rows]

    def get_timeline(self, subject: str, limit: int = 50) -> list[TemporalTriple]:
        rows = self._run(
            f"MATCH (s:Entity {{name: $subject}})-[r]->(o:Entity) "
            f"RETURN {_PROJECTION} ORDER BY r.valid_at ASC LIMIT $limit",
            subject=subject, limit=limit,
        )
        return [self._to_triple(rec) for rec in rows]

    def temporal_search(
        self, query_entities: list[str], at_time: str | None = None, limit: int = 20
    ) -> list[TemporalTriple]:
        if not query_entities:
            return []
        if at_time:
            time_clause = "r.valid_at <= $at AND (r.invalid_at IS NULL OR r.invalid_at > $at)"
            params: dict[str, Any] = {"ents": list(query_entities), "at": at_time, "limit": limit}
        else:
            time_clause = "r.invalid_at IS NULL"
            params = {"ents": list(query_entities), "limit": limit}
        rows = self._run(
            f"MATCH (s:Entity)-[r]->(o:Entity) "
            "WHERE (s.name IN $ents OR o.name IN $ents) AND "
            f"{time_clause} RETURN {_PROJECTION} LIMIT $limit",
            **params,
        )
        return [self._to_triple(rec) for rec in rows]

    def get_stats(self) -> dict[str, Any]:
        edges = self._run("MATCH ()-[r]->() RETURN count(r) AS c")
        nodes = self._run("MATCH (n:Entity) RETURN count(n) AS c")
        current = self._run("MATCH ()-[r]->() WHERE r.invalid_at IS NULL RETURN count(r) AS c")
        return {
            "backend": "neo4j",
            "triples": int(edges[0]["c"]) if edges else 0,
            "entities": int(nodes[0]["c"]) if nodes else 0,
            "current_triples": int(current[0]["c"]) if current else 0,
        }

    def flush(self) -> None:
        # Neo4j 自动提交事务，无缓冲需要落盘。
        return None

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception as e:  # pragma: no cover
            logger.debug("Neo4jGraphStore close 忽略: %s", e)
