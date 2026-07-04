"""图构建、三元组提取、实体链接。

职责：
  - SQLite 知识图谱的初始化与写入
  - 实体 upsert、relationships 同步、冲突检测
  - 从记忆内容自动抽取并存储三元组

三元组抽取实现见 extraction.py，实体抽取实现见 entity.py。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnimem.deep.kg.entity import _classify_entity_poleo, extract_entities
from omnimem.deep.kg.extraction import extract_triples, infer_relations
from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)


# ─── KnowledgeGraph ────────────────────────────────────────────


class KnowledgeGraph:
    """SQLite 知识图谱，支持实体抽取、关系推理和图谱检索。"""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "knowledge_graph.db"
        self._conn: sqlite3.Connection | None = None
        self._triple_count = 0
        self._lock = threading.RLock()
        # ★ TTL 查询缓存：减少重复实体查询的 SQLite IO
        self._CACHE_TTL = 30.0
        self._query_cache: dict[str, tuple[Any, float]] = {}
        self._init_db()

    def _init_db(self) -> None:
        """初始化知识图谱数据库。"""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")

        # 三元组表（含时序有效性）
        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="triples",
            create_sql="""
                CREATE TABLE IF NOT EXISTS triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    source_memory_id TEXT,
                    confidence REAL DEFAULT 1.0,
                    is_negation INTEGER DEFAULT 0,
                    valid_from TEXT,
                    valid_to TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """,
            migrations=[],
        )
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_object ON triples(object)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predicate ON triples(predicate)
        """)

        # 实体表
        migrator.migrate(
            table_name="entities",
            create_sql="""
                CREATE TABLE IF NOT EXISTS entities (
                    name TEXT PRIMARY KEY,
                    entity_type TEXT DEFAULT 'unknown',
                    mention_count INTEGER DEFAULT 1,
                    first_seen TEXT,
                    last_seen TEXT
                )
            """,
            migrations=[],
        )

        # 关系表（从三元组派生的实体间关系，含强度）
        migrator.migrate(
            table_name="relationships",
            create_sql="""
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(source_id, target_id, relation_type)
                )
            """,
            migrations=[],
        )
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id)
        """)

        self._conn.commit()

    # ─── 三元组操作 ────────────────────────────────────────────

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source_memory_id: str = "",
        confidence: float = 1.0,
        is_negation: bool = False,
        valid_from: str = "",
        valid_to: str = "",
    ) -> int:
        """添加三元组。"""
        with self._lock:
            assert self._conn is not None
            try:
                # 冲突检测：如果已有否定关系，不再添加肯定关系
                if not is_negation:
                    existing = self._conn.execute(
                        "SELECT id FROM triples WHERE subject = ? AND predicate = ? AND object = ? AND is_negation = 1",
                        (subject, predicate, obj),
                    ).fetchone()
                    if existing:
                        logger.warning(
                            "Triple blocked by negation: %s %s %s", subject, predicate, obj
                        )
                        return -1

                now = datetime.now(timezone.utc).isoformat()
                cursor = self._conn.execute(
                    """INSERT INTO triples (subject, predicate, object, source_memory_id,
                       confidence, is_negation, valid_from, valid_to, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        subject,
                        predicate,
                        obj,
                        source_memory_id,
                        confidence,
                        1 if is_negation else 0,
                        valid_from,
                        valid_to,
                        now,
                    ),
                )
                self._conn.commit()
                self._triple_count += 1

                # 同步更新实体表
                self._upsert_entity_locked(subject)
                self._upsert_entity_locked(obj)

                # ★ 同步更新 relationships 表
                self._sync_relationship_locked(subject, obj, predicate, confidence)

                # 数据变更后清除查询缓存
                self._invalidate_cache()

                return cursor.lastrowid if cursor.lastrowid is not None else -1
            except Exception as e:
                logger.warning("Triple add failed: %s", e)
                return -1

    def add_triple_with_negation_check(
        self,
        subject: str,
        predicate: str,
        obj: str,
        content: str,
        source_memory_id: str = "",
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """添加三元组并自动检测否定关系。

        Returns:
            操作结果，包含是否有冲突
        """
        with self._lock:
            assert self._conn is not None
            # 检查内容是否包含否定
            is_negation = any(
                neg_word in content
                for neg_word in [
                    "不",
                    "并非",
                    "没有",
                    "无法",
                    "不能",
                    "不是",
                    "don't",
                    "not",
                    "no longer",
                ]
            )

            # 如果是新三元组，检查与已有三元组的否定冲突
            conflict = None
            if not is_negation:
                # 检查是否已有否定关系
                existing_neg = self._conn.execute(
                    "SELECT id, source_memory_id FROM triples WHERE subject = ? AND predicate = ? AND object = ? AND is_negation = 1",
                    (subject, predicate, obj),
                ).fetchone()
                if existing_neg:
                    conflict = {"type": "negation_exists", "triple_id": existing_neg[0]}
            else:
                # 否定关系：标记已有肯定关系为失效
                self._conn.execute(
                    "UPDATE triples SET valid_to = ? WHERE subject = ? AND predicate = ? AND object = ? AND is_negation = 0 AND valid_to = ''",
                    (datetime.now(timezone.utc).isoformat(), subject, predicate, obj),
                )
                self._conn.commit()
                self._invalidate_cache()

        triple_id = self.add_triple(
            subject,
            predicate,
            obj,
            source_memory_id=source_memory_id,
            confidence=confidence,
            is_negation=is_negation,
        )

        return {
            "triple_id": triple_id,
            "is_negation": is_negation,
            "conflict": conflict,
        }

    def delete_by_memory_id(self, memory_id: str) -> int:
        """删除指定记忆来源的所有三元组（含推理产生的三元组）。

        Args:
            memory_id: 来源记忆 ID

        Returns:
            删除的三元组数量
        """
        if not memory_id:
            return 0
        with self._lock:
            assert self._conn is not None
            try:
                cursor = self._conn.execute(
                    "DELETE FROM triples WHERE source_memory_id = ? OR source_memory_id LIKE ?",
                    (memory_id, f"inferred-from:{memory_id}"),
                )
                self._conn.commit()
                self._triple_count = max(0, self._triple_count - cursor.rowcount)
                self._invalidate_cache()
                return cursor.rowcount
            except Exception as e:
                logger.warning("KnowledgeGraph delete_by_memory_id failed: %s", e)
                return 0

    def _cached(self, key: str, fetch_fn: Callable[[], Any]) -> Any:
        """带 TTL 的查询缓存（CPython 下单个 dict 操作原子性足够）。"""
        now = time.monotonic()
        cached = self._query_cache.get(key)
        if cached:
            result, ts = cached
            if now - ts < self._CACHE_TTL:
                return result
            del self._query_cache[key]
        result = fetch_fn()
        self._query_cache[key] = (result, now)
        return result

    def _invalidate_cache(self) -> None:
        """数据变更后清除查询缓存。"""
        self._query_cache.clear()

    # ─── 从记忆中自动抽取 ─────────────────────────────────────

    def extract_and_store(
        self, content: str, memory_id: str = "", confidence: float = 0.8
    ) -> dict[str, Any]:
        """从记忆内容中抽取实体和三元组并存储。

        Returns:
            抽取统计
        """
        # 提取实体
        entities = extract_entities(content)
        for entity in entities:
            self._upsert_entity(entity)

        # 提取三元组
        raw_triples = extract_triples(content)
        stored_triples = []
        conflicts = []

        for subj, pred, obj in raw_triples:
            result = self.add_triple_with_negation_check(
                subj,
                pred,
                obj,
                content=content,
                source_memory_id=memory_id,
                confidence=confidence,
            )
            if result["triple_id"] > 0:
                stored_triples.append(result)
            if result["conflict"]:
                conflicts.append(result["conflict"])

        # ★ P1方案四：增量局部推理（替代全表扫描）
        # 对新三元组的主语和宾语做 2-hop 邻居查询，仅对局部子图推理
        inferred_stored = []
        seen_inferred: set[tuple[str, str, str]] = set()
        for subj, pred, obj in raw_triples:
            local_triples: list[dict[str, Any]] = []
            try:
                local_triples.extend(self.query_by_subject(subj))
                local_triples.extend(self.query_by_object(subj))
                local_triples.extend(self.query_by_subject(obj))
                local_triples.extend(self.query_by_object(obj))
            except Exception as e:
                logger.warning("extract_and_store local query failed: %s", e)
                continue

            inferred = infer_relations(local_triples)
            assert self._conn is not None
            for isubj, ipred, iobj in inferred:
                key = (isubj, ipred, iobj)
                if key in seen_inferred:
                    continue
                seen_inferred.add(key)
                existing = self._conn.execute(
                    "SELECT id FROM triples WHERE subject = ? AND predicate = ? AND object = ?",
                    (isubj, ipred, iobj),
                ).fetchone()
                if not existing:
                    tid = self.add_triple(
                        isubj,
                        ipred,
                        iobj,
                        source_memory_id=f"inferred-from:{memory_id}",
                        confidence=0.5,
                    )
                    if tid > 0:
                        inferred_stored.append(
                            {"subject": isubj, "predicate": ipred, "object": iobj}
                        )

        return {
            "entities_extracted": len(entities),
            "triples_extracted": len(raw_triples),
            "triples_stored": len(stored_triples),
            "conflicts_found": len(conflicts),
            "inferred_triples": len(inferred_stored),
        }

    # ─── 内部方法 ─────────────────────────────────────────────

    def _upsert_entity(self, name: str) -> None:
        """更新或插入实体（外部调用，加锁）。"""
        with self._lock:
            self._upsert_entity_locked(name)

    def _upsert_entity_locked(self, name: str) -> None:
        """更新或插入实体（内部已持有锁时调用）。"""
        try:
            assert self._conn is not None
            now = datetime.now(timezone.utc).isoformat()
            existing = self._conn.execute(
                "SELECT name FROM entities WHERE name = ?", (name,)
            ).fetchone()
            if existing:
                self._conn.execute(
                    "UPDATE entities SET mention_count = mention_count + 1, last_seen = ? WHERE name = ?",
                    (now, name),
                )
            else:
                # 推断实体类型
                entity_type = self._infer_entity_type(name)
                self._conn.execute(
                    "INSERT INTO entities (name, entity_type, mention_count, first_seen, last_seen) VALUES (?, ?, 1, ?, ?)",
                    (name, entity_type, now, now),
                )
            self._conn.commit()
        except Exception as e:
            logger.warning("Entity upsert failed: %s", e)

    def _infer_entity_type(self, name: str) -> str:
        """推断实体类型 → POLE+O 五类 (Person/Organization/Location/Event/Object)。

        统一使用 _classify_entity_poleo 规则引擎，与 extract_entities_llm 保持一致。
        """
        poleo = _classify_entity_poleo(name)
        poleo_label = {
            "person": "Person",
            "org": "Organization",
            "location": "Location",
            "event": "Event",
            "object": "Object",
        }.get(poleo, "Object")
        return poleo_label

    def _get_all_triples(self, limit: int = 5000) -> list[dict[str, Any]]:
        """获取所有有效三元组。"""
        try:
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT * FROM triples WHERE (valid_to = '' OR valid_to IS NULL) LIMIT ?",
                (limit,),
            ).fetchall()
            return self._rows_to_dicts(rows)
        except Exception as e:
            logger.warning("_get_all_triples failed: %s", e)
            return []

    def _rows_to_dicts(self, rows: list[Any]) -> list[dict[str, Any]]:
        """将行转为字典。"""
        keys = [
            "id",
            "subject",
            "predicate",
            "object",
            "source_memory_id",
            "confidence",
            "is_negation",
            "valid_from",
            "valid_to",
            "created_at",
        ]
        return [dict(zip(keys, row, strict=False)) for row in rows]

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None


# ─── 挂载 query / temporal / relationships 模块方法到 KnowledgeGraph ──
from omnimem.deep.kg import query as _query_module  # noqa: E402
from omnimem.deep.kg import relationships as _relationships_module  # noqa: E402
from omnimem.deep.kg import temporal as _temporal_module  # noqa: E402

KnowledgeGraph.query_by_subject = _query_module.query_by_subject
KnowledgeGraph.query_by_object = _query_module.query_by_object
KnowledgeGraph.query_by_predicate = _query_module.query_by_predicate
KnowledgeGraph.get_neighbors = _query_module.get_neighbors
KnowledgeGraph.find_path = _query_module.find_path
KnowledgeGraph.find_path_context = _query_module.find_path_context
KnowledgeGraph.graph_search = _query_module.graph_search
KnowledgeGraph.graph_rag_context = _query_module.graph_rag_context
KnowledgeGraph.graph_rag_search = _query_module.graph_rag_search
KnowledgeGraph.get_entity = _query_module.get_entity
KnowledgeGraph.get_all_entities = _query_module.get_all_entities
KnowledgeGraph.get_entity_graph = _query_module.get_entity_graph
KnowledgeGraph.shortest_path = _query_module.shortest_path
KnowledgeGraph.connected_components = _query_module.connected_components

KnowledgeGraph.get_timeline = _temporal_module.get_timeline
KnowledgeGraph.get_entity_timeline_text = _temporal_module.get_entity_timeline_text
KnowledgeGraph.get_recent_changes = _temporal_module.get_recent_changes

KnowledgeGraph.get_stats = _relationships_module.get_stats
KnowledgeGraph._sync_relationship_locked = _relationships_module._sync_relationship_locked
KnowledgeGraph.sync_relationships_from_triples = _relationships_module.sync_relationships_from_triples
