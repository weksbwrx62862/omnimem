"""KnowledgeGraph — SQLite 知识图谱。

参考 MemOS 的知识图谱设计 + MemPalace 的时序三元组：
  - 实体自动抽取：从记忆内容中提取实体和关系
  - 关系推理：基于已有三元组推断隐含关系
  - 时序有效性：valid_from/valid_to 时间过滤
  - 图谱检索通道：1-hop 扩展 + 关系网络发现

Phase 3 完整实现。
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─── 实体抽取模式 ────────────────────────────────────────────

# 中文实体模式：人名/地名/机构/技术术语
_ZH_ENTITY_PATTERNS = [
    r"[\u4e00-\u9fff]{2,4}(?=公司|团队|项目|系统|框架|平台|模块|服务|接口|数据库)",  # 组织/系统名
    r"(?<=用户|客户|同事|老板|领导|朋友)[\u4e00-\u9fff]{2,3}",  # 人名
]

# 英文实体模式
_EN_ENTITY_PATTERNS = [
    r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b",  # CamelCase
    r"\b[A-Z]{2,}\b",  # 缩写 API, SQL, etc.
    r"\b[a-z]+(?:-[a-z]+)+\b",  # kebab-case
    # 技术名词：不用 \b（中英混合时无效），但用前后断言防止子串匹配
    # 例：匹配 "Docker部署" 中的 "Docker"，但不匹配 "Pythonic" 中的 "Python"
    r"(?<![A-Za-z])(Python|Java|Go|Rust|TypeScript|React|Vue|Docker|K8s|Redis|MySQL|PostgreSQL|MongoDB|Neo4j|ChromaDB|SQLite)(?![A-Za-z])",
]

# 通用实体模式：从关系三元组中提取的实体
_GENERIC_ENTITY_PATTERNS = [
    # 中文关键词前面的名词（如"前端使用React"中的"前端"）
    r"(?<=[，。、\s])[\u4e00-\u9fff]{2,6}(?=使用|采用|选用|基于|依赖|运行)",
    # 中文关键词后面的英文技术名词
    r"(?:使用|采用|选用|基于|依赖)\s*([A-Z][A-Za-z0-9_.-]*)",
]

# 关系模式：从文本中提取 (主语, 关系, 宾语) 三元组
_RELATION_PATTERNS = [
    # 中文关系
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:使用|采用|选用|基于|依赖|运行在)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "uses",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:属于|归入|隶属于)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "belongs_to",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:导致|引起|造成|触发)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "causes",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:替代|取代|替换|升级为)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "replaces",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:连接|关联|对应|映射到)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "connects_to",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:优于|胜过|好于)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "better_than",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:包含|包括|由.*组成)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "contains",
    ),
    (r"([\u4e00-\u9fff]{2,8})\s*(?:在|于)\s*([\u4e00-\u9fff]{2,8})\s*(?:中|里|上)", "located_in"),
    # 英文关系
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:uses?|depends?\s+on|relies?\s+on)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "uses",
    ),
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:causes?|leads?\s+to|triggers?)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "causes",
    ),
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:replaces?|supersedes?)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "replaces",
    ),
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:contains?|includes?)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "contains",
    ),
]

# 否定关系模式
_NEGATION_PATTERNS = [
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:不|并非|没有|无法|不能)\s*(?:使用|采用|依赖|支持)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "not_uses",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:不同于|区别于|不是)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "differs_from",
    ),
]


# ─── 实体抽取函数 ────────────────────────────────────────────


def extract_entities(text: str) -> list[str]:
    """从文本中提取实体。

    ★ P1方案四：优先使用 jieba 分词 + 词性标注（若可用），
    与规则正则互补，提升通用命名实体覆盖率。
    """
    entities: set[str] = set()

    # 优先路径：jieba NER（人名nr/地名ns/机构名nt/其他专名nz）
    try:
        import jieba.posseg as pseg

        for word, flag in pseg.lcut(text):
            if flag in ("nr", "ns", "nt", "nz") and len(word) >= 2:
                entities.add(word)
    except ImportError:
        pass

    # 中文实体（规则正则，与 jieba 互补）
    for pattern in _ZH_ENTITY_PATTERNS:
        matches = re.findall(pattern, text)
        entities.update(matches)

    # 英文实体
    for pattern in _EN_ENTITY_PATTERNS:
        matches = re.findall(pattern, text)
        entities.update(matches)

    # 通用实体模式
    for pattern in _GENERIC_ENTITY_PATTERNS:
        matches = re.findall(pattern, text)
        entities.update(matches)

    # 从三元组中提取的实体（主语和宾语也是实体）
    triples = extract_triples(text)
    for subj, _, obj in triples:
        entities.add(subj)
        entities.add(obj)

    # 去除太短的实体
    return [e for e in entities if len(e) >= 2]


def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """从文本中提取 (主语, 关系, 宾语) 三元组。

    Returns:
        List of (subject, predicate, object) tuples
    """
    triples: list[tuple[str, str, str]] = []

    for pattern, predicate in _RELATION_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                subj, obj = match[0], match[1]
                if subj and obj and subj != obj:
                    triples.append((subj, predicate, obj))

    # 否定关系
    for pattern, predicate in _NEGATION_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                subj, obj = match[0], match[1]
                if subj and obj and subj != obj:
                    triples.append((subj, predicate, obj))

    return triples


def infer_relations(existing_triples: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """基于已有三元组推理隐含关系。

    推理规则:
      - 传递性: A uses B, B uses C → A uses C (transitive)
      - 互逆: A belongs_to B → B contains A
      - 替代链: A replaces B, B replaces C → A replaces C
    """
    inferred: list[tuple[str, str, str]] = []

    # 建立主语→(关系→宾语)的索引
    subj_map: dict[str, dict[str, list[str]]] = {}
    for t in existing_triples:
        s, p, o = t.get("subject", ""), t.get("predicate", ""), t.get("object", "")
        if not s or not p or not o:
            continue
        subj_map.setdefault(s, {}).setdefault(p, []).append(o)

    # 传递性推理: uses, causes, replaces
    transitive_preds = {"uses", "causes", "replaces"}
    for subj, pred_map in subj_map.items():
        for pred in transitive_preds:
            if pred in pred_map:
                for obj in pred_map[pred]:
                    # obj 的关系传递到 subj
                    if obj in subj_map and pred in subj_map[obj]:
                        for trans_obj in subj_map[obj][pred]:
                            if trans_obj != subj:  # 避免循环
                                inferred.append((subj, pred, trans_obj))

    # 互逆推理: belongs_to ↔ contains
    for subj, pred_map in subj_map.items():
        if "belongs_to" in pred_map:
            for obj in pred_map["belongs_to"]:
                inferred.append((obj, "contains", subj))

    return inferred


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
        self._conn.execute("""
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
        """)
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
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                entity_type TEXT DEFAULT 'unknown',
                mention_count INTEGER DEFAULT 1,
                first_seen TEXT,
                last_seen TEXT
            )
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
                        logger.debug(
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

                # 数据变更后清除查询缓存
                self._invalidate_cache()

                return cursor.lastrowid if cursor.lastrowid is not None else -1
            except Exception as e:
                logger.debug("Triple add failed: %s", e)
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

    def query_by_subject(self, subject: str, include_expired: bool = False) -> list[dict[str, Any]]:
        """按主语查询三元组。"""

        def _fetch() -> list[dict[str, Any]]:
            assert self._conn is not None
            try:
                if include_expired:
                    rows = self._conn.execute(
                        "SELECT * FROM triples WHERE subject = ?",
                        (subject,),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT * FROM triples WHERE subject = ? AND (valid_to = '' OR valid_to IS NULL)",
                        (subject,),
                    ).fetchall()
                return self._rows_to_dicts(rows)
            except Exception:
                return []

        return self._cached(f"subj:{subject}:{include_expired}", _fetch)  # type: ignore[no-any-return]

    def query_by_object(self, obj: str, include_expired: bool = False) -> list[dict[str, Any]]:
        """按宾语查询三元组。"""

        def _fetch() -> list[dict[str, Any]]:
            assert self._conn is not None
            try:
                if include_expired:
                    rows = self._conn.execute(
                        "SELECT * FROM triples WHERE object = ?",
                        (obj,),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT * FROM triples WHERE object = ? AND (valid_to = '' OR valid_to IS NULL)",
                        (obj,),
                    ).fetchall()
                return self._rows_to_dicts(rows)
            except Exception:
                return []

        return self._cached(f"obj:{obj}:{include_expired}", _fetch)  # type: ignore[no-any-return]

    def query_by_predicate(self, predicate: str, limit: int = 50) -> list[dict[str, Any]]:
        """按谓词查询三元组。"""
        try:
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT * FROM triples WHERE predicate = ? AND (valid_to = '' OR valid_to IS NULL) LIMIT ?",
                (predicate, limit),
            ).fetchall()
            return self._rows_to_dicts(rows)
        except Exception:
            return []

    def get_neighbors(self, entity: str, depth: int = 1) -> list[dict[str, Any]]:
        """获取实体的邻居（递归扩展查询），带 TTL 缓存。"""

        def _fetch() -> list[dict[str, Any]]:
            results = []
            visited: set[str] = set()

            def _expand(e: str, d: int) -> None:
                if d <= 0 or e in visited:
                    return
                visited.add(e)
                as_subj = self.query_by_subject(e)
                results.extend(as_subj)
                as_obj = self.query_by_object(e)
                results.extend(as_obj)
                if d > 1:
                    for t in as_subj:
                        _expand(t.get("object", ""), d - 1)
                    for t in as_obj:
                        _expand(t.get("subject", ""), d - 1)

            _expand(entity, depth)
            seen_ids: set[int] = set()
            unique_results = []
            for r in results:
                rid = r.get("id")
                if rid is not None:
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        unique_results.append(r)
            return unique_results

        return self._cached(f"neighbors:{entity}:{depth}", _fetch)  # type: ignore[no-any-return]

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
            except Exception:
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

    # ─── 图谱检索通道 ─────────────────────────────────────────

    def graph_search(self, query: str, max_depth: int = 2, limit: int = 20) -> list[dict[str, Any]]:
        """图谱检索通道：从查询中提取实体，然后扩展搜索。

        用于检索引擎的第6通道 (Graph Retriever)。
        """
        # 从查询中提取可能的实体
        query_entities = extract_entities(query)

        if not query_entities:
            # 尝试直接关键词匹配（转义 LIKE 通配符防止注入/误匹配）
            try:
                assert self._conn is not None
                escaped = query.replace("%", "\\%").replace("_", "\\_")
                rows = self._conn.execute(
                    "SELECT * FROM triples WHERE subject LIKE ? ESCAPE '\\' OR object LIKE ? ESCAPE '\\' LIMIT ?",
                    (f"%{escaped}%", f"%{escaped}%", limit),
                ).fetchall()
                return self._rows_to_dicts(rows)
            except Exception:
                return []

        # 对每个实体进行扩展搜索
        all_results: list[dict[str, Any]] = []
        for entity in query_entities[:3]:  # 最多3个实体
            neighbors = self.get_neighbors(entity, depth=max_depth)
            all_results.extend(neighbors)

        # 去重
        seen_ids: set[int] = set()
        unique = []
        for r in all_results:
            rid = r.get("id")
            if rid not in seen_ids:
                seen_ids.add(rid)  # type: ignore[arg-type]
                unique.append(r)

        return unique[:limit]

    # ─── 实体操作 ─────────────────────────────────────────────

    def get_entity(self, name: str) -> dict[str, Any] | None:
        """获取实体信息。"""
        try:
            assert self._conn is not None
            row = self._conn.execute("SELECT * FROM entities WHERE name = ?", (name,)).fetchone()
            if row:
                keys = ["name", "entity_type", "mention_count", "first_seen", "last_seen"]
                return dict(zip(keys, row, strict=False))
            return None
        except Exception:
            return None

    def get_all_entities(self, limit: int = 100) -> list[dict[str, Any]]:
        """获取所有实体。"""
        try:
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT * FROM entities ORDER BY mention_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
            keys = ["name", "entity_type", "mention_count", "first_seen", "last_seen"]
            return [dict(zip(keys, row, strict=False)) for row in rows]
        except Exception:
            return []

    # ─── 统计 ─────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """获取图谱统计。"""
        stats: dict[str, Any] = {"total_triples": 0, "total_entities": 0}
        if self._conn:
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM triples").fetchone()
                stats["total_triples"] = row[0] if row else 0

                row = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()
                stats["total_entities"] = row[0] if row else 0

                # 按谓词统计
                pred_rows = self._conn.execute(
                    "SELECT predicate, COUNT(*) as cnt FROM triples GROUP BY predicate ORDER BY cnt DESC LIMIT 10"
                ).fetchall()
                stats["predicates"] = {r[0]: r[1] for r in pred_rows}
            except Exception:
                pass
        return stats

    # ─── 图算法 ─────────────────────────────────────────────

    def shortest_path(
        self,
        start: str,
        end: str,
        max_depth: int = 5,
    ) -> list[dict[str, Any]]:
        """返回两实体间的最短关系路径（BFS）。

        Args:
            start: 起始实体
            end: 目标实体
            max_depth: 最大搜索深度

        Returns:
            路径上的三元组列表（从 start 到 end）
        """
        if not self._conn:
            return []
        try:
            from collections import deque

            visited: dict[str, tuple[Any, ...]] = {start: ()}  # entity -> (prev_entity, triple_dict)
            queue: deque[str] = deque([start])
            depth = 0

            while queue and depth < max_depth:
                for _ in range(len(queue)):
                    current = queue.popleft()
                    if current == end:
                        # 回溯路径
                        path = []
                        node = end
                        while node != start:
                            prev, triple = visited[node]
                            path.append(triple)
                            node = prev
                        return list(reversed(path))

                    # 扩展邻居：作为 subject 或 object
                    rows = self._conn.execute(
                        "SELECT subject, predicate, object, confidence FROM triples "
                        "WHERE (subject = ? OR object = ?) AND (valid_to = '' OR valid_to IS NULL)",
                        (current, current),
                    ).fetchall()
                    for subj, pred, obj, conf in rows:
                        neighbor = obj if subj == current else subj
                        if neighbor not in visited:
                            visited[neighbor] = (
                                current,
                                {
                                    "subject": subj,
                                    "predicate": pred,
                                    "object": obj,
                                    "confidence": conf,
                                },
                            )
                            queue.append(neighbor)
                depth += 1
            return []
        except Exception as e:
            logger.debug("Shortest path failed: %s", e)
            return []

    def connected_components(self, min_size: int = 3, limit: int = 500) -> list[list[str]]:
        """发现知识社区（连通分量）。

        Args:
            min_size: 社区最小实体数
            limit: 最大扫描实体数

        Returns:
            每个社区是一个实体名称列表
        """
        if not self._conn:
            return []
        try:
            rows = self._conn.execute(
                "SELECT DISTINCT subject, object FROM triples "
                "WHERE valid_to = '' OR valid_to IS NULL LIMIT ?",
                (limit * 2,),
            ).fetchall()
            from collections import defaultdict

            graph: defaultdict[str, set[str]] = defaultdict(set)
            all_entities: set[str] = set()
            for subj, obj in rows:
                graph[subj].add(obj)
                graph[obj].add(subj)
                all_entities.add(subj)
                all_entities.add(obj)

            visited: set[str] = set()
            components: list[list[str]] = []
            for entity in all_entities:
                if entity in visited:
                    continue
                stack = [entity]
                comp: list[str] = []
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    comp.append(node)
                    stack.extend(graph[node] - visited)
                if len(comp) >= min_size:
                    components.append(comp)
            return components
        except Exception as e:
            logger.debug("Connected components failed: %s", e)
            return []

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None

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
            logger.debug("Entity upsert failed: %s", e)

    def _infer_entity_type(self, name: str) -> str:
        """推断实体类型。"""
        # 技术术语
        tech_terms = {
            "Python",
            "Java",
            "Go",
            "Rust",
            "TypeScript",
            "React",
            "Vue",
            "Docker",
            "K8s",
            "Redis",
            "MySQL",
            "PostgreSQL",
            "MongoDB",
            "Neo4j",
            "ChromaDB",
            "SQLite",
            "API",
            "SQL",
            "REST",
            "GraphQL",
        }
        if name in tech_terms:
            return "technology"

        # CamelCase → 可能是类名/组件名
        if re.match(r"^[A-Z][a-z]+(?:[A-Z][a-z]+)+$", name):
            return "component"

        # 全大写 → 缩写
        if re.match(r"^[A-Z]{2,}$", name):
            return "abbreviation"

        # kebab-case → 工具/包名
        if re.match(r"^[a-z]+(?:-[a-z]+)+$", name):
            return "package"

        # 中文 → 默认概念
        if re.match(r"^[\u4e00-\u9fff]+$", name):
            return "concept"

        return "unknown"

    def _get_all_triples(self, limit: int = 5000) -> list[dict[str, Any]]:
        """获取所有有效三元组。"""
        try:
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT * FROM triples WHERE (valid_to = '' OR valid_to IS NULL) LIMIT ?",
                (limit,),
            ).fetchall()
            return self._rows_to_dicts(rows)
        except Exception:
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
