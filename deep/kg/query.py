"""图搜索、Graph RAG、路径与社区发现。

职责：
  - 按主语/宾语/谓词查询三元组
  - 邻居扩展、最短路径、连通分量
  - 基于实体的 Graph RAG 上下文生成
  - 实体列表与 POLE+O 分组查询
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

logger = logging.getLogger(__name__)


def query_by_subject(self, subject: str, include_expired: bool = False) -> list[dict[str, Any]]:
    """按主语查询三元组（大小写不敏感）。"""

    def _fetch() -> list[dict[str, Any]]:
        assert self._conn is not None
        try:
            if include_expired:
                rows = self._conn.execute(
                    "SELECT * FROM triples WHERE LOWER(subject) = LOWER(?)",
                    (subject,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM triples WHERE LOWER(subject) = LOWER(?) AND (valid_to = '' OR valid_to IS NULL)",
                    (subject,),
                ).fetchall()
            return self._rows_to_dicts(rows)
        except Exception as e:
            logger.warning("query_by_subject failed for %s: %s", subject, e)
            return []

    return self._cached(f"subj:{subject.lower()}:{include_expired}", _fetch)  # type: ignore[no-any-return]


def query_by_object(self, obj: str, include_expired: bool = False) -> list[dict[str, Any]]:
    """按宾语查询三元组（大小写不敏感）。"""

    def _fetch() -> list[dict[str, Any]]:
        assert self._conn is not None
        try:
            if include_expired:
                rows = self._conn.execute(
                    "SELECT * FROM triples WHERE LOWER(object) = LOWER(?)",
                    (obj,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM triples WHERE LOWER(object) = LOWER(?) AND (valid_to = '' OR valid_to IS NULL)",
                    (obj,),
                ).fetchall()
            return self._rows_to_dicts(rows)
        except Exception as e:
            logger.warning("query_by_object failed for %s: %s", obj, e)
            return []

    return self._cached(f"obj:{obj.lower()}:{include_expired}", _fetch)  # type: ignore[no-any-return]


def query_by_predicate(self, predicate: str, limit: int = 50) -> list[dict[str, Any]]:
    """按谓词查询三元组。"""
    try:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM triples WHERE predicate = ? AND (valid_to = '' OR valid_to IS NULL) LIMIT ?",
            (predicate, limit),
        ).fetchall()
        return self._rows_to_dicts(rows)
    except Exception as e:
        logger.warning("query_by_predicate failed for %s: %s", predicate, e)
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


def find_path(self, start: str, end: str, max_depth: int = 3) -> list[dict[str, Any]]:
    """BFS 最短路径查找：从 start 到 end 的知识链条。

    Returns:
        路径上的三元组列表，按顺序排列。空列表表示无路径。
    """
    if start == end:
        return []

    # BFS queue: (current_entity, path_so_far)
    queue: deque[tuple[str, list[dict[str, Any]]]] = deque([(start, [])])
    visited: set[str] = {start}

    while queue:
        current, path = queue.popleft()
        if len(path) >= max_depth:
            continue

        # 获取当前实体的所有关系
        as_subj = self.query_by_subject(current)
        as_obj = self.query_by_object(current)

        for triple in as_subj + as_obj:
            subj = triple.get("subject", "")
            obj = triple.get("object", "")
            neighbor = obj if subj == current else subj

            if neighbor in visited:
                continue

            new_path = path + [triple]

            if neighbor == end:
                return new_path

            visited.add(neighbor)
            queue.append((neighbor, new_path))

    return []  # 无路径


def find_path_context(self, start: str, end: str, max_depth: int = 3) -> str:
    """将 find_path 结果格式化为可读的推理链文本。"""
    path = self.find_path(start, end, max_depth)
    if not path:
        return f"未找到 {start} 和 {end} 之间的知识路径。"

    relation_labels = {
        "uses": "使用", "belongs_to": "属于", "causes": "导致",
        "replaces": "替代", "connects_to": "关联到", "contains": "包含",
        "located_in": "位于", "better_than": "优于", "is_a": "是一种",
        "part_of": "是...的一部分", "related": "相关于",
    }

    lines = [f"从 {start} 到 {end} 的知识链路（{len(path)} 跳）："]
    for i, triple in enumerate(path):
        subj = triple.get("subject", "")
        pred = triple.get("predicate", "")
        obj = triple.get("object", "")
        label = relation_labels.get(pred, pred)
        conf = triple.get("confidence", 1.0)
        lines.append(f"  {i+1}. {subj} —[{label}]→ {obj}  (置信度: {conf:.1f})")

    return "\n".join(lines)


def graph_search(self, query: str, max_depth: int = 2, limit: int = 20) -> list[dict[str, Any]]:
    """图谱检索通道：从查询中提取实体，然后扩展搜索。

    用于检索引擎的第6通道 (Graph Retriever)。
    """
    from omnimem.deep.kg.entity import extract_entities

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
        except Exception as e:
            logger.warning("graph_search keyword query failed: %s", e)
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


def graph_rag_context(self, entity: str, depth: int = 1) -> str:
    """Graph RAG: 生成实体子图的可读上下文文本。

    将子图中的三元组格式化为自然语言描述，可直接注入LLM上下文窗口。
    参考 Cognee/Zep 的 Graph RAG 模式。

    Args:
        entity: 起始实体名
        depth: 扩展深度（1-hop/2-hop）

    Returns:
        格式化的子图上下文文本，无结果返回空字符串
    """
    neighbors = self.get_neighbors(entity, depth=max(depth, 1))
    if not neighbors:
        # 尝试用部分匹配
        try:
            assert self._conn is not None
            escaped = entity.replace("%", "\\%").replace("_", "\\_")
            rows = self._conn.execute(
                "SELECT * FROM triples WHERE subject LIKE ? ESCAPE '\\' LIMIT 5",
                (f"%{escaped}%",),
            ).fetchall()
            neighbors = self._rows_to_dicts(rows)
        except Exception as e:
            logger.warning("graph_rag_context partial match failed: %s", e)

    if not neighbors:
        return ""

    # 按关系类型分组
    grouped: dict[str, list[tuple[str, str]]] = {}
    for t in neighbors:
        subj = t.get("subject", "")
        obj = t.get("object", "")
        pred = t.get("predicate", "")
        if subj and obj:
            grouped.setdefault(pred, []).append((subj, obj))
        elif subj:
            grouped.setdefault("related", []).append((subj, "?"))

    # 生成自然语言上下文
    lines = []
    relation_labels = {
        "uses": "使用", "belongs_to": "属于", "causes": "导致",
        "replaces": "替代", "connects_to": "关联到", "contains": "包含",
        "located_in": "位于", "better_than": "优于",
        "not_uses": "不使用", "differs_from": "不同于",
        "related": "相关于",
    }
    seen_pairs: set[tuple[str, str, str]] = set()
    for pred, pairs in grouped.items():
        label = relation_labels.get(pred, pred)
        for subj, obj in pairs[:3]:  # Max 3 per relation type
            key = (subj, pred, obj)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            if obj == "?":
                lines.append(f"- {subj} {label}")
            else:
                lines.append(f"- {subj} {label} {obj}")

    if not lines:
        return ""
    return f"[Knowledge Graph: {entity}]\n" + "\n".join(lines)


def graph_rag_search(self, query: str, max_depth: int = 2, _limit: int = 10) -> str:
    """完整的 Graph RAG 搜索：提取实体→扩展子图→生成上下文文本。

    Args:
        query: 自然语言查询
        max_depth: 扩展深度
        limit: 最多返回的三元组数

    Returns:
        Graph RAG 上下文字符串，可注入LLM
    """
    from omnimem.deep.kg.entity import extract_entities

    query_entities = extract_entities(query)
    if not query_entities:
        return ""

    contexts = []
    for entity in query_entities[:3]:
        ctx = self.graph_rag_context(entity, depth=max_depth)
        if ctx:
            contexts.append(ctx)

    return "\n\n".join(contexts)


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
    except Exception as e:
        logger.warning("get_entity failed for %s: %s", name, e)
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
    except Exception as e:
        logger.warning("get_all_entities failed: %s", e)
        return []


def get_entity_graph(self, limit: int = 100) -> dict[str, list[dict[str, Any]]]:
    """获取按 POLE+O 类型分组的实体图谱摘要。

    Returns:
        {"Person": [...], "Organization": [...], "Location": [...],
         "Event": [...], "Object": [...]}
    """
    entities = self.get_all_entities(limit)
    graph: dict[str, list[dict[str, Any]]] = {
        "Person": [],
        "Organization": [],
        "Location": [],
        "Event": [],
        "Object": [],
    }
    for e in entities:
        etype = e.get("entity_type", "Object")
        if etype in graph:
            graph[etype].append(e)
        else:
            graph["Object"].append(e)
    return graph


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
        logger.warning("Shortest path failed: %s", e)
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
        logger.warning("Connected components failed: %s", e)
        return []
