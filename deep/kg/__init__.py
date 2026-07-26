"""deep/kg 子包 —— 知识图谱核心能力。

向后兼容：原 omnimem.deep.knowledge_graph 的公共 API 已全部迁移至此，
旧导入路径通过 deep/knowledge_graph.py shim 继续可用。
"""

from __future__ import annotations

from omnimem.deep.kg.builder import KnowledgeGraph
from omnimem.deep.kg.entity import (
    _POLEO_TYPES,
    _classify_entity_poleo,
    extract_entities,
    extract_entities_llm,
)
from omnimem.deep.kg.extraction import extract_triples, infer_relations
from omnimem.deep.kg.query import (
    connected_components,
    find_path,
    find_path_context,
    get_all_entities,
    get_entity,
    get_entity_graph,
    get_neighbors,
    graph_rag_context,
    graph_rag_search,
    graph_search,
    query_by_object,
    query_by_predicate,
    query_by_subject,
    shortest_path,
)
from omnimem.deep.kg.temporal import (
    get_entity_timeline_text,
    get_recent_changes,
    get_timeline,
)

__all__ = [
    "KnowledgeGraph",
    "extract_entities",
    "extract_triples",
    "extract_entities_llm",
    "infer_relations",
    "_classify_entity_poleo",
    "_POLEO_TYPES",
    "query_by_subject",
    "query_by_object",
    "query_by_predicate",
    "get_neighbors",
    "find_path",
    "find_path_context",
    "graph_search",
    "graph_rag_context",
    "graph_rag_search",
    "get_entity",
    "get_all_entities",
    "get_entity_graph",
    "shortest_path",
    "connected_components",
    "get_timeline",
    "get_entity_timeline_text",
    "get_recent_changes",
]
