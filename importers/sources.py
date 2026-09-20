"""各上游项目的导出结构适配（改进项 #8）。

每个适配器负责其项目的容器路径与字段怪癖（如 Letta core block 的
label/value、Graphiti 的 node/edge 分列），把记录拆平成一组 dict 后交给
_base.convert 统一归一，避免在通用层里塞满 if source == ...。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import _base

# ─── mem0 ─────────────────────────────────────────────
# Memory.get_all() -> {"results": [{id, memory, user_id, hash, metadata, created_at}]}
# memory_add / search 结果同样带 memory 文本字段。


def _unwrap(data: Any, *keys: str) -> Any:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in keys:
            val = data.get(k)
            if isinstance(val, list):
                return val
    return data


def mem0(data: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in _base.normalize_list(_unwrap(data, "results", "memories", "items")):
        if isinstance(item, str):
            out.append({"memory": item})
            continue
        rec = {
            "id": item.get("id"),
            "memory": item.get("memory") or item.get("text") or item.get("content"),
            "created_at": item.get("created_at"),
            "metadata": item.get("metadata"),
        }
        cats = (item.get("metadata") or {}).get("categories") if isinstance(item.get("metadata"), dict) else None
        if cats:
            rec["tags"] = cats
        out.append(rec)
    return out


# ─── Letta (MemGPT) ───────────────────────────────────
# archival passages: [{id, text, tags, created_at}]  或 {"passages": [...]}
# core memory blocks: [{label, value}]


def letta(data: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    blocks = _unwrap(data, "passages", "results", "records", "archival_memories")
    if isinstance(data, dict) and isinstance(data.get("memory_blocks"), list):
        for blk in data["memory_blocks"]:
            value = blk.get("value") or blk.get("contents")
            if value:
                out.append({
                    "content": value,
                    "id": blk.get("id") or blk.get("label"),
                    "memory_type": "preference" if str(blk.get("label", "")).lower() in {"persona", "human", "self"} else "fact",
                })
    for item in _base.normalize_list(blocks):
        if isinstance(item, str):
            out.append({"content": item})
            continue
        out.append({
            "id": item.get("id"),
            "content": item.get("text") or item.get("content") or item.get("value"),
            "tags": item.get("tags"),
            "created_at": item.get("created_at") or item.get("last_accessed_at"),
        })
    return out


# ─── Zep ──────────────────────────────────────────────
# user facts: [{id, text, created_at}] 或 {"facts": [...]}
# thread messages: [{content, created_at}]


def zep(data: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in _base.normalize_list(_unwrap(data, "facts", "memories", "messages", "results")):
        if isinstance(item, str):
            out.append({"content": item})
            continue
        out.append({
            "id": item.get("id") or item.get("fact_id"),
            "content": item.get("text") or item.get("fact") or item.get("content"),
            "memory_type": item.get("type"),
            "created_at": item.get("created_at") or item.get("timestamp"),
            "metadata": item.get("metadata"),
        })
    return out


# ─── Graphiti (Zep 底层时序知识图谱) ──────────────────
# {"nodes": [{name, summary, labels}], "edges": [{fact, name, valid_at, invalid_at}]}


def graphiti(data: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(data, dict):
        for edge in _base.normalize_list(data.get("edges")):
            if isinstance(edge, dict):
                out.append({
                    "id": edge.get("uuid") or edge.get("id"),
                    "content": edge.get("fact") or edge.get("name") or edge.get("summary"),
                    "created_at": edge.get("valid_at") or edge.get("created_at"),
                    "tags": edge.get("name"),
                })
        for node in _base.normalize_list(data.get("nodes")):
            if isinstance(node, dict):
                out.append({
                    "id": node.get("uuid") or node.get("id"),
                    "content": node.get("summary") or node.get("name"),
                    "memory_type": "entity",
                    "tags": node.get("labels") or node.get("name"),
                    "created_at": node.get("created_at"),
                })
    if not out:
        out = zep(data)
    return [r for r in out if _base._coerce_text(r.get("content"))]


# ─── Cognee ───────────────────────────────────────────
# 数据点 / 图导出: {"nodes":[{type,text/name}], "edges":[...]} 或 [{text/description}]


def cognee(data: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(data, dict) and (isinstance(data.get("nodes"), list) or isinstance(data.get("edges"), list)):
        return graphiti(data)
    for item in _base.normalize_list(_unwrap(data, "data_points", "nodes", "results", "memories")):
        if isinstance(item, str):
            out.append({"content": item})
            continue
        out.append({
            "id": item.get("id"),
            "content": item.get("text") or item.get("description") or item.get("name"),
            "memory_type": item.get("type"),
            "created_at": item.get("created_at"),
            "metadata": item.get("metadata"),
        })
    return [r for r in out if _base._coerce_text(r.get("content"))]


SOURCE_ADAPTERS: dict[str, Callable[[Any], list[dict[str, Any]]]] = {
    "mem0": mem0,
    "letta": letta,
    "zep": zep,
    "graphiti": graphiti,
    "cognee": cognee,
    "auto": _base.normalize_list,
}

SUPPORTED_SOURCES = ("mem0", "letta", "zep", "graphiti", "cognee", "auto")
