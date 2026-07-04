"""
QueryPlanner — 多跳查询规划侧边栏。

不修改核心检索引擎（HybridRetriever/BM25/Vector/RRF），
作为编排层在 recall handler 与 retriever 之间插一层。

原理（来自论文 Are We Ready For An Agent-Native Memory System?）：
  检索的目标不应该是 "先找一条最像的记忆"，
  而应该是 "先定位可能相关的证据区域，再把互补证据组装完整"。

流程：
  1. 从查询中提取实体（复用 knowledge_graph.extract_entities）
  2. 如实体数 < 2 → 返回 None（走标准单查询路径）
  3. 如实体数 ≥ 2.
     生成子查询：每个实体 + 原始查询上下文
     并行执行所有子查询（retriever.search）
     合并结果：跨子查询命中的记忆提升权重
     标签 _hop_depth 标记跳数
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)

# 模块级线程池（复用，不每次新建）
_planner_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="omnimem-planner")

# 跨子查询命中提升倍率
_CROSS_QUERY_BOOST = 1.5


def plan_and_search(
    query: str,
    retriever: Any,
    max_tokens: int = 1500,
    mode: str = "rag",
    enable_trace: bool = False,
) -> list[dict[str, Any]] | None:
    """多跳查询规划主入口。

    如查询含多个实体，生成子查询并合并结果。
    如查询无/仅一个实体，返回 None 让调用方走标准路径。

    Args:
        query: 原始查询
        retriever: HybridRetriever 实例（或兼容 search() 接口的对象）
        max_tokens: 每个子查询的 token 预算
        mode: 检索模式 (rag/llm)
        enable_trace: 是否记录检索轨迹

    Returns:
        合并后的结果列表，或 None（无需多跳）
    """
    # 从查询中提取实体
    try:
        from omnimem.deep.kg import extract_entities
        entities = extract_entities(query)
    except Exception as e:
        logger.debug("QueryPlanner: entity extraction failed (%s), falling through", e)
        return None

    # 清理过短的实体
    entities = [e for e in entities if len(e) >= 2]
    if len(entities) < 2:
        logger.debug(
            "QueryPlanner: only %d entity/ies, standard search sufficient",
            len(entities),
        )
        return None

    logger.info(
        "QueryPlanner: multi-hop query with %d entities: %s",
        len(entities), entities,
    )

    # 生成子查询：每个实体 + 原始查询扩展
    sub_queries = _generate_sub_queries(query, entities)
    if not sub_queries:
        return None

    # 并行执行子查询
    all_results: list[list[dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 4)) as executor:
        future_map = {
            executor.submit(
                retriever.search, sq,
                max_tokens=max_tokens, mode=mode, enable_trace=enable_trace,
            ): sq
            for sq in sub_queries
        }
        for future in as_completed(future_map):
            try:
                results = future.result()
                if results:
                    all_results.append(results)
                    logger.debug(
                        "QueryPlanner: sub-query '%s' returned %d results",
                        future_map[future], len(results),
                    )
            except Exception as e:
                logger.warning(
                    "QueryPlanner: sub-query '%s' failed: %s",
                    future_map[future], e,
                )

    if not all_results:
        logger.debug("QueryPlanner: no sub-query returned results")
        return None

    # 合并：跨子查询提升
    merged = _merge_and_boost(all_results)

    logger.info(
        "QueryPlanner: merged %d sub-query results into %d unique items",
        sum(len(r) for r in all_results),
        len(merged),
    )
    return merged


def _generate_sub_queries(query: str, entities: list[str]) -> list[str]:
    """生成子查询列表。

    策略：
      - 原始查询本身作为第一路（保证上下文完整性）
      - 每个实体单独作为一路子查询（精确目标定位）
      - 实体对查询（跨实体关联发现）
    """
    sub_queries = [query]  # 原始完整查询作为基线

    # 每个实体的独立查询
    for entity in entities:
        sub_queries.append(f"{entity} {query}")

    # 实体对查询（发现跨实体关联）
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            sub_queries.append(f"{entities[i]} {entities[j]}")

    # 去重
    seen = set()
    unique = []
    for sq in sub_queries:
        if sq not in seen:
            seen.add(sq)
            unique.append(sq)

    # 限制子查询总数（避免过多并发降低精度）
    max_sub_queries = 6
    return unique[:max_sub_queries]


def _merge_and_boost(
    all_results: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """合并多路子查询结果并应用跨查询命中提升。

    - memory_id 去重
    - 在 ≥2 路子查询中命中的记忆，score 提升 _CROSS_QUERY_BOOST
    - 标记 _hop_depth = 命中该记忆的子查询数
    """
    merged: dict[str, dict[str, Any]] = {}
    hit_count: dict[str, int] = {}

    for channel_results in all_results:
        for r in channel_results:
            mid = r.get("memory_id", "") or r.get("content", "")[:64]
            if not mid:
                continue
            if mid not in merged:
                merged[mid] = dict(r)
                hit_count[mid] = 0
            hit_count[mid] += 1

    for mid, count in hit_count.items():
        merged[mid]["_hop_depth"] = count
        if count >= 2:
            # 跨查询命中提升
            original_score = merged[mid].get("score", 0)
            merged[mid]["score"] = original_score * _CROSS_QUERY_BOOST
            merged[mid]["_cross_query_boosted"] = True

    # 按 score 降序排列
    sorted_results = sorted(
        merged.values(),
        key=lambda x: x.get("score", 0),
        reverse=True,
    )

    return sorted_results
