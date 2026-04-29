"""OmniMem recall 处理器。

从 provider.py 的 _handle_recall() 方法提取，通过 provider 参数访问实例组件。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ★ R26优化：提取公共正则常量，避免4处硬编码重复
_CJK_KEYWORD_RE = re.compile(
    r"[\u4e00-\u9fff]{2,}|[\uac00-\ud7af]{2,}|[\u3040-\u309f\u30a0-\u30ff]{2,}|[a-zA-Z]{3,}"
)

# ★ R27优化：模块级同义词映射，避免 llm 模式每次调用重建字典
_SYNONYM_MAP: dict[str, list[str]] = {
    "宠物": ["猫咪", "狗狗", "兔子", "仓鼠", "小鸟", "小鱼"],
    "饮食": ["食用", "喂食", "饲料", "鸡胸肉", "猫粮", "狗粮"],
    "编程": ["代码", "开发", "程序", "coding"],
    "部署": ["deploy", "上线", "发布", "运维"],
    "数据库": ["mysql", "postgres", "mongodb", "redis"],
}


def _extract_query_keywords(query: str) -> set[str]:
    """从查询中提取关键词集合，含 CJK 长词窗口切分。"""
    _raw_kw = _CJK_KEYWORD_RE.findall(query.lower())
    keywords = set()
    for kw in _raw_kw:
        # ★ R25修复：连续汉字 >4 字时按2-4字窗口切分
        if re.match(r"[\u4e00-\u9fff]+$", kw) and len(kw) > 4:
            for i in range(len(kw)):
                for n in (4, 3, 2):
                    if i + n <= len(kw):
                        keywords.add(kw[i : i + n])
        else:
            keywords.add(kw)
    return keywords


def handle_recall(provider, args: dict[str, Any]) -> str:
    """主动检索记忆 — 经 ContextManager 精炼后返回精简摘要。

    与 prefetch 不同，recall 是 Agent 主动调用，预算更宽松。
    但仍然经过精炼/去重，并保留 original_content 供 omni_detail 按需拉取。

    检索流程（三种模式）:
      rag 模式（默认）:
        1. HybridRetriever.search() 执行向量+BM25+RRF融合检索
        2. 图谱检索通道补充（知识图谱三元组）
        3. 时间衰减 + 隐私过滤
        4. 主存储验证（过滤索引残留）
        5. 最低相关性过滤（关键词验证）
        6. ContextManager 精炼

      llm 模式（深度检索）:
        1. 同 rag 模式基础流程
        2. 额外 store 内容搜索补充通道（同义词扩展 + 关键词重叠过滤）
        3. 图谱检索通道补充
        4. 后续过滤和精炼同 rag

      无结果 fallback:
        向量+BM25均无结果时，回退到 store 全量关键词匹配

    Args:
        provider: OmniMemProvider 实例，用于访问子组件
        args: 工具调用参数，包含 query/mode(rag|llm)/max_tokens

    Returns:
        JSON 字符串，status 可能为:
          found — 找到相关记忆，包含精炼后的摘要列表
          no_results — 未找到任何相关记忆
    """
    query = args["query"]
    mode = args.get("mode", "rag")
    max_tokens = args.get("max_tokens", 1500)

    # ★ R27优化：预提取查询关键词，避免同一函数内4次重复正则匹配与CJK切分
    _query_keywords = _extract_query_keywords(query)

    results = provider._retriever.search(query, max_tokens=max_tokens, mode=mode)

    # ★ llm 模式补充通道：从 store 做内容搜索，弥补向量/BM25 可能遗漏的
    # 但需要过滤：只保留与 query 有关键词重叠的结果，避免噪音
    if mode == "llm":
        try:
            # ★ 同义词扩展：常见中文近义词/上下位词，弥补 BM25 词袋模型的语义鸿沟
            # 注意：单字会被 _tokenize 丢弃，所以用2+字词
            expanded_queries = [query]
            for key, synonyms in _SYNONYM_MAP.items():
                if key in query:
                    for syn in synonyms:
                        expanded_queries.append(query.replace(key, syn))

            all_store_results = []
            existing_ids = {r.get("memory_id", "") for r in results}
            for eq in expanded_queries:
                all_store_results.extend(provider._store.search_by_content(eq, limit=5))

            # 去重
            seen = set(existing_ids)
            query_keywords = _query_keywords
            for sr in all_store_results:
                mid = sr.get("memory_id", "")
                if mid in seen:
                    continue
                seen.add(mid)
                sr_content = sr.get("content", "").lower()
                # ★ 关键词重叠过滤：至少1个关键词在结果内容中出现（宽松，因为同义词已扩展）
                if query_keywords:
                    overlap_count = sum(1 for kw in query_keywords if kw in sr_content)
                    if overlap_count >= 1:
                        sr["_source"] = "store_supplement"
                        sr["score"] = 0.3
                        results.append(sr)
                # query 无关键词时不追加
        except (TimeoutError, ConnectionError) as e:
            logger.debug("OmniMem llm store supplement failed: %s", e)

    # 图谱检索通道
    if provider._knowledge_graph:
        try:
            graph_results = provider._knowledge_graph.graph_search(query, max_depth=2, limit=10)
            if graph_results:
                for gr in graph_results[:5]:
                    gr[
                        "content"
                    ] = f"{gr.get('subject', '')} {gr.get('predicate', '')} {gr.get('object', '')}"
                    gr["type"] = "graph_triple"
                    gr["confidence"] = gr.get("confidence", 0.5)
                results.extend(graph_results[:5])
        except (RuntimeError, ValueError) as e:
            logger.debug("OmniMem graph recall failed: %s", e)

    results = provider._temporal_decay.apply(results)
    results = provider._privacy.filter(results, session_id=provider._session_id)

    # ★ 主存储验证：过滤掉向量/BM25索引中残留但主存储已删除的条目
    # sync- 条目被归档后主存储已无，但向量/BM25索引中可能残留
    valid_results = []
    for r in results:
        mid = r.get("memory_id", "")
        if mid and not provider._store.get(mid):
            # 主存储中不存在 → 索引残留，跳过
            continue
        valid_results.append(r)
    results = valid_results

    # ★ 最低相关性过滤 — 统一所有来源的结果
    # 来源分类：
    #   RRF 融合: score = rrf_score (0.02-0.05), 已在 rrf.py 中过滤 < 0.015
    #   store_supplement: score = 0.3, 已做关键词重叠过滤
    #   graph_triple: confidence = 0.5, 无关键词过滤
    # ★ 关键词验证：对每条结果检查内容与 query 是否有实质关联
    # ★ R25修复：连续汉字需按2-4字窗口切分，避免6字整体匹配不到2字词
    query_keywords = _query_keywords
    filtered = []
    for r in results:
        score = r.get("score", 0)
        if score <= 0:
            continue
        source = r.get("_source", "")
        # ★ store_supplement 已在上方做过关键词过滤，直接通过
        if source == "store_supplement":
            filtered.append(r)
            continue
        # ★ graph_triple: 检查内容是否与 query 关键词有重叠
        if r.get("type") == "graph_triple":
            content = r.get("content", "").lower()
            if query_keywords and any(kw in content for kw in query_keywords):
                filtered.append(r)
            continue
        # ★ RRF 融合结果: score = rrf_score
        # rrf_score < 0.015 的已在 rrf.py 中过滤
        # 但如果 rrf_score 很低（0.015-0.025），可能是单路低排名的噪音
        # 进一步验证：内容是否与 query 有任何关键词重叠
        if score < 0.025:
            if query_keywords:
                content = r.get("content", "").lower()
                has_overlap = any(kw in content for kw in query_keywords)
                if not has_overlap:
                    continue  # 低分且无关键词重叠 → 噪音，跳过
            else:
                # ★ R24修复QUAL-1：无关键词的垃圾查询（如 zzzzzxyz123），
                # 低分结果一定是噪音，直接跳过
                continue
        filtered.append(r)
    results = filtered

    if not results:
        # ★ R17修复QUAL-2：向量+BM25均无结果时，fallback到store关键词匹配
        # 解决长文本语义稀释问题（similarity=0.2998 被阈值0.3过滤的边界情况）
        # 注意：不用 search_by_content（精确子串），而是搜索后做关键词级匹配
        query_keywords = _query_keywords
        if query_keywords:
            store_all = provider._store.search(limit=50)
            for sf in store_all:
                sf_mid = sf.get("memory_id", "")
                if sf_mid in {r.get("memory_id", "") for r in results}:
                    continue
                sf_content = sf.get("content", "").lower()
                keyword_hits = sum(1 for kw in query_keywords if kw in sf_content)
                if keyword_hits >= 1:
                    sf["_source"] = "store_fallback"
                    sf["score"] = min(0.15 + keyword_hits * 0.05, 0.35)
                    results.append(sf)
                    if len(results) >= 5:
                        break

    # ★ R25优化：结果不足时补充 store 关键词匹配
    # 解决中英混合查询时语义检索遗漏问题（如 "基础设施管理" 搜不到 "Terraform"）
    if len(results) < 3:
        query_keywords = _query_keywords
        existing_ids = {r.get("memory_id", "") for r in results}
        if query_keywords:
            try:
                store_all = provider._store.search(limit=50)
                for sf in store_all:
                    sf_mid = sf.get("memory_id", "")
                    if sf_mid in existing_ids:
                        continue
                    sf_content = sf.get("content", "").lower()
                    # 关键词命中或原文包含查询核心词
                    keyword_hits = sum(1 for kw in query_keywords if kw in sf_content)
                    if keyword_hits >= 1:
                        sf["_source"] = "store_supplement"
                        sf["score"] = sf.get("score", 0) or min(0.12 + keyword_hits * 0.03, 0.25)
                        results.append(sf)
                        existing_ids.add(sf_mid)
                        if len(results) >= 5:
                            break
            except Exception:
                pass
        if not results:
            return json.dumps(
                {
                    "status": "no_results",
                    "query": query,
                    "message": "No relevant memories found.",
                }
            )

    # ★ 经 ContextManager 精炼 — 精简摘要 + 保留原文供 detail 拉取
    refined = provider._context_manager.refine_recall_results(results, max_tokens=max_tokens)

    return json.dumps(
        {
            "status": "found",
            "query": query,
            "count": len(refined),
            "memories": refined,
            "hint": "Use omni_detail with a memory_id to fetch full content.",
        },
        ensure_ascii=False,
    )
