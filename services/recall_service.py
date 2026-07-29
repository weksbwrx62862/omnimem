"""OmniMem 记忆召回 Service。

将 handlers/recall.py 中的核心业务逻辑下沉：
- 检索编排（retriever / QueryPlanner / 图谱 / 时序图谱）
- 多源结果合并与过滤
- 主存储验证
- 最低相关性过滤
- 证据组富化与分组
- ContextManager 精炼
- 召回反馈循环
- 异步召回路径
"""

from __future__ import annotations

import asyncio
import atexit
import functools
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from omnimem.handlers.deps import HandlerDependencies, RecallResult
from omnimem.utils.logging import sanitize_for_log
from omnimem.utils.metrics import record_recall_duration

logger = logging.getLogger(__name__)

# ★ P1修复：模块级共享 executor，避免每次 recall 调用创建新线程池
_recall_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="omnimem-recall")
atexit.register(_recall_executor.shutdown, wait=False)

# ★ 时序关键词
_TEMPORAL_KEYWORDS = (
    "现在", "之前", "之后", "上个月", "下个月", "去年", "今年", "目前",
    "当前", "最近", "以前", "后来", "后来呢", "什么时候", "什么时候开始",
    "什么时候结束", "什么时候换", "什么时候改", "变化", "变了", "换了",
    "更新了", "升级了", "改成了", "变成了", "之前是", "原来是", "以前是",
    "last", "now", "before", "after", "previously", "currently",
    "used to", "changed", "updated", "replaced",
)

# ★ R26优化：提取公共正则常量
_CJK_KEYWORD_RE = re.compile(
    r"[\u4e00-\u9fff]{2,}|[\uac00-\ud7af]{2,}|[\u3040-\u309f\u30a0-\u30ff]{2,}|[a-zA-Z]{3,}"
)

# ★ R27优化：模块级同义词映射
_DEFAULT_SYNONYM_MAP: dict[str, list[str]] = {
    "宠物": ["猫咪", "狗狗", "兔子", "仓鼠", "小鸟", "小鱼"],
    "饮食": ["食用", "喂食", "饲料", "鸡胸肉", "猫粮", "狗粮"],
    "编程": ["代码", "开发", "程序", "coding"],
    "部署": ["deploy", "上线", "发布", "运维"],
    "数据库": ["mysql", "postgres", "mongodb", "redis"],
}


def _load_synonyms_from_config() -> dict[str, list[str]]:
    """从 config/synonyms.json 加载同义词表，合并默认值。"""
    import json as _json
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config" / "synonyms.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                user_synonyms = _json.load(f)
            merged = {**_DEFAULT_SYNONYM_MAP, **user_synonyms}
            logger.debug("Loaded %d synonym entries from %s", len(merged), config_path)
            return merged
        except Exception as e:
            logger.warning("Failed to load synonyms from %s: %s", config_path, e)
    return _DEFAULT_SYNONYM_MAP


_SYNONYM_MAP: dict[str, list[str]] = _load_synonyms_from_config()


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


def _project_matches(entry: dict[str, Any], query_project: str, strict: bool = False) -> bool:
    """项目命名空间匹配判定(召回硬隔离)。

    - query_project 为空: 不过滤(未指定项目的查询看到全部);
    - entry.project == query_project: 同项目, 保留;
    - entry.project 为空(未标记):
        * strict=False(默认/向后兼容): 视为全局记忆, 保留;
        * strict=True: 也排除(严格隔离, 指定项目时仅同名 project 可见);
    - entry.project 非空且不等: 属于其他项目, 排除。
    """
    if not query_project:
        return True
    entry_project = (entry.get("project", "") or "").strip()
    if not entry_project:
        return not strict
    return entry_project == query_project


class RecallService:
    """记忆召回服务实现。"""

    def __init__(self, deps: HandlerDependencies) -> None:
        self.deps = deps

    # ------------------------------------------------------------------
    # 同步入口
    # ------------------------------------------------------------------
    def handle(self, args: dict[str, Any]) -> RecallResult:
        """同步主动检索记忆，返回结构化结果。"""
        query = args["query"]
        mode = args.get("mode", "rag")
        max_tokens = args.get("max_tokens", 1500)
        user_id = args.get("user_id", "default")
        enable_trace = args.get("enable_trace", False)

        if self.deps.rbac and not self.deps.rbac.check_permission(user_id, "read"):
            return {"status": "blocked", "reason": f"User '{user_id}' lacks 'read' permission"}

        _query_keywords = _extract_query_keywords(query)
        # ★ 项目命名空间: 贯穿主召回/兜底/补充全链路做硬隔离
        _project = (args.get("project", "") or "").strip()

        recall_timeout = self.deps.config.get("recall_timeout_ms", 5000) / 1000.0
        top_k = args.get("top_k", 40)
        import time as _time

        _recall_start = _time.monotonic()

        # P2-1 多跳查询规划
        _planned = None
        if self.deps.knowledge_graph:
            try:
                from omnimem.handlers.query_planner import plan_and_search as _plan

                _planned = _plan(
                    query,
                    self.deps.retriever,
                    max_tokens=max_tokens,
                    mode=mode,
                    enable_trace=enable_trace,
                )
            except Exception as e:
                logger.debug("QueryPlanner skipped (%s), using standard path", e)

        if _planned:
            results = _planned
            _recall_latency_ms = (_time.monotonic() - _recall_start) * 1000.0
        else:
            future = _recall_executor.submit(
                self.deps.retriever.search,
                query,
                max_tokens=max_tokens,
                mode=mode,
                top_k=top_k,
                enable_trace=enable_trace,
            )
            try:
                results = future.result(timeout=recall_timeout)
            except TimeoutError:
                future.cancel()
                logger.warning(
                    "OmniMem recall timed out (%.1fs) query=%s, returning empty",
                    recall_timeout,
                    sanitize_for_log(query),
                )
                results = []
            except Exception as e:
                logger.error("OmniMem recall failed: %s query=%s", e, sanitize_for_log(query))
                results = []
            _recall_latency_ms = (_time.monotonic() - _recall_start) * 1000.0

        # ★ 自动联想扩散
        _should_spread = (
            mode == "associative"
            or (mode == "rag" and len(results) < 3)
        )
        if _should_spread:
            try:
                from omnimem.associative import AssociativeSpreader

                spreader = AssociativeSpreader(
                    knowledge_graph=self.deps.knowledge_graph,
                    retriever=self.deps.retriever,
                )
                _existing_ids = {r.get("memory_id", "") for r in results if r.get("memory_id")}
                _assocs = spreader.spread(
                    query=query,
                    existing_ids=_existing_ids,
                    top_k=5,
                )
                if _assocs:
                    logger.debug("AssociativeSpreader: %d associations found", len(_assocs))
                    results.extend(_assocs)
            except Exception as e:
                logger.warning("Associative spread failed (non-fatal): %s", e)

        # llm 模式补充通道
        if mode == "llm":
            results = self._apply_llm_store_supplement(
                results, query, _query_keywords, project=_project,
            )

        # 图谱检索通道
        results = self._apply_graph_channel(results, query)

        # 时序图谱检索通道
        _has_temporal_intent = any(kw in query for kw in _TEMPORAL_KEYWORDS)
        results = self._apply_temporal_channel(results, query, _has_temporal_intent)

        results = self.deps.temporal_decay.apply(results) if self.deps.temporal_decay else results
        results = self.deps.privacy.filter(results, session_id=self.deps.session_id) if self.deps.privacy else results

        # 主存储验证与过滤(含项目硬隔离)
        results = self._validate_store_entries(results, project=_project)

        # ★ 增强通道分数锚定: 防图谱/时序/联想裸分抢占直接命中的 Top-1
        results = self._rescale_enhancement_scores(results)

        # 最低相关性过滤
        results = self._filter_by_relevance(results, _query_keywords)

        # 结果不足 fallback
        results = self._fallback_if_few(results, query, _query_keywords, project=_project)

        if not results:
            return {
                "status": "no_results",
                "query": query,
                "message": "No relevant memories found.",
            }

        # 证据组富化
        results = self._enrich_evidence(results)

        # 启动效应加成
        results = self._apply_priming_boost(results)

        # 证据分组
        results = self._group_by_entities(results)

        # 冲突标注
        results = self._annotate_conflicts(results)

        # 按类型过滤
        type_filter = args.get("type_filter")
        if type_filter:
            results = self._filter_by_type(results, type_filter)

        # ContextManager 精炼
        refined = self.deps.context_manager.refine_recall_results(
            results, max_tokens=max_tokens, explain=args.get("explain", False)
        )

        # 提取检索轨迹
        trace_data = None
        if enable_trace and results:
            trace_data = results[-1].pop("_trace", None)

        # 质量评估
        quality_data = self._record_quality(
            query, results, refined, _recall_latency_ms, enable_trace
        )

        if self.deps.audit_logger:
            self.deps.audit_logger.log(
                "recall",
                details={"query": sanitize_for_log(query), "mode": mode, "count": len(refined)},
                result="success",
                instance_id=self.deps.instance_id,
            )

        # 召回反馈循环
        self._record_recall_feedback(refined)

        # 记录启动效应
        self._record_priming(results)

        response: RecallResult = {
            "status": "found",
            "query": query,
            "count": len(refined),
            "memories": refined,
            "hint": "Use omni_detail with a memory_id to fetch full content.",
        }
        if trace_data:
            response["trace"] = trace_data
        if quality_data:
            response["_quality"] = quality_data
        return response

    # ------------------------------------------------------------------
    # 异步入口
    # ------------------------------------------------------------------
    async def async_handle(self, args: dict[str, Any]) -> RecallResult:
        """异步主动检索记忆，返回结构化结果。"""
        import time as _time

        query = args["query"]
        mode = args.get("mode", "rag")
        max_tokens = args.get("max_tokens", 1500)
        user_id = args.get("user_id", "default")
        enable_trace = args.get("enable_trace", False)

        if self.deps.rbac and not self.deps.rbac.check_permission(user_id, "read"):
            return {"status": "blocked", "reason": f"User '{user_id}' lacks 'read' permission"}

        _query_keywords = _extract_query_keywords(query)
        # ★ 项目命名空间(异步路径, 与同步一致)
        _project = (args.get("project", "") or "").strip()

        recall_timeout = self.deps.config.get("recall_timeout_ms", 5000) / 1000.0
        _recall_start = _time.monotonic()
        try:
            results = await asyncio.wait_for(
                self.deps.retriever.async_search(
                    query,
                    max_tokens=max_tokens,
                    mode=mode,
                    enable_trace=enable_trace,
                ),
                timeout=recall_timeout,
            )
        except TimeoutError:
            logger.warning(
                "OmniMem async recall timed out (%.1fs) query=%s, returning empty",
                recall_timeout,
                sanitize_for_log(query),
            )
            results = []
        except Exception as e:
            logger.error("OmniMem async recall failed: %s query=%s", e, sanitize_for_log(query))
            results = []
        _recall_latency_ms = (_time.monotonic() - _recall_start) * 1000.0
        record_recall_duration(_recall_latency_ms / 1000.0)

        if mode == "llm":
            results = await self._async_apply_llm_store_supplement(
                results, query, _query_keywords, project=_project,
            )

        graph_results, temporal_results = await asyncio.gather(
            self._async_graph_search(query),
            self._async_temporal_search(query, _has_temporal_intent=any(kw in query for kw in _TEMPORAL_KEYWORDS)),
        )
        results.extend(graph_results)
        results.extend(temporal_results)

        if self.deps.temporal_decay:
            results = await asyncio.to_thread(self.deps.temporal_decay.apply, results)
        if self.deps.privacy:
            results = await asyncio.to_thread(self.deps.privacy.filter, results, session_id=self.deps.session_id)

        results = await self._async_validate_store_entries(results, project=_project)
        # ★ 增强通道分数锚定(与同步一致)
        results = self._rescale_enhancement_scores(results)
        results = self._filter_by_relevance(results, _query_keywords)
        results = await self._async_fallback_if_few(results, query, _query_keywords, project=_project)

        # ★ 自动联想扩散（异步路径）
        _should_spread = (
            mode == "associative"
            or (mode == "rag" and len(results) < 3)
        )
        if _should_spread:
            try:
                from omnimem.associative import AssociativeSpreader

                _spreader = AssociativeSpreader(
                    knowledge_graph=self.deps.knowledge_graph,
                    retriever=self.deps.retriever,
                )
                _existing_ids = {r.get("memory_id", "") for r in results if r.get("memory_id")}
                _assocs = _spreader.spread(
                    query=query,
                    existing_ids=_existing_ids,
                    top_k=5,
                )
                if _assocs:
                    logger.debug("AssociativeSpreader (async): %d associations found", len(_assocs))
                    results.extend(_assocs)
                    # 重新过滤+fallback（联想结果也要经过校验）
                    results = self._filter_by_relevance(results, _query_keywords)
                    results = await self._async_fallback_if_few(results, query, _query_keywords)
            except Exception as e:
                logger.warning("Async associative spread failed (non-fatal): %s", e)

        # 按类型过滤（异步路径）
        type_filter = args.get("type_filter")
        if type_filter:
            results = self._filter_by_type(results, type_filter)

        if not results:
            return {
                "status": "no_results",
                "query": query,
                "message": "No relevant memories found.",
            }

        refined = await asyncio.to_thread(
            functools.partial(
                self.deps.context_manager.refine_recall_results,
                results, max_tokens=max_tokens, explain=args.get("explain", False),
            )
        )

        trace_data = None
        if enable_trace and results:
            trace_data = results[-1].pop("_trace", None)

        quality_data = None
        if enable_trace and self.deps.quality_evaluator:
            try:
                from dataclasses import asdict

                from omnimem.retrieval.quality_eval import RetrievalQualityEvaluator

                relevant_ids = RetrievalQualityEvaluator.infer_relevant_ids(results)
                metrics = await asyncio.to_thread(
                    self.deps.quality_evaluator.evaluate,
                    query=query,
                    results=results,
                    relevant_ids=relevant_ids,
                    latency_ms=_recall_latency_ms,
                )
                await asyncio.to_thread(self.deps.quality_evaluator.record_evaluation, metrics)
                quality_data = asdict(metrics)
            except Exception as e:
                logger.warning("异步质量评估记录失败: %s", e)

        if self.deps.audit_logger:
            await asyncio.to_thread(
                self.deps.audit_logger.log,
                "recall",
                details={"query": sanitize_for_log(query), "mode": mode, "count": len(refined)},
                result="success",
                instance_id=self.deps.instance_id,
            )

        for r in refined:
            mid = r.get("memory_id", "")
            if mid:
                try:
                    await asyncio.to_thread(self.deps.forgetting.record_access, mid)
                except Exception as e:
                    logger.debug("async recall feedback record_access failed for %s: %s", mid, e)

        response: RecallResult = {
            "status": "found",
            "query": query,
            "count": len(refined),
            "memories": refined,
            "hint": "Use omni_detail with a memory_id to fetch full content.",
        }
        if trace_data:
            response["trace"] = trace_data
        if quality_data:
            response["_quality"] = quality_data
        return response

    # ------------------------------------------------------------------
    # 结果处理辅助（同步）
    # ------------------------------------------------------------------
    def _apply_llm_store_supplement(
        self,
        results: list[dict[str, Any]],
        query: str,
        query_keywords: set[str],
        project: str = "",
    ) -> list[dict[str, Any]]:
        """llm 模式下从 store 补充关键词相关结果。

        ★ 项目硬隔离: 若调用方指定 project, 仅补充同 project 或未标记(全局)
        的记忆; 其他项目的记忆被排除, 根治跨项目泛化混淆。
        """
        try:
            expanded_queries = [query]
            for key, synonyms in _SYNONYM_MAP.items():
                if key in query:
                    for syn in synonyms:
                        expanded_queries.append(query.replace(key, syn))

            all_store_results = []
            existing_ids = {r.get("memory_id", "") for r in results}
            for eq in expanded_queries:
                all_store_results.extend(self.deps.store.search_by_content(eq, limit=5))

            seen = set(existing_ids)
            for sr in all_store_results:
                mid = sr.get("memory_id", "")
                if mid in seen:
                    continue
                seen.add(mid)
                if not _project_matches(sr, project, self._is_project_strict()):
                    continue
                sr_content = sr.get("content", "").lower()
                if query_keywords:
                    overlap_count = sum(1 for kw in query_keywords if kw in sr_content)
                    if overlap_count >= 1:
                        sr["_source"] = "store_supplement"
                        sr["score"] = 0.3
                        results.append(sr)
        except (TimeoutError, ConnectionError) as e:
            logger.warning("OmniMem llm store supplement failed: %s", e)
        return results

    def _apply_graph_channel(
        self, results: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """合并知识图谱检索结果。"""
        if not self.deps.knowledge_graph:
            return results
        try:
            graph_rag_ctx = self.deps.knowledge_graph.graph_rag_search(query, max_depth=2)
            if graph_rag_ctx:
                results.append({
                    "content": graph_rag_ctx,
                    "type": "graph_rag",
                    "confidence": 0.6,
                    "score": 0.5,
                    "_source": "graph_rag",
                })
                return results
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.debug("graph_rag_search failed, fallback to graph_search: %s", e)

        try:
            graph_results = self.deps.knowledge_graph.graph_search(query, max_depth=2, limit=10)
            if graph_results:
                for gr in graph_results[:5]:
                    gr["content"] = (
                        f"{gr.get('subject', '')} {gr.get('predicate', '')} {gr.get('object', '')}"
                    )
                    gr["type"] = "graph_triple"
                    gr["confidence"] = gr.get("confidence", 0.5)
                results.extend(graph_results[:5])
        except (RuntimeError, ValueError) as e2:
            logger.warning("OmniMem graph recall failed: %s", e2)
        return results

    def _apply_temporal_channel(
        self,
        results: list[dict[str, Any]],
        query: str,
        _has_temporal_intent: bool,
    ) -> list[dict[str, Any]]:
        """合并时序图谱检索结果。"""
        if not (_has_temporal_intent and self.deps.temporal_kg):
            return results
        try:
            from omnimem.deep.kg import extract_entities as _kg_extract_entities

            query_entities = _kg_extract_entities(query)
            if query_entities:
                temporal_ctx = self.deps.temporal_kg.temporal_rag_context(query_entities)
                if temporal_ctx:
                    results.append({
                        "content": temporal_ctx,
                        "type": "temporal_kg",
                        "confidence": 0.7,
                        "score": 0.55,
                        "_source": "temporal_kg",
                    })
        except (RuntimeError, ValueError, AttributeError, ImportError) as e:
            logger.warning("OmniMem temporal KG recall failed: %s", e)
        return results

    def _validate_store_entries(
        self, results: list[dict[str, Any]], project: str = ""
    ) -> list[dict[str, Any]]:
        """过滤索引残留，封存记忆降权保留(判定逻辑见 _apply_lifecycle)。"""
        valid_results = []
        for r in results:
            # ★ 主路径结果补默认来源标记(联想/兜底等增强路径各自带标)
            r.setdefault("_source", "fusion")
            mid = r.get("memory_id", "")
            if mid:
                entry = self.deps.store.get(mid)
                if not self._apply_lifecycle(r, mid, entry, project):
                    continue
            valid_results.append(r)
        return valid_results

    # 增强通道(图谱/时序/联想)标识: 这些结果在 recall 后直接追加,
    # 携带裸分(0.5~0.55), 与 RRF 主路径分(~0.05)不同量级。
    _ENHANCEMENT_SOURCES: frozenset[str] = frozenset(
        {"graph_rag", "graph_triple", "temporal_kg", "association"}
    )

    @classmethod
    def _is_enhancement(cls, r: dict[str, Any]) -> bool:
        return (
            r.get("_source") in cls._ENHANCEMENT_SOURCES
            or r.get("type") in cls._ENHANCEMENT_SOURCES
        )

    def _rescale_enhancement_scores(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """将增强通道分数锚定到主路径 Top-1 之下, 修复尺度错配导致的抢位。

        背景: 图谱/时序/联想结果携带 0.5~0.55 裸分, 而 RRF 主检索 rank-1 仅 ~0.05
        (未启用 reranker 时), 导致增强通道在按 score 排序时压过直接语义命中(R11 #4/#5)。

        策略: 以主路径(_source=='fusion')的最高分为锚, 将增强通道封顶在 anchor*0.95
        之下——直接命中稳坐 Top-1, 增强结果作为紧随其后的补充; 若主路径无强结果
        (稀疏/空), 保留增强原分, 不破坏"结果不足时联想填补"的能力。
        """
        anchor = max(
            (
                float(r.get("score", 0) or 0)
                for r in results
                if r.get("_source") == "fusion" and not self._is_enhancement(r)
            ),
            default=0.0,
        )
        if anchor <= 0:
            return results
        cap = anchor * 0.95
        for r in results:
            if self._is_enhancement(r) and float(r.get("score", 0) or 0) > cap:
                r["score"] = round(cap, 5)
        return results

    def _is_project_strict(self) -> bool:
        """读取项目召回严格隔离开关(空标签是否也排除)。"""
        _cfg = getattr(self.deps, "config", None)
        if _cfg is None:
            return False
        try:
            return bool(_cfg.get("project_recall_strict", False))
        except Exception:
            return False

    def _apply_lifecycle(
        self, r: dict[str, Any], mid: str, entry: Any, project: str = ""
    ) -> bool:
        """生命周期判定唯一实现(同步/异步 validate 共享, 防双实现漂移): 返回是否保留。

        entry 缺失(索引残留)剔除; 跨项目(project 不匹配)剔除; forgotten 剔除;
        archived 按 archive_recall_policy:
          - downweight(默认): 降权 0.3x + sealed 标记(数据可见但排序靠后)
          - exclude: 彻底排除(九轮测试实锤: 封存记忆残留持续污染稀疏库召回)
        """
        if not entry:
            return False
        # ★ 项目硬隔离: 主召回路径(vector+BM25 融合/联想/图谱)统一在此过滤,
        #   查询指定 project 时排除其他项目记忆(strict 开启时未标记条目也排除)。
        if not _project_matches(entry, project, self._is_project_strict()):
            return False
        _cfg = getattr(self.deps, "config", None)
        _exclude_archived = (
            _cfg.get("archive_recall_policy", "downweight") == "exclude"
            if _cfg is not None else False
        )
        if entry.get("archived"):
            if _exclude_archived:
                return False
            r["score"] = r.get("score", 0) * 0.3
            r["sealed"] = True
            return True
        # store 条目不含 archived 字段时, 权威生命周期状态在 forgetting_state
        if self.deps.forgetting is not None:
            try:
                _stage = self.deps.forgetting.get_stage(mid)
            except Exception:
                _stage = "active"
            if _stage == "forgotten":
                return False
            if _stage == "archived":
                if _exclude_archived:
                    return False
                r["score"] = r.get("score", 0) * 0.3
                r["sealed"] = True
        return True

    def _filter_by_relevance(
        self, results: list[dict[str, Any]], query_keywords: set[str]
    ) -> list[dict[str, Any]]:
        """统一最低相关性过滤。"""
        filtered = []
        for r in results:
            score = r.get("score", 0)
            if score <= 0:
                continue
            source = r.get("_source", "")
            if source == "store_supplement":
                filtered.append(r)
                continue
            if r.get("type") == "graph_triple":
                content = r.get("content", "").lower()
                if query_keywords and any(kw in content for kw in query_keywords):
                    filtered.append(r)
                continue
            if source == "temporal_kg":
                filtered.append(r)
                continue
            # ★ 联想扩散结果绕过融合层语义地板, 须与查询有关键词重叠才保留
            #   (与 graph_triple 同等要求, 防止无关查询经实体扩散捞回弱邻居)
            if source == "association":
                content = r.get("content", "").lower()
                if query_keywords and any(kw in content for kw in query_keywords):
                    filtered.append(r)
                continue
            # ★ 缺陷3修复: 偏好类记忆查询相关性门控 —— 覆盖主检索之外的增强路径
            #   (联想扩散/图谱/兜底). 泛化偏好与查询无关键词重叠时一律过滤,
            #   避免"用户偏好使用中文进行交互"这类记忆污染无关查询。
            if r.get("type") in ("preference", "preferences"):
                content = r.get("content", "").lower()
                has_overlap = bool(query_keywords) and any(kw in content for kw in query_keywords)
                if not has_overlap:
                    continue
            if score < 0.025:
                if query_keywords:
                    content = r.get("content", "").lower()
                    has_overlap = any(kw in content for kw in query_keywords)
                    if not has_overlap:
                        continue
                else:
                    continue
            filtered.append(r)
        return filtered

    def _is_inactive(self, memory_id: str) -> bool:
        """生命周期检查: forgotten/archived 记忆不得经兜底路径复活(R5-Q9 旧污染泄漏)。

        主召回路径对 archived 降权保留(见 _apply_lifecycle), 但兜底路径是
        "结果不足"时的救援通道, 若放行 archived 会以 0.2-0.35 兜底分全权重
        复活封存记忆, 在负对照/稀疏查询中污染结果(八轮测试实锤缺陷)。
        """
        if not memory_id or self.deps.forgetting is None:
            return False
        try:
            return self.deps.forgetting.get_stage(memory_id) in ("archived", "forgotten")
        except Exception:
            return False

    def _admit_fts_fallback(
        self, sf: dict[str, Any], existing_ids: set[str], project: str = ""
    ) -> bool:
        """FTS 兜底准入判定唯一实现(同步/异步共享): 去重+跨项目+archived/forgotten 拦截+打标。"""
        sf_mid = sf.get("memory_id", "")
        if sf_mid in existing_ids or self._is_inactive(sf_mid):
            return False
        if not _project_matches(sf, project, self._is_project_strict()):
            return False
        sf["_source"] = "store_fts_fallback"
        sf["score"] = sf.get("score", 0) or 0.2
        return True

    def _admit_store_fallback(
        self, sf: dict[str, Any], existing_ids: set[str], query_keywords: set[str],
        project: str = "",
    ) -> bool:
        """store 全量扫描兜底准入判定唯一实现: 须关键词命中、同项目且非 archived/forgotten。"""
        sf_mid = sf.get("memory_id", "")
        if sf_mid in existing_ids or self._is_inactive(sf_mid):
            return False
        if not _project_matches(sf, project, self._is_project_strict()):
            return False
        sf_content = sf.get("content", "").lower()
        keyword_hits = sum(1 for kw in query_keywords if kw in sf_content)
        if keyword_hits < 1:
            return False
        sf["_source"] = "store_fallback"
        sf["score"] = min(0.15 + keyword_hits * 0.05, 0.35)
        return True

    def _fallback_if_few(
        self,
        results: list[dict[str, Any]],
        query: str,
        query_keywords: set[str],
        project: str = "",
    ) -> list[dict[str, Any]]:
        """结果不足时 fallback 到 store 关键词匹配(准入判定见 _admit_*)。"""
        if len(results) >= 5 or not query_keywords:
            return results

        existing_ids = {r.get("memory_id", "") for r in results}

        # 优先 FTS5
        try:
            fts_results = self.deps.store.search_by_content(query, limit=10)
            for sf in fts_results:
                if self._admit_fts_fallback(sf, existing_ids, project):
                    results.append(sf)
                    existing_ids.add(sf.get("memory_id", ""))
                    if len(results) >= 5:
                        return results
        except Exception as e:
            logger.warning("OmniMem FTS fallback failed: %s", e)

        # 补充 store 全量扫描
        if len(results) < 5:
            try:
                store_all = self.deps.store.search(limit=50)
                for sf in store_all:
                    if self._admit_store_fallback(sf, existing_ids, query_keywords, project):
                        results.append(sf)
                        existing_ids.add(sf.get("memory_id", ""))
                        if len(results) >= 5:
                            break
            except Exception as e:
                logger.warning("OmniMem store fallback failed: %s", e)

        return results

    def _enrich_evidence(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """为每条结果富化证据元数据。"""
        for r in results:
            mid = r.get("memory_id", "")
            if mid and self.deps.store:
                try:
                    entry = self.deps.store.get(mid)
                    if entry:
                        r.setdefault("stored_at", entry.get("stored_at", ""))
                        r.setdefault("entities", entry.get("entities", []))
                        r.setdefault("provenance", entry.get("provenance", ""))
                        r.setdefault("type", entry.get("type", r.get("type", "fact")))
                        r.setdefault("confidence", entry.get("confidence", r.get("confidence", 3)))
                        r.setdefault("privacy", entry.get("privacy", ""))
                        r.setdefault("scope", entry.get("scope", ""))
                        r["_evidence_enriched"] = True
                except Exception:
                    pass
        return results

    def _apply_priming_boost(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """启动效应加成。"""
        try:
            from omnimem.handlers.priming import get_priming_state

            _priming = get_priming_state(self.deps.session_id)
            _priming.apply_boost(results)
        except Exception as e:
            logger.debug("Priming boost skipped: %s", e)
        return results

    def _group_by_entities(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按共享实体分组。"""
        if len(results) < 3:
            return results

        entity_groups: dict[str, list[int]] = {}
        orphan_indices: list[int] = []
        for i, r in enumerate(results):
            ents = r.get("entities", [])
            if isinstance(ents, str):
                try:
                    ents = json.loads(ents)
                except (json.JSONDecodeError, TypeError):
                    ents = [ents] if ents else []
            if ents:
                key_entity = ents[0]
                if key_entity not in entity_groups:
                    entity_groups[key_entity] = []
                entity_groups[key_entity].append(i)
            else:
                orphan_indices.append(i)

        grouped_results: list[dict[str, Any]] = []
        for _, indices in entity_groups.items():
            group_items = [results[i] for i in indices]
            group_items.sort(key=lambda x: x.get("stored_at", ""), reverse=True)
            if len(group_items) > 1:
                group_items[0]["_group_start"] = True
                group_items[0]["_group_size"] = len(group_items)
            grouped_results.extend(group_items)
        for i in orphan_indices:
            grouped_results.append(results[i])
        return grouped_results

    def _annotate_conflicts(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """标注已记录的冲突。"""
        for r in results:
            mid = r.get("memory_id", "")
            if mid and self.deps.store:
                try:
                    entry = self.deps.store.get(mid)
                    if entry:
                        conflict_target = entry.get("conflicting_with", "")
                        if conflict_target:
                            r["_conflict"] = {
                                "conflicting_with": conflict_target,
                                "conflict_type": entry.get("conflict_type", "unknown"),
                            }
                except Exception:
                    pass
        return results

    def _filter_by_type(self, results: list[dict[str, Any]], type_filter: list[str]) -> list[dict[str, Any]]:
        """按记忆类型过滤结果，保留指定类型的条目。

        Args:
            results: 检索结果列表
            type_filter: 要保留的记忆类型列表（如 ['preference', 'fact']）

        Returns:
            过滤后的结果列表（保留系统类型如 graph_rag, temporal_kg, association 等）
        """
        type_set = set(type_filter)
        filtered = []
        for r in results:
            rtype = r.get("type", "")
            # 系统内部类型（图谱、时序、联想）总是保留
            if rtype in {"graph_rag", "graph_triple", "temporal_kg", "association", "store_supplement", "store_fts_fallback", "store_fallback"} or rtype in type_set or not rtype:
                filtered.append(r)
        logger.debug(
            "_filter_by_type: %d -> %d results (filter=%s)",
            len(results), len(filtered), type_filter,
        )
        return filtered

    def _record_quality(
        self,
        query: str,
        results: list[dict[str, Any]],
        refined: list[dict[str, Any]],
        latency_ms: float,
        enable_trace: bool,
    ) -> dict[str, Any] | None:
        """记录检索质量评估。"""
        if not (enable_trace and self.deps.quality_evaluator):
            return None
        try:
            from dataclasses import asdict

            from omnimem.retrieval.quality_eval import RetrievalQualityEvaluator

            relevant_ids = RetrievalQualityEvaluator.infer_relevant_ids(results)
            metrics = self.deps.quality_evaluator.evaluate(
                query=query,
                results=results,
                relevant_ids=relevant_ids,
                latency_ms=latency_ms,
            )
            self.deps.quality_evaluator.record_evaluation(metrics)
            return asdict(metrics)
        except Exception as e:
            logger.warning("质量评估记录失败: %s", e)
        return None

    def _record_recall_feedback(self, refined: list[dict[str, Any]]) -> None:
        """记录召回反馈到遗忘曲线。"""
        for r in refined:
            mid = r.get("memory_id", "")
            if mid:
                try:
                    self.deps.forgetting.record_access(mid)
                except Exception as e:
                    logger.debug("recall feedback record_access failed for %s: %s", mid, e)

    def _record_priming(self, results: list[dict[str, Any]]) -> None:
        """记录本次命中实体到启动效应缓存。"""
        try:
            from omnimem.handlers.priming import get_priming_state

            _priming = get_priming_state(self.deps.session_id)
            _collected = []
            for _r in results or []:
                _ents = _r.get("entities", [])
                if isinstance(_ents, str):
                    try:
                        import json as _json

                        _ents = _json.loads(_ents)
                    except Exception:
                        _ents = [_ents] if _ents else []
                if isinstance(_ents, list):
                    _collected.extend(_ents)
            if _collected:
                _priming.record(_collected)
        except Exception as e:
            logger.debug("Priming record skipped: %s", e)

    # ------------------------------------------------------------------
    # 异步辅助
    # ------------------------------------------------------------------
    async def _async_apply_llm_store_supplement(
        self,
        results: list[dict[str, Any]],
        query: str,
        query_keywords: set[str],
        project: str = "",
    ) -> list[dict[str, Any]]:
        """异步 llm 模式 store 补充(项目隔离语义与同步版共享 _project_matches)。"""
        try:
            expanded_queries = [query]
            for key, synonyms in _SYNONYM_MAP.items():
                if key in query:
                    for syn in synonyms:
                        expanded_queries.append(query.replace(key, syn))

            store_tasks = [
                asyncio.to_thread(self.deps.store.search_by_content, eq, limit=5)
                for eq in expanded_queries
            ]
            store_results_list = await asyncio.gather(*store_tasks, return_exceptions=True)

            all_store_results: list[dict[str, Any]] = []
            for sr_list in store_results_list:
                if isinstance(sr_list, list):
                    all_store_results.extend(sr_list)

            existing_ids = {r.get("memory_id", "") for r in results}
            seen = set(existing_ids)
            for sr in all_store_results:
                mid = sr.get("memory_id", "")
                if mid in seen:
                    continue
                seen.add(mid)
                if not _project_matches(sr, project, self._is_project_strict()):
                    continue
                sr_content = sr.get("content", "").lower()
                if query_keywords:
                    overlap_count = sum(1 for kw in query_keywords if kw in sr_content)
                    if overlap_count >= 1:
                        sr["_source"] = "store_supplement"
                        sr["score"] = 0.3
                        results.append(sr)
        except (TimeoutError, ConnectionError) as e:
            logger.warning("OmniMem async llm store supplement failed: %s", e)
        return results

    async def _async_graph_search(self, query: str) -> list[dict[str, Any]]:
        """异步图谱检索通道。"""
        graph_results: list[dict[str, Any]] = []
        if not self.deps.knowledge_graph:
            return graph_results
        try:
            graph_rag_ctx = await asyncio.to_thread(
                self.deps.knowledge_graph.graph_rag_search, query, max_depth=2
            )
            if graph_rag_ctx:
                graph_results.append({
                    "content": graph_rag_ctx,
                    "type": "graph_rag",
                    "confidence": 0.6,
                    "score": 0.5,
                    "_source": "graph_rag",
                })
                return graph_results
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.debug("async graph_rag_search failed, fallback: %s", e)

        try:
            raw_graph_results = await asyncio.to_thread(
                self.deps.knowledge_graph.graph_search, query, max_depth=2, limit=10
            )
            if raw_graph_results:
                for gr in raw_graph_results[:5]:
                    gr["content"] = (
                        f"{gr.get('subject', '')} {gr.get('predicate', '')} {gr.get('object', '')}"
                    )
                    gr["type"] = "graph_triple"
                    gr["confidence"] = gr.get("confidence", 0.5)
                graph_results.extend(raw_graph_results[:5])
        except (RuntimeError, ValueError) as e2:
            logger.warning("OmniMem async graph recall failed: %s", e2)
        return graph_results

    async def _async_temporal_search(
        self, query: str, *, _has_temporal_intent: bool
    ) -> list[dict[str, Any]]:
        """异步时序图谱检索通道。"""
        temporal_results: list[dict[str, Any]] = []
        if not (_has_temporal_intent and self.deps.temporal_kg):
            return temporal_results
        try:
            from omnimem.deep.kg import extract_entities as _kg_extract_entities

            query_entities = await asyncio.to_thread(_kg_extract_entities, query)
            if query_entities:
                temporal_ctx = await asyncio.to_thread(
                    self.deps.temporal_kg.temporal_rag_context, query_entities
                )
                if temporal_ctx:
                    temporal_results.append({
                        "content": temporal_ctx,
                        "type": "temporal_kg",
                        "confidence": 0.7,
                        "score": 0.55,
                        "_source": "temporal_kg",
                    })
        except (RuntimeError, ValueError, AttributeError, ImportError) as e:
            logger.warning("OmniMem async temporal KG recall failed: %s", e)
        return temporal_results

    async def _async_validate_store_entries(
        self, results: list[dict[str, Any]], project: str = ""
    ) -> list[dict[str, Any]]:
        """异步主存储验证(判定逻辑与同步版共享 _apply_lifecycle, 杜绝漂移)。"""
        valid_results = []
        for r in results:
            r.setdefault("_source", "fusion")
            mid = r.get("memory_id", "")
            if mid:
                entry = await asyncio.to_thread(self.deps.store.get, mid)
                if not self._apply_lifecycle(r, mid, entry, project):
                    continue
            valid_results.append(r)
        return valid_results

    async def _async_fallback_if_few(
        self,
        results: list[dict[str, Any]],
        query: str,
        query_keywords: set[str],
        project: str = "",
    ) -> list[dict[str, Any]]:
        """异步结果不足 fallback。"""
        if len(results) >= 5 or not query_keywords:
            return results

        existing_ids = {r.get("memory_id", "") for r in results}
        try:
            fts_results = await asyncio.to_thread(self.deps.store.search_by_content, query, limit=10)
            for sf in fts_results:
                if self._admit_fts_fallback(sf, existing_ids, project):
                    results.append(sf)
                    existing_ids.add(sf.get("memory_id", ""))
                    if len(results) >= 5:
                        return results
        except Exception as e:
            logger.warning("OmniMem async FTS fallback failed: %s", e)

        if len(results) < 5:
            try:
                store_all = await asyncio.to_thread(self.deps.store.search, limit=50)
                for sf in store_all:
                    if self._admit_store_fallback(sf, existing_ids, query_keywords, project):
                        results.append(sf)
                        existing_ids.add(sf.get("memory_id", ""))
                        if len(results) >= 5:
                            break
            except Exception as e:
                logger.warning("OmniMem async store fallback failed: %s", e)
        return results
