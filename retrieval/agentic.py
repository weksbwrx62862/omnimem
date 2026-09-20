"""Agentic 迭代检索（改进项 #9）。

在单次混合检索之上加一层 ReAct 式「检索→判据→再检索」循环：把复杂/多跳问题
分解为子查询，逐轮补检并 RRF 融合，直到满足充分性判据或触顶步数。

设计约束（保持纯增量、可离线测）：
- 不改动 retrieval/engine.py 与 handlers/recall.py；通过注入 retriever
  （任何带 .search(query, top_k=, mode=) 的对象，HybridRetriever 即符合）组合使用；
- 分解与判据默认走确定性启发式，可注入 rewrite_fn/judge_fn 接 LLM（AsyncLLMClient），
  无凭证环境自动降级；
- 融合复用 retrieval.rrf.RRFFusion，评分口径与主检索一致。
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from omnimem.retrieval.rrf import RRFFusion

logger = logging.getLogger(__name__)

# 分解时剔除的中英停用词（保守小集合，避免过度工程）。
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "for", "on", "with", "that", "this", "it", "as", "at", "by", "be", "从", "到",
    "的", "了", "和", "与", "是", "在", "吗", "什么", "如何", "为什么", "请",
}
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+")


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS and len(t) >= 1]


@dataclass
class AgenticConfig:
    """迭代检索参数。"""

    max_steps: int = 3            # 最多检索轮次（含首轮）
    queries_per_step: int = 3     # 每轮并行的子查询上限
    top_k: int = 20               # 每个子查询的候选数
    final_k: int = 10             # 融合后返回条数
    mode: str = "hybrid"          # 透传给底层检索
    min_hits: int = 3             # 启发式判据：去重后命中的最少条数


@dataclass
class StepRecord:
    step: int
    queries: list[str]
    new_hits: int
    total_hits: int


@dataclass
class AgenticResult:
    results: list[dict[str, Any]]
    steps: list[StepRecord] = field(default_factory=list)

    def to_recall_dict(self, *, explain: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "ok",
            "results": self.results,
            "count": len(self.results),
            "mode": "agentic",
            "steps": len(self.steps),
        }
        if explain:
            payload["_agentic"] = {
                "steps": [
                    {
                        "step": s.step,
                        "queries": s.queries,
                        "new_hits": s.new_hits,
                        "total_hits": s.total_hits,
                    }
                    for s in self.steps
                ]
            }
        return payload


# 可注入的分解/判据函数签名
RewriteFn = Callable[[str, list[dict[str, Any]], int], list[str]]
JudgeFn = Callable[[str, list[dict[str, Any]]], bool]


class AgenticRetriever:
    """迭代式 agentic 检索编排器。"""

    def __init__(
        self,
        retriever: Any,
        *,
        rewrite_fn: RewriteFn | None = None,
        judge_fn: JudgeFn | None = None,
        fusion: RRFFusion | None = None,
        config: AgenticConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._rewrite = rewrite_fn or self._default_rewrite
        self._judge = judge_fn or self._default_judge
        # ★ 跨查询融合：每路 result_list 已是底层混合检索的成品（各自已过
        #   min_rrf/rerank 质量门），此处只做「多视角投票」合并，故不再叠加
        #   阈值过滤（min_rrf=0.0），避免过度筛掉后续轮次补检到的新命中。
        self._fusion = fusion or RRFFusion(k=60, min_rrf=0.0)
        self._cfg = config or AgenticConfig()

    # ── 启发式默认实现 ────────────────────────────────
    def _default_judge(self, query: str, gathered: list[dict[str, Any]]) -> bool:
        return len(gathered) >= self._cfg.min_hits

    def _default_rewrite(self, query: str, gathered: list[dict[str, Any]], step: int) -> list[str]:
        """无 LLM 时的确定性多跳分解：用已检索命中的高频新词补出后续查询。"""
        q_tokens = set(_tokens(query))
        freq: dict[str, int] = {}
        for doc in gathered:
            content = str(doc.get("content", ""))
            for tok in _tokens(content):
                freq[tok] = freq.get(tok, 0) + 1
        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        bridges = [tok for tok, _cnt in ranked if tok not in q_tokens][: self._cfg.queries_per_step]
        return [f"{query} {b}".strip() for b in bridges]

    # ── 核心循环 ──────────────────────────────────────
    def run(self, query: str) -> AgenticResult:
        if not (query or "").strip():
            return AgenticResult(results=[], steps=[])

        seen_queries: set[str] = set()
        # 每个 memory_id 保留跨查询累加前的最优单查询结果，供 RRF 融合排名。
        by_id: dict[str, dict[str, Any]] = {}
        result_lists: list[list[dict[str, Any]]] = []
        steps: list[StepRecord] = []

        queue = [query]
        for step in range(1, self._cfg.max_steps + 1):
            batch = [q for q in queue if q not in seen_queries][: self._cfg.queries_per_step]
            if not batch:
                break
            for q in batch:
                seen_queries.add(q)
            before = len(by_id)
            batch_lists: list[list[dict[str, Any]]] = []
            for q in batch:
                hits = self._search(q)
                batch_lists.append(hits)
                for rank, doc in enumerate(hits):
                    doc_id = str(doc.get("memory_id") or doc.get("id") or f"hash-{hash(str(doc.get('content','')))}")
                    if doc_id not in by_id:
                        by_id[doc_id] = dict(doc)
                        by_id[doc_id].setdefault("memory_id", doc_id)
            result_lists.extend(batch_lists)
            gathered = list(by_id.values())
            steps.append(StepRecord(step=step, queries=list(batch), new_hits=len(by_id) - before, total_hits=len(by_id)))

            if self._judge(query, gathered):
                break
            if step >= self._cfg.max_steps:
                break
            nxt = self._rewrite(query, gathered, step) or []
            queue = [q for q in nxt if q not in seen_queries]

        merged = (
            self._fusion.merge(result_lists, weights=[1.0] * len(result_lists))
            if result_lists
            else []
        )
        # merge 已按融合分排序并按 memory_id 去重；截断到 final_k。
        return AgenticResult(results=merged[: self._cfg.final_k], steps=steps)

    def _search(self, query: str) -> list[dict[str, Any]]:
        try:
            hits = self._retriever.search(query, top_k=self._cfg.top_k, mode=self._cfg.mode)
        except Exception as e:  # 底层检索异常不应中断整轮编排
            logger.warning("agentic search 失败 query=%r: %s", query, e)
            return []
        return list(hits or [])
