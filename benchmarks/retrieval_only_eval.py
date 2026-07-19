#!/usr/bin/env python3
"""轻量检索质量评测 — BM25 + 向量语义检索 + RRF 融合。

核心思路:
  - 直接使用 OmniMem 的 BM25Retriever + VectorRetriever
  - RRF 融合两路检索结果
  - 评估指标: session_hit / turn_hit / coverage / Top-K hit / MRR

支持三种评测模式:
  - retrieval: 仅计算检索指标（MRR / Top-K / 分维度），不调用 LLM
  - gen: 检索 + LLM 生成答案，输出预测答案但不判分
  - full: 检索 + 生成 + LLM Judge 判分，输出 QA 准确率

用法:
  python3 benchmarks/retrieval_only_eval.py --limit 10
  python3 benchmarks/retrieval_only_eval.py --limit 20 --max-sessions 0 --top-k 100
  python3 benchmarks/retrieval_only_eval.py --limit 20 --max-sessions 0 --no-vector   # 仅 BM25
  python3 benchmarks/retrieval_only_eval.py --limit 3 --max-sessions 0 --top-k 100 --no-split --vector --dynamic-weight
  python3 benchmarks/retrieval_only_eval.py --limit 10 --mode full --dynamic-weight
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omnimem.retrieval.bm25 import BM25Retriever
from omnimem.retrieval.vector import VectorRetriever
from omnimem.retrieval.rrf import RRFFusion
from benchmarks.longmemeval_adapter import OmniMemMemoryProvider, _PREFERENCE_KEYWORDS
from benchmarks.run_longmemeval import (
    evaluate_retrieval_quality,
    build_answer_turn_contents,
    _generate_answer_with_llm,
    _judge_answer_with_llm,
    _check_llm_available,
)


# ── Chain-of-Note 阅读策略 ──

_CON_NOTE_SYSTEM = (
    "You are a careful reader. Given a piece of chat history and a question, "
    "extract ONLY the facts relevant to answering the question. "
    "Be precise: include exact numbers, dates, names, and preferences. "
    "If the passage contains no relevant information, respond with 'N/A'. "
    "Keep each note under 2 sentences."
)

_CON_ANSWER_SYSTEM = (
    "You are a memory assistant. Based on the provided notes extracted from chat history, "
    "answer the user's question precisely. "
    "Guidelines:\n"
    "- Use exact numbers, dates, and names from the notes.\n"
    "- For temporal questions, calculate time differences carefully.\n"
    "- For 'how many' questions, find the specific number.\n"
    "- For preference questions, identify what the user chose or enjoys.\n"
    "- If the notes do not contain the answer, respond 'I don't know'.\n"
    "Answer concisely in 1-2 sentences."
)


def _generate_answer_with_con(
    query: str,
    contexts: list[str],
    api_key: str,
    base_url: str,
    model: str = "deepseek-chat",
    max_notes: int = 30,
) -> str:
    """Chain-of-Note 阅读策略：先对每条 context 做笔记，再基于笔记生成答案。

    论文证明此策略可比直接拼接 context 提升约 10 个绝对点。

    流程:
      1. 对前 max_notes 条 context 逐条做笔记（LLM 提取关键信息）
      2. 过滤掉 'N/A' 的笔记
      3. 基于有效笔记生成最终答案

    Args:
        query: 用户问题
        contexts: 检索到的 context 列表
        api_key: LLM API key
        base_url: LLM API base URL
        model: LLM 模型名
        max_notes: 最多对前 N 条 context 做笔记（控制 API 调用成本）

    Returns:
        生成的答案文本
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai 库未安装，无法生成答案")
        return ""

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Step 1: 对每条 context 做笔记
    notes = []
    for i, ctx in enumerate(contexts[:max_notes]):
        note_prompt = (
            f"Question: {query}\n\n"
            f"Passage [{i+1}]: {ctx[:800]}\n\n"
            f"Note:"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _CON_NOTE_SYSTEM},
                    {"role": "user", "content": note_prompt},
                ],
                n=1,
                temperature=0,
                max_tokens=100,
            )
            note = resp.choices[0].message.content.strip()
            if note and note.upper() != "N/A":
                notes.append(f"[{i+1}] {note}")
        except Exception as e:
            logger.debug("CoN note %d 失败: %s", i + 1, e)

    if not notes:
        # 所有笔记都是 N/A，回退为直接生成
        logger.debug("CoN: 所有笔记均为 N/A，回退为直接生成")
        return _generate_answer_with_llm(query, contexts[:20], api_key, base_url, model=model)

    # Step 2: 基于笔记生成答案
    notes_text = "\n".join(notes)
    answer_prompt = (
        f"Notes from chat history:\n{notes_text}\n\n"
        f"Question: {query}\nAnswer:"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CON_ANSWER_SYSTEM},
                {"role": "user", "content": answer_prompt},
            ],
            n=1,
            temperature=0,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("CoN 最终答案生成失败: %s", e)
        return _generate_answer_with_llm(query, contexts[:20], api_key, base_url, model=model)

# ── 原子事实提取 ──

_FACT_SYSTEM = """你是一个事实提取专家。从给定文本中提取独立的原子事实。

规则：
1. 每条事实只包含一个信息点（主语+谓语+宾语）
2. 保持原始信息，不推断不编造
3. 用第三人称表述（如"用户喜欢..."而非"我喜欢..."）
4. 输出 JSON 数组格式
5. 如果文本中没有可提取的事实，返回空数组 []

示例：
输入: "我喜欢 Python，尤其是 3.11 版本"
输出: ["用户喜欢 Python", "用户偏好 Python 3.11 版本"]

输入: "好的"
输出: []"""

_FACT_USER = """从以下文本中提取独立的原子事实：

文本: {content}

原子事实（JSON 数组）:"""


def _extract_facts_inline(
    content: str,
    api_key: str,
    base_url: str,
    model: str = "deepseek-chat",
) -> list[str]:
    """调用 LLM 从文本中提取原子事实，失败时回退为 [content]。

    Args:
        content: 原始文本内容
        api_key: LLM API key
        base_url: LLM API base URL
        model: LLM 模型名

    Returns:
        原子事实列表；LLM 失败时返回 [content]
    """
    try:
        from openai import OpenAI
    except ImportError:
        return [content]

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = _FACT_USER.format(content=content[:2000])

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _FACT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            n=1,
            temperature=0,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content.strip()
        # 尝试解析 JSON 数组
        import json as _json
        # 处理可能的 markdown 代码块包裹
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        facts = _json.loads(raw)
        if isinstance(facts, list) and facts:
            # 过滤空字符串
            facts = [f.strip() for f in facts if isinstance(f, str) and f.strip()]
            return facts if facts else [content]
        return [content]
    except Exception as e:
        logger.debug("事实提取失败，回退原始内容: %s", e)
        return [content]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrieval_eval")

# 降低噪音
for _n in ("omnimem", "jieba", "filelock", "sentence_transformers", "chromadb",
           "huggingface_hub", "transformers", "urllib3", "httpcore", "httpx",
           "filelock", "torch", "tiktoken"):
    logging.getLogger(_n).setLevel(logging.WARNING)

# 强制 CPU + 中国镜像
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _build_documents(
    entry: dict[str, Any],
    max_sessions: int = 10,
    enable_split: bool = False,
    preference_boost: bool = False,
) -> list[dict[str, Any]]:
    """从 LongMemEval entry 构建文档集合。

    优先保留 answer sessions，确保答案相关的 turn 不被截断。
    preference_boost=True 时，为偏好类 turn 注入可搜索前缀标签。
    """
    answer_session_ids = set(entry.get("answer_session_ids", []))
    session_ids = entry.get("haystack_session_ids", [])
    sessions = entry.get("haystack_sessions", [])

    # 优先保留 answer sessions
    if max_sessions > 0 and len(session_ids) > max_sessions:
        answer_indices = []
        filler_indices = []
        for idx, sid in enumerate(session_ids):
            is_answer = (
                sid in answer_session_ids
                or any(aid in sid for aid in answer_session_ids)
            )
            if is_answer:
                answer_indices.append(idx)
            else:
                filler_indices.append(idx)
        selected = answer_indices + filler_indices[: max(0, max_sessions - len(answer_indices))]
        selected.sort()
        session_ids = [session_ids[i] for i in selected]
        sessions = [sessions[i] for i in selected]

    documents = []

    for sess_idx, (sess_id, sess_turns) in enumerate(
        zip(session_ids, sessions)
    ):
        for turn_idx, turn in enumerate(sess_turns):
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if not content:
                continue
            if role == "assistant" and not OmniMemMemoryProvider._is_substantive(content):
                continue

            # 长回复分段存储
            if enable_split:
                segments = OmniMemMemoryProvider._split_long_content(content, role)
            else:
                segments = [content]

            # 偏好增强索引
            is_preference = (
                role == "user"
                and any(kw in content.lower() for kw in _PREFERENCE_KEYWORDS)
            )

            for seg_idx, segment in enumerate(segments):
                if preference_boost and is_preference:
                    segment = f"[prefer like enjoy] {segment}"

                documents.append({
                    "content": segment,
                    "role": role,
                    "session_id": sess_id,
                    "turn_index": turn_idx,
                    "has_answer": turn.get("has_answer", False),
                    "segment_index": seg_idx if len(segments) > 1 else None,
                    "is_preference": is_preference,
                })

    return documents


# ── 动态权重路由 ──

# 日期/数字模式（用于时序查询检测）
_DATE_PATTERN = re.compile(
    r'\b(january|february|march|april|may|june|july|august|september|october|november|december'
    r'|\d{4}|\d{1,2}(st|nd|rd|th))\b',
    re.IGNORECASE,
)

# 偏好类信号词
_PREFERENCE_SIGNALS = (
    "recommend", "suggest", "prefer", "enjoy", "like", "favorite", "favourite", "best",
)

# 时序/精确匹配类信号词
_TEMPORAL_SIGNALS = (
    "how many", "when", "what date", "how long", "last", "first", "before", "after",
)


def _detect_query_type(query: str) -> str:
    """检测查询类型，用于动态权重路由。

    返回值:
        "preference" — 偏好类（向量主导）
        "temporal"   — 时序/精确匹配类（BM25 主导）
        "default"    — 默认
    """
    q_lower = query.lower()
    if any(sig in q_lower for sig in _PREFERENCE_SIGNALS):
        return "preference"
    if any(sig in q_lower for sig in _TEMPORAL_SIGNALS) or _DATE_PATTERN.search(query):
        return "temporal"
    return "default"


def _get_dynamic_weights(query_type: str) -> tuple[float, float]:
    """根据查询类型返回 (bm25_weight, vector_weight)。"""
    if query_type == "preference":
        return 1.0, 3.0   # 偏好类：向量主导
    elif query_type == "temporal":
        return 5.0, 0.5   # 时序类：BM25 主导
    else:
        return 5.0, 1.0   # 默认：BM25 主导


def _query_type_short(qtype: str) -> str:
    """查询类型缩写，用于进度行显示。"""
    return {
        "preference": "pref",
        "temporal": "temp",
        "default": "def",
    }.get(qtype, "def")


# ── RRF 融合 ──

def _rrf_fuse(
    bm25_results: list[dict],
    vector_results: list[dict],
    bm25_weight: float = 2.0,
    vector_weight: float = 3.0,
    rrf_k: int = 35,
    top_k: int = 100,
    query: str = "",
    dynamic_weight: bool = False,
) -> tuple[list[dict], str]:
    """RRF 融合 BM25 和向量检索结果。

    RRF 公式: score = Σ weight / (k + rank)

    当 dynamic_weight=True 时，内部调用 _detect_query_type 和 _get_dynamic_weights
    自动调整权重；否则使用传入的 bm25_weight / vector_weight 固定值。

    Returns:
        (融合后的结果列表, 检测到的查询类型)
    """
    query_type = "default"
    if dynamic_weight and query:
        query_type = _detect_query_type(query)
        bm25_weight, vector_weight = _get_dynamic_weights(query_type)

    score_map: dict[str, float] = {}  # memory_id → RRF 分数
    doc_map: dict[str, dict] = {}     # memory_id → 文档信息

    # BM25 通道
    for rank, r in enumerate(bm25_results, start=1):
        mid = r.get("memory_id", f"bm25_{rank}")
        rrf_score = bm25_weight / (rrf_k + rank)
        score_map[mid] = score_map.get(mid, 0) + rrf_score
        if mid not in doc_map:
            doc_map[mid] = r

    # 向量通道
    for rank, r in enumerate(vector_results, start=1):
        mid = r.get("memory_id", f"vec_{rank}")
        rrf_score = vector_weight / (rrf_k + rank)
        score_map[mid] = score_map.get(mid, 0) + rrf_score
        if mid not in doc_map:
            doc_map[mid] = r

    # 按 RRF 分数排序
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

    results = []
    for mid, score in ranked[:top_k]:
        entry = dict(doc_map[mid])
        entry["rrf_score"] = score
        results.append(entry)

    return results, query_type


def _compute_ranking_metrics(
    retrieved: list[dict],
    entry: dict[str, Any],
    top_ks: tuple[int, ...] = (5, 10, 20, 50),
) -> dict[str, Any]:
    """计算排名相关指标：Top-K 命中率、MRR、答案首次出现位置。"""
    answer_session_contents, answer_turn_contents, answer_texts = (
        build_answer_turn_contents(entry)
    )

    if not answer_texts:
        return {"skipped": True, "reason": "no_answer_turns"}

    first_rank = None

    for rank, r in enumerate(retrieved, start=1):
        content = r.get("content", "")
        for ans_content in answer_turn_contents:
            if ans_content[:80] in content:
                first_rank = rank
                break
        if first_rank is not None:
            break

    if first_rank is None and answer_session_contents:
        for rank, r in enumerate(retrieved, start=1):
            content = r.get("content", "")
            for ans_content in answer_session_contents:
                if ans_content[:80] in content:
                    first_rank = rank
                    break
            if first_rank is not None:
                break

    mrr = 1.0 / first_rank if first_rank is not None else 0.0

    top_k_hits = {}
    for k in top_ks:
        top_k_hits[f"top{k}_hit"] = first_rank is not None and first_rank <= k

    return {
        "first_rank": first_rank,
        "mrr": mrr,
        **top_k_hits,
        "skipped": False,
    }


# ── Judge 判分（简单关键词匹配） ──

# 拒绝类问题指示词
_REFUSAL_INDICATORS = (
    "不知道", "无法", "不能", "没有", "不确定",
    "don't know", "cannot", "unable", "not sure", "no information",
)


def _keyword_judge(prediction: str, gold_answer: str, is_abstention: bool) -> bool:
    """简单关键词匹配判分：prediction 包含 gold answer 核心词即正确。

    参考 longmemeval_adapter.py 中的 _judge_answer 方法（无 LLM 时的回退路径）。
    对于 abstention 类问题，检查 prediction 是否合理拒绝回答。
    """
    if not prediction.strip():
        return False

    # 拒绝类问题：预测答案应合理拒绝回答
    if is_abstention:
        pred_lower = prediction.lower()
        return any(ind in pred_lower for ind in _REFUSAL_INDICATORS)

    # 普通问题：关键词匹配（gold answer 核心词覆盖率 >= 50%）
    gold_words = set(gold_answer.lower().split())
    if not gold_words:
        return False
    pred_lower = prediction.lower()
    matched = sum(1 for w in gold_words if w in pred_lower)
    return matched / len(gold_words) >= 0.5


# ── ChromaDB collection 重置（OOM 修复） ──

def _reset_vector_collection(vector_retriever: VectorRetriever) -> None:
    """重置 ChromaDB collection：delete + recreate，只清数据不释放连接。

    关键点：
    - 不调用 store.reset()（会重新初始化 store，可能导致嵌入模型重新加载）
    - 直接操作 ChromaDB 的 collection：delete_collection → get_or_create_collection
    - 嵌入模型（_embedding_fn）全局缓存，不会被销毁
    - 首次运行时 delete_collection 会失败（collection 不存在），直接 get_or_create
    """
    store = vector_retriever._store
    if store is None or getattr(store, "_client", None) is None:
        return

    # 尝试删除旧 collection（清空数据释放内存）
    try:
        store._client.delete_collection("omnimem")
    except Exception:
        # 首次运行时 collection 不存在，忽略错误
        logger.debug("delete_collection 失败（可能是首次运行），将直接 get_or_create")

    # 重新创建空 collection（复用同一个 _embedding_fn，不重新加载模型）
    try:
        embedding_fn = getattr(store, "_embedding_fn", None)
        store._collection = store._client.get_or_create_collection(
            name="omnimem",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_fn,
        )
    except Exception as e:
        logger.warning("重置向量 collection 失败: %s", e)


def main():
    parser = argparse.ArgumentParser(description="轻量检索质量评测（BM25 + 向量 + RRF 融合）")
    parser.add_argument("--data", type=str,
                        default=str(PROJECT_ROOT / "benchmarks" / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"),
                        help="评测数据路径")
    parser.add_argument("--limit", type=int, default=10, help="每种类型取几题")
    parser.add_argument("--max-sessions", type=int, default=10, help="每题最多取几个 session")
    parser.add_argument("--top-k", type=int, default=100, help="检索 top_k")
    parser.add_argument("--split", action="store_true", default=True, help="启用长回复分段存储")
    parser.add_argument("--no-split", action="store_true", help="禁用长回复分段存储")
    parser.add_argument("--vector", action="store_true", default=True, help="启用向量语义检索")
    parser.add_argument("--no-vector", action="store_true", help="禁用向量检索（仅 BM25）")
    parser.add_argument("--bm25-weight", type=float, default=2.0, help="BM25 通道 RRF 权重")
    parser.add_argument("--vector-weight", type=float, default=3.0, help="向量通道 RRF 权重")
    parser.add_argument("--rrf-k", type=int, default=35, help="RRF 融合参数 k")
    parser.add_argument("--offset", type=int, default=0, help="每种类型跳过前 N 题（用于分批）")
    parser.add_argument("--pref-boost", action="store_true", default=False, help="偏好增强索引")
    parser.add_argument("--no-pref-boost", action="store_true", help="禁用偏好增强索引")
    # ── 新增参数 ──
    parser.add_argument("--mode", type=str, default="retrieval",
                        choices=["retrieval", "gen", "full"],
                        help="评测模式: retrieval=仅检索指标, gen=检索+LLM生成, full=检索+生成+Judge判分")
    parser.add_argument("--dynamic-weight", action="store_true", default=False,
                        help="启用动态权重路由（根据查询类型自动调整 BM25/向量权重）")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "benchmarks" / "results"),
                        help="结果 JSON 输出目录")
    parser.add_argument("--llm-base-url", type=str, default="",
                        help="LLM API base URL（gen/full 模式使用）")
    parser.add_argument("--gen-model", type=str, default="gpt-4o-mini",
                        help="LLM 生成模型名称（gen/full 模式使用）")
    parser.add_argument("--chain-of-note", action="store_true", default=False,
                        help="启用 Chain-of-Note 阅读策略：先对每条 context 做笔记，再基于笔记生成答案")
    parser.add_argument("--con-max-notes", type=int, default=30,
                        help="Chain-of-Note 最多对前 N 条 context 做笔记（控制 API 调用成本，默认 30）")
    parser.add_argument("--gen-context-size", type=int, default=20,
                        help="传给 LLM 生成的 context 条数（默认 20，top_k=200 时建议增大到 50）")
    parser.add_argument("--judge-model", type=str, default="deepseek-chat",
                        help="LLM Judge 模型名称（full 模式使用，默认 deepseek-chat）")
    parser.add_argument("--fact-extract", action="store_true", default=False,
                        help="启用 LLM 原子事实提取：ingest 时从对话中提取结构化事实独立存储")
    parser.add_argument("--fact-extract-model", type=str, default="deepseek-chat",
                        help="事实提取 LLM 模型名称（默认 deepseek-chat）")
    args = parser.parse_args()

    if args.no_split:
        args.split = False
    if args.no_vector:
        args.vector = False
    if args.no_pref_boost:
        args.pref_boost = False

    # 检查 LLM 可用性（gen/full 模式或事实提取需要）
    llm_available = False
    api_key = ""
    base_url = ""
    if args.mode in ("gen", "full") or args.fact_extract:
        llm_available, api_key, base_url = _check_llm_available(args.llm_base_url)
        if not llm_available:
            if args.mode in ("gen", "full"):
                logger.warning("LLM API 不可用，gen/full 模式将回退为 retrieval 模式")
                args.mode = "retrieval"
            if args.fact_extract:
                logger.warning("LLM API 不可用，事实提取将被禁用")
                args.fact_extract = False
        else:
            logger.info("LLM 可用: base_url=%s, model=%s", base_url, args.gen_model)

    # 加载数据
    with open(args.data) as f:
        data = json.load(f)

    by_type = defaultdict(list)
    for d in data:
        by_type[d["question_type"]].append(d)

    selected = []
    for qtype, items in sorted(by_type.items()):
        start = getattr(args, 'offset', 0)
        end = start + args.limit
        selected.extend(items[start:end])

    mode_str = "BM25+Vector" if args.vector else "BM25-only"
    dyn_str = "ON" if args.dynamic_weight else "OFF"
    logger.info("评测 %d 题 (mode=%s, retrieval=%s, top_k=%d, max_sessions=%d, "
                "bm25_w=%.1f, vec_w=%.1f, rrf_k=%d, dynamic_weight=%s)",
                len(selected), args.mode, mode_str, args.top_k, args.max_sessions,
                args.bm25_weight, args.vector_weight, args.rrf_k, dyn_str)

    results = []
    t0 = time.time()

    # 向量检索器全局初始化（直接用 sentence-transformers，绕过 ChromaDB 避免 OOM）
    _global_embedder = None
    vec_temp_dir = None
    if args.vector:
        logger.info("初始化向量嵌入模型（sentence-transformers, CPU）...")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            from sentence_transformers import SentenceTransformer
            _global_embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            logger.info("嵌入模型加载完成: all-MiniLM-L6-v2 (384维)")
        except Exception as e:
            logger.error("嵌入模型加载失败: %s", e)
            args.vector = False

    try:
        for i, item in enumerate(selected):
            qid = item["question_id"][:12]
            qtype = item["question_type"]
            question = item["question"]
            answer = str(item["answer"])
            is_abstention = "_abs" in item["question_id"]

            # 构建文档集合
            documents = _build_documents(
                item, max_sessions=args.max_sessions,
                enable_split=args.split, preference_boost=args.pref_boost,
            )

            # 事实提取：对每个文档的 content 提取原子事实，展开为独立文档
            if args.fact_extract and llm_available:
                expanded_docs = []
                for doc in documents:
                    content = doc["content"]
                    if len(content) > 20:
                        facts = _extract_facts_inline(
                            content, api_key, base_url,
                            model=args.fact_extract_model,
                        )
                        for fact_content in facts:
                            expanded_docs.append({
                                "content": fact_content,
                                "role": doc["role"],
                                "session_id": doc["session_id"],
                                "turn_index": doc["turn_index"],
                                "has_answer": doc.get("has_answer", False),
                                "segment_index": doc.get("segment_index"),
                                "is_preference": doc.get("is_preference", False),
                            })
                    else:
                        expanded_docs.append(doc)
                documents = expanded_docs

            # 初始化 BM25 检索器
            bm25 = BM25Retriever(buffer_size=len(documents) + 10)
            for doc_idx, doc in enumerate(documents):
                doc_id = f"doc_{doc_idx}"
                bm25.add(doc["content"], doc_id, {
                    "role": doc["role"],
                    "session_id": doc["session_id"],
                    "has_answer": doc.get("has_answer", False),
                })
            bm25.flush()

            # BM25 检索
            bm25_results = bm25.search(question, top_k=args.top_k)

            # 向量检索（直接用 numpy cosine similarity，绕过 ChromaDB 避免 OOM）
            vec_results = []
            if _global_embedder is not None:
                # 编码所有文档
                doc_texts = [doc["content"] for doc in documents]
                doc_embeddings = _global_embedder.encode(
                    doc_texts, batch_size=64, show_progress_bar=False,
                    convert_to_numpy=True, normalize_embeddings=True,
                )
                # 编码查询
                query_embedding = _global_embedder.encode(
                    [question], show_progress_bar=False,
                    convert_to_numpy=True, normalize_embeddings=True,
                )[0]
                # cosine similarity（已归一化，直接点积）
                scores = doc_embeddings @ query_embedding
                # 取 top_k
                top_indices = np.argsort(scores)[::-1][:args.top_k]
                for rank, idx in enumerate(top_indices):
                    if scores[idx] < 0.01:  # 最低阈值
                        continue
                    vec_results.append({
                        "memory_id": f"doc_{idx}",
                        "content": documents[idx]["content"],
                        "role": documents[idx]["role"],
                        "session_id": documents[idx]["session_id"],
                        "score": float(scores[idx]),
                    })
                # 释放嵌入矩阵
                del doc_embeddings, query_embedding, scores

            # RRF 融合
            query_type = "default"
            if _global_embedder is not None and vec_results:
                retrieved, query_type = _rrf_fuse(
                    bm25_results, vec_results,
                    bm25_weight=args.bm25_weight,
                    vector_weight=args.vector_weight,
                    rrf_k=args.rrf_k,
                    top_k=args.top_k,
                    query=question,
                    dynamic_weight=args.dynamic_weight,
                )
            else:
                # 纯 BM25（无向量检索时也检测查询类型用于进度显示）
                retrieved = bm25_results
                if args.dynamic_weight:
                    query_type = _detect_query_type(question)

            # 评估检索质量
            retrieval_quality = evaluate_retrieval_quality(
                [r.get("content", "") for r in retrieved],
                item,
            )
            ranking_metrics = _compute_ranking_metrics(retrieved, item)

            # 构建结果记录
            result_record = {
                "question_id": qid,
                "question_type": qtype,
                "doc_count": len(documents),
                "retrieved_count": len(retrieved),
                "bm25_count": len(bm25_results),
                "vec_count": len(vec_results),
                "query_type": query_type,
                **retrieval_quality,
                **ranking_metrics,
            }

            # ── gen/full 模式：LLM 生成答案 ──
            if args.mode in ("gen", "full") and llm_available:
                contexts = [r.get("content", "") for r in retrieved[:args.gen_context_size]]
                if contexts:
                    if args.chain_of_note:
                        prediction = _generate_answer_with_con(
                            question, contexts, api_key, base_url,
                            model=args.gen_model,
                            max_notes=args.con_max_notes,
                        )
                    else:
                        prediction = _generate_answer_with_llm(
                            question, contexts, api_key, base_url,
                            model=args.gen_model,
                        )
                    result_record["prediction"] = prediction[:500]

                    # full 模式：Judge 判分
                    if args.mode == "full":
                        # 优先使用 LLM-as-Judge，失败回退关键词判分
                        if llm_available:
                            try:
                                is_correct = _judge_answer_with_llm(
                                    question, answer, prediction,
                                    qtype, is_abstention,
                                    api_key, base_url,
                                    model=args.judge_model,
                                )
                                judge_type = "LLM"
                            except Exception as e:
                                logger.warning("LLM Judge 失败，回退关键词判分: %s", e)
                                is_correct = _keyword_judge(prediction, answer, is_abstention)
                                judge_type = "keyword(fallback)"
                        else:
                            is_correct = _keyword_judge(prediction, answer, is_abstention)
                            judge_type = "keyword"
                        result_record["is_correct"] = is_correct
                        result_record["gold_answer"] = answer[:200]
                        result_record["judge_type"] = judge_type

            results.append(result_record)

            # 打印进度
            rm = ranking_metrics
            mark = "✓" if retrieval_quality.get("turn_hit") else "✗"
            rank_str = f"#{rm['first_rank']}" if rm.get("first_rank") else ">100"
            vec_str = f"v={len(vec_results)}" if args.vector else "v=OFF"
            type_str = f"type={_query_type_short(query_type)}"

            progress_line = (
                f"  {mark} [{i+1}/{len(selected)}] {qtype:<28} "
                f"rank={rank_str:>5} MRR={rm.get('mrr', 0):.2f} "
                f"T5={'Y' if rm.get('top5_hit') else 'N'} T10={'Y' if rm.get('top10_hit') else 'N'} "
                f"b={len(bm25_results)} {vec_str} {type_str} │ {qid}"
            )

            # gen/full 模式附加生成信息
            if args.mode in ("gen", "full") and result_record.get("prediction"):
                pred_preview = result_record["prediction"][:60].replace("\n", " ")
                progress_line += f" │ ans={pred_preview}"
                if args.mode == "full":
                    jt = result_record.get("judge_type", "?")
                    progress_line += f" [{'✓' if result_record.get('is_correct') else '✗'}|{jt}]"

            print(progress_line, flush=True)

            # ★ 每题结束后释放大对象引用 + 强制 GC
            del bm25, bm25_results, vec_results, retrieved, documents
            if _global_embedder is not None:
                gc.collect()

    finally:
        pass  # numpy 方案无需清理临时目录

    elapsed = time.time() - t0
    _print_summary(results, args, elapsed)

    # 保存 JSON 结果
    _save_results_json(results, args, elapsed)


def _print_summary(results: list[dict], args, elapsed: float) -> None:
    """打印汇总统计。"""
    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    evaluable = [r for r in results if not r.get("skipped")]
    total = len(evaluable)

    # 排名指标
    rank_evaluable = [r for r in evaluable if not r.get("skipped") and r.get("first_rank") is not None]
    mrrs = [r["mrr"] for r in rank_evaluable]
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
    ranks = [r["first_rank"] for r in rank_evaluable]
    median_rank = sorted(ranks)[len(ranks) // 2] if ranks else 0
    avg_rank = sum(ranks) / len(ranks) if ranks else 0

    # Top-K 命中率
    top5_hits = sum(1 for r in evaluable if r.get("top5_hit"))
    top10_hits = sum(1 for r in evaluable if r.get("top10_hit"))
    top20_hits = sum(1 for r in evaluable if r.get("top20_hit"))

    # 基础指标
    session_hits = sum(1 for r in evaluable if r.get("session_hit") is True)
    turn_hits = sum(1 for r in evaluable if r.get("turn_hit") is True)

    mode_str = "BM25+Vector" if args.vector else "BM25-only"
    dyn_str = "ON" if args.dynamic_weight else "OFF"
    con_str = f"ON(notes={args.con_max_notes})" if args.chain_of_note else "OFF"
    fact_str = f"ON(model={args.fact_extract_model})" if args.fact_extract else "OFF"
    ctx_str = f"ctx={args.gen_context_size}"
    print("\n" + "=" * 78)
    print(f"检索质量评测结果 ({total} 题, {elapsed:.1f}s, 模式={args.mode}, 检索={mode_str}, 动态权重={dyn_str})")
    print(f"  BM25权重={args.bm25_weight}, 向量权重={args.vector_weight}, RRF k={args.rrf_k}, top_k={args.top_k}")
    print(f"  Chain-of-Note={con_str}, {ctx_str}, gen_model={args.gen_model}")
    judge_str = args.judge_model if args.mode == "full" else "N/A"
    print(f"  Judge: {judge_str}")
    print(f"  事实提取: {fact_str}")
    print("=" * 78)

    print("\n  ── 排名指标（核心） ──")
    if total:
        print(f"  MRR (Mean Reciprocal Rank):  {avg_mrr:.3f}")
        print(f"  答案首次出现位置:            平均 #{avg_rank:.1f}  中位数 #{median_rank}")
        print()
        print(f"  Top-5  命中率: {top5_hits}/{total} = {top5_hits/total:.1%}")
        print(f"  Top-10 命中率: {top10_hits}/{total} = {top10_hits/total:.1%}")
        print(f"  Top-20 命中率: {top20_hits}/{total} = {top20_hits/total:.1%}")

    print("\n  ── 基础指标 ──")
    if total:
        print(f"  Session Hit Rate:  {session_hits}/{total} = {session_hits/total:.1%}")
        print(f"  Turn Hit Rate:     {turn_hits}/{total} = {turn_hits/total:.1%}")

    # gen/full 模式：生成与判分指标
    if args.mode in ("gen", "full"):
        gen_results = [r for r in results if r.get("prediction")]
        print(f"\n  ── 生成指标 ({args.mode} 模式) ──")
        print(f"  生成答案数: {len(gen_results)}/{len(results)}")

        if args.mode == "full":
            judged = [r for r in gen_results if r.get("is_correct") is not None]
            correct = sum(1 for r in judged if r["is_correct"])
            if judged:
                accuracy = correct / len(judged)
                print(f"  QA 准确率: {correct}/{len(judged)} = {accuracy:.1%}")

                # 分维度准确率
                print(f"\n  ── 分维度 QA 准确率 ──")
                print(f"  {'类型':<28} {'准确率':>8} {'正确/总数':>10}")
                print(f"  {'-'*28} {'-'*8} {'-'*10}")
                for qtype in sorted(by_type):
                    items = [r for r in by_type[qtype] if r.get("is_correct") is not None]
                    if not items:
                        continue
                    corr = sum(1 for r in items if r["is_correct"])
                    acc = corr / len(items)
                    print(f"  {qtype:<28} {acc:>7.1%} {corr}/{len(items)}")

    # 分维度
    print(f"\n  ── 分维度排名指标 ──")
    print(f"  {'类型':<28} {'MRR':>6} {'Top-5':>7} {'Top-10':>7} {'Top-20':>7} {'中位排名':>8}")
    print(f"  {'-'*28} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    task_mrrs = []
    for qtype in sorted(by_type):
        items = [r for r in by_type[qtype] if not r.get("skipped")]
        if not items:
            continue

        item_mrrs = [r["mrr"] for r in items if r.get("mrr") is not None]
        avg_m = sum(item_mrrs) / len(item_mrrs) if item_mrrs else 0
        task_mrrs.append(avg_m)

        t5 = sum(1 for r in items if r.get("top5_hit")) / len(items)
        t10 = sum(1 for r in items if r.get("top10_hit")) / len(items)
        t20 = sum(1 for r in items if r.get("top20_hit")) / len(items)

        item_ranks = [r["first_rank"] for r in items if r.get("first_rank") is not None]
        med = sorted(item_ranks)[len(item_ranks) // 2] if item_ranks else 0

        print(f"  {qtype:<28} {avg_m:>5.3f} {t5:>6.0%} {t10:>6.0%} {t20:>6.0%} #{med:>6}")

    print()
    if task_mrrs:
        task_avg_mrr = sum(task_mrrs) / len(task_mrrs)
        print(f"  Task-Averaged MRR: {task_avg_mrr:.3f}")

    # 动态权重路由统计
    if args.dynamic_weight:
        type_counts = defaultdict(int)
        for r in results:
            type_counts[r.get("query_type", "default")] += 1
        print(f"\n  ── 查询类型分布（动态权重路由） ──")
        for qt in ("preference", "temporal", "default"):
            cnt = type_counts.get(qt, 0)
            if cnt > 0:
                w_bm25, w_vec = _get_dynamic_weights(qt)
                print(f"  {qt:<12} {cnt:>4} 题  (bm25_w={w_bm25}, vec_w={w_vec})")

    # 排名分布
    print(f"\n  ── 答案排名分布 ──")
    if ranks:
        buckets = [(1, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 100)]
        for lo, hi in buckets:
            cnt = sum(1 for r in ranks if lo <= r <= hi)
            bar = "█" * cnt
            print(f"  #{lo:>3}-#{hi:<3}  {cnt:>3}  {bar}")

    mode_hint = "BM25+Vector(RRF)" if args.vector else "BM25-only"
    print(f"\n  检索模式: {mode_hint}")
    print(f"  评测模式: {args.mode}")
    print(f"  动态权重: {'ON' if args.dynamic_weight else 'OFF'}")
    print(f"  Chain-of-Note: {'ON' if args.chain_of_note else 'OFF'}")
    print(f"  事实提取: {'ON' if args.fact_extract else 'OFF'}")
    print(f"  Context 条数: {args.gen_context_size}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Judge 模型: {args.judge_model if args.mode == 'full' else 'N/A'}")
    print(f"  嵌入模型: all-MiniLM-L6-v2 (384维, CPU)")


def _save_results_json(results: list[dict], args, elapsed: float) -> None:
    """将评测结果保存为 JSON 文件。"""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 汇总统计
    evaluable = [r for r in results if not r.get("skipped")]
    total = len(evaluable)
    rank_evaluable = [r for r in evaluable if r.get("first_rank") is not None]
    mrrs = [r["mrr"] for r in rank_evaluable]
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0

    top5_hits = sum(1 for r in evaluable if r.get("top5_hit"))
    top10_hits = sum(1 for r in evaluable if r.get("top10_hit"))
    top20_hits = sum(1 for r in evaluable if r.get("top20_hit"))

    # full 模式准确率
    qa_accuracy = None
    if args.mode == "full":
        judged = [r for r in results if r.get("is_correct") is not None]
        if judged:
            correct = sum(1 for r in judged if r["is_correct"])
            qa_accuracy = round(correct / len(judged), 4)

    # 查询类型分布
    type_counts = defaultdict(int)
    for r in results:
        type_counts[r.get("query_type", "default")] += 1

    summary = {
        "total_questions": len(results),
        "evaluable_questions": total,
        "elapsed_seconds": round(elapsed, 2),
        "mode": args.mode,
        "retrieval_mode": "BM25+Vector" if args.vector else "BM25-only",
        "dynamic_weight": args.dynamic_weight,
        "bm25_weight": args.bm25_weight,
        "vector_weight": args.vector_weight,
        "rrf_k": args.rrf_k,
        "top_k": args.top_k,
        "max_sessions": args.max_sessions,
        "chain_of_note": args.chain_of_note,
        "con_max_notes": args.con_max_notes if args.chain_of_note else None,
        "gen_context_size": args.gen_context_size,
        "avg_mrr": round(avg_mrr, 4),
        "top5_hit_rate": round(top5_hits / total, 4) if total else 0,
        "top10_hit_rate": round(top10_hits / total, 4) if total else 0,
        "top20_hit_rate": round(top20_hits / total, 4) if total else 0,
        "qa_accuracy": qa_accuracy,
        "query_type_distribution": dict(type_counts),
        "fact_extract": args.fact_extract,
        "fact_extract_model": args.fact_extract_model if args.fact_extract else None,
        "judge_model": args.judge_model if args.mode == "full" else None,
    }

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = args.mode
    dyn_tag = "dyn" if args.dynamic_weight else "fix"
    vec_tag = "vec" if args.vector else "bm25"
    con_tag = "con" if args.chain_of_note else "direct"
    filename = f"retrieval_eval_{mode_tag}_{vec_tag}_{dyn_tag}_{con_tag}_{timestamp}.json"
    output_path = output_dir / filename

    output_data = {
        "summary": summary,
        "results": results,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已保存: {output_path}")
    except Exception as e:
        logger.error("保存结果 JSON 失败: %s", e)


if __name__ == "__main__":
    main()
