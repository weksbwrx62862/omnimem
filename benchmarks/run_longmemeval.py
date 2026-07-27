"""OmniMem → LongMemEval-S 评测主脚本。

评测流程:
  1. 加载 longmemeval_s_cleaned.json 数据集
  2. 对每条问题:
     a. 使用 OmniMemMemoryProvider.ingest_sessions() 写入 haystack_sessions
     b. 使用 OmniMemMemoryProvider.search() 检索相关 context
     c. 评估检索质量（是否命中 has_answer=true 的 turn / answer_session_ids）
     d. （可选）使用 LLM 基于检索 context 生成答案
     e. （可选）使用 LLM Judge 评判答案正确性
  3. 按 question_type 维度统计分数
  4. 输出综合分数到 scores.json

支持两种模式:
  - retrieval-only: 仅运行 ingest + retrieval，评估检索质量（无需 LLM API）
  - full: 运行完整流程 ingest + retrieval + generation + evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.longmemeval_adapter import (
    OmniMemMemoryProvider,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("longmemeval_eval")

# 降低 OmniMem 内部日志噪音
for _logger_name in (
    "omnimem", "httpx", "sentence_transformers", "jieba",
    "chromadb", "huggingface_hub", "transformers", "filelock",
    "urllib3", "httpcore",
):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)


# ======================================================================
# 检索质量评估
# ======================================================================

def build_answer_turn_contents(
    entry: dict[str, Any],
) -> tuple[set[str], set[str], list[str]]:
    """构建答案 turn 的文本集合，用于检索命中率评估。

    Returns:
        (answer_session_contents, answer_turn_contents, answer_texts)
        - answer_session_contents: answer_session_ids 对应的所有 turn 内容集合
        - answer_turn_contents: has_answer=true 的 turn 内容集合
        - answer_texts: 所有答案 turn 的内容列表（用于模糊匹配）
    """
    answer_session_ids = set(entry.get("answer_session_ids", []))
    sessions = entry.get("haystack_sessions", [])
    session_ids = entry.get("haystack_session_ids", [])

    answer_session_contents: set[str] = set()
    answer_turn_contents: set[str] = set()
    answer_texts: list[str] = []

    for sess_id, sess_turns in zip(session_ids, sessions):
        # 标准化 session_id（LongMemEval 中 answer_ 前缀表示包含答案的 session）
        normalized_id = sess_id.replace("answer_", "").replace("noans_", "")
        is_answer_session = (
            sess_id in answer_session_ids
            or normalized_id in answer_session_ids
            or any(aid in sess_id for aid in answer_session_ids)
        )

        for turn in sess_turns:
            content = turn.get("content", "").strip()
            if not content:
                continue

            # answer session 中的所有 turn 内容
            if is_answer_session:
                answer_session_contents.add(content)
                answer_texts.append(content)

            # has_answer=true 的 turn 内容
            if turn.get("has_answer"):
                answer_turn_contents.add(content)
                if content not in answer_texts:
                    answer_texts.append(content)

    return answer_session_contents, answer_turn_contents, answer_texts


def evaluate_retrieval_quality(
    retrieved_contexts: list[str],
    entry: dict[str, Any],
) -> dict[str, Any]:
    """评估检索质量：检索到的 context 是否包含答案信息。

    评估指标:
      - session_hit: 检索结果中是否包含 answer_session_ids 中的 turn 内容
      - turn_hit: 检索结果中是否包含 has_answer=true 的 turn 内容
      - has_answer_coverage: 命中的 has_answer turn 占总数的比例
    """
    answer_session_contents, answer_turn_contents, answer_texts = (
        build_answer_turn_contents(entry)
    )

    if not answer_texts:
        # 没有明确的答案 turn（如 abstention 问题），跳过
        return {
            "session_hit": None,
            "turn_hit": None,
            "has_answer_coverage": None,
            "skipped": True,
            "reason": "no_answer_turns",
        }

    # 将检索结果拼接为一个长文本用于匹配
    retrieved_text = " ".join(retrieved_contexts) if retrieved_contexts else ""

    # 检查 session 命中
    session_hits = 0
    for content in answer_session_contents:
        if content[:80] in retrieved_text:  # 取前 80 字符做模糊匹配
            session_hits += 1

    # 检查 has_answer turn 命中
    turn_hits = 0
    for content in answer_turn_contents:
        if content[:80] in retrieved_text:
            turn_hits += 1

    total_answer_turns = len(answer_turn_contents) if answer_turn_contents else len(answer_session_contents)

    return {
        "session_hit": session_hits > 0 if answer_session_contents else None,
        "turn_hit": turn_hits > 0 if answer_turn_contents else None,
        "session_hit_count": session_hits,
        "session_total": len(answer_session_contents),
        "turn_hit_count": turn_hits,
        "turn_total": len(answer_turn_contents),
        "has_answer_coverage": (
            round(turn_hits / total_answer_turns, 4) if total_answer_turns > 0 else None
        ),
        "skipped": False,
    }


# ======================================================================
# LLM 生成与评判（可选）
# ======================================================================

def _check_llm_available(llm_base_url: str = "") -> tuple[bool, str, str]:
    """检查 LLM API 是否可用，返回 (available, api_key, base_url)。

    优先使用环境变量 OPENAI_API_KEY / OPENAI_BASE_URL；
    如果环境变量未设置，则从 hermes config.yaml 中读取 deepseek 或 dashscope 配置。
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")

    # 如果环境变量已设置，直接使用
    if api_key.strip():
        return True, api_key, base_url or llm_base_url or "https://api.openai.com/v1"

    # 回退：从 hermes config.yaml 读取 deepseek 或 dashscope 配置
    try:
        import yaml
        config_path = Path.home() / ".hermes" / "config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            providers = cfg.get("providers", {})

            # 优先查找 deepseek 配置
            ds_cfg = providers.get("deepseek", {})
            ds_key = ds_cfg.get("api_key", "")
            if ds_key:
                ds_url = ds_cfg.get("base_url", "https://api.deepseek.com")
                logger.info("从 hermes config.yaml 读取 DeepSeek 凭证")
                return True, ds_key, llm_base_url or ds_url

            # 回退到 dashscope 配置
            dashscope_cfg = providers.get("dashscope", {})
            ds_key = dashscope_cfg.get("api_key", "")
            if ds_key:
                ds_url = dashscope_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
                logger.info("从 hermes config.yaml 读取 DashScope 凭证")
                return True, ds_key, llm_base_url or ds_url
    except Exception as e:
        logger.debug("读取 hermes config.yaml 失败: %s", e)

    return False, "", llm_base_url or base_url or "https://api.openai.com/v1"


def _generate_answer_with_llm(
    query: str,
    contexts: list[str],
    api_key: str,
    base_url: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> str:
    """使用 LLM 基于检索到的 context 生成答案。"""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai 库未安装，无法生成答案")
        return ""

    client = OpenAI(api_key=api_key, base_url=base_url)

    context_text = "\n\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts))
    prompt = (
        "I will give you several history chats between you and a user. "
        "Please answer the question based on the relevant chat history.\n\n"
        "Guidelines:\n"
        "- Look carefully at ALL entries for numbers, dates, names, and specific details.\n"
        "- For temporal questions, identify date expressions and calculate time differences.\n"
        "- For 'how many' questions, find the specific number mentioned in the history.\n"
        "- For questions about schedules/assignments, look for the specific entry in tables or lists.\n"
        "- For preference questions, identify what the user explicitly prefers or chooses.\n"
        "- If the information exists in the history, answer precisely. Do NOT say it's unavailable if it's there.\n\n"
        f"History Chats:\n\n{context_text}\n\n"
        f"Question: {query}\nAnswer:"
    )

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=0,
                max_tokens=500,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning("LLM 生成重试 (%d/%d): %s", attempt + 1, max_retries, e)
                time.sleep(wait)
            else:
                logger.error("LLM 生成失败: %s", e)
                return ""


def _judge_answer_with_llm(
    question: str,
    gold_answer: str,
    prediction: str,
    question_type: str,
    is_abstention: bool,
    api_key: str,
    base_url: str,
    model: str = "gpt-4o",
    max_retries: int = 3,
) -> bool:
    """使用 LLM-as-Judge 评判答案正确性。"""
    try:
        from openai import OpenAI
    except ImportError:
        return False

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 参照 LongMemEval 官方 evaluate_qa.py 的 prompt 模板
    if is_abstention:
        prompt = (
            "I will give you an unanswerable question, an explanation, and a response "
            "from a model. Please answer yes if the model correctly identifies the "
            "question as unanswerable. The model could say that the information is "
            "incomplete, or some other information is given but the asked information "
            "is not.\n\n"
            f"Question: {question}\n\nExplanation: {gold_answer}\n\n"
            f"Model Response: {prediction}\n\n"
            "Does the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )
    elif question_type == "temporal-reasoning":
        prompt = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. In addition, do not penalize off-by-one errors for the number "
            "of days.\n\n"
            f"Question: {question}\n\nCorrect Answer: {gold_answer}\n\n"
            f"Model Response: {prediction}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif question_type == "knowledge-update":
        prompt = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. If the "
            "response contains some previous information along with an updated answer, "
            "the response should be considered as correct as long as the updated answer "
            "is the required answer.\n\n"
            f"Question: {question}\n\nCorrect Answer: {gold_answer}\n\n"
            f"Model Response: {prediction}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif question_type == "single-session-preference":
        prompt = (
            "I will give you a question, a rubric for desired personalized response, "
            "and a response from a model. Please answer yes if the response satisfies "
            "the desired response. The model does not need to reflect all the points "
            "in the rubric. The response is correct as long as it recalls and utilizes "
            "the user's personal information correctly.\n\n"
            f"Question: {question}\n\nRubric: {gold_answer}\n\n"
            f"Model Response: {prediction}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    else:
        prompt = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response is equivalent to the correct answer or contains "
            "all the intermediate steps to get the correct answer, you should also "
            "answer yes.\n\n"
            f"Question: {question}\n\nCorrect Answer: {gold_answer}\n\n"
            f"Model Response: {prediction}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=0,
                max_tokens=10,
            )
            verdict = completion.choices[0].message.content.strip().lower()
            return "yes" in verdict
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning("LLM Judge 重试 (%d/%d): %s", attempt + 1, max_retries, e)
                time.sleep(wait)
            else:
                logger.error("LLM Judge 失败: %s", e)
                return False


# ======================================================================
# 核心评测流程
# ======================================================================

def _prepare_entry_for_ingest(
    entry: dict[str, Any],
    max_sessions: int = 0,
    user_only: bool = False,
) -> dict[str, Any]:
    """准备评测用的 entry 数据（限制 sessions 数量、过滤 turn）。

    Args:
        entry: 原始 LongMemEval entry。
        max_sessions: 每题最大 session 数，0=全部。优先保留 answer sessions。
        user_only: 是否只保留 user 角色的 turn。默认 False，写入所有角色
            （含 assistant turn，memory_type 映射为 "action"）。

    Returns:
        处理后的 entry（不修改原始数据）。
    """
    import copy
    prepared = copy.deepcopy(entry)

    if max_sessions > 0:
        answer_ids = set(entry.get("answer_session_ids", []))
        session_ids = entry.get("haystack_session_ids", [])
        sessions = entry.get("haystack_sessions", [])
        dates = entry.get("haystack_dates", [])

        # 优先保留 answer sessions
        answer_indices = []
        filler_indices = []
        for idx, sid in enumerate(session_ids):
            # 检查是否为 answer session
            is_answer = (
                sid in answer_ids
                or any(aid in sid for aid in answer_ids)
            )
            if is_answer:
                answer_indices.append(idx)
            else:
                filler_indices.append(idx)

        # 取 answer sessions + 部分 filler sessions
        selected = answer_indices + filler_indices[: max(0, max_sessions - len(answer_indices))]
        selected.sort()  # 保持原始顺序

        prepared["haystack_session_ids"] = [session_ids[i] for i in selected]
        prepared["haystack_sessions"] = [sessions[i] for i in selected]
        prepared["haystack_dates"] = [dates[i] for i in selected]

    if user_only:
        for sess in prepared.get("haystack_sessions", []):
            # 原地过滤，只保留 user turn
            to_remove = [i for i, t in enumerate(sess) if t.get("role") != "user"]
            for i in reversed(to_remove):
                sess.pop(i)

    return prepared


def run_single_question(
    entry: dict[str, Any],
    top_k: int = 10,
    max_sessions: int = 0,
    user_only: bool = False,
    rrf_k: int = 10,
    llm_available: bool = False,
    api_key: str = "",
    base_url: str = "",
    gen_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    no_gen: bool = False,
) -> dict[str, Any]:
    """对单条问题执行完整评测流程。

    每个问题使用独立的临时目录，确保数据隔离。
    """
    question_id = entry["question_id"]
    question = entry["question"]
    question_type = entry["question_type"]
    gold_answer = entry.get("answer", "")
    is_abstention = "_abs" in question_id

    result: dict[str, Any] = {
        "question_id": question_id,
        "question_type": question_type,
        "question": question,
        "gold_answer": gold_answer,
        "is_abstention": is_abstention,
        "max_sessions": max_sessions,
        "user_only": user_only,
    }

    # 准备 ingest 数据（限制 sessions、过滤 turn）
    prepared_entry = _prepare_entry_for_ingest(entry, max_sessions, user_only)
    n_sessions_actual = len(prepared_entry.get("haystack_sessions", []))
    n_turns_actual = sum(len(s) for s in prepared_entry.get("haystack_sessions", []))
    result["actual_sessions"] = n_sessions_actual
    result["actual_turns"] = n_turns_actual

    # 使用独立临时目录，确保每题数据隔离
    tmp_dir = tempfile.mkdtemp(prefix=f"omnimem_lme_{question_id}_")

    try:
        # 步骤 1: Ingest — 将 haystack_sessions 写入 OmniMem
        print(f"  [{question_id[:8]}] ingest 开始 ({n_sessions_actual} sessions, {n_turns_actual} turns)...", end="", flush=True)
        t0 = time.perf_counter()
        sdk_config = {
            "rrf_k": rrf_k,
            "enable_reranker": True,  # 启用 Cross-Encoder 精排
            "vector_weight": 2.0,     # 向量通道权重降低（原 3.0）
            "bm25_weight": 2.0,       # BM25 通道权重提升（原 1.0）
            "recall_timeout_ms": 30000,  # 检索超时 30s（需配合 reranker 候选数限制）
        }
        provider = OmniMemMemoryProvider(storage_dir=tmp_dir, config=sdk_config)
        if not provider.is_ready:
            result["error"] = "SDK 初始化失败"
            result["ingest_count"] = 0
            result["retrieval_quality"] = {"skipped": True, "reason": "sdk_init_failed"}
            print(" FAILED", flush=True)
            return result

        ingest_count = provider.ingest_sessions([prepared_entry])
        ingest_time = time.perf_counter() - t0
        result["ingest_count"] = ingest_count
        result["ingest_time_s"] = round(ingest_time, 2)
        print(f" {ingest_time:.1f}s ({ingest_count}条)", end="", flush=True)

        # 步骤 2: Retrieval — 检索相关 context
        print(" → retr...", end="", flush=True)
        t1 = time.perf_counter()
        contexts = provider.search(question, top_k=top_k)
        retrieval_time = time.perf_counter() - t1
        result["retrieved_count"] = len(contexts)
        result["retrieval_time_s"] = round(retrieval_time, 2)
        result["retrieved_contexts"] = [c[:200] for c in contexts]  # 截断存储
        print(f" {retrieval_time:.1f}s ({len(contexts)}条)", end="", flush=True)

        # 步骤 2.5: 评估检索质量
        retrieval_quality = evaluate_retrieval_quality(contexts, entry)
        result["retrieval_quality"] = retrieval_quality

        # 步骤 3 & 4: Generation + Evaluation（需要 LLM API）
        if no_gen:
            # --no-gen 模式：跳过 LLM 生成，直接拼接检索结果
            result["prediction"] = "\n".join(contexts[:5])[:500] if contexts else ""
            result["is_correct"] = None  # 未评判
            result["note"] = "no-gen 模式，跳过 LLM 生成"
            print(" → skip-gen", flush=True)
        elif llm_available and contexts:
            # 生成答案
            print(" → gen...", end="", flush=True)
            t2 = time.perf_counter()
            prediction = _generate_answer_with_llm(
                question, contexts, api_key, base_url, model=gen_model,
            )
            gen_time = time.perf_counter() - t2
            result["prediction"] = prediction[:500]
            result["generation_time_s"] = round(gen_time, 2)
            print(f" {gen_time:.1f}s", end="", flush=True)

            # 评判正确性
            print(" → judge...", end="", flush=True)
            t3 = time.perf_counter()
            is_correct = _judge_answer_with_llm(
                question, gold_answer, prediction, question_type,
                is_abstention, api_key, base_url, model=judge_model,
            )
            judge_time = time.perf_counter() - t3
            result["is_correct"] = is_correct
            result["judge_time_s"] = round(judge_time, 2)
            mark = "✓" if is_correct else "✗"
            print(f" {judge_time:.1f}s → {mark}", flush=True)
        elif llm_available and not contexts:
            result["prediction"] = ""
            result["is_correct"] = False
            result["note"] = "检索无结果，跳过生成"
            print(" → no-ctx ✗", flush=True)
        else:
            # retrieval-only 模式：使用检索质量作为近似指标
            result["prediction"] = ""
            result["is_correct"] = None  # 未评判
            print(" → retrieval-only", flush=True)

        # 关闭 SDK
        provider.close()

    except Exception as e:
        logger.error("问题 %s 评测异常: %s", question_id, e, exc_info=True)
        result["error"] = str(e)
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return result


# ======================================================================
# 汇总与统计
# ======================================================================

# question_type → 能力维度映射（复用适配器逻辑）
_QUESTION_TYPE_TO_DIMENSION = {
    "single-session-user": "single-session-user",
    "single-session-assistant": "single-session-assistant",
    "single-session-preference": "single-session-preference",
    "multi-session": "multi-session",
    "temporal-reasoning": "temporal-reasoning",
    "knowledge-update": "knowledge-update",
}


def compute_scores(
    results: list[dict[str, Any]],
    llm_available: bool,
) -> dict[str, Any]:
    """计算各维度分数。"""
    # 按 question_type 分组
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        qt = r.get("question_type", "unknown")
        by_type[qt].append(r)

    scores: dict[str, Any] = {
        "mode": "full" if llm_available else "retrieval-only",
        "total_questions": len(results),
        "config": {
            "max_sessions": results[0].get("max_sessions", 0) if results else 0,
            "user_only": results[0].get("user_only", False) if results else False,
        },
    }

    # ── QA 正确率（仅 full 模式）──
    if llm_available:
        qa_by_type: dict[str, dict] = {}
        all_correct = []
        for qt, items in by_type.items():
            correct = [1 for r in items if r.get("is_correct") is True]
            total = [1 for r in items if r.get("is_correct") is not None]
            if total:
                acc = sum(correct) / len(total)
            else:
                acc = 0.0
            qa_by_type[qt] = {
                "accuracy": round(acc, 4),
                "correct": sum(correct),
                "total": len(total),
            }
            all_correct.extend(correct)
        scores["qa_accuracy"] = {
            "overall": round(sum(all_correct) / len(all_correct), 4) if all_correct else 0.0,
            "by_type": qa_by_type,
            "task_averaged": round(
                sum(v["accuracy"] for v in qa_by_type.values()) / len(qa_by_type), 4
            ) if qa_by_type else 0.0,
        }

    # ── 检索质量指标（两种模式都有）──
    retrieval_by_type: dict[str, dict] = {}
    all_session_hits = []
    all_turn_hits = []
    all_coverages = []

    for qt, items in by_type.items():
        # 跳过 abstention 问题（无明确答案 turn）
        evaluable = [r for r in items if not r.get("retrieval_quality", {}).get("skipped")]
        if not evaluable:
            retrieval_by_type[qt] = {"note": "no_evaluable_items"}
            continue

        session_hit_count = sum(
            1 for r in evaluable if r.get("retrieval_quality", {}).get("session_hit") is True
        )
        turn_hit_count = sum(
            1 for r in evaluable if r.get("retrieval_quality", {}).get("turn_hit") is True
        )
        coverages = [
            r["retrieval_quality"]["has_answer_coverage"]
            for r in evaluable
            if r.get("retrieval_quality", {}).get("has_answer_coverage") is not None
        ]
        avg_coverage = round(sum(coverages) / len(coverages), 4) if coverages else 0.0

        retrieval_by_type[qt] = {
            "session_hit_rate": round(session_hit_count / len(evaluable), 4),
            "turn_hit_rate": round(turn_hit_count / len(evaluable), 4),
            "avg_coverage": avg_coverage,
            "session_hits": session_hit_count,
            "turn_hits": turn_hit_count,
            "total_evaluable": len(evaluable),
        }

        all_session_hits.extend(
            [1 for r in evaluable if r.get("retrieval_quality", {}).get("session_hit") is True]
        )
        all_turn_hits.extend(
            [1 for r in evaluable if r.get("retrieval_quality", {}).get("turn_hit") is True]
        )
        all_coverages.extend(coverages)

    scores["retrieval_quality"] = {
        "overall_session_hit_rate": (
            round(sum(all_session_hits) / len(all_session_hits), 4) if all_session_hits else 0.0
        ),
        "overall_turn_hit_rate": (
            round(sum(all_turn_hits) / len(all_turn_hits), 4) if all_turn_hits else 0.0
        ),
        "overall_avg_coverage": (
            round(sum(all_coverages) / len(all_coverages), 4) if all_coverages else 0.0
        ),
        "by_type": retrieval_by_type,
    }

    # ── 性能指标 ──
    ingest_times = [r["ingest_time_s"] for r in results if "ingest_time_s" in r]
    retrieval_times = [r["retrieval_time_s"] for r in results if "retrieval_time_s" in r]
    ingest_counts = [r["ingest_count"] for r in results if "ingest_count" in r]

    scores["performance"] = {
        "avg_ingest_time_s": round(sum(ingest_times) / len(ingest_times), 2) if ingest_times else 0,
        "avg_retrieval_time_s": round(sum(retrieval_times) / len(retrieval_times), 2) if retrieval_times else 0,
        "avg_ingest_count": round(sum(ingest_counts) / len(ingest_counts), 1) if ingest_counts else 0,
        "total_ingest_time_s": round(sum(ingest_times), 2),
        "total_retrieval_time_s": round(sum(retrieval_times), 2),
    }

    return scores


# ======================================================================
# 入口
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniMem → LongMemEval-S 评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="LongMemEval-S 数据文件路径 (longmemeval_s_cleaned.json)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="结果输出目录",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="限制评测题目数 (0=全部)",
    )
    parser.add_argument(
        "--top-k", type=int, default=100,
        help="检索 top-k 数量 (默认 100)",
    )
    parser.add_argument(
        "--max-sessions", type=int, default=0,
        help="每题最大 session 数 (0=全部, 建议快速测试设为 10)",
    )
    parser.add_argument(
        "--user-only", action="store_true",
        help="只写入 user 角色的 turn（跳过 assistant turn，加速 ingest）",
    )
    parser.add_argument(
        "--rrf-k", type=int, default=10,
        help="RRF 融合参数 k（默认 10，越小高置信度结果权重越高）",
    )
    parser.add_argument(
        "--gen-model", type=str, default="deepseek-chat",
        help="生成答案使用的模型 (默认 deepseek-chat)",
    )
    parser.add_argument(
        "--judge-model", type=str, default="deepseek-chat",
        help="评判使用的模型 (默认 deepseek-chat)",
    )
    parser.add_argument(
        "--llm-base-url", type=str, default="https://api.deepseek.com",
        help="LLM API base URL (默认 DeepSeek 官方)",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="跳过 ingest 步骤（调试用，将导致检索无结果）",
    )
    parser.add_argument(
        "--no-gen", action="store_true",
        help="禁用 LLM 答案生成步骤（调试用，直接拼接检索结果）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ★ 抑制噪音日志：Saga/BM25/VectorStore 的 WARNING 太多，降到 ERROR
    for noisy in ("omnimem.core.saga", "omnimem.retrieval.bm25",
                  "omnimem.retrieval.vector_store", "omnimem.memory.meta_store",
                  "omnimem.services.memory_write_service", "jieba"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    # 检查数据文件
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("数据文件不存在: %s", data_path)
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查 LLM API
    llm_available, api_key, base_url = _check_llm_available(args.llm_base_url)
    mode_str = "full (检索+生成+评判)" if llm_available else "retrieval-only (仅检索评估)"
    logger.info("评测模式: %s", mode_str)

    # 加载数据
    logger.info("加载数据: %s", data_path)
    raw_data = OmniMemMemoryProvider.load_raw_data(data_path)
    logger.info("数据条数: %d", len(raw_data))

    # 统计 question_type 分布
    type_counter = Counter(d["question_type"] for d in raw_data)
    logger.info("question_type 分布: %s", dict(type_counter))

    # 限制题目数（分层采样：确保每种 question_type 都有代表）
    limit = args.limit if args.limit > 0 else len(raw_data)
    if limit >= len(raw_data):
        eval_data = raw_data
    else:
        # 按类型分组，每类型取 min(per_type, available) 题
        by_type: dict[str, list[dict]] = defaultdict(list)
        for d in raw_data:
            by_type[d["question_type"]].append(d)
        n_types = len(by_type)
        per_type = max(1, limit // n_types)
        eval_data: list[dict[str, Any]] = []
        for qtype, items in by_type.items():
            n_take = min(per_type, len(items))
            eval_data.extend(items[:n_take])
        # 如果有剩余名额，从最多的类型中补充
        remaining = limit - len(eval_data)
        if remaining > 0:
            for qtype, items in by_type.items():
                already_taken = min(per_type, len(items))
                extra = min(remaining, len(items) - already_taken)
                if extra > 0:
                    eval_data.extend(items[already_taken:already_taken + extra])
                    remaining -= extra
                if remaining <= 0:
                    break
    logger.info("评测题目数: %d (limit=%d, 类型数=%d)", len(eval_data), limit, n_types)

    # 逐题评测
    all_results: list[dict[str, Any]] = []
    total = len(eval_data)

    logger.info("=" * 50)
    logger.info("开始评测 (%d 题, top_k=%d, max_sessions=%s, user_only=%s, rrf_k=%d)",
                total, args.top_k,
                args.max_sessions if args.max_sessions > 0 else "全部",
                args.user_only, args.rrf_k)
    logger.info("=" * 50)

    for i, entry in enumerate(eval_data):
        qid = entry.get("question_id", f"unknown-{i}")
        qtype = entry.get("question_type", "unknown")
        n_sessions = len(entry.get("haystack_sessions", []))
        n_turns = sum(len(s) for s in entry.get("haystack_sessions", []))

        logger.info(
            "[%d/%d] question_id=%s type=%s sessions=%d turns=%d",
            i + 1, total, qid, qtype, n_sessions, n_turns,
        )

        # ★ 步骤计时：ingest → retrieval → generation → judge
        t_q_start = time.perf_counter()

        result = run_single_question(
            entry,
            top_k=args.top_k,
            max_sessions=args.max_sessions,
            user_only=args.user_only,
            rrf_k=args.rrf_k,
            llm_available=llm_available,
            api_key=api_key,
            base_url=base_url,
            gen_model=args.gen_model,
            judge_model=args.judge_model,
            no_gen=args.no_gen,
        )

        t_q_total = time.perf_counter() - t_q_start
        all_results.append(result)

        # ★ 立即打印本题结果（print 不走 logger 缓冲）
        _qid_short = qid[:12]
        _ingest_t = result.get("ingest_time_s", "?")
        _retr_t = result.get("retrieval_time_s", "?")
        _gen_t = result.get("generation_time_s", "?")
        _judge_t = result.get("judge_time_s", "?")
        _n_ctx = result.get("retrieved_count", 0)
        _qa_mark = "?"
        if result.get("is_correct") is True:
            _qa_mark = "✓"
        elif result.get("is_correct") is False:
            _qa_mark = "✗"

        # 检索质量
        rq = result.get("retrieval_quality", {})
        _sess_hit = "Y" if rq.get("session_hit") else "N"
        _turn_hit = "Y" if rq.get("turn_hit") else "N"

        # 累计准确率
        correct_count = sum(1 for r in all_results if r.get("is_correct") is True)
        judged_count = sum(1 for r in all_results if r.get("is_correct") is not None)
        acc_str = f"{correct_count}/{judged_count}={correct_count/judged_count*100:.1f}%" if judged_count > 0 else "N/A"

        # 进度条
        bar_len = 30
        filled = int(bar_len * (i + 1) / total)
        bar = "█" * filled + "░" * (bar_len - filled)

        # ★ print + sys.stdout.flush() 确保立即可见
        print(
            f"▌{bar}▌ {i+1}/{total} │ QA: {_qa_mark} │ 累计: {acc_str} │ "
            f"ingest={_ingest_t}s retr={_retr_t}s gen={_gen_t}s judge={_judge_t}s │ "
            f"ctx={_n_ctx} sess={_sess_hit} turn={_turn_hit} │ "
            f"total={t_q_total:.1f}s │ {qtype} {_qid_short}",
            flush=True,
        )

        # 同时写 logger（持久化到文件时有用）
        logger.warning(
            "▌%s▌ %d/%d │ QA: %s │ 累计: %s │ %s %s",
            bar, i + 1, total, _qa_mark, acc_str, qtype, _qid_short,
        )

        # 每 10 题输出进度摘要
        if (i + 1) % 10 == 0:
            done = [r for r in all_results if "error" not in r]
            session_hits = sum(
                1 for r in done
                if r.get("retrieval_quality", {}).get("session_hit") is True
            )
            turn_hits = sum(
                1 for r in done
                if r.get("retrieval_quality", {}).get("turn_hit") is True
            )
            logger.info(
                "--- 进度 %d/%d | session_hit=%d turn_hit=%d ---",
                i + 1, total, session_hits, turn_hits,
            )

    # 计算汇总分数
    logger.info("=" * 50)
    logger.info("计算汇总分数...")
    scores = compute_scores(all_results, llm_available)

    # 输出分数
    logger.info("=" * 50)
    logger.info("评测结果:")
    logger.info("  模式: %s", scores["mode"])
    logger.info("  总题数: %d", scores["total_questions"])

    # 检索质量
    rq = scores["retrieval_quality"]
    logger.info("  检索质量:")
    logger.info("    session_hit_rate: %.2f%%", rq["overall_session_hit_rate"] * 100)
    logger.info("    turn_hit_rate: %.2f%%", rq["overall_turn_hit_rate"] * 100)
    logger.info("    avg_coverage: %.2f%%", rq["overall_avg_coverage"] * 100)

    # 分维度
    logger.info("  分维度检索质量:")
    for qt, qt_scores in rq["by_type"].items():
        if "note" in qt_scores:
            logger.info("    %s: %s", qt, qt_scores["note"])
        else:
            logger.info(
                "    %s: session_hit=%.2f%% turn_hit=%.2f%% coverage=%.2f%%",
                qt,
                qt_scores["session_hit_rate"] * 100,
                qt_scores["turn_hit_rate"] * 100,
                qt_scores["avg_coverage"] * 100,
            )

    # QA 正确率（仅 full 模式）
    if llm_available and "qa_accuracy" in scores:
        qa = scores["qa_accuracy"]
        logger.info("  QA 正确率:")
        logger.info("    overall: %.2f%%", qa["overall"] * 100)
        logger.info("    task_averaged: %.2f%%", qa["task_averaged"] * 100)
        for qt, qa_scores in qa["by_type"].items():
            logger.info(
                "    %s: %.2f%% (%d/%d)",
                qt, qa_scores["accuracy"] * 100,
                qa_scores["correct"], qa_scores["total"],
            )

    # 性能
    perf = scores["performance"]
    logger.info("  性能:")
    logger.info("    平均 ingest 时间: %.2fs", perf["avg_ingest_time_s"])
    logger.info("    平均 retrieval 时间: %.2fs", perf["avg_retrieval_time_s"])

    # 保存结果
    scores_path = output_dir / "scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    logger.info("分数已保存: %s", scores_path)

    details_path = output_dir / "details.jsonl"
    with open(details_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("详细结果已保存: %s", details_path)

    logger.info("=" * 50)
    logger.info("评测完成!")


if __name__ == "__main__":
    main()
