"""LongMemEval 错题三分归因脚本（诊断 Task 2 核心交付物）。

对评测输出目录（details.jsonl）中的每道错题做三分归因：
  - 阅读丢失（reading_loss）  : 答案证据已进入检索上下文，但 LLM 生成/判分错误
  - 检索丢失（retrieval_loss） : 证据已写入记忆库，但未被检索回
        · turn 级：答案 session 有内容被检回，但 has_answer 答案 turn 本身未检回
        · session 级：答案 session 完全未进 top-k
  - 构造丢失（construction_loss）: 证据在写入阶段即丢失，三种子情况：
        · write_filter：答案 turn 是低密度 assistant 回复，被 _is_substantive 过滤跳过
        · session_cutoff：答案 turn 所在 session 被 --max-sessions 截断，从未参与写入
        · ingest_ratio：整体写入量不足（ingest_count/actual_turns 低于阈值）
  - 不确定（uncertain）       : 启发式信号矛盾或处于边界（见下方判据）

判据均为启发式，主要局限：
  1. ingest_count 只反映「成功写入条数」，无法证明特定答案 turn 一定写入成功
     （去重、分段、异常静默失败都可能丢内容），故「写入完整但检索未命中」时
     无法 100% 排除构造问题——本脚本将 ingest 比例处于 [low, hi) 边界带的
     这类题归为「不确定」；
  2. turn_hit / coverage 基于前 80 字符模糊匹配，截断的检索上下文（200 字符）
     可能低估真实重叠；
  3. details.jsonl 中 retrieved_contexts 被截断至 200 字符，重叠复算只作辅助信号。

用法：
    .venv/bin/python benchmarks/diagnosis/attribution.py \
        --data-dir benchmarks/results/diagnosis/lme_baseline \
        --data benchmarks/results/diagnosis/lme_sample_60.json

输出（默认写入 <data-dir>/attribution/）：
    attribution_details.jsonl  逐题归因明细（含全部中间信号）
    attribution_summary.json    按题型汇总（机器可读）
    attribution_summary.md     按题型归因分布表（人读）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# 将项目根目录加入 sys.path，以便复用评测脚本中的既有逻辑
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 复用适配器的 assistant 过滤规则（判断答案 turn 是否会在写入时被跳过）
from benchmarks.longmemeval_adapter import (  # noqa: E402
    OmniMemMemoryProvider,
)

# 复用评测主脚本的 session 选择与答案 turn 构建（保证与评测时口径完全一致）
from benchmarks.run_longmemeval import (  # noqa: E402
    _prepare_entry_for_ingest,
    build_answer_turn_contents,
)

# 归因类别常量
READING = "reading_loss"            # 阅读丢失：证据在上下文中但答错
RETRIEVAL = "retrieval_loss"         # 检索丢失：已写入未检回
CONSTRUCTION = "construction_loss"   # 构造丢失：写入阶段即丢失
UNCERTAIN = "uncertain"              # 不确定：边界/矛盾
UNATTRIBUTABLE = "unattributable"    # 无法归因（跳过题/缺数据/评测异常）

# 类别中文名（用于 markdown 输出）
CN_NAME = {
    READING: "阅读丢失",
    RETRIEVAL: "检索丢失",
    CONSTRUCTION: "构造丢失",
    UNCERTAIN: "不确定",
    UNATTRIBUTABLE: "无法归因",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="LongMemEval 错题三分归因")
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="评测输出目录（含 details.jsonl）",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="评测用的 LongMemEval 数据文件（json，含 haystack_sessions 原始数据）",
    )
    parser.add_argument(
        "--output-dir", type=str, default="",
        help="归因输出目录（默认 <data-dir>/attribution）",
    )
    parser.add_argument(
        "--low-ratio", type=float, default=0.8,
        help="ingest_count/actual_turns 低于该阈值视为疑似构造丢失（默认 0.8）",
    )
    parser.add_argument(
        "--hi-ratio", type=float, default=0.9,
        help="ingest 比例处于 [low, hi) 且检索未命中时归为不确定（默认 0.9）",
    )
    return parser.parse_args()


def fuzzy_in(content: str, retrieved_text: str) -> bool:
    """复刻评测脚本的模糊匹配：取内容前 80 字符做子串匹配。"""
    return bool(content) and content[:80] in retrieved_text


def collect_answer_turns(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    """收集 prepared 数据中所有 has_answer=true 且内容非空的 turn。"""
    turns = []
    for sess in prepared.get("haystack_sessions", []):
        for turn in sess:
            if turn.get("has_answer") and turn.get("content", "").strip():
                turns.append(turn)
    return turns


def estimate_writable(answer_turns: list[dict[str, Any]]) -> tuple[int, int, list[str]]:
    """估算答案 turn 的「可写入率」。

    复用适配器写入路径的跳过规则：assistant 角色且未通过
    _is_substantive 实质性过滤的 turn 不会被写入记忆库。

    Returns:
        (可写入数, 总数, 被跳过原因列表)
    """
    writable = 0
    reasons: list[str] = []
    for turn in answer_turns:
        role = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if not content:
            reasons.append("empty_content")
            continue
        if role == "assistant" and not OmniMemMemoryProvider._is_substantive(content):
            reasons.append("assistant_not_substantive")
            continue
        writable += 1
    return writable, len(answer_turns), reasons


def classify_question(
    record: dict[str, Any],
    entry: dict[str, Any] | None,
    low_ratio: float,
    hi_ratio: float,
) -> dict[str, Any]:
    """对单道错题做三分归因，返回含全部中间信号的归因记录。

    判定顺序（先排除、后细分）：
      0. 非错题 / 数据缺失 / 检索质量被跳过 / 无答案 turn → 不归因或无法归因
      1. turn_hit=True 或复算证据重叠>0 → 阅读丢失（证据在上下文中）
      2. 构造丢失：答案 turn 被 max_sessions 截断 / 被写入过滤跳过 / 写入量不足
      3. session_hit=True → 检索丢失（turn 级）
      4. session_hit=False 时按 ingest 比例细分：
         < low_ratio → 构造丢失（写入量不足，疑似）
         [low, hi)   → 不确定（边界带，无法完全区分构造/检索）
         >= hi_ratio → 检索丢失（session 级）
    """
    qid = record.get("question_id", "unknown")
    qtype = record.get("question_type", "unknown")
    rq = record.get("retrieval_quality", {}) or {}

    out: dict[str, Any] = {
        "question_id": qid,
        "question_type": qtype,
        "question": record.get("question", ""),
        "is_correct": record.get("is_correct"),
        "signals": {},       # 全部中间信号（供人工复核）
        "attribution": None,  # 归因类别
        "attribution_sub": "",  # 归因子类
        "reason": "",          # 归因依据说明
    }

    # ---- 信号收集 ----
    ingest_count = record.get("ingest_count")
    actual_turns = record.get("actual_turns")
    ingest_ratio = (
        round(ingest_count / actual_turns, 4)
        if isinstance(ingest_count, (int, float))
        and isinstance(actual_turns, (int, float)) and actual_turns > 0
        else None
    )
    turn_hit = rq.get("turn_hit")
    session_hit = rq.get("session_hit")
    coverage = rq.get("has_answer_coverage")

    # 复算证据重叠（对照被截断的 retrieved_contexts，仅作辅助/下限信号）
    evidence_overlap = None
    session_overlap = None
    if entry is not None:
        retrieved_text = " ".join(record.get("retrieved_contexts") or [])
        session_contents, answer_turn_contents, _ = build_answer_turn_contents(entry)
        if answer_turn_contents:
            evidence_overlap = round(
                sum(1 for c in answer_turn_contents if fuzzy_in(c, retrieved_text))
                / len(answer_turn_contents), 4,
            )
        if session_contents:
            session_overlap = round(
                sum(1 for c in session_contents if fuzzy_in(c, retrieved_text))
                / len(session_contents), 4,
            )

    # 复刻评测时的 session 选择，检查答案 turn 是否被 max_sessions 截断
    answer_turns_in_prepared: list[dict[str, Any]] = []
    total_answer_turns = 0
    writable = writable_total = 0
    skipped_reasons: list[str] = []
    prep_consistent = None
    if entry is not None:
        prepared = _prepare_entry_for_ingest(
            entry,
            max_sessions=record.get("max_sessions", 0) or 0,
            user_only=record.get("user_only", False),
        )
        # 一致性校验：复刻结果应与评测时记录的 actual_sessions/turns 一致
        prep_consistent = (
            len(prepared.get("haystack_sessions", [])) == record.get("actual_sessions")
            and sum(len(s) for s in prepared.get("haystack_sessions", []))
            == record.get("actual_turns")
        )
        answer_turns_in_prepared = collect_answer_turns(prepared)
        # 全量数据中 has_answer turn 数（跨所有 session，含被截断的 filler session）
        total_answer_turns = sum(
            1
            for sess in entry.get("haystack_sessions", [])
            for t in sess
            if t.get("has_answer") and t.get("content", "").strip()
        )
        writable, writable_total, skipped_reasons = estimate_writable(
            answer_turns_in_prepared,
        )

    out["signals"] = {
        "turn_hit": turn_hit,
        "session_hit": session_hit,
        "has_answer_coverage": coverage,
        "ingest_count": ingest_count,
        "actual_turns": actual_turns,
        "ingest_ratio": ingest_ratio,
        "evidence_overlap_recomputed": evidence_overlap,
        "session_overlap_recomputed": session_overlap,
        "answer_turns_total_in_data": total_answer_turns,
        "answer_turns_in_prepared": len(answer_turns_in_prepared),
        "answer_turns_writable": writable,
        "answer_turns_writable_ratio": (
            round(writable / total_answer_turns, 4) if total_answer_turns else None
        ),
        "answer_turns_skipped_reasons": skipped_reasons,
        "prepared_replication_consistent": prep_consistent,
        "eval_error": record.get("error"),
    }

    # ---- 归因判定 ----
    # 0a. 非错题：不做归因（信号已保留供参考）
    if record.get("is_correct") is not False:
        out["reason"] = "非错题（is_correct != False），不做归因"
        return out
    # 0b. 数据/评测层面的无法归因
    if entry is None:
        out["attribution"] = UNATTRIBUTABLE
        out["reason"] = "数据文件中找不到该题（question_id 不匹配）"
        return out
    if record.get("error"):
        out["attribution"] = UNATTRIBUTABLE
        out["reason"] = f"评测本身异常: {record.get('error')}"
        return out
    if rq.get("skipped"):
        out["attribution"] = UNATTRIBUTABLE
        out["reason"] = f"检索质量评估被跳过: {rq.get('reason', 'unknown')}"
        return out
    if not total_answer_turns:
        out["attribution"] = UNATTRIBUTABLE
        out["reason"] = "数据中无 has_answer=true 的答案 turn"
        return out

    # 1. 阅读丢失：证据已在检索上下文中（记录值或复算重叠任一命中）
    if turn_hit is True or (coverage or 0) > 0 or (evidence_overlap or 0) > 0:
        out["attribution"] = READING
        out["reason"] = "答案证据已进入检索上下文，但生成答案错误"
        if turn_hit is not True:
            out["attribution_sub"] = "overlap_only"
            out["reason"] += "（turn_hit=False 但复算重叠>0，命中强度较弱）"
        elif 0 < (coverage or 0) < 0.5:
            out["attribution_sub"] = "partial_coverage"
            out["reason"] += "（coverage<0.5，证据仅部分进入上下文）"
        # 附加提示：部分答案 turn 未写入（阅读段之外还叠加了构造段问题）
        if writable < total_answer_turns:
            out["reason"] += (
                f"；另注意 {total_answer_turns - writable}/{total_answer_turns} 个"
                "答案 turn 未进入写入范围（构造段也部分受损）"
            )
        return out

    # 2. 构造丢失：答案 turn 在写入阶段即丢失
    # 2a. 答案 turn 所在 session 被 max_sessions 截断（从未参与写入）
    if len(answer_turns_in_prepared) < total_answer_turns:
        out["attribution"] = CONSTRUCTION
        out["attribution_sub"] = "session_cutoff"
        out["reason"] = (
            f"答案 turn 被截断：数据中 {total_answer_turns} 个答案 turn，"
            f"仅 {len(answer_turns_in_prepared)} 个位于实际写入的 session 内"
        )
        return out
    # 2b. 答案 turn 全部被写入过滤跳过（低密度 assistant 回复）
    if writable_total > 0 and writable == 0:
        out["attribution"] = CONSTRUCTION
        out["attribution_sub"] = "write_filter"
        out["reason"] = "全部答案 turn 为低密度 assistant 回复，写入时被过滤跳过"
        return out
    # 2c. 部分答案 turn 被过滤（多跳证据部分缺失）
    if writable < writable_total:
        out["attribution"] = CONSTRUCTION
        out["attribution_sub"] = "write_filter_partial"
        out["reason"] = (
            f"{writable_total - writable}/{writable_total} 个答案 turn "
            "会被写入过滤跳过，多跳证据部分缺失"
        )
        return out
    # 2d. 整体写入量不足（按比例阈值，疑似构造丢失）
    if ingest_ratio is not None and ingest_ratio < low_ratio:
        out["attribution"] = CONSTRUCTION
        out["attribution_sub"] = "ingest_ratio"
        out["reason"] = (
            f"写入量不足：ingest_ratio={ingest_ratio} < {low_ratio}，"
            "证据可能未写入记忆库（疑似）"
        )
        return out

    # 3. 检索丢失（turn 级）：session 命中但答案 turn 未检回
    if session_hit is True:
        out["attribution"] = RETRIEVAL
        out["attribution_sub"] = "turn_level"
        out["reason"] = "答案 session 有内容被检回，但含答案的 turn 本身未进上下文"
        return out

    # 4. session 也未命中：按 ingest 比例区分构造 / 检索 / 不确定
    if ingest_ratio is not None and ingest_ratio < hi_ratio:
        out["attribution"] = UNCERTAIN
        out["reason"] = (
            f"检索未命中且 ingest_ratio={ingest_ratio} 处于边界带 "
            f"[{low_ratio}, {hi_ratio})，无法完全区分构造/检索"
        )
        return out
    out["attribution"] = RETRIEVAL
    out["attribution_sub"] = "session_level"
    out["reason"] = (
        "写入量充分（"
        + (f"ingest_ratio={ingest_ratio}" if ingest_ratio is not None else "ingest_ratio 未知")
        + "）但答案 session 完全未进 top-k"
    )
    return out


def summarize(attributions: list[dict[str, Any]], total_by_type: dict[str, int]) -> dict[str, Any]:
    """按题型汇总归因分布，并标注主导损失段。"""
    wrong_by_type: dict[str, list[dict]] = defaultdict(list)
    for a in attributions:
        if a.get("is_correct") is False:
            wrong_by_type[a["question_type"]].append(a)

    summary: dict[str, Any] = {}
    for qtype in sorted(set(total_by_type) | set(wrong_by_type)):
        items = wrong_by_type.get(qtype, [])
        counts = Counter(a["attribution"] for a in items)
        n_wrong = len(items)
        total = total_by_type.get(qtype, 0)
        n_correct = total - n_wrong

        # 主导损失段：三类损失中占比最高者（平票并列；不确定/无法归因不参与
        # 主导判定，但若其数量超过三类损失最大值，在 note 中说明）
        loss_counts = {
            READING: counts.get(READING, 0),
            RETRIEVAL: counts.get(RETRIEVAL, 0),
            CONSTRUCTION: counts.get(CONSTRUCTION, 0),
        }
        if n_wrong > 0 and max(loss_counts.values()) > 0:
            top_n = max(loss_counts.values())
            dominant = [k for k, v in loss_counts.items() if v == top_n]
        else:
            dominant = []

        note = ""
        if counts.get(UNCERTAIN, 0) > max(loss_counts.values(), default=0):
            note = "不确定类占比超过三类损失，主导判定置信度低"

        # 检索丢失细分（turn 级 / session 级）
        retr = counts.get(RETRIEVAL, 0)
        retr_sub = Counter(
            a["attribution_sub"] for a in items if a["attribution"] == RETRIEVAL
        )
        summary[qtype] = {
            "total": total,
            "correct": n_correct,
            "wrong": n_wrong,
            "accuracy": round(n_correct / total, 4) if total else None,
            "wrong_distribution": {
                k: {
                    "count": counts.get(k, 0),
                    "ratio_of_wrong": round(counts.get(k, 0) / n_wrong, 4) if n_wrong else 0.0,
                    "ratio_of_type": round(counts.get(k, 0) / total, 4) if total else 0.0,
                }
                for k in (READING, RETRIEVAL, CONSTRUCTION, UNCERTAIN, UNATTRIBUTABLE)
                if counts.get(k, 0) > 0
            },
            "retrieval_sub": dict(retr_sub) if retr else {},
            "dominant_loss": dominant,
            "note": note,
        }
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    """渲染按题型的归因分布 markdown 表。"""
    lines: list[str] = []
    lines.append("# LongMemEval 错题三分归因汇总")
    lines.append("")
    lines.append(
        "归因类别：阅读丢失=证据已进上下文但答错；检索丢失=已写入未检回"
        "（session 级/turn 级）；构造丢失=写入阶段即丢失"
        "（写入过滤/max_sessions 截断/写入量不足）；不确定=启发式信号边界。"
        "判据为启发式，局限见 attribution.py 头部 docstring。"
    )
    lines.append("")
    lines.append(
        "| 题型 | 正确/总数 | 错题数 | 阅读丢失 | 检索丢失 | 构造丢失 | 不确定 | 无法归因 | 主导损失段 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for qtype, s in sorted(summary.items()):
        dist = s["wrong_distribution"]

        def fmt(key: str, _dist: dict = dist) -> str:
            d = _dist.get(key)
            return f"{d['count']} ({d['ratio_of_wrong'] * 100:.0f}%)" if d else "0"

        dominant = (
            "、".join(CN_NAME[d] for d in s["dominant_loss"]) if s["dominant_loss"] else "—"
        )
        acc = f"{s['correct']}/{s['total']}"
        lines.append(
            f"| {qtype} | {acc} | {s['wrong']} | {fmt(READING)} | {fmt(RETRIEVAL)} "
            f"| {fmt(CONSTRUCTION)} | {fmt(UNCERTAIN)} | {fmt(UNATTRIBUTABLE)} | {dominant} |"
        )
    lines.append("")
    # 检索丢失细分注记
    sub_notes = [(q, s["retrieval_sub"]) for q, s in summary.items() if s["retrieval_sub"]]
    if sub_notes:
        lines.append(
            "**检索丢失细分**（turn 级 = 答案 session 有内容被检回但答案 turn 未检回；"
            "session 级 = 答案 session 完全未进 top-k）："
        )
        for q, sub in sub_notes:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(sub.items()))
            lines.append(f"- {q}: {parts}")
        lines.append("")
    notes = [(q, s["note"]) for q, s in summary.items() if s.get("note")]
    if notes:
        lines.append("**注记**：")
        for q, n in notes:
            lines.append(f"- {q}: {n}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """主流程：读评测明细 → 逐题归因 → 汇总输出。"""
    args = parse_args()

    data_dir = Path(args.data_dir)
    details_path = data_dir / "details.jsonl"
    if not details_path.exists():
        print(f"[错误] 找不到 {details_path}", file=sys.stderr)
        sys.exit(1)

    data_path = Path(args.data)
    with open(data_path, encoding="utf-8") as f:
        data_by_qid = {d["question_id"]: d for d in json.load(f)}

    # 读取评测明细
    records: list[dict[str, Any]] = []
    with open(details_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"读取评测明细: {len(records)} 条（来自 {details_path}）")
    print(f"数据文件: {data_path}（{len(data_by_qid)} 题）")

    # 逐题归因（只归因错题；正确题与未判分题跳过）
    attributions: list[dict[str, Any]] = []
    n_wrong = 0
    for record in records:
        entry = data_by_qid.get(record.get("question_id"))
        a = classify_question(record, entry, args.low_ratio, args.hi_ratio)
        if record.get("is_correct") is False:
            n_wrong += 1
            print(
                f"  [错题] {a['question_type']:<26} {a['question_id'][:12]:<12} "
                f"→ {CN_NAME.get(a['attribution'], a['attribution'])}"
                f"{'/' + a['attribution_sub'] if a['attribution_sub'] else ''}"
            )
        attributions.append(a)

    # 题型总数（以评测明细为准）
    total_by_type: dict[str, int] = dict(Counter(
        r.get("question_type", "unknown") for r in records
    ))

    summary = summarize(attributions, total_by_type)
    markdown = render_markdown(summary)

    # 输出
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "attribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    details_out = output_dir / "attribution_details.jsonl"
    with open(details_out, "w", encoding="utf-8") as f:
        for a in attributions:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    summary_out = output_dir / "attribution_summary.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_out = output_dir / "attribution_summary.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"\n错题总数: {n_wrong}")
    print(f"归因明细: {details_out}")
    print(f"汇总(json): {summary_out}")
    print(f"汇总(md):  {md_out}")
    print()
    print(markdown)


if __name__ == "__main__":
    main()
