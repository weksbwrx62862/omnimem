#!/usr/bin/env python3
"""OmniMem 抽取质量评测脚本。

加载 extraction_quality_eval.json 数据集，对每条对话执行规则提取，
计算精确率、召回率、F1、误提取率，输出 CI 友好的 JSON 报告。

用法:
    python benchmarks/run_extraction_eval.py              # 规则基线
    python benchmarks/run_extraction_eval.py --output out.json  # 指定输出
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ★ M5-2: --mode hybrid 时注入的 LLM 客户端（None = 纯规则模式）
_LLM_CLIENT = None

# 数据集中文关键词 → 事实归一化映射（用于模糊匹配打分）
_SYNONYM_MAP: dict[str, list[str]] = {
    "不喜欢": ["讨厌", "反感", "不爱"],
    "喜欢": ["爱", "偏好", "倾向"],
    "住": ["居住", "住在"],
    "养": ["饲养", "有"],
}


def _build_llm_client():
    """★ M5-2: 从 hermes 安装目录读取 LLM 凭证构建客户端（DEEPSEEK_API_KEY）。"""
    import os

    from omnimem.utils.llm_client import AsyncLLMClient

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    model = os.environ.get("OMNIMEM_LLM_MODEL", "")
    if not api_key:
        hermes_env = Path(os.environ.get("HERMES_HOME", r"C:\Users\13104\AppData\Local\hermes")) / ".env"
        if hermes_env.exists():
            for line in hermes_env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip(chr(34)).strip(chr(39))
                    break
    if not api_key:
        raise SystemExit("hybrid 模式需要 DEEPSEEK_API_KEY（环境变量或 hermes/.env）")
    if not model:
        model = "deepseek-chat"
    base_url = os.environ.get("OMNIMEM_LLM_BASE_URL", "https://api.deepseek.com")
    logger.warning("hybrid 模式: model=%s base_url=%s (key len=%d)", model, base_url, len(api_key))
    return AsyncLLMClient(api_key=api_key, base_url=base_url, model=model, timeout=60.0)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    """加载评测数据集。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    logger.info("加载数据集: %s, %d 条对话", data.get("name"), len(items))
    return items


def _extract_facts_rule(dialogue: str) -> list[str]:
    """规则提取：使用 AtomicFactExtractor 提取原子事实。"""
    try:
        from omnimem.perception.fact_extractor import AtomicFactExtractor

        extractor = AtomicFactExtractor(llm_client=_LLM_CLIENT)  # 无 LLM client → 纯规则模式
        return extractor.extract_facts(dialogue)
    except ImportError as e:
        logger.warning("AtomicFactExtractor 导入失败: %s，使用简单回退", e)
        return _simple_extract(dialogue)


def _simple_extract(text: str) -> list[str]:
    """简单回退提取：按句号/换行划分短句作为候选事实。"""
    import re

    parts = re.split(r"[。！？.!?\n]", text)
    return [p.strip() for p in parts if len(p.strip()) >= 5 and len(p.strip()) <= 100]


def _normalize(text: str) -> str:
    """归一化文本用于比较：去标点、去多余空格、转小写。"""
    import re

    text = text.strip().lower()
    # 移除常见标点
    text = re.sub(r"[，。！？、；：\u201c\u201d\u2018\u2019（）【】《》\s]+", "", text)
    return text


def _fact_match(extracted: str, expected: str) -> bool:
    """判断提取的事实是否与期望事实匹配（模糊匹配）。

    策略：
      1. 精确归一化匹配
      2. 子串包含（任一方向）
      3. 关键词交集 >= 50%
    """
    e_norm = _normalize(extracted)
    x_norm = _normalize(expected)

    if e_norm == x_norm:
        return True
    if e_norm in x_norm or x_norm in e_norm:
        return True

    # 关键词交集匹配
    e_words = set(e_norm)
    x_words = set(x_norm)
    if not e_words or not x_words:
        return False
    intersection = e_words & x_words
    # 任一方匹配率超过 50%
    return len(intersection) / min(len(e_words), len(x_words)) >= 0.5


def _evaluate_single(
    item: dict[str, Any],
) -> dict[str, Any]:
    """评估单条对话。"""
    dialogue = item["dialogue"]
    expected = item.get("should_extract", [])
    should_not = item.get("should_not_extract", [])

    extracted = _extract_facts_rule(dialogue)

    # 计算匹配
    matched_expected: set[int] = set()  # 被匹配的期望事实索引
    matched_extracted: set[int] = set()  # 正确提取的索引
    false_positives: list[str] = []

    for ei, ext_fact in enumerate(extracted):
        matched = False
        for xi, exp_fact in enumerate(expected):
            if xi in matched_expected:
                continue
            if _fact_match(ext_fact, exp_fact):
                matched_expected.add(xi)
                matched_extracted.add(ei)
                matched = True
                break
        if not matched:
            false_positives.append(ext_fact)

    true_positives = len(matched_extracted)
    false_negatives = len(expected) - len(matched_expected)
    false_positive_count = len(false_positives)

    # 不应提取但被提取了
    should_not_extracted: list[str] = []
    for sn in should_not:
        sn_norm = _normalize(sn)
        for ext_fact in extracted:
            if sn_norm in _normalize(ext_fact) or _normalize(ext_fact) in sn_norm:
                should_not_extracted.append(ext_fact)
                break

    return {
        "id": item["id"],
        "dialogue": dialogue[:100] + ("..." if len(dialogue) > 100 else ""),
        "capability": item.get("capability", ""),
        "extracted": extracted,
        "expected": expected,
        "true_positives": true_positives,
        "false_positives": false_positive_count,
        "false_negatives": false_negatives,
        "should_not_extracted": len(should_not_extracted),
        "should_not_count": len(should_not),
    }


def run_evaluation(dataset_path: Path, mode: str = "rule") -> dict[str, Any]:
    """运行全量评测。"""
    items = _load_dataset(dataset_path)
    results: list[dict[str, Any]] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_should_not_hit = 0
    total_should_not = 0

    by_capability: dict[str, dict[str, int]] = {}

    for item in items:
        r = _evaluate_single(item)
        results.append(r)

        total_tp += r["true_positives"]
        total_fp += r["false_positives"]
        total_fn += r["false_negatives"]
        total_should_not_hit += r["should_not_extracted"]
        total_should_not += r["should_not_count"]

        cap = item.get("capability", "unknown")
        if cap not in by_capability:
            by_capability[cap] = {"tp": 0, "fp": 0, "fn": 0, "count": 0}
        by_capability[cap]["tp"] += r["true_positives"]
        by_capability[cap]["fp"] += r["false_positives"]
        by_capability[cap]["fn"] += r["false_negatives"]
        by_capability[cap]["count"] += 1

    # 计算总体指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    false_positive_rate = (
        total_should_not_hit / total_should_not if total_should_not > 0 else 0.0
    )

    # 按能力维度统计
    capability_metrics = {}
    for cap, m in by_capability.items():
        p = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0.0
        r = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        capability_metrics[cap] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "count": m["count"],
        }

    report = {
        "dataset": "extraction_quality_eval_v1",
        "mode": mode,
        "total_items": len(items),
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_hit_rate": round(false_positive_rate, 4),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
        },
        "by_capability": capability_metrics,
        "details": results,
    }

    return report


def print_summary(report: dict[str, Any]) -> None:
    """打印人类可读的评测摘要。"""
    overall = report["overall"]
    print()
    print("=" * 60)
    print("  OmniMem 抽取质量评测报告")
    print("=" * 60)
    print(f"  数据集: {report['dataset']}")
    print(f"  抽取模式: {report['mode']}")
    print(f"  对话数: {report['total_items']}")
    print()
    print("  ── 总体指标 ──")
    print(f"  精确率 (Precision): {overall['precision']:.2%}")
    print(f"  召回率 (Recall):    {overall['recall']:.2%}")
    print(f"  F1 分数:            {overall['f1']:.2%}")
    print(f"  误提取率 (FP Rate): {overall['false_positive_hit_rate']:.2%}")
    print(f"  TP={overall['true_positives']}  FP={overall['false_positives']}  FN={overall['false_negatives']}")
    print()
    print("  ── 按能力维度 ──")
    for cap in sorted(report.get("by_capability", {}).keys()):
        cm = report["by_capability"][cap]
        print(
            f"  {cap:30s}  P={cm['precision']:.2%}  R={cm['recall']:.2%}  F1={cm['f1']:.2%}  n={cm['count']}"
        )
    print()
    # 通过判定（★ 不使用 emoji：Windows GBK 控制台会 UnicodeEncodeError）
    if overall["f1"] >= 0.50:
        print("  [PASS] 规则基线达成（F1 >= 50%）")
    else:
        print(f"  [WARN] 规则基线未达标（F1={overall['f1']:.2%} < 50%）")
    print("=" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniMem 抽取质量评测")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "extraction_quality_eval.json",
        help="评测数据集 JSON 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="结果输出 JSON 路径（默认输出到 stdout）",
    )
    parser.add_argument(
        "--mode",
        choices=["rule", "hybrid"],
        default="rule",
        help="rule=纯规则, hybrid=LLM 驱动（凭证自动从 hermes 读取）",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI 模式：仅输出 JSON，不打印摘要",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    if args.mode == "hybrid":
        globals()["_LLM_CLIENT"] = _build_llm_client()
    report = run_evaluation(args.data, mode=args.mode)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告已写入: {args.output}")

    if not args.ci:
        print_summary(report)

    # CI 模式下根据 F1 判断通过/失败
    f1 = report["overall"]["f1"]
    if args.ci and f1 < 0.50:
        sys.exit(1)


if __name__ == "__main__":
    main()
