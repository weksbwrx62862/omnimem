"""LongMemEval 分层抽样脚本（诊断 Task 2 用）。

从 longmemeval_s_cleaned.json 按 question_type 分层抽样：
  - 每题型固定 N 题（默认 10），固定随机种子保证可复现
  - 排除 abstention 题（question_id 含 "_abs"）：
    其答案 turn 缺失导致检索质量评估被跳过（skipped），
    无法参与「构造/检索/阅读」三分归因，故不纳入本基线
  - 输出保持原数据格式不变（仅子集），可直接喂给 run_longmemeval.py

用法:
    .venv/bin/python benchmarks/diagnosis/make_stratified_sample.py \
        --data benchmarks/LongMemEval/data/longmemeval_s_cleaned.json \
        --output benchmarks/results/diagnosis/lme_sample_60.json \
        --per-type 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="LongMemEval 分层抽样")
    parser.add_argument(
        "--data", type=str, required=True,
        help="原始数据文件路径 (longmemeval_s_cleaned.json)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="抽样输出文件路径 (json，格式与原数据一致)",
    )
    parser.add_argument(
        "--per-type", type=int, default=10,
        help="每题型抽取题数（默认 10）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42，保证可复现）",
    )
    parser.add_argument(
        "--include-abstention", action="store_true",
        help="是否包含 abstention 题（默认排除，因其无法做三分归因）",
    )
    parser.add_argument(
        "--type-order", type=str, default="",
        help="输出文件中的题型顺序（逗号分隔）；默认按题型名排序。"
        "影响输出排序，不影响抽样成员（成员由 --seed 决定）",
    )
    parser.add_argument(
        "--sacrificial", type=int, default=0,
        help="在文件末尾追加 N 道「牺牲题」（取最后一个题型的剩余题）。"
        "用于规避 run_longmemeval.py 的启动 bug：当 --limit >= 数据条数时 "
        "n_types 未定义直接崩溃，导致无法全量评测；追加牺牲题后可用 "
        "--limit <正式题数> 让每题型恰好取满，牺牲题永不被评测",
    )
    return parser.parse_args()


def main() -> None:
    """主流程：按题型分层抽样并写出。"""
    args = parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    # 过滤 abstention 题（除非显式要求包含）
    if not args.include_abstention:
        pool = [d for d in data if "_abs" not in d.get("question_id", "")]
        n_excluded = len(data) - len(pool)
    else:
        pool = list(data)
        n_excluded = 0

    # 按题型分组
    by_type: dict[str, list[dict]] = defaultdict(list)
    for d in pool:
        by_type[d["question_type"]].append(d)

    # 分层抽样（固定种子）
    rng = random.Random(args.seed)
    sample: list[dict] = []
    for qtype in sorted(by_type.keys()):
        items = by_type[qtype]
        # 先打乱再取前 N，避免固定取数据文件头部导致样本偏置
        shuffled = list(items)
        rng.shuffle(shuffled)
        take = shuffled[: args.per_type]
        sample.extend(take)
        print(
            f"  {qtype:<28} 可抽 {len(items):>3} → 抽取 {len(take)}"
            f"{' (不足)' if len(take) < args.per_type else ''}"
        )

    # 按题型排序后重排样本顺序（题型内保持随机序），
    # 与 run_longmemeval.py 内部 limit 分层逻辑解耦
    sample_by_type: dict[str, list[dict]] = defaultdict(list)
    for d in sample:
        sample_by_type[d["question_type"]].append(d)
    if args.type_order:
        # 用户显式指定题型顺序（未列出的题型排最后，按名排序）
        specified = [t.strip() for t in args.type_order.split(",") if t.strip()]
        rest = sorted(t for t in sample_by_type if t not in specified)
        type_order = [t for t in specified if t in sample_by_type] + rest
    else:
        type_order = sorted(sample_by_type.keys())
    ordered: list[dict] = []
    for qtype in type_order:
        ordered.extend(sample_by_type[qtype])

    # 追加牺牲题（不会被评测，仅用于规避 --limit 启动 bug，见 --sacrificial 说明）
    n_sacrificial = 0
    if args.sacrificial > 0 and type_order:
        last_type = type_order[-1]
        chosen_ids = {d["question_id"] for d in ordered}
        leftovers = [
            d for d in by_type[last_type] if d["question_id"] not in chosen_ids
        ]
        # 用同一 rng 继续抽取，保持可复现
        n_sacrificial = min(args.sacrificial, len(leftovers))
        if n_sacrificial:
            extra_pick = random.Random(args.seed + 1).sample(leftovers, n_sacrificial)
            ordered.extend(extra_pick)

    # 写出（保持原格式：顶层为 list，元素结构不变）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False)

    print(f"\n抽样完成: 共 {len(ordered)} 题 → {output_path}")
    print(f"题型分布: {dict(Counter(d['question_type'] for d in ordered))}")
    if n_sacrificial:
        print(
            f"含牺牲题 {n_sacrificial} 道（置于文件末尾，"
            f"评测时用 --limit {len(ordered) - n_sacrificial} 规避启动 bug，不会被评测）"
        )
    if n_excluded:
        print(f"已排除 abstention 题: {n_excluded} 题（可用 --include-abstention 保留）")


if __name__ == "__main__":
    main()
