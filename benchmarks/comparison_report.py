#!/usr/bin/env python3
"""
OmniMem 与开源项目 LongMemEval 评测对比报告生成器。

功能：
1. 内置开源项目的 LongMemEval 评测分数（硬编码，基于公开调研数据）。
2. 读取 OmniMem 最新评测结果 JSON 文件（benchmarks/results/ 下最新的 scores.json）。
3. 生成 Markdown 对比报告，包含总分对比表、分维度对比表、评测条件标注与重要警示。
4. 支持命令行参数 --result-dir 与 --output。

重要警示：不同项目的分数不可直接比较，因为使用了不同的 generator LLM、judge LLM、retrieval budget。
本脚本独立运行，不依赖 OmniMem 的其他模块。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ======================================================================
# 开源项目 LongMemEval 评测分数（基于公开调研数据硬编码）
# ======================================================================
OPEN_SOURCE_SCORES: dict[str, dict[str, Any]] = {
    "Mem0 Platform (v3)": {
        "overall": 94.4,
        "generator": "GPT-5 (内部)",
        "judge": "GPT-5",
        "top_k": 200,
        "oracle": False,
        "by_type": {
            "single-session-user": 98.6,
            "single-session-assistant": 98.2,
            "single-session-preference": 96.7,
            "knowledge-update": 93.6,
            "temporal-reasoning": 97.0,
            "multi-session": 88.0,
        },
        "source": "mem0ai/memory-benchmarks (2026/04)",
    },
    "Mem0 OSS (GPT-5 抽取)": {
        "overall": 91.0,
        "generator": "GPT-5",
        "judge": "GPT-5",
        "top_k": 200,
        "oracle": False,
        "by_type": {
            "single-session-user": 95.7,
            "single-session-assistant": 92.9,
            "single-session-preference": 93.3,
            "knowledge-update": 91.0,
            "temporal-reasoning": 94.7,
            "multi-session": 83.5,
        },
        "source": "mem0ai/memory-benchmarks (2026/04)",
    },
    "HeLa-Mem": {
        "overall": 65.4,
        "generator": "gpt-4o-mini",
        "judge": "gpt-4o-mini",
        "top_k": "N/A",
        "oracle": False,
        "by_type": {},
        "source": "ACL 2026 论文",
    },
    "A-MEM": {
        "overall": 62.6,
        "generator": "gpt-4o-mini",
        "judge": "gpt-4o-mini",
        "top_k": "N/A",
        "oracle": False,
        "by_type": {},
        "source": "HeLa-Mem 论文复现",
    },
    "NaiveRAG": {
        "overall": 61.0,
        "generator": "gpt-4o-mini",
        "judge": "gpt-4o-mini",
        "top_k": "N/A",
        "oracle": False,
        "by_type": {},
        "source": "HeLa-Mem 论文复现",
    },
    "FullText": {
        "overall": 56.8,
        "generator": "gpt-4o-mini",
        "judge": "gpt-4o-mini",
        "top_k": "N/A",
        "oracle": False,
        "by_type": {},
        "source": "HeLa-Mem 论文复现",
    },
    "Mem0 (HeLa-Mem 复现)": {
        "overall": 53.61,
        "generator": "gpt-4o-mini",
        "judge": "gpt-4o-mini",
        "top_k": "N/A",
        "oracle": False,
        "by_type": {},
        "source": "HeLa-Mem 论文复现",
    },
    "MemoryOS": {
        "overall": 44.8,
        "generator": "gpt-4o-mini",
        "judge": "gpt-4o-mini",
        "top_k": "N/A",
        "oracle": False,
        "by_type": {},
        "source": "HeLa-Mem 论文复现",
    },
    "LangMem": {
        "overall": 37.2,
        "generator": "gpt-4o-mini",
        "judge": "gpt-4o-mini",
        "top_k": "N/A",
        "oracle": False,
        "by_type": {},
        "source": "HeLa-Mem 论文复现",
    },
    "ChatGPT (论文 baseline)": {
        "overall": 57.73,
        "generator": "GPT-4o",
        "judge": "GPT-4o",
        "top_k": "N/A",
        "oracle": False,
        "by_type": {},
        "source": "LongMemEval 论文 Figure 3a",
    },
    "Coze (论文 baseline)": {
        "overall": 32.99,
        "generator": "GPT-4o",
        "judge": "GPT-4o",
        "top_k": "N/A",
        "oracle": False,
        "by_type": {},
        "source": "LongMemEval 论文 Figure 3a",
    },
    "Offline Reading (上限)": {
        "overall": 91.84,
        "generator": "GPT-4o",
        "judge": "GPT-4o",
        "top_k": "N/A",
        "oracle": True,
        "by_type": {},
        "source": "LongMemEval 论文 Figure 3a",
    },
}

# 分维度类型展示顺序与中文名映射
TYPE_ORDER: list[str] = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "temporal-reasoning",
    "multi-session",
]

TYPE_LABELS: dict[str, str] = {
    "single-session-user": "单会话-用户",
    "single-session-assistant": "单会话-助手",
    "single-session-preference": "单会话-偏好",
    "knowledge-update": "知识更新",
    "temporal-reasoning": "时序推理",
    "multi-session": "多会话",
}


# ======================================================================
# OmniMem 结果读取
# ======================================================================
def find_latest_scores(result_dir: Path) -> Path | None:
    """在 result_dir 下递归查找最新的 scores.json 文件。

    按文件修改时间排序，返回最近修改的 scores.json 路径。
    若目录不存在或无匹配文件，返回 None。
    """
    if not result_dir.exists() or not result_dir.is_dir():
        return None

    candidates = sorted(
        result_dir.rglob("scores.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _to_percentage(value: Any) -> float | None:
    """将数值转换为百分比（0-100）。

    规则：
    - 若值 <= 1.0 且 >= 0，视为 0-1 比例，乘以 100。
    - 若值 > 1.0，视为已是百分比，直接返回。
    - None 或非数值返回 None。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if 0 <= value <= 1.0:
            return round(value * 100, 2)
        return round(float(value), 2)
    return None


def load_omnimem_scores(path: Path) -> dict[str, Any] | None:
    """加载并归一化 OmniMem 评测结果。

    兼容两种 JSON 格式：
    - 旧格式（当前实际）：qa_accuracy.overall (0-1)、qa_accuracy.by_type、retrieval_quality 等。
    - 新格式（未来可能）：顶层 overall_qa_acc、overall_mrr、overall_top5、overall_top10、by_type。

    返回归一化后的字典，包含:
      overall, generator, judge, top_k, oracle, by_type, source, raw
    若文件无法解析返回 None。
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"警告: 无法解析 OmniMem 结果文件 {path}: {exc}", file=sys.stderr)
        return None

    # ── 解析总分 ──
    # 优先级：顶层 overall_qa_acc > qa_accuracy.overall
    overall: float | None = None
    if "overall_qa_acc" in raw:
        overall = _to_percentage(raw.get("overall_qa_acc"))
    if overall is None:
        qa = raw.get("qa_accuracy")
        if isinstance(qa, dict):
            overall = _to_percentage(qa.get("overall"))

    # ── 解析分维度 ──
    by_type: dict[str, float] = {}
    # 优先顶层 by_type，其次 qa_accuracy.by_type
    raw_by_type = raw.get("by_type")
    if not isinstance(raw_by_type, dict):
        qa = raw.get("qa_accuracy")
        if isinstance(qa, dict):
            raw_by_type = qa.get("by_type")

    if isinstance(raw_by_type, dict):
        for qtype, val in raw_by_type.items():
            if isinstance(val, dict):
                acc = _to_percentage(val.get("accuracy"))
            else:
                acc = _to_percentage(val)
            if acc is not None:
                by_type[qtype] = acc

    # ── 解析配置信息 ──
    config = raw.get("config", {}) if isinstance(raw.get("config"), dict) else {}
    max_sessions = config.get("max_sessions", "N/A")
    user_only = config.get("user_only", "N/A")
    total_questions = raw.get("total_questions", "N/A")
    mode = raw.get("mode", "N/A")

    # ── 排名指标（如有）──
    overall_mrr = raw.get("overall_mrr")
    overall_top5 = raw.get("overall_top5")
    overall_top10 = raw.get("overall_top10")

    # 构建评测条件描述
    conditions_parts: list[str] = []
    conditions_parts.append(f"模式={mode}")
    conditions_parts.append(f"题数={total_questions}")
    if isinstance(max_sessions, (int, float)) and max_sessions:
        conditions_parts.append(f"max_sessions={max_sessions}")
    conditions_parts.append(f"user_only={user_only}")

    result: dict[str, Any] = {
        "overall": overall,
        "generator": "N/A (本地评测)",
        "judge": "N/A (本地评测)",
        "top_k": "N/A",
        "oracle": False,
        "by_type": by_type,
        "source": f"OmniMem 本地评测 ({path.parent.name}/{path.name})",
        "conditions": " | ".join(conditions_parts),
        "overall_mrr": overall_mrr,
        "overall_top5": overall_top5,
        "overall_top10": overall_top10,
        "raw": raw,
    }
    return result


# ======================================================================
# Markdown 报告生成
# ======================================================================
def _fmt_score(value: Any) -> str:
    """格式化分数为字符串。"""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def _fmt_oracle(oracle: bool) -> str:
    """格式化 oracle 标记。"""
    return "是 ⚠️" if oracle else "否"


def generate_overall_table(
    omnimem: dict[str, Any] | None,
    open_scores: dict[str, dict[str, Any]],
) -> str:
    """生成总分对比表（按分数降序排列）。"""
    # 汇总所有项目分数
    rows: list[tuple[str, float | None, str, str, Any, bool, str]] = []
    for name, info in open_scores.items():
        rows.append(
            (
                name,
                info.get("overall"),
                info.get("generator", "N/A"),
                info.get("judge", "N/A"),
                info.get("top_k", "N/A"),
                info.get("oracle", False),
                info.get("source", "N/A"),
            )
        )
    if omnimem is not None and omnimem.get("overall") is not None:
        rows.append(
            (
                "OmniMem (本地评测)",
                omnimem.get("overall"),
                omnimem.get("generator", "N/A"),
                omnimem.get("judge", "N/A"),
                omnimem.get("top_k", "N/A"),
                omnimem.get("oracle", False),
                omnimem.get("source", "N/A"),
            )
        )

    # 按 overall 降序排列（None 排最后）
    rows.sort(key=lambda r: (r[1] is None, -(r[1] if r[1] is not None else 0)))

    lines = [
        "| 排名 | 项目 | 总分 | Generator LLM | Judge LLM | Top-K | Oracle | 数据来源 |",
        "|------|------|------|---------------|-----------|-------|--------|----------|",
    ]
    for idx, (name, overall, gen, judge, top_k, oracle, source) in enumerate(rows, 1):
        lines.append(
            f"| {idx} | {name} | {_fmt_score(overall)} | {gen} | {judge} | "
            f"{top_k} | {_fmt_oracle(oracle)} | {source} |"
        )
    return "\n".join(lines)


def generate_by_type_table(
    omnimem: dict[str, Any] | None,
    open_scores: dict[str, dict[str, Any]],
) -> str:
    """生成分维度对比表（仅含有分维度数据的项目）。"""
    # 收集所有有 by_type 数据的项目
    projects: list[tuple[str, dict[str, float]]] = []
    for name, info in open_scores.items():
        by_type = info.get("by_type", {})
        if by_type:
            projects.append((name, by_type))
    if omnimem is not None and omnimem.get("by_type"):
        projects.append(("OmniMem (本地评测)", omnimem["by_type"]))

    if not projects:
        return "*暂无项目提供分维度数据。*"

    # 表头：维度列 + 每个项目列
    header = "| 维度 | " + " | ".join(name for name, _ in projects) + " |"
    separator = "|------|" + "|".join("------" for _ in projects) + "|"

    lines = [header, separator]
    for qtype in TYPE_ORDER:
        label = TYPE_LABELS.get(qtype, qtype)
        cells = []
        for _, by_type in projects:
            val = by_type.get(qtype)
            cells.append(_fmt_score(val))
        lines.append("| " + label + " | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def generate_conditions_table(
    omnimem: dict[str, Any] | None,
    open_scores: dict[str, dict[str, Any]],
) -> str:
    """生成评测条件标注表。"""
    lines = [
        "| 项目 | Generator LLM | Judge LLM | Top-K | Oracle | 备注 |",
        "|------|---------------|-----------|-------|--------|------|",
    ]
    for name, info in open_scores.items():
        note = info.get("source", "N/A")
        lines.append(
            f"| {name} | {info.get('generator', 'N/A')} | {info.get('judge', 'N/A')} | "
            f"{info.get('top_k', 'N/A')} | {_fmt_oracle(info.get('oracle', False))} | {note} |"
        )
    if omnimem is not None:
        note = omnimem.get("conditions", "N/A")
        lines.append(
            f"| OmniMem (本地评测) | {omnimem.get('generator', 'N/A')} | "
            f"{omnimem.get('judge', 'N/A')} | {omnimem.get('top_k', 'N/A')} | "
            f"{_fmt_oracle(omnimem.get('oracle', False))} | {note} |"
        )
    return "\n".join(lines)


def generate_omnimem_detail_section(omnimem: dict[str, Any] | None) -> str:
    """生成 OmniMem 详细指标段落（含检索质量、排名指标、性能指标）。"""
    if omnimem is None:
        return ""

    raw = omnimem.get("raw", {})
    lines: list[str] = ["### OmniMem 详细指标", ""]

    # 检索质量
    retrieval = raw.get("retrieval_quality")
    if isinstance(retrieval, dict):
        lines.append("**检索质量：**")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|------|")
        lines.append(
            f"| Session Hit Rate | {_fmt_score(_to_percentage(retrieval.get('overall_session_hit_rate')))}% |"
        )
        lines.append(
            f"| Turn Hit Rate | {_fmt_score(_to_percentage(retrieval.get('overall_turn_hit_rate')))}% |"
        )
        lines.append(
            f"| Avg Coverage | {_fmt_score(_to_percentage(retrieval.get('overall_avg_coverage')))}% |"
        )
        lines.append("")

    # 排名指标（如有）
    mrr = omnimem.get("overall_mrr")
    top5 = omnimem.get("overall_top5")
    top10 = omnimem.get("overall_top10")
    if any(v is not None for v in (mrr, top5, top10)):
        lines.append("**排名指标：**")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|------|")
        if mrr is not None:
            lines.append(f"| Overall MRR | {_fmt_score(_to_percentage(mrr))} |")
        if top5 is not None:
            lines.append(f"| Overall Top-5 命中率 | {_fmt_score(_to_percentage(top5))}% |")
        if top10 is not None:
            lines.append(f"| Overall Top-10 命中率 | {_fmt_score(_to_percentage(top10))}% |")
        lines.append("")

    # 性能指标
    perf = raw.get("performance")
    if isinstance(perf, dict):
        lines.append("**性能指标：**")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|------|")
        lines.append(f"| 平均写入耗时 | {_fmt_score(perf.get('avg_ingest_time_s'))} s |")
        lines.append(f"| 平均检索耗时 | {_fmt_score(perf.get('avg_retrieval_time_s'))} s |")
        lines.append(f"| 平均写入条数 | {_fmt_score(perf.get('avg_ingest_count'))} |")
        lines.append(f"| 总写入耗时 | {_fmt_score(perf.get('total_ingest_time_s'))} s |")
        lines.append(f"| 总检索耗时 | {_fmt_score(perf.get('total_retrieval_time_s'))} s |")
        lines.append("")

    return "\n".join(lines)


def generate_report(
    omnimem: dict[str, Any] | None,
    open_scores: dict[str, dict[str, Any]],
    omnimem_path: Path | None,
) -> str:
    """生成完整的 Markdown 对比报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    # ── 标题 ──
    lines.append("# OmniMem 与开源项目 LongMemEval 评测对比报告")
    lines.append("")
    lines.append(f"> 生成时间：{now}")
    if omnimem_path is not None:
        lines.append(f"> OmniMem 结果文件：`{omnimem_path}`")
    else:
        lines.append("> OmniMem 结果文件：**未找到**（仅展示开源项目分数）")
    lines.append("")

    # ── 重要警示 ──
    lines.append("## ⚠️ 重要警示")
    lines.append("")
    lines.append(
        "> **分数不可直接比较。** 以下各项目的分数使用了不同的 generator LLM、"
        "judge LLM 和 retrieval budget（top-k），评测条件存在显著差异，"
        "跨项目排名仅供参考，不能作为唯一的性能评判依据。"
    )
    lines.append("")
    lines.append("主要差异包括：")
    lines.append("")
    lines.append(
        "- **Generator LLM 不同**：Mem0 Platform/OSS 使用 GPT-5，HeLa-Mem 系列复现使用 gpt-4o-mini，论文 baseline 使用 GPT-4o。更强的 generator LLM 会显著拉高分数。"
    )
    lines.append(
        "- **Judge LLM 不同**：不同项目使用不同的 judge 模型（GPT-5 / GPT-4o / gpt-4o-mini），评判尺度可能不一致。"
    )
    lines.append(
        "- **Retrieval Budget 不同**：Mem0 Platform/OSS 使用 top_k=200，而多数论文复现项目未公开 retrieval budget。"
    )
    lines.append(
        "- **Oracle 设置**：Offline Reading 为 oracle 上限（直接提供标准答案所在上下文），非真实检索场景。"
    )
    lines.append("- **评测范围不同**：OmniMem 为本地评测，题数与配置可能与其他项目不同。")
    lines.append("")

    # ── 总分对比表 ──
    lines.append("## 一、总分对比")
    lines.append("")
    lines.append("按总分降序排列：")
    lines.append("")
    lines.append(generate_overall_table(omnimem, open_scores))
    lines.append("")

    # ── 分维度对比表 ──
    lines.append("## 二、分维度对比")
    lines.append("")
    lines.append("仅展示提供分维度数据的项目：")
    lines.append("")
    lines.append(generate_by_type_table(omnimem, open_scores))
    lines.append("")

    # ── 评测条件标注 ──
    lines.append("## 三、评测条件标注")
    lines.append("")
    lines.append(generate_conditions_table(omnimem, open_scores))
    lines.append("")

    # ── OmniMem 详细指标 ──
    detail = generate_omnimem_detail_section(omnimem)
    if detail:
        lines.append(detail)

    # ── 数据来源说明 ──
    lines.append("## 四、数据来源说明")
    lines.append("")
    lines.append(
        "- **Mem0 Platform (v3) / Mem0 OSS**：来自 mem0ai/memory-benchmarks 仓库（2026/04），使用 GPT-5 作为 generator 与 judge，top_k=200。"
    )
    lines.append(
        "- **HeLa-Mem / A-MEM / NaiveRAG / FullText / Mem0(复现) / MemoryOS / LangMem**：来自 HeLa-Mem 论文（ACL 2026）及其复现实验，统一使用 gpt-4o-mini 作为 generator 与 judge。"
    )
    lines.append(
        "- **ChatGPT / Coze / Offline Reading**：来自 LongMemEval 论文 Figure 3a，使用 GPT-4o 作为 generator 与 judge。Offline Reading 为 oracle 上限。"
    )
    if omnimem is not None:
        lines.append("- **OmniMem**：本地 LongMemEval 评测结果，generator/judge 配置见评测脚本。")
    lines.append("")

    # ── 脚注 ──
    lines.append("---")
    lines.append("")
    lines.append("*本报告由 `benchmarks/comparison_report.py` 自动生成。*")
    lines.append("")

    return "\n".join(lines)


# ======================================================================
# CLI 入口
# ======================================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="生成 OmniMem 与开源项目 LongMemEval 评测对比报告（Markdown）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="OmniMem 结果目录（默认: benchmarks/results/）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evaluation_report_full.md"),
        help="输出报告路径（默认: docs/evaluation_report_full.md）",
    )
    return parser.parse_args()


def main() -> int:
    """主入口：读取结果、生成报告、写入文件。"""
    args = parse_args()

    # 查找最新的 OmniMem 结果文件
    omnimem_path = find_latest_scores(args.result_dir)
    omnimem: dict[str, Any] | None = None
    if omnimem_path is not None:
        print(f"找到 OmniMem 结果文件: {omnimem_path}")
        omnimem = load_omnimem_scores(omnimem_path)
        if omnimem is None:
            print("警告: OmniMem 结果文件解析失败，将仅输出开源项目分数表。", file=sys.stderr)
    else:
        print(
            f"未在 {args.result_dir} 下找到 scores.json 文件，" "将仅输出开源项目分数表。",
            file=sys.stderr,
        )

    # 生成报告
    report = generate_report(omnimem, OPEN_SOURCE_SCORES, omnimem_path)

    # 确保输出目录存在
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"对比报告已生成: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
