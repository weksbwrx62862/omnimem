#!/usr/bin/env python3
"""omni_import.py — 从外部记忆项目导入记忆（改进项 #8）。

支持 mem0 / Letta / Zep / Graphiti / Cognee 的导出 JSON，转换为 OmniMem
原生导入信封后，复用 OmniMemSDK.import_memories 落库。

两种模式：
  1) 仅转换（离线，不触碰 store/LLM）：给定 --out 时只写原生信封并退出；
  2) 直接导入：初始化 SDK 并落库（需可写存储目录与依赖环境）。

用法示例：
  # 只看转换结果、先落一个原生信封文件
  python scripts/omni_import.py --source mem0 mem0_export.json --out native.json --preview 5

  # 转换并导入到指定存储目录
  python scripts/omni_import.py --source letta letta_passages.json --storage-dir ~/.omnimem

  # 干跑（打印统计与样例，不落库）
  python scripts/omni_import.py --source zep zep_facts.json --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from omnimem.importers import SUPPORTED_SOURCES, convert_file  # noqa: E402


def _preview(envelope: dict, limit: int) -> None:
    records = envelope.get("memories", [])
    print(f"[convert] source={envelope.get('source')} 原始={envelope.get('source_count')} → 有效={len(records)}")
    for rec in records[:limit]:
        text = rec.get("content", "")
        if len(text) > 80:
            text = text[:80] + "…"
        print(f"  - ({rec.get('type')}) {text}")
    if len(records) > limit:
        print(f"  … 其余 {len(records) - limit} 条省略")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="导入 mem0/Letta/Zep/Graphiti/Cognee 导出到 OmniMem",
    )
    parser.add_argument("input", help="上游导出的 JSON 文件路径")
    parser.add_argument("--source", choices=SUPPORTED_SOURCES, default="auto", help="来源项目（默认 auto 尽力解析）")
    parser.add_argument("--out", help="仅转换：把原生导入信封写到该路径后退出")
    parser.add_argument("--preview", type=int, default=5, help="打印的样例条数（默认 5）")
    parser.add_argument("--dry-run", action="store_true", help="仅转换并预览，不落库")
    parser.add_argument("--storage-dir", help="OmniMem 存储目录（导入模式）")
    parser.add_argument("--wing", default="personal", help="写入的 wing（记忆宫殿分区）")
    parser.add_argument("--room", default="imported", help="写入的 room")
    parser.add_argument("--privacy", default="personal", help="privacy 等级")
    parser.add_argument("--confidence", type=int, default=3, help="初始置信度（原生 1-5 量纲）")
    parser.add_argument("--keep-native", help="导入模式下保留临时原生信封的目录")
    parser.add_argument("--no-dedup", action="store_true", help="跳过去重")
    parser.add_argument("--no-conflict", action="store_true", help="跳过冲突处理")
    args = parser.parse_args(argv)

    convert_kwargs = {
        "wing": args.wing,
        "room": args.room,
        "privacy": args.privacy,
        "confidence": args.confidence,
    }

    if args.out:
        envelope = convert_file(args.input, source=args.source, output_path=args.out, **convert_kwargs)
        _preview(envelope, args.preview)
        print(f"[ok] 原生信封已写入 {args.out}")
        return 0

    envelope_path = Path(args.out) if args.out else Path(args.keep_native or ".") / f".omni_import_{args.source}.tmp.json"
    if args.dry_run:
        envelope = convert_file(args.input, source=args.source, **convert_kwargs)
        _preview(envelope, args.preview)
        return 0

    from omnimem.importers import import_file
    from omnimem.sdk import OmniMemSDK

    sdk = OmniMemSDK(storage_dir=args.storage_dir)
    result = import_file(
        sdk,
        args.input,
        source=args.source,
        skip_duplicates=not args.no_dedup,
        resolve_conflicts=not args.no_conflict,
        workdir=args.keep_native,
        **convert_kwargs,
    )
    print(f"[import] {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
