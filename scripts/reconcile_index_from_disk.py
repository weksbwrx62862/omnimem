#!/usr/bin/env python3
"""reconcile_index_from_disk.py — 以磁盘抽屉为事实来源对账 index.db（P1-1）。

线上缺口：3 135 个 drawer / 712 行 index / 交集 539 —— 2 596 条记忆写抽屉成功、
写索引失败被静默吞掉（P1-3/P1-4 已止血，存量要在这里清）。这些记忆关键词与语义
检索都召不回，且系统里没有任何机制会发现该缺口。

默认 **dry-run**：只报差额，不写任何文件。加 ``--apply`` 才补索引行。

用法：
    # 1) 先看差额（安全，只读）
    python scripts/reconcile_index_from_disk.py --data-dir ~/.hermes/omnimem

    # 2) 补索引行（仍不动向量库；向量积压交给 drain/rebuild）
    python scripts/reconcile_index_from_disk.py --data-dir ~/.hermes/omnimem --apply

    # 3) 同时清理「有 index 行、磁盘无抽屉」的幽灵行
    python scripts/reconcile_index_from_disk.py --data-dir ~/.hermes/omnimem \\
        --apply --prune-ghosts

对账后仍需重建向量通道才能语义召回，见 doctor 的 rebuild 路径与修复报告顺序说明。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from omnimem.config import resolve_default_data_dir  # noqa: E402
from omnimem.governance.reconciler import reconcile_index  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="磁盘抽屉 ↔ index.db 对账")
    parser.add_argument("--data-dir", default=None, help="OmniMem 数据根目录（默认按 HERMES_HOME 解析）")
    parser.add_argument("--apply", action="store_true", help="真正补写缺失的索引行（默认只报告）")
    parser.add_argument("--prune-ghosts", action="store_true", help="删除幽灵索引行（需配合 --apply）")
    args = parser.parse_args(argv)

    data_dir = resolve_default_data_dir(Path(args.data_dir) if args.data_dir else None)
    if not (data_dir / "palace").exists():
        print(f"错误：{data_dir} 下没有 palace/ 目录，请确认 --data-dir", file=sys.stderr)
        return 1

    if args.prune_ghosts and not args.apply:
        print("错误：--prune-ghosts 需要与 --apply 同时使用（dry-run 不删任何东西）", file=sys.stderr)
        return 1

    report = reconcile_index(data_dir, apply=args.apply, prune_ghosts=args.prune_ghosts)
    print(f"data-dir: {data_dir}")
    print("  " + report.summary())
    if report.failed:
        print(f"\n失败 {len(report.failed)} 条（详见日志），重跑本命令会继续补这些")
    if report.dry_run:
        if report.missing_in_index:
            print("\n下一步：加 --apply 补索引行（补完再重建向量通道）")
        if report.ghosts_in_index:
            print("清理幽灵行：--apply --prune-ghosts（幽灵行会占 index 但无抽屉可回源）")
        if not (report.missing_in_index or report.ghosts_in_index):
            print("索引行已与磁盘一致；向量通道积压请走 drain/rebuild")
    else:
        print(f"  本次：补 {report.repaired} 行，清 {report.pruned} 行")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
