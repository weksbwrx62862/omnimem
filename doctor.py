#!/usr/bin/env python3
"""omni-doctor — OmniMem 健康检查与诊断工具。

用法:
    python -m omnimem.doctor              # 完整检查
    python -m omnimem.doctor --quick      # 快速检查
    python -m omnimem.doctor --config     # 仅检查配置
    python -m omnimem.doctor --deps       # 仅检查依赖
    python -m omnimem.doctor reconcile    # 磁盘抽屉 ↔ index.db 对账（dry-run）
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ANSI colors
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _status(ok: bool, msg: str) -> str:
    icon = f"{_GREEN}✅{_RESET}" if ok else f"{_RED}❌{_RESET}"
    return f"  {icon} {msg}"


def _warn(msg: str) -> str:
    return f"  {_YELLOW}⚠️  {_RESET}{msg}"


class Doctor:
    """OmniMem 健康检查器。"""

    def __init__(self, data_dir: Path | None = None, quick: bool = False):
        # ★ P2-3g：默认目录必须与插件真实写入位置一致（原先硬编码 ~/.omnimem，
        #   于是 doctor 检查的是另一个实例：报 palace 0.1 MB，真实 254 MB）。
        from omnimem.config._config import resolve_default_data_dir

        self.data_dir = resolve_default_data_dir(data_dir)
        self.quick = quick
        self.issues: list[str] = []
        self.warnings: list[str] = []

    def run_all(self) -> bool:
        """运行所有检查，返回 True 表示全部通过。"""
        print(f"\n{_BOLD}🩺 OmniMem Health Check{_RESET}\n")

        self.check_python_version()
        self.check_dependencies()
        self.check_config()
        self.check_data_dir()
        if not self.quick:
            self.check_encryption()
            self.check_vector_backend()
            self.check_bm25()
            self.check_sqlite()
            self.check_permissions()

        self._print_summary()
        return len(self.issues) == 0

    def check_python_version(self) -> None:
        """检查 Python 版本。"""
        print(f"{_BOLD}[Python Version]{_RESET}")
        v = sys.version_info
        # ★ P2-3d：原先硬编码上界 3.12，而 pyproject 声明的是 requires-python >=3.10
        #   （无上界）。生产网关跑 3.11.15、插件 .venv 跑 3.13.13，仓库 2 707 例在
        #   3.13 全绿 —— doctor 却把 3.13 报成问题，等于每次自检都有一条假警报。
        #   改为与声明一致：只检查下界。
        ok = (v.major, v.minor) >= (3, 10)
        print(_status(ok, f"Python {v.major}.{v.minor}.{v.micro} (要求 ≥3.10，与 pyproject 一致)"))
        if not ok:
            self.issues.append(f"Python {v.major}.{v.minor} 低于最低要求 3.10")

    def check_dependencies(self) -> None:
        """检查关键依赖。"""
        print(f"\n{_BOLD}[Dependencies]{_RESET}")

        required = [
            ("rank_bm25", "BM25 关键词检索"),
            ("sqlite3", "SQLite 元数据存储"),
        ]
        optional = [
            ("chromadb", "向量检索 (ChromaDB)"),
            ("cryptography", "加密支持"),
            ("sentence_transformers", "嵌入模型"),
            ("torch", "深度学习 (LoRA/KV Cache)"),
            ("jieba", "中文分词"),
            ("qdrant_client", "Qdrant 向量后端"),
        ]

        for mod, desc in required:
            try:
                importlib.import_module(mod)
                print(_status(True, f"{mod} — {desc}"))
            except ImportError:
                print(_status(False, f"{mod} — {desc} (必需)"))
                self.issues.append(f"缺少必需依赖: {mod}")

        for mod, desc in optional:
            try:
                importlib.import_module(mod)
                print(_status(True, f"{mod} — {desc}"))
            except ImportError:
                print(_warn(f"{mod} — {desc} (可选，未安装)"))
                self.warnings.append(f"可选依赖未安装: {mod}")

    def check_config(self) -> None:
        """检查配置文件。"""
        print(f"\n{_BOLD}[Configuration]{_RESET}")

        config_paths = [
            self.data_dir / "config.yaml",
            Path.home() / ".hermes" / "omnimem" / "config.yaml",
            Path.home() / ".omnimem" / "config.yaml",
            Path.cwd() / "omnimem.yaml",
            Path.cwd() / "config.yaml",
        ]

        found = False
        for p in config_paths:
            if p.exists():
                print(_status(True, f"配置文件: {p}"))
                found = True
                break
        if not found:
            print(_warn("未找到配置文件，将使用默认配置"))

        # ★ P2-3h：每个 storage_dir 各自生成 api_key 是设计（OmniMemConfig 在 key 为空
        #   时会 secrets.token_hex(32) 并 save()），不是缺陷 —— 缺的是可观测性：出问题时
        #   看不出「这个 store 的 key 是哪一个」。这里只打印指纹（sha256 前 8 位）。
        #   注意：不能用 OmniMemConfig(data_dir) 来读 —— 它的构造函数会 mkdir 数据目录、
        #   并在缺 key 时写 config.yaml，健康检查不该有这个副作用。
        try:
            import hashlib

            api_key = ""
            for p in config_paths:
                if not p.exists():
                    continue
                try:
                    import yaml

                    loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                except Exception:
                    continue
                if isinstance(loaded, dict) and loaded.get("api_key"):
                    api_key = str(loaded["api_key"])
                    digest = hashlib.sha256(api_key.encode()).hexdigest()[:8]
                    print(_status(True, f"api_key 指纹: {digest} (来自 {p})"))
                    if p != self.data_dir / "config.yaml":
                        print(_warn(f"当前 store（{self.data_dir}）自己没有 config.yaml，"
                                    f"沿用了上面这份配置的 key"))
                    break
            if not api_key:
                print(_warn(f"未在配置文件里找到 api_key（storage_dir={self.data_dir}）："
                            f"首次实例化该 store 时会自动生成并写入 config.yaml"))
        except Exception as e:
            print(_warn(f"读取 api_key 失败: {e}"))

        # Check encryption key
        env_key = os.environ.get("OMNIMEM_ENCRYPTION_KEY", "")
        if env_key:
            print(_status(True, "OMNIMEM_ENCRYPTION_KEY 已设置"))
        else:
            print(_warn("OMNIMEM_ENCRYPTION_KEY 未设置，加密将被禁用"))

    def check_data_dir(self) -> None:
        """检查数据目录。"""
        print(f"\n{_BOLD}[Data Directory]{_RESET}")

        if self.data_dir.exists():
            print(_status(True, f"数据目录: {self.data_dir}"))
            # Check subdirectories
            subdirs = ["palace", ".meta", "governance"]
            for d in subdirs:
                p = self.data_dir / d
                if p.exists():
                    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    size_mb = size / (1024 * 1024)
                    print(_status(True, f"  {d}/ ({size_mb:.1f} MB)"))
                else:
                    print(_warn(f"  {d}/ 不存在（首次运行时自动创建）"))
        else:
            print(_warn(f"数据目录不存在: {self.data_dir}（首次运行时自动创建）"))

    def check_encryption(self) -> None:
        """检查加密子系统。"""
        print(f"\n{_BOLD}[Encryption]{_RESET}")
        try:
            from omnimem.governance.encryption import MemoryEncryption

            enc = MemoryEncryption()
            if enc.is_available():
                print(_status(True, "Fernet 加密可用"))
                # Test encrypt/decrypt roundtrip
                test_text = "omni-doctor-test"
                encrypted = enc.encrypt(test_text)
                decrypted = enc.decrypt(encrypted)
                if decrypted == test_text:
                    print(_status(True, "加密/解密往返测试通过"))
                else:
                    print(_status(False, "加密/解密往返测试失败"))
                    self.issues.append("加密往返测试失败")
            else:
                print(_warn("加密不可用（cryptography 未安装或未配置密钥）"))
        except Exception as e:
            print(_status(False, f"加密检查异常: {e}"))
            self.issues.append(f"加密检查异常: {e}")

    def check_vector_backend(self) -> None:
        """检查向量后端。"""
        print(f"\n{_BOLD}[Vector Backend]{_RESET}")
        try:
            from omnimem.retrieval.vector_store import ChromaVectorStore

            store = ChromaVectorStore(data_dir=self.data_dir / "vectors")
            count = store.count()
            print(_status(True, f"ChromaDB 可用，向量数: {count}"))
        except ImportError:
            print(_warn("ChromaDB 未安装，向量检索不可用"))
        except Exception as e:
            print(_status(False, f"ChromaDB 检查异常: {e}"))
            self.issues.append(f"ChromaDB 异常: {e}")

    def check_bm25(self) -> None:
        """检查 BM25 索引。"""
        print(f"\n{_BOLD}[BM25 Index]{_RESET}")
        try:
            from omnimem.retrieval.bm25 import BM25Retriever

            r = BM25Retriever(data_dir=self.data_dir / "bm25")
            print(_status(True, f"BM25 可用，已索引文档: {r.document_count}"))
        except ImportError:
            print(_status(False, "rank_bm25 未安装"))
            self.issues.append("rank_bm25 未安装")
        except Exception as e:
            print(_status(False, f"BM25 检查异常: {e}"))

    def check_sqlite(self) -> None:
        """检查 SQLite 存储。"""
        print(f"\n{_BOLD}[SQLite Storage]{_RESET}")
        try:
            from omnimem.memory.meta_store import MetaStore

            # ★ MetaStore 实际住在 palace/.meta（drawer_closet 建的）。原先指向
            #   data_dir/.meta，每次健康检查都在数据根目录凭空建一个 0 字节的
            #   meta_store.db，然后报「记录数: 0」——把「检查」写成了「新建一个空库」。
            palace_dir = self.data_dir / "palace"
            meta_dir = (palace_dir / ".meta") if palace_dir.exists() else (self.data_dir / ".meta")
            ms = MetaStore(meta_dir, palace_dir=palace_dir if palace_dir.exists() else None)
            count = ms.count()
            print(_status(True, f"MetaStore 可用，记录数: {count}（{meta_dir}）"))
            ms.close()
        except Exception as e:
            print(_status(False, f"MetaStore 检查异常: {e}"))
            self.issues.append(f"MetaStore 异常: {e}")

    def check_permissions(self) -> None:
        """检查文件权限。"""
        print(f"\n{_BOLD}[Permissions]{_RESET}")
        if self.data_dir.exists():
            writable = os.access(self.data_dir, os.W_OK)
            print(_status(writable, f"数据目录可写: {writable}"))
            if not writable:
                self.issues.append(f"数据目录不可写: {self.data_dir}")

    def _print_summary(self) -> None:
        """打印总结。"""
        print(f"\n{'─' * 50}")
        if self.issues:
            print(f"{_RED}{_BOLD}❌ 发现 {len(self.issues)} 个问题:{_RESET}")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        if self.warnings:
            print(f"{_YELLOW}{_BOLD}⚠️  {len(self.warnings)} 个警告:{_RESET}")
            for w in self.warnings:
                print(f"  - {w}")
        if not self.issues and not self.warnings:
            print(f"{_GREEN}{_BOLD}✅ 所有检查通过！{_RESET}")
        elif not self.issues:
            print(f"{_GREEN}{_BOLD}✅ 核心检查通过（有 {len(self.warnings)} 个警告）{_RESET}")
        print()


def _cmd_migrate_index(args: argparse.Namespace) -> None:
    """执行 UnifiedMemoryIndex 迁移。"""
    from omnimem.config._config import resolve_default_data_dir
    from omnimem.memory.migration_tool import IndexMigrationTool

    data_dir = resolve_default_data_dir(args.data_dir)

    tool = IndexMigrationTool(
        index_dir=data_dir / "index",
        meta_dir=data_dir / ".meta",
        unified_dir=data_dir / "index",  # 目标路径与 index/ 同目录，输出 unified_index.db
    )

    result = tool.migrate(dry_run=args.dry_run, skip_backup=args.no_backup)
    tool.print_report(result)

    if not result.get("success"):
        sys.exit(1)


def _cmd_reconcile(args: argparse.Namespace) -> None:
    """★ P1-1: 以磁盘抽屉为事实来源对账 index.db。

    默认 dry-run —— 只报差额，不写任何东西。「有 drawer 文件、无 index 行」的记忆
    关键词和语义检索都召不回，而写索引失败原先被静默吞掉，所以这个缺口不会自己
    被发现；必须有人（人或 doctor）主动数一次。
    """
    from omnimem.config._config import resolve_default_data_dir
    from omnimem.governance.reconciler import reconcile_index

    data_dir = resolve_default_data_dir(args.data_dir)
    if not (data_dir / "palace").exists():
        print(f"{_RED}错误：{data_dir} 下没有 palace/ 目录，请确认 --data-dir{_RESET}")
        sys.exit(1)
    if args.prune_ghosts and not args.apply:
        print(f"{_RED}错误：--prune-ghosts 需要与 --apply 同时使用{_RESET}")
        sys.exit(1)

    print(f"\n{_BOLD}🔍 磁盘 ↔ index.db 对账{_RESET}\n")
    print(f"  data-dir: {data_dir}")
    report = reconcile_index(data_dir, apply=args.apply, prune_ghosts=args.prune_ghosts)
    print("  " + report.summary().replace("\n  ", "\n  "))
    if report.dry_run:
        if report.missing_in_index:
            print("\n  下一步：omni-doctor reconcile --apply 补索引行（补完再重建向量通道）")
        if report.ghosts_in_index:
            print("  清理幽灵行：--apply --prune-ghosts（有 index 无 drawer，召回到处也回不出内容）")
        if not (report.missing_in_index or report.ghosts_in_index):
            print("  索引行已与「应可召回」的抽屉集合一致；向量通道积压走 drain/rebuild")
        if report.skipped_archived:
            print(
                f"  另有 {len(report.skipped_archived)} 条抽屉属于已归档/已遗忘，"
                "按遗忘语义**不补索引**（开机审计会持续删它们）"
            )
    elif args.apply:
        print(f"  本次：补 {report.repaired} 行，清 {report.pruned} 行")
        if report.skipped_archived:
            print(f"  跳过已归档/已遗忘 {len(report.skipped_archived)} 条（不重新索引）")
    if report.failed:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniMem 健康检查与维护工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 默认：健康检查
    parser.add_argument("--quick", action="store_true", help="快速检查（跳过子系统测试）")
    parser.add_argument("--config", action="store_true", help="仅检查配置")
    parser.add_argument("--deps", action="store_true", help="仅检查依赖")
    parser.add_argument("--data-dir", type=Path, help="数据目录路径")

    # migrate-index 子命令
    migrate_parser = subparsers.add_parser(
        "migrate-index",
        help="将 ThreeLevelIndex + MetaStore 迁移到 UnifiedMemoryIndex",
    )
    migrate_parser.add_argument("--data-dir", type=Path, help="数据目录路径")
    migrate_parser.add_argument(
        "--dry-run", action="store_true", help="仅模拟，不实际写入"
    )
    migrate_parser.add_argument(
        "--no-backup", action="store_true", help="跳过源数据库备份"
    )

    # reconcile 子命令
    reconcile_parser = subparsers.add_parser(
        "reconcile",
        help="以磁盘抽屉为事实来源对账 index.db（默认 dry-run，只报差额）",
    )
    reconcile_parser.add_argument("--data-dir", type=Path, help="数据目录路径")
    reconcile_parser.add_argument(
        "--apply", action="store_true", help="补写缺失的索引行（默认只报告）"
    )
    reconcile_parser.add_argument(
        "--prune-ghosts", action="store_true", help="删除幽灵索引行（需配合 --apply）"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    if args.command == "migrate-index":
        _cmd_migrate_index(args)
        return

    if args.command == "reconcile":
        _cmd_reconcile(args)
        return

    doctor = Doctor(data_dir=args.data_dir, quick=args.quick)

    if args.config:
        doctor.check_python_version()
        doctor.check_config()
        doctor._print_summary()
    elif args.deps:
        doctor.check_python_version()
        doctor.check_dependencies()
        doctor._print_summary()
    else:
        success = doctor.run_all()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
