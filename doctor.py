#!/usr/bin/env python3
"""omni-doctor — OmniMem 健康检查与诊断工具。

用法:
    python -m omnimem.doctor              # 完整检查
    python -m omnimem.doctor --quick      # 快速检查
    python -m omnimem.doctor --config     # 仅检查配置
    python -m omnimem.doctor --deps       # 仅检查依赖
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
        self.data_dir = data_dir or Path.home() / ".omnimem"
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
        ok = (3, 10) <= (v.major, v.minor) <= (3, 12)
        print(_status(ok, f"Python {v.major}.{v.minor}.{v.micro} (要求 3.10-3.12)"))
        if not ok:
            self.issues.append(f"Python {v.major}.{v.minor} 不在支持范围 3.10-3.12")

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

            ms = MetaStore(self.data_dir / ".meta")
            count = ms.count()
            print(_status(True, f"MetaStore 可用，记录数: {count}"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniMem 健康检查工具")
    parser.add_argument("--quick", action="store_true", help="快速检查（跳过子系统测试）")
    parser.add_argument("--config", action="store_true", help="仅检查配置")
    parser.add_argument("--deps", action="store_true", help="仅检查依赖")
    parser.add_argument("--data-dir", type=Path, help="数据目录路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

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
