#!/usr/bin/env python3
"""check_dependency_sync.py — 依赖单一来源校验（M9-22）。

以 pyproject.toml 为唯一事实来源，校验 requirements.txt 与 plugin.yaml
的 pip 依赖是否与之一致，防止三处声明再度漂移（历史上出现过
chromadb 版本上界不一致、datasketch 声明未用等问题）。

约定的映射关系:
    requirements.txt   = [project.dependencies]
                       + [project.optional-dependencies].vector
                       + [project.optional-dependencies].nlp
                       (+ 允许附加 setuptools 钉版等环境约束行)
    plugin.yaml        = 同 requirements.txt（pip_dependencies 列表）

用法:
    python scripts/check_dependency_sync.py          # 校验，漂移则 exit 1
    python scripts/check_dependency_sync.py --quiet  # 仅输出错误
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent

# requirements.txt 中允许存在、但不要求出现在 pyproject 的环境约束包
_ALLOWED_EXTRA_PACKAGES = {"setuptools"}


def _pkg_name(spec: str) -> str:
    """从依赖声明中提取归一化包名（小写，- 与 _ 等价）。"""
    m = re.match(r"^[A-Za-z0-9_.\-]+", spec.strip())
    name = m.group(0) if m else spec.strip()
    return name.lower().replace("_", "-")


def _normalize(spec: str) -> str:
    """归一化完整声明：包名小写化 + 去空白。"""
    spec = spec.strip().replace(" ", "")
    name = _pkg_name(spec)
    rest = spec[len(re.match(r"^[A-Za-z0-9_.\-]+", spec).group(0)):] if spec else ""
    return f"{name}{rest}"


def load_pyproject_expected() -> dict[str, str]:
    """从 pyproject.toml 构造期望依赖集 {包名: 归一化声明}。"""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})
    specs: list[str] = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    for extra in ("vector", "nlp"):
        specs.extend(optional.get(extra, []))
    return {_pkg_name(s): _normalize(s) for s in specs}


def load_requirements() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        result[_pkg_name(line)] = _normalize(line)
    return result


def load_plugin_yaml() -> dict[str, str]:
    """轻量解析 plugin.yaml 的 pip_dependencies（避免引入 yaml 依赖差异）。"""
    result: dict[str, str] = {}
    in_deps = False
    for line in (ROOT / "plugin.yaml").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("pip_dependencies:"):
            in_deps = True
            continue
        if in_deps:
            if stripped.startswith("- "):
                spec = stripped[2:].strip().strip('"').strip("'")
                result[_pkg_name(spec)] = _normalize(spec)
            elif stripped.startswith("#") or not stripped:
                continue
            else:
                break  # 下一个顶层 key，pip_dependencies 结束
    return result


def diff_sets(expected: dict[str, str], actual: dict[str, str], label: str,
              allow_extra: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    allow_extra = allow_extra or set()
    for name, spec in expected.items():
        if name not in actual:
            errors.append(f"[{label}] 缺少依赖: {spec}")
        elif actual[name] != spec:
            errors.append(f"[{label}] 版本约束漂移: {actual[name]} (期望 {spec})")
    for name, spec in actual.items():
        if name not in expected and name not in allow_extra:
            errors.append(f"[{label}] 多余依赖（pyproject 未声明）: {spec}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="依赖单一来源校验")
    parser.add_argument("--quiet", action="store_true", help="仅输出错误")
    args = parser.parse_args()

    expected = load_pyproject_expected()
    errors: list[str] = []
    errors += diff_sets(expected, load_requirements(), "requirements.txt",
                        allow_extra=_ALLOWED_EXTRA_PACKAGES)
    errors += diff_sets(expected, load_plugin_yaml(), "plugin.yaml",
                        allow_extra=_ALLOWED_EXTRA_PACKAGES)

    if errors:
        print("依赖声明漂移（唯一事实来源: pyproject.toml）:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"[PASS] 依赖一致性校验通过（{len(expected)} 个包，3 处声明同步）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
