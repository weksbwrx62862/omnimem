"""P2-3e: 跨包/跨插件隐藏依赖契约测试。

背景：本插件与兄弟插件之间存在**没有声明的硬依赖**，任一侧改名都会静默炸掉另一侧
（症状往往是 except 吞异常后的功能失效，而不是报错）。本轮实测已经抓到一例：
provider_middleware 从 `agent.skill_commands` 导入三个符号，而活框架里它们根本不在
那个模块（真实出处 `agent.skill_utils`，公开名 `parse_frontmatter`），ImportError 被
`logger.debug` 吞掉 —— skill 预注入从未生效。

本文件把这些依赖钉成测试，两侧改动会立刻变红而不是静默退化：
  A. 插件 → 框架：扫描源码里的 `from agent.*`，逐个符号在活框架上解析。
  B. 兄弟插件 → 插件：扫描 plugins 根目录下的 `from omnimem.*`，逐个符号解析。
  两者都只在对应根目录存在时才跑（用环境变量覆盖，缺则 skip），不绑架 CI。
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
import sys
import types
import typing
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AGENT_ROOT = Path("/home/xxh/apps/hermes-v021")
DEFAULT_PLUGINS_ROOT = Path("/mnt/sdb2/xxh-data/.hermes/plugins")

AGENT_ROOT = Path(os.environ.get("HERMES_AGENT_ROOT", str(DEFAULT_AGENT_ROOT)))
PLUGINS_ROOT = Path(os.environ.get("HERMES_PLUGINS_ROOT", str(DEFAULT_PLUGINS_ROOT)))


def _collect_from_imports(root: Path, top_level: str, *, exclude_dirs: tuple[str, ...] = ()) -> list[tuple[str, str, str]]:
    """AST 扫描 root 下所有 .py，返回 (文件, 模块, 符号) 三元组。"""
    found: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*.py")):
        rel_parts = path.relative_to(root).parts
        if any(part in exclude_dirs for part in rel_parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module.split(".")[0] != top_level:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                found.append((str(path.relative_to(root)), node.module, alias.name))
    return found


def _resolve(module: str, symbol: str) -> bool:
    """符号存在（或本身是可导入子模块）即为通过。"""
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return False
    if hasattr(mod, symbol):
        return True
    try:
        importlib.import_module(f"{module}.{symbol}")
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# A. 插件 → Hermes 框架
# ─────────────────────────────────────────────────────────────────────────────
class TestPluginToFrameworkContract:
    @pytest.fixture(autouse=True)
    def _require_framework(self, monkeypatch):
        if not (AGENT_ROOT / "agent").is_dir():
            pytest.skip(f"未找到 Hermes 框架: {AGENT_ROOT}/agent")
        import sys

        if str(AGENT_ROOT) not in sys.path:
            sys.path.insert(0, str(AGENT_ROOT))
        # sys.modules 里的 "agent" 常常不是真框架：
        #   - 仓库 tests/conftest.py 塞的是 MagicMock；
        #   - omnimem/__init__.py 在框架不可导入时会合成一个空的假 agent 模块。
        # 两者都不是包，importlib 在它们下面找 agent.skill_utils 会报
        # "'agent' is not a package"。这里按 __file__ 归属判定并临时摘掉非框架条目，
        # monkeypatch 在测试结束后原样装回。
        agent_root_resolved = str(AGENT_ROOT.resolve())
        for name in [n for n in list(sys.modules) if n == "agent" or n.startswith("agent.")]:
            module_file = getattr(sys.modules[name], "__file__", None)
            if isinstance(module_file, str) and module_file.startswith(agent_root_resolved):
                continue
            monkeypatch.delitem(sys.modules, name, raising=False)

    def test_skill_utils_symbols_exist(self):
        """provider_middleware 依赖的三个框架符号必须在 agent.skill_utils 上。"""
        mod = importlib.import_module("agent.skill_utils")
        for symbol in ("parse_frontmatter", "get_all_skills_dirs", "iter_skill_index_files"):
            assert hasattr(mod, symbol), f"agent.skill_utils 缺少 {symbol}"

    def test_memory_provider_is_subclassable(self):
        """OmniMemProvider 直接继承框架 ABC。"""
        from agent.memory_provider import MemoryProvider

        assert isinstance(MemoryProvider, type)

    def test_every_agent_import_in_plugin_source_resolves(self):
        """扫描插件源码中全部 `from agent.*`，逐个符号必须在活框架里解析成功。"""
        imports = _collect_from_imports(PACKAGE_ROOT, "agent", exclude_dirs=("tests", ".gitnexus"))
        assert imports, "扫描不到任何 agent.* 导入，扫描逻辑可能失效"
        unresolved = [
            f"{file}: from {module} import {symbol}"
            for file, module, symbol in imports
            if not _resolve(module, symbol)
        ]
        assert not unresolved, "以下框架符号在活框架中不存在：\n" + "\n".join(unresolved)


# ─────────────────────────────────────────────────────────────────────────────
# B. 兄弟插件 → OmniMem
# ─────────────────────────────────────────────────────────────────────────────
class TestExternalConsumerContract:
    @pytest.fixture(autouse=True)
    def _require_plugins_root(self):
        if not PLUGINS_ROOT.is_dir():
            pytest.skip(f"未找到插件根目录: {PLUGINS_ROOT}")

    def test_every_omnimem_import_by_sibling_plugins_resolves(self):
        """兄弟插件（adaptive_multi_agent 等）从 omnimem 导入的符号必须仍然存在。"""
        external_imports: list[tuple[str, str, str]] = []
        for plugin_dir in sorted(p for p in PLUGINS_ROOT.iterdir() if p.is_dir() and p.name != "omnimem"):
            external_imports.extend(
                (f"{plugin_dir.name}/{f}", m, s)
                for f, m, s in _collect_from_imports(plugin_dir, "omnimem")
            )
        if not external_imports:
            pytest.skip("当前插件根目录下没有第三方插件导入 omnimem")
        unresolved = [
            f"{file}: from {module} import {symbol}"
            for file, module, symbol in external_imports
            if not _resolve(module, symbol)
        ]
        assert not unresolved, "以下 omnimem 符号被外部插件引用但已不存在：\n" + "\n".join(unresolved)

    def test_sdk_constructor_accepts_storage_dir(self):
        """adaptive_multi_agent 以 OmniMemSDK(storage_dir=...) 初始化。"""
        from omnimem.sdk import OmniMemSDK

        names = list(inspect.signature(OmniMemSDK.__init__).parameters)
        assert "storage_dir" in names

    def test_sdk_recall_tolerates_unknown_kwargs(self):
        """消费方传 mode/max_tokens 等关键字，签名必须保持 **kwargs 弹性。"""
        from omnimem.sdk import OmniMemSDK

        assert hasattr(OmniMemSDK, "recall")
        params = inspect.signature(OmniMemSDK.recall).parameters
        assert "query" in params
        assert inspect.Parameter.VAR_KEYWORD in [p.kind for p in params.values()], (
            "OmniMemSDK.recall 丢掉 **kwargs 会打断外部调用方"
        )

    def test_sdk_memorize_tolerates_unknown_kwargs(self):
        from omnimem.sdk import OmniMemSDK

        params = inspect.signature(OmniMemSDK.memorize).parameters
        assert "content" in params
        assert inspect.Parameter.VAR_KEYWORD in [p.kind for p in params.values()]


# ─────────────────────────────────────────────────────────────────────────────
# C. 顶层命名空间不被插件抢占
# ─────────────────────────────────────────────────────────────────────────────
class TestNoTopLevelNamespaceHijack:
    """omnimem/__init__.py 曾把包目录 sys.path.insert(0, ...)。

    包目录下有 utils/config/core/memory/services/... 等通用名，插到头部会让它们在
    整个 gateway 进程里遮蔽 Hermes 框架的同名顶层包（实测 agent.skill_utils 里的
    `from utils import file_signature` 因此失败）。这条约束必须由测试钉住。
    """

    def test_init_does_not_prepend_package_root(self):
        source = (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")
        assert "sys.path.insert(0, str(_PROJECT_ROOT))" not in source
        assert "sys.path.append(str(_PROJECT_ROOT))" in source

    def test_package_root_generic_names_are_known(self):
        """列出会被暴露成顶层名的目录，提醒后续维护者这些是碰撞面。"""
        generic = {
            p.name for p in PACKAGE_ROOT.iterdir()
            if p.is_dir() and (p / "__init__.py").exists()
        }
        assert {"utils", "config", "core", "services"} <= generic

    def test_skill_preinject_is_off_by_default(self):
        """修好框架导入后 skill 预注入会真生效，默认必须保持线上现状（关闭）。"""
        from omnimem.config._config import _CONFIG_SCHEMA

        assert _CONFIG_SCHEMA["skill_preinject_enabled"]["default"] is False
        assert _CONFIG_SCHEMA["skill_preinject_enabled"]["type"] is bool


# ─────────────────────────────────────────────────────────────────────────────
# D. recall 响应形状（外部按 dict 键取值）
# ─────────────────────────────────────────────────────────────────────────────
class TestRecallResponseShape:
    def test_recall_result_keys(self):
        """engine.py 读 res["status"] / res["memories"]。"""
        from omnimem.handlers.deps import RecallResult

        keys = set(typing.get_type_hints(RecallResult).keys())
        assert {"status", "memories"} <= keys

    def test_recall_memory_keys(self):
        """engine.py 读 memory["memory_id"] / memory["content"]。"""
        from omnimem.handlers.deps import RecallMemory

        keys = set(typing.get_type_hints(RecallMemory).keys())
        assert {"memory_id", "content"} <= keys


# ─────────────────────────────────────────────────────────────────────────────
# E. skill 预注入的框架调用形状
# ─────────────────────────────────────────────────────────────────────────────
class TestSkillPreinjectWiring:
    """provider_middleware 必须走 agent.skill_utils 的真实公开名。

    原代码写的是 `from agent.skill_commands import _parse_frontmatter, ...`，活框架里
    这三个符号都在 agent.skill_utils（公开名 parse_frontmatter），ImportError 被
    logger.debug 吞掉后功能静默休眠。
    """

    def _mixin(self, config: dict[str, Any]):
        from omnimem.core.provider_middleware import ProviderMiddlewareMixin

        class Target(ProviderMiddlewareMixin):
            def __init__(self) -> None:
                self._config = config
                self._skill_index_built = False
                self._skill_index_cache = []

        return Target()

    def test_disabled_config_skips_scan(self):
        """默认关闭时不得扫描技能目录（保持线上现状）。"""
        target = self._mixin({"skill_preinject_enabled": False})
        target._build_skill_index()
        assert target._skill_index_cache == []
        assert target._skill_index_built is True

    def test_enabled_scan_uses_skill_utils_symbols(self, tmp_path, monkeypatch):
        """开关打开后按 agent.skill_utils 的三个公开名建索引。"""
        skill_dir = tmp_path / "demo"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: 处理 sqlite 并发与 database lock\n---\nbody\n",
            encoding="utf-8",
        )

        agent_pkg = type(sys)("agent")
        skill_utils = types.ModuleType("agent.skill_utils")

        def get_all_skills_dirs() -> list[Path]:
            return [tmp_path]

        def iter_skill_index_files(skills_dir: Path, filename: str):
            return sorted(skills_dir.rglob(filename))

        def parse_frontmatter(content: str):
            fm: dict[str, str] = {}
            for line in content.splitlines()[1:]:
                if line.strip() == "---":
                    break
                if ":" in line:
                    key, _, value = line.partition(":")
                    fm[key.strip()] = value.strip()
            return fm, content

        skill_utils.get_all_skills_dirs = get_all_skills_dirs
        skill_utils.iter_skill_index_files = iter_skill_index_files
        skill_utils.parse_frontmatter = parse_frontmatter
        agent_pkg.skill_utils = skill_utils
        monkeypatch.setitem(sys.modules, "agent", agent_pkg)
        monkeypatch.setitem(sys.modules, "agent.skill_utils", skill_utils)

        target = self._mixin({"skill_preinject_enabled": True})
        target._build_skill_index()

        assert [s["name"] for s in target._skill_index_cache] == ["demo-skill"]
        keywords = target._skill_index_cache[0]["keywords"]
        assert "sqlite" in keywords and "database" in keywords

    def test_broken_framework_api_warns_instead_of_debug(self, monkeypatch, caplog):
        """框架接口变更必须 warn（此前是 debug，等于静默失效）。"""
        import logging

        agent_pkg = type(sys)("agent")
        broken = types.ModuleType("agent.skill_utils")
        agent_pkg.skill_utils = broken
        monkeypatch.setitem(sys.modules, "agent", agent_pkg)
        monkeypatch.setitem(sys.modules, "agent.skill_utils", broken)

        target = self._mixin({"skill_preinject_enabled": True})
        with caplog.at_level(logging.WARNING, logger="omnimem.core.provider_middleware"):
            target._build_skill_index()

        assert target._skill_index_cache == []
        assert any("skill 预注入不可用" in r.getMessage() for r in caplog.records)
