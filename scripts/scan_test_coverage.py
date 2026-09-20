"""扫描运行时模块的测试覆盖情况，产出 docs/pending_work.md §0/§2 的全部数字。

口径要点（改动前先读 docs/pending_work.md §2「refs 计数口径」）：
- 运行时模块 = 排除 tests/ benchmarks/ examples/ scripts/ 、各级 __init__.py、
  以及仓库根的一次性脚本后剩下的 .py。
- covered  = 被 tests/ 下任一文件 import（同时识别 `omnimem.x.y` 与 `x.y` 两种风格）。
- refs     = 有多少个非测试运行时模块 import 了它，**不计 __init__.py 的聚合 re-export**。
"""

import ast
import os
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {"__pycache__", "node_modules"}
# top-level dirs never counted as runtime modules
NON_RUNTIME_TOP = {"benchmarks", "examples", "scripts", "tests"}
# one-off scripts sitting at repo root
ROOT_THROWAWAY = {"doctor.py", "async_sdk.py", "langchain_memory.py"}

def excluded_part(d):
    # 隐藏目录（.venv/.git/.mypy_cache/…）与任何备份目录（*.bak、*.bak.<ts>）都要跳过：
    # 重命名过的 venv 备份（如 .venv.broken.bak）内含上千个第三方 .py，
    # 只按名字精确匹配会让它们混进模块表并污染全部统计。
    return d in EXCLUDE_DIRS or d.startswith(".") or ".bak" in d

def modpath_of(rel):
    mp = str(rel.with_suffix("")).replace("/", ".")
    return mp[:-9] if mp.endswith(".__init__") else mp

def norm(mp):
    return mp[len("omnimem."):] if mp.startswith("omnimem.") else mp

def imports_of(p):
    out = set()
    try:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(norm(a.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                m = norm(node.module)
                out.add(m)
                for a in node.names:
                    out.add(f"{m}.{a.name}")
    return out

files = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if not excluded_part(d)]
    for fn in filenames:
        if not fn.endswith(".py"):
            continue
        p = pathlib.Path(dirpath) / fn
        files.append((p, p.relative_to(root)))

def is_runtime(rel):
    if rel.parts[0] in NON_RUNTIME_TOP:
        return False
    if rel.name == "__init__.py":
        return False
    # root-level files: keep only the real package entry points
    if len(rel.parts) == 1:
        return rel.name not in ROOT_THROWAWAY and not rel.name.startswith("_")
    return True

mods = {}
for p, rel in files:
    if not is_runtime(rel):
        continue
    lines = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
    mods[modpath_of(rel)] = (rel, lines)

test_imports = set()
main_importers = {}
for p, rel in files:
    if rel.parts[0] in {"benchmarks", "examples", "scripts"}:
        continue
    if rel.name == "__init__.py" and len(rel.parts) == 1:
        continue
    if len(rel.parts) == 1 and rel.name.startswith("_"):
        continue
    is_test = (rel.parts[0] == "tests") or rel.name.startswith("test_")
    imps = imports_of(p)
    if is_test:
        test_imports |= imps
    elif rel.name == "__init__.py":
        # __init__.py 只做聚合 re-export。把它算作"主链引用"会让任何被包导出的模块
        # 都显得有人用，从而掩盖"仅被导出、从未被真正 import"的模块。
        # 与本文把 __init__.py 归为空壳的口径保持一致，故不计入引用方。
        continue
    else:
        src = modpath_of(rel)
        for i in imps:
            main_importers.setdefault(i, set()).add(src)

def covered(mp):
    return mp in test_imports or any(i.startswith(mp + ".") for i in test_imports)

def refs(mp):
    srcs = set(main_importers.get(mp, set()))
    for i, ss in main_importers.items():
        if i.startswith(mp + "."):
            srcs |= ss
    srcs.discard(mp)
    return len({s for s in srcs if not s.rsplit(".", 1)[-1].startswith("test_")})

rows = [(refs(mp), lines, str(rel), mp)
        for mp, (rel, lines) in mods.items() if not covered(mp)]
rows.sort(key=lambda r: (-r[0], -r[1]))
cov = sum(1 for mp in mods if covered(mp))
t1 = [r for r in rows if r[0] >= 2]
t2 = [r for r in rows if r[0] == 1]
t3 = [r for r in rows if r[0] == 0]
print(f"runtime modules: {len(mods)}   covered: {cov} ({cov*100//len(mods)}%)   uncovered: {len(rows)}")
print(f"Tier1(refs>=2)={len(t1)}  Tier2(refs==1)={len(t2)}  Tier3(refs==0)={len(t3)}")
print(f"Tier1+Tier2 = {len(t1)+len(t2)}")
print()
for tag, group in (("T1", t1), ("T2", t2), ("T3", t3)):
    print(f"===== {tag} =====")
    for refs_n, lines, rel, mp in group:
        print(f"{lines:5d}  refs={refs_n}  {rel}")
