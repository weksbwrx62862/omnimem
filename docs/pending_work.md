# OmniMem 未完成工作清单

> 生成时间：2026-09-20（16:16 补入全量回归数字；16:40 复核 AST 扫描并修正若干条目；
> 17:30 重建 `.venv`，§1 由"绕过"升级为"根治"；18:10 本轮产出分批入库并清理
> `.venv.broken.bak`；19:05 **fresh clone 复验暴露"提交测试依赖未提交 WIP"，已修复**）
> 仓库：`/home/xxh/.hermes/plugins/omnimem`（git 分支 `master`）。
> **本轮产出已提交 7 个 commit，均未 push**：
> `cf7388a`（文档+工具）→ `ac799fb`（62 测试）→ `69efd86`（4 游离测试）→
> `500c7c2`（文档同步）→ `37c5f59`（**22 个未入库源模块 + `.gitignore` 修复3**）→
> `09b4adf`（**撤回 8 个依赖未提交 WIP 的测试**）→ `3f55027`（本次文档收口）。详见 §3.1。
> **既有 WIP 63 modified + 21 deleted 一律未碰**，详见 §3.2。
> 数据来源：AST 静态扫描运行时模块 × `tests/` 全部 import 的真实引用图，非凭记忆
> 回归基线：全量 suite **2680 passed / 0 failed**，经四次独立运行确认
> （`272.30s` 系统解释器 + `sys.path` 修复 → `/tmp/full_suite_fixed.log`；
> `275.03s` 经 `run_tests.sh` → `/tmp/run_tests_sh_full.log`；
> `267.45s` **重建后的 `.venv`，无任何路径 hack** → `/tmp/venv_rebuild_suite.log`；
> `255.63s` **3 个 commit 落地后**复验已提交状态 → `/tmp/post_commit_suite.log`）
> **⚠️ 上述四次全部在"带 WIP 的工作树"里跑**，因此会掩盖"已提交内容能否独立跑通"。
> 补做的 fresh clone 复验见 §3.1 末尾，结论：HEAD 需 `37c5f59` + `09b4adf` 才自洽。

---

## 0. 现状快照

> **覆盖率必须分两个状态报，否则数字会自相矛盾。** `scan_test_coverage.py` 走的是
> **文件系统**而非 git index，因此它会把"未跟踪的 WIP 源文件"和"已撤回的 8 个测试"
> 一并计入。下表左列是**已提交状态（代码状态 = `09b4adf`，fresh clone 实测；其后的
> `3f55027` 是纯文档 commit，不改任何数字）**，才是
> "还欠什么"的真实口径；右列是**当前工作树（含 63 modified + 未跟踪文件）**，
> 只是本地跑测试时看到的样子。

| 指标 | 已提交状态（clone 实测） | 当前工作树 |
|---|---|---|
| 运行时模块（**本文口径**） | **196** | 197 |
| 已被至少一个测试文件 import | **142（72%）** | 145（73%） |
| 未被任何测试 import | **54**（Tier1 **7** + Tier2 28 + Tier3 19） | 52（Tier1 6 + Tier2 28 + Tier3 18） |
| Tier1+Tier2（"值得补"） | **35** | 34 |
| 仓库内 `.py` 总数（排除隐藏目录与 `*.bak` 备份目录） | — | 489 |
| `tests/` 下 `.py` 文件数 | **131** | 133（131 `test_*.py` + `__init__.py` + `conftest.py`） |
| `tests/` 下仍未跟踪的文件 | — | **10 个**：2 个 `.bak.20260730_155405` 陈旧快照 + **8 个已撤回测试**（见 §3.1） |
| 本轮 commit | **7 个**（`cf7388a`/`ac799fb`/`69efd86`/`500c7c2`/`37c5f59`/`09b4adf`/`3f55027`，均未 push） | 同左 |
| 全量 suite 结果 | **2462 passed / 0 failed / 0 skipped**（fresh clone，36.93s） | 2680 passed / 0 failed |
| 本地跑测试的推荐命令 | `.venv/bin/python -m pytest tests/`（**无需任何路径 hack**，见 §1） | 同左 |
| `.venv` 状态 | 2026-09-20 重建：uv + CPython 3.13.13，136 包 / 1.5G，`torch 2.14.0+cpu`，0 个 CUDA 包 | 同左 |

> `tests/` 计数也对得上：工作树 133 − 8（撤回，不在 HEAD）+ 6（WIP 在磁盘上删掉、
> **但 HEAD 里还在**，故 clone 会带出来）= **131**。那 6 个是
> `test_drawer_closet.py`、`test_drawer_closet_saga.py`、`test_llm_summary.py`、
> `test_mermaid_canvas.py`、`test_provider_initializer.py`、`test_provider_lifecycle.py`，
> 属 §3.2 的既有 WIP，**不要动**；但要知道它们让 clone 的测试数比工作树多。
>
> **两列的差额已逐项核实**（不是估算）。未覆盖集合的差异：
> ```
> 已提交独有（= 因撤回测试而"重新欠下"的）  3 个
>   + multimodal/types.py            → 进 Tier1（refs=3）
>   + governance/forgetting_stages.py → 进 Tier2（refs=1）
>   + retrieval/planner.py           → 进 Tier3（refs=0）
> 工作树独有                              1 个
>   - core/test_engram_bridge.py      → 未跟踪文件，clone 里没有
> 另有 1 个纯 tier 迁移（两侧都未覆盖）
>   deep/kg/backends/factory.py：工作树 Tier2（refs=1）→ 已提交 Tier3（refs=0）
> ```
> 对账：`52 + 3 − 1 = 54`；`Tier1 6+1=7`、`Tier2 28+1−1=28`、`Tier3 18+2−1=19`，
> `7+28+19 = 54` ✓。
>
> ⚠️ 注意 `deep/kg/backends/factory.py` 的 refs 由 1 变 0：它唯一的引用方是未提交的
> WIP，所以**在已提交状态里它是"零引用"死代码候选**。这类模块补测前必须先做
> 死代码判定（见 §2 Tier3 的说明）。
>
> `core/test_engram_bridge.py` 名字带 `test_` 前缀却放在 `core/` 下，被扫描器当成
> 运行时模块 —— 位置本身可疑，见 §4。
>
> 复现两列的命令：
> ```bash
> # 已提交状态：在 fresh clone 里跑（clone 必须放 ext4，见 §3.1 的 /tmp 陷阱）
> cd <clone> && <repo>/.venv/bin/python scripts/scan_test_coverage.py
> # 工作树状态：直接在仓库里跑
> cd /home/xxh/.hermes/plugins/omnimem && .venv/bin/python scripts/scan_test_coverage.py
> ```

### 489 个 `.py` 的完整拆分（口径必须写清，否则数字对不上）

| 类别 | 数量 | 是否计入"运行时模块" |
|---|---|---|
| `benchmarks/**` | 105 | ✗ 范围外 |
| `tests/**` | 133 | ✗ 测试本身 |
| `__init__.py`（**上述 4 类目录之外**的各级聚合导出） | 24 | ✗ 空壳，无逻辑可测 |
| 仓库根一次性脚本（`_verify_*`/`_patch*`/`_repro*`/`_t_*`/`_list_api`/`doctor.py`/`async_sdk.py`/`langchain_memory.py`） | 20 | ✗ 范围外 |
| `examples/**` | 5 | ✗ 范围外 |
| `scripts/**` | 5 | ✗ 范围外 |
| **其余 = 运行时模块** | **197** | ✓ |
| 合计 | **489** | |

> 各行**互斥且穷尽**，加总严格等于 489。上一版此表写的是「合计 488、benchmarks 95、
> `__init__.py` 35」，有两处错：
> 1. **重复计数** —— 全仓共 35 个 `__init__.py`，其中 **11 个位于 `benchmarks/`、`tests/` 等
>    已被单独列行的目录内**，被同时算进了两行。本表只计目录外的 **24** 个。
> 2. **总数过时** —— 本轮新增 `scripts/scan_test_coverage.py`，`scripts/` 由 4 变 5，
>    总数由 488 变 **489**。
>
> 旧版各行的错误恰好相互抵消（benchmarks 少算 10、`__init__.py` 多算 11、scripts 少算 1，
> 净差 0），所以合计仍凑成 489 而看不出来 —— 这正是"分项必须互斥"的原因。
> 运行时模块 **197** 与覆盖率数字**不受此修正影响**（`is_runtime()` 从来不看这两行）。

> ⚠️ **此前记录的"249 个运行时模块"无法复现**。按上表口径实测为 **197**；
> 即便把 35 个 `__init__.py` 加回也只有 232，把 20 个根级一次性脚本再加回是 252，
> 都对不上 249。相应地，旧数字「170 已覆盖 / 79 未覆盖 / ≈45 值得补」应以本文
> 复核后的数字为准 —— **工作树「145 已覆盖 / 52 未覆盖 / Tier1+Tier2 = 34」**，
> **已提交状态「142 / 54 / 35」**（两状态之别见 §0 表）。
> 两者结论方向一致（旧 ≈45 vs 复核 34~35），但**逐项 tier 归属有出入**，详见 §2。
>
> 复核中还发现 `tests/` 内**混用两种 import 风格**：既有 `from omnimem.retrieval.fusion import …`
> （走 editable install 的 `.pth` → `/home/xxh/.hermes/plugins`；旧环境是
> `__editable__.omnimem-1.0.0.pth`，重建后的 `.venv` 是 `__editable__.omnimem-1.2.0.pth`，
> 指向同一目录。版本号差异来自 `pyproject.toml` 已升到 1.2.0，而该文件处于 modified/WIP 状态），
> 也有 `from governance import triple_extractor as …`（走 repo 根）。只匹配其中一种风格
> 会大幅高估未覆盖数（本次复核中途就因此得出过 8% 和 199 个未覆盖的错误结果），
> 扫描脚本必须先归一化掉 `omnimem.` 前缀。

本轮（最近一段会话）新增、当时 pytest + ruff 双绿的 5 个代表测试文件 ——
**但其中 3 个已随 `09b4adf` 撤回，现不在 HEAD 里**：

| 文件 | 用例数 | 被测模块 | 是否仍在 HEAD |
|---|---|---|---|
| `tests/test_context_manager.py` | 138 | `context/manager.py` | ✅ 在（clone 内复验 138 collected） |
| `tests/test_screening_engine.py` | 53 | `governance/screening_engine.py` | ✅ 在（clone 内复验 53 collected） |
| `tests/test_fusion_mixin.py` | 66 | `retrieval/fusion.py` | ❌ **已撤回**（依赖 WIP 的 `rrf_fuse(planner_weights=…)`） |
| `tests/test_forgetting_stages.py` | 49 | `governance/forgetting_stages.py` | ❌ **已撤回**（依赖 WIP 的 `_ACCESS_RETRY_COUNT`） |
| `tests/test_hybrid_orchestrator.py` | 43 | `retrieval/hybrid_orchestrator.py` | ❌ **已撤回**（依赖 WIP 的 `_planner_enabled`） |

> 这 3 个撤回文件**仍在磁盘上**（未跟踪），代码本身没坏，只是跑在未提交的 WIP 之上；
> 回库条件见 §3.1「仍未决」。这也是"覆盖率必须分两个状态报"的直接原因（见 §0）。

---

## 1. 【最高优先级 → 已根治】测试执行环境的 PYTHONPATH 冲突（`.venv` 已重建并三次验证）

### 症状
用长期沿用的命令跑全量测试：

```bash
PYTHONPATH=/home/xxh/.hermes/plugins:/usr/lib/python3/dist-packages \
  python3.13 -m pytest tests/ -q
```

得到 **25 failed + 1 collection error**，全部是同一个根因：

```
ImportError: cannot import name 'exceptions' from 'cryptography.hazmat.bindings._rust'
```

`tests/test_governance_service.py` 直接无法收集，导致整个 suite 被 `Interrupted`。

### 根因（已验证）
机器上存在两份 `cryptography`：

| 位置 | 版本 | `_rust` 形态 |
|---|---|---|
| `/home/xxh/.local/lib/python3.13/site-packages/cryptography` | 47.0.0 | `hazmat/bindings/_rust.abi3.so`（正常） |
| `/usr/lib/python3/dist-packages/cryptography` | 46.0.5 | `hazmat/bindings/_rust/`（**空目录 → 被识别为 namespace package**） |

`PYTHONPATH` 的条目会被插到 `sys.path` 中 **site-packages 之前**，于是 Debian 版 46.0.5 的
`cryptography/__init__.py` 胜出，而它的 `_rust` 是个没有 `.so` 的空目录，导入即失败。

### 已验证的修复方案
把 `dist-packages` 追加到 `sys.path` **末尾**而不是放进 `PYTHONPATH`：

```bash
cd /home/xxh/.hermes/plugins/omnimem
HF_HUB_OFFLINE=1 ANONYMIZED_TELEMETRY=False PYTHONPATH=/home/xxh/.hermes/plugins \
  /home/xxh/.local/bin/python3.13 -c "
import sys; sys.path.append('/usr/lib/python3/dist-packages')
import pytest; sys.exit(pytest.main(['tests/', '-q']))
"
```

不能简单删掉 `dist-packages`：`packaging` 只存在于该目录，pytest 启动即 `ModuleNotFoundError`。

用该方式重跑此前失败的 6 个文件（`test_security.py` / `test_sdk.py` / `test_governance_ext.py` /
`test_governance_weak_paths.py` / `test_recall_accuracy_fixes.py` / `test_governance_service.py`）：
**92 passed，0 failed**。

用同一方式重跑**全量** `tests/`（后台任务已完成，日志 `/tmp/full_suite_fixed.log`）：

```
platform linux -- Python 3.13.13, pytest-9.0.3, pluggy-1.6.0
configfile: pyproject.toml
plugins: asyncio-1.3.0, xdist-3.8.0, anyio-4.13.0, typeguard-4.4.4
================ 2680 passed, 199 warnings in 272.30s (0:04:32) ================
```

即此前的 **25 failed + 1 collection error 全部是环境噪音**，与本轮代码/测试改动无关；
修复路径后 suite 完全干净（199 条 warnings 均为第三方弃用提示：`pkg_resources`、
Pydantic V2.11 `model_fields`，非本仓问题）。

### 待办与本轮复核结论

**(b) `tests/conftest.py` 方案已排除** —— 不再是"需实测"，而是**确定无效**。pytest 在加载任何
conftest 之前就会失败：

```
File ".../site-packages/_pytest/config/__init__.py", line 1420, in _checkversion
    from packaging.version import Version
ModuleNotFoundError: No module named 'packaging'
```

`packaging` 只存在于 `/usr/lib/python3/dist-packages`，而 `_checkversion()` 发生在 conftest
导入之前，所以 conftest 里做任何 `sys.path` 重排都来不及。（`tests/conftest.py` 本身是
clean 状态、非 WIP，可以安全修改 —— 但改它解决不了这个问题。）

**原 `.venv` 已损坏 —— 已重建并验证（本节问题由此根治）**：

| 项 | 旧 `.venv`（已备份为 `.venv.broken.bak`） | 新 `.venv`（本轮重建） |
|---|---|---|
| `pyvenv.cfg` | `version = 3.13.7`、`executable = /usr/bin/python3.13`（该文件已不存在） | `home = ~/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/bin`、`version_info = 3.13.13`、`uv = 0.11.9` |
| `/usr/bin/python3` 现指向 | **Python 3.14** —— 与 venv 的 3.13 包目录错位 | 不再依赖系统解释器 |
| `sys.path` | `python314.zip` / `python3.14` / `lib-dynload`（找不到 `lib/python3.13/site-packages`） | 正常，且 **`dist-packages` 完全不出现** |
| 包数 | 123 | 136（1.5G） |
| `cryptography` | 需回落到系统 46.0.5（空 `_rust`）→ 病因 | **50.0.1**，来自 venv 本地 |
| `packaging` | `ModuleNotFoundError` | **26.3**，来自 venv 本地 |
| `.venv/bin/python -m pytest` | `No module named pytest` | 直接可用 |

重建过程有一个**必须记录的坑**：`requirements-dev.txt` 默认解析会拉入 `torch==2.14.0` 及
约 20 个 `nvidia-*` / `cuda-*` / `triton` 包（数 GB）。本机是 GTX 1050 Mobile（Pascal 架构）
且**未装驱动**，而 CUDA 13 已放弃 Pascal，这些包既装不下也用不上。规避方式：先从
`https://download.pytorch.org/whl/cpu` 预装 `torch==2.14.0+cpu`（已确认阿里云镜像**不提供**
`+cpu` 本地版本），再装 `-r requirements-dev.txt`。结果：`torch 2.14.0+cpu`，
`nvidia|cuda|triton` 包数 = **0**。

**根治验证**（无 `PYTHONPATH`、无 `sys.path.append`，`env -u PYTHONPATH` 显式清空）：

```
$ env -u PYTHONPATH .venv/bin/python -m pytest tests/ -q
================ 2680 passed, 204 warnings in 267.45s (0:04:27) ================
```

与 §1 修复后在系统解释器上跑出的 **2680 passed** 完全一致 —— 说明 25 failed + 1 collection
error 的唯一成因确实是 `dist-packages` 遮蔽，健康 venv 下该问题不再存在。
（日志：`/tmp/venv_rebuild_suite.log`）

**CI 不受影响**：`.github/workflows/ci.yml` 用 `actions/setup-python` + `pip install -r
requirements-dev.txt` 的干净环境，`packaging`/`cryptography` 均来自 pip，不存在 dist-packages
遮蔽。本问题**纯本地**。

落地选项（按重建后的实际推荐度）：

- [x] (d) **重建 `.venv` —— 已完成，是当前推荐路径**。此后本地跑测试只需
      `.venv/bin/python -m pytest tests/`，**不需要任何路径 hack**。见上表与验证输出。
- [x] (a) 在仓库根加 `run_tests.sh` —— **已完成、已实测、已提交（`cf7388a`），并决定保留**。
      它针对"用系统解释器 `/home/xxh/.local/bin/python3.13` 跑测试"这一旧路径，通过在解释器内
      `sys.path.append` 绕过遮蔽（注意不能用 `PYTHONPATH` 表达本修复 —— PYTHONPATH 必然排在
      site-packages 之前，正是病因）。重建 venv 后它**降级为兜底**：36 行、已入库、删除是净损失，
      且是 venv 缺失/损坏时唯一不依赖 venv 的复现路径 —— 而 venv 损坏正是本轮真实发生过的事。
- [ ] (c) 卸载 `/usr/lib/python3/dist-packages/cryptography` 46.0.5 —— **动系统包，需用户授权**，
      未执行。重建 venv 后已无必要：venv 内 `include-system-site-packages = false`，
      系统包不再参与解析。仅在"有人继续用系统解释器裸跑 pytest"时才需要。
- [x] (e) 清理 `.venv.broken.bak`（1.4G）—— **已删除**，释放 1.4G。删除依据见 §3.1。
- [x] ~~更新 `AGENTS.md` / `CLAUDE.md` 中记录的测试命令~~ —— **原记录有误，已作废**：两个文件
      内容完全相同，且从头到尾都是 GitNexus 自动生成块（`<!-- gitnexus:start -->` …
      `<!-- gitnexus:end -->`），**并不含任何测试命令**，手改会在下次 `analyze` 时被覆盖。
      真正记载测试命令的是 `CONTRIBUTING.md`（第 76–82 行，`pytest tests/ -v` 等），
      那里的写法对 CI/干净环境正确，无需改；如需提示本地路径问题，应加在 `run_tests.sh` 的注释里。

#### `run_tests.sh`（本轮新增，仓库根，已 chmod +x，已实测；重建 venv 后降级为兜底）

> 本文**不再内嵌脚本全文** —— 内嵌副本已与真实文件漂移过一次。以仓库根的 `run_tests.sh` 为准。

需要记住的只有三点机制：

1. 用 bash 而非 `PYTHONPATH` 表达修复，核心是内联 python 启动器里的
   `sys.path.append("/usr/lib/python3/dist-packages")` —— 必须**排在末尾**才有效。
2. 默认目标写死为 `tests/`，不能省略：`pyproject.toml` 的 `testpaths = ["tests", "."]`
   会让裸跑扫到 `core/test_engram_bridge.py` 等非 `tests/` 测试。
3. 离线环境变量（`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / 三个 telemetry 开关）
   在脚本内 `export`，避免测试触发联网。

实测记录（本轮全部跑过，非纸面推断）：

| 调用 | 结果 |
|---|---|
| `bash -n run_tests.sh` | 语法 OK |
| `./run_tests.sh tests/test_security.py tests/test_governance_service.py` | 28 passed（同样两文件用旧命令 → collection error + Interrupted） |
| `./run_tests.sh tests/ -k "fusion or trust"` | 82 passed, 2598 deselected（确认 `-q --tb=short` 放在路径**之前**，`-k` 才不会吞掉路径参数） |
| `./run_tests.sh`（无参，全量） | **2680 passed / 0 failed，275.03s** |

> 脚本硬编码了 `/home/xxh/.local/bin/python3.13` 与 `/home/xxh/.hermes/plugins`，
> 因为本问题是**这台机器特有的**；CI 走 `ci.yml`，不经过此脚本。
> 一个 bash 坑已规避：`"${@:-}"` 在无参时会展开成**一个空字符串参数**，pytest 会把它当路径，
> 故改用 `target` 数组并在无参时默认 `("tests/")`。

---

## 2. 未被测试覆盖的运行时模块（按价值排序）

判据：**主链引用数** = 有多少个非测试模块 import 了它。数值越高，说明它在真实链路上越关键。

> **下面三张 tier 表一律以「已提交状态（代码状态 = `09b4adf`）」为准** —— 因为本文是"还欠什么"
> 的清单，欠的是仓库里的账，不是本地工作树的账。仅存在于工作树的行用 ~~删除线~~ 标出并注明，
> 仅存在于已提交状态的行会写明"仅已提交口径"。两状态的完整差额见 §0 表下注。
> 表内行数若两状态不同（WIP modified 所致），写成 `HEAD / 工作树` 双值。
> 三张表共 **56 行 = 54 个已提交状态未覆盖模块 + 2 个删除线行**（工作树独有），
> 已用脚本逐行比对 `scan_test_coverage.py` 输出，**成员与行数均精确一致**。

### 复核后的修正（与旧版 §2 的差异，逐条列出）

**(A) 旧版列为"未覆盖"、实则已有测试 —— 应从名单删除**

| 模块 | 行数 | 实际覆盖它的测试 |
|---|---|---|
| `governance/temporal_kg.py` | 708 | `tests/test_refcount_logs.py`（函数内 `from governance.temporal_kg import …`，AST 扫描需走到函数体才能发现） |
| `governance/triple_extractor.py` | 480 | `tests/test_triple_extractor.py`（`from governance import triple_extractor as te`） |
| `governance/adaptive_optimizer.py` | 572 | `tests/test_adaptive_optimizer.py` |
| `governance/semantic_clusterer.py` | 405 | `tests/test_semantic_clusterer.py` |
| `governance/knowledge_graph_enhancer.py` | 399 | `tests/test_knowledge_graph_enhancer.py` |
| `governance/personalized_fsrs.py` | 358 | `tests/test_personalized_fsrs.py` |
| `governance/performance_optimizer.py` | 313 | `tests/test_performance_optimizer.py` |
| `governance/wiki_upgrade.py` | 257 | `tests/test_wiki_upgrade.py` |
| `governance/api.py` | 204 | `tests/test_governance*.py` |

> 这 9 个全在 `governance/` 下，且全部用**不带 `omnimem.` 前缀**的 import 风格。
> 旧扫描只认前缀风格，于是整片 `governance/` 被误判为未覆盖 —— 这是 249 vs 197 口径差的主要来源。

**(B) Tier 归属变动**（"复核后"一律为**严格口径**，即不计 `__init__.py` 引用）

| 模块 | 旧版 | 复核后 |
|---|---|---|
| `memory/unified_index.py`(554) | Tier 1（refs=3） | **Tier 2**（refs=1） |
| `memory/migration_tool.py`(457) | Tier 1（refs=2） | **Tier 2**（refs=1） |
| `deep/reflect/pipeline.py`(417) | Tier 1（refs=2） | **Tier 3**（refs=0，唯一引用方是 `deep/reflect/__init__.py`） |
| `deep/kg/query.py`(482) | Tier 2 | **Tier 2**（refs=1；宽口径下会升到 T1） |
| `deep/kg/temporal.py`(93) | Tier 2 | **Tier 2**（refs=1；同上） |
| `facades/sync_facade.py`(95) | Tier 2 | **Tier 3**（refs=0） |
| `facades/deep_memory.py`(53) | Tier 2 | **Tier 3**（refs=0） |
| `services/base.py`(51) | Tier 2 | **Tier 3**（refs=0） |
| `deep/kg/relationships.py`(100) | Tier 3 | **Tier 2**（refs=1） |
| `multimodal/embedders.py`(110) | Tier 3 | **Tier 3**（refs=0，旧版归属正确） |
| `multimodal/parse.py`(110) | Tier 2 | **Tier 3**（refs=0） |
| `deep/reflect/writer.py`(47) | Tier 3 | **Tier 2**（refs=1） |

**(C) 旧版遗漏、复核新增**（括号内为严格口径 tier）

- Tier 1：`governance/forgetting_core.py`(242)
- Tier 2：`core/provider_middleware.py`(404)、`facades/governance.py`(201)、`core/async_provider.py`(191)、
  `core/plur_config.py`(157)、`handlers/compat_handler.py`(145)、`facades/retrieval.py`(138)、
  `facades/storage.py`(108)、`memory/markdown_store.py`(108)、`multimodal/store.py`(98)
- Tier 3：`storage/milvus_store.py`(192)、`embedding/onnx_provider.py`(131)、
  `handlers/_compat.py`(20)、`deep/knowledge_graph.py`(11)

> 其中 `handlers/compat_handler.py`、`facades/governance.py`、`governance/forgetting_core.py`
> 三个都是 **modified 状态的既有 WIP**，补测前须先确认改动意图（见 §3.2）。

**(D) 已复核为准确、继续沿用的旧版备注**

- `deep/kg/extraction.py`：`tests/test_triple_extractor.py` 测的是 `governance/triple_extractor.py`，
  **不是**这个文件 —— 备注成立。
- `deep/kg/query.py`：`tests/test_kg_query.py` 只 `from omnimem.deep.kg import entity` +
  `from omnimem.deep.kg.builder import KnowledgeGraph`，确实没碰 `query.py` —— 备注成立。
- `memory/migration_tool.py`：`tests/test_migration.py` 只 `from omnimem.utils.migration import SchemaMigrator`，
  两个 migration 文件都真实存在，测的是另一个 —— 备注成立。
- `deep/reflect/pipeline.py`：全仓 `tests/` 无任何 `reflect.pipeline` import；
  `synthesis`/`prompts`/`disposition` 已测而编排层缺 —— 备注成立。

### refs 计数口径（已定案，影响全部 Tier 归属）

**主链引用数不计 `xxx/__init__.py` 的聚合 re-export。**

理由：`__init__.py` 只是导出壳，把它算作引用方会让"被包导出过"等价于"被真正依赖"，
从而掩盖那些**除了自己包的 `__init__.py` 之外无人 import** 的模块。本文既然把
`__init__.py` 归为空壳（见 §0 拆分表），就不该反过来让它给别的模块贡献引用数。

这条口径已由 `scripts/scan_test_coverage.py` 固化（见该文件内注释），可复现。
**采用严格口径前，同一份数据会得出 Tier1=17 / Tier2=22 / Tier3=13**；
下文一律为严格口径结果 —— **已提交状态 Tier1=7 / Tier2=28 / Tier3=19（合计 54）**，
**工作树 Tier1=6 / Tier2=28 / Tier3=18（合计 52）**，两者差额逐项见 §0 表下注。
跨会话比对时若看到 17/22/13，说明对方用的是宽口径；若看到 6/28/18，说明对方扫的是
**带 WIP 的工作树**而非 clone —— 都不是数据变了。

### Tier 1 — 主链引用 ≥ 2，建议优先补（已提交状态 **7 个** / 工作树 6 个）

| 行数 | refs | 模块 | 备注 |
|---|---|---|---|
| 273 | 6 | `utils/llm_client.py` | 全仓引用最多的未测模块；LLM 调用封装，可用假 transport 离线测 |
| 261 | 3 | `deep/kg/extraction.py` | 见上 (D)：别被同名的 `test_triple_extractor.py` 误导 |
| 94 | 3 | `multimodal/types.py` | **仅存在于已提交口径**：工作树里被 `tests/test_multimodal_ingest.py` 覆盖，但该测试已随 `09b4adf` 撤回（依赖 WIP 的 `RecallService` 新属性）。`37c5f59` 刚把它入库，是 5 个 `multimodal/*` 模块的类型底座，refs=3 来自同包的 `parse.py:17`/`store.py:14`/`embedders.py:16`。内容为 2 个 dataclass（`MediaPart`/`MediaRef`）+ 1 个 `MediaEmbedder` Protocol + 1 个 mime 辅助函数（**无枚举**）。**已实跑验证**：`MediaPart.sha256`/`.has_inline` 是 **`@property` 不是方法**（写测试时别加括号），`MediaRef.to_dict()`/`from_dict()` 往返相等、且 `from_dict({})` 有防御性默认值（`kind='file'`/`size=0`）值得单独测，`_kind_from_mime` 只映射到 `image`/`audio`/`file` 三种。**零外部依赖、可离线秒测 —— Tier1 里性价比最高的一个** |
| 236 / 242 | 2 | `governance/forgetting_core.py` | **modified（既有 WIP）**，补测前先确认改动意图。行数两状态不同：HEAD **236** / 工作树 **242** |
| 194 | 2 | `handlers/query_planner.py` | **同名陷阱（已核实）**：`tests/test_query_planner.py` 存在，但它 import 的是 `retrieval.planner.QueryPlanner`，与本文件无关。refs=2 来自 `core/provider_lifecycle.py:282` 与 `services/recall_service.py:196` |
| 131 | 2 | `core/warmup_manager.py` | 冷启动预热，纯逻辑可测 |
| 110 | 2 | `core/llm_initializer.py` | 注意：`tests/test_provider_initializer.py` 已被删除 |

> Tier 1 从宽口径的 17 个收缩到 6~7 个 —— 这正是严格口径的价值：
> 其余 11 个此前看着"有 2 处依赖"，实际有一处是 `__init__.py`，真实耦合只有 1。
> 6 与 7 的差别只由 `multimodal/types.py` 一个模块造成，根因是 §3.1 撤回的 8 个测试。

### Tier 2 — 主链引用 = 1（两状态都是 **28 个**，但成员差 1 个）

> 已提交口径含 `governance/forgetting_stages.py`、不含 `deep/kg/backends/factory.py`
> （后者 refs 掉到 0，移入 Tier 3）；工作树口径正好相反。数量巧合都是 28。

| 行数 | 模块 | 备注 |
|---|---|---|
| 554 | `memory/unified_index.py` | 最大的未测运行时模块；**modified（既有 WIP）** |
| 482 | `deep/kg/query.py` | 见上 (D)：`test_kg_query.py` 只 import 了 `entity` + `builder` |
| 457 | `memory/migration_tool.py` | 见上 (D)：`test_migration.py` 测的是 `utils/migration.py` |
| 404 | `core/provider_middleware.py` | |
| 340 | `governance/forgetting_stages.py` | **仅已提交口径**，且 **modified（既有 WIP）**。工作树里被 `tests/test_forgetting_stages.py`（49 用例）覆盖，但该测试已随 `09b4adf` 撤回（依赖 WIP 的 `_ACCESS_RETRY_COUNT`）。补测前先确认 WIP 改动意图 |
| 307 | `core/llm_memory_manager.py` | |
| 304 | `governance/auditor.py` | |
| 295 | `governance/forgetting_ops.py` | |
| 257 | `core/reflection_trigger.py` | |
| 228 | `deep/kg/backends/neo4j_backend.py` | 需假 driver，离线可测；`37c5f59` 入库 |
| 217 | `core/llm_client_manager.py` | |
| 200 / 201 | `facades/governance.py` | **modified（既有 WIP）**；HEAD **200** / 工作树 **201** |
| 200 | `handlers/priming.py` | |
| 197 | `utils/async_llm.py` | |
| 191 | `core/async_provider.py` | |
| 169 | `internalize/train_loop.py` | **modified（既有 WIP）** |
| 167 | `core/system_prompt_builder.py` | |
| 157 | `core/plur_config.py` | |
| 140 / 145 | `handlers/compat_handler.py` | **modified（既有 WIP）**；HEAD **140** / 工作树 **145** |
| 138 | `facades/retrieval.py` | |
| 114 | `compression/priority.py` | 宽口径 refs=2 的另一处是 `compression/__init__.py`；真实引用方只有 `compression/pipeline.py` |
| 108 | `facades/storage.py` | |
| 108 | `memory/markdown_store.py` | |
| 101 | `compat/provider_proxy.py` | |
| 100 | `deep/kg/relationships.py` | |
| 93 | `deep/kg/temporal.py` | |
| 63 | `core/backup_manager.py` | |
| 47 | `deep/reflect/writer.py` | |
| ~~34~~ | `deep/kg/backends/factory.py` | **仅工作树口径**（已提交状态下 refs=0 → 移入 Tier 3）。34 行，**最小、成本最低，适合第一个动手验证流程** |

### Tier 3 — 主链引用 = 0（已提交状态 **19 个** / 工作树 18 个，补测前必须先做死代码判定）

| 行数 | 模块 | 说明 |
|---|---|---|
| 417 | `deep/reflect/pipeline.py` | **仅被 `deep/reflect/__init__.py` 导出**，无真实调用方；宽口径下曾被列进 Tier 1 |
| 286 | `core/plur_client.py` | PLUR 客户端 |
| 214 | `core/plur_server.py` | PLUR 服务端 |
| 212 | `retrieval/planner.py` | **仅已提交口径**。`37c5f59` 刚入库；工作树里被 `tests/test_query_planner.py` 覆盖，但该测试已随 `09b4adf` 撤回。**refs=0 是因为唯一引用方 `retrieval/hybrid_orchestrator.py` 的 planner 接线属未提交 WIP** —— 典型的"WIP 提交后 tier 会自动上升"，现在别当死代码删 |
| 192 | `storage/milvus_store.py` | 需外部服务；考虑只测参数校验/异常分支 |
| 166 | `governance/distributed_sync.py` | **已确认废弃**，主链路未使用 —— 不要补测 |
| 147 | `utils/llm_backend.py` | |
| 131 | `embedding/onnx_provider.py` | |
| 118 | `protocols.py` | |
| 110 | `multimodal/parse.py` | `37c5f59` 入库 |
| 110 | `multimodal/embedders.py` | `37c5f59` 入库；注意 `multimodal/__init__.py:34` 是 `__getattr__` 里的**惰性 import**，静态扫描看不见调用方 |
| 98 | `multimodal/store.py` | `37c5f59` 入库 |
| 95 | `facades/sync_facade.py` | 仅被 `facades/__init__.py` 导出 |
| 86 | `core/cross_db_coordinator.py` | |
| 53 | `facades/deep_memory.py` | 仅被 `facades/__init__.py` 导出 |
| 51 | `services/base.py` | 仅被 `services/__init__.py` 导出；基类，可能被动态继承 |
| 34 | `deep/kg/backends/factory.py` | **仅已提交口径**，且 refs 由工作树的 1 变 **0** —— 唯一引用方是未提交 WIP。`37c5f59` 入库 |
| 20 | `handlers/_compat.py` | 私有兼容 shim；`.gitignore` 修复2 的受害者，已入库 |
| 11 | `deep/knowledge_graph.py` | 极小，疑似仅 re-export |
| ~~318~~ | `core/test_engram_bridge.py` | **仅工作树口径**（未跟踪，clone 里没有）。放在 `core/` 下的旧测试脚本，**不是运行时模块**，建议移出或删除 |

> 已提交状态 19 个 = 工作树 18 个 − `core/test_engram_bridge.py`（未跟踪）
> \+ `retrieval/planner.py` + `deep/kg/backends/factory.py`。

> ⚠️ **refs=0 ≠ 死代码。** 上表 3 个标注"仅被 `__init__.py` 导出"的模块
> （`sync_facade` / `deep_memory` / `services.base`，外加 Tier 2 里的 `deep/reflect/pipeline.py`，
> 共 4 个）都是 facade/基类/编排层，
> 典型用法是外部拿到包对象后按属性访问，AST 静态扫描看不到这类动态分发。
> 判定死代码前**必须**用 grep 或 GraphNexus `impact`（`risk: UNKNOWN` 要当未决处理）二次确认，
> 别仅凭本表就删代码或断言无用。
>
> `async_sdk.py`(86) 与 `langchain_memory.py`(41) 在本文口径里归为**仓库根 SDK 入口**、
> 不计入 197 个运行时模块；旧版把 `langchain_memory.py` 列进 Tier 3 属口径不一致。
> 若认定它们是对外 API，应单独按"公开 SDK 契约测试"对待，而不是塞进 Tier 3。

### 明确不做（范围外）

| 类别 | 条目 |
|---|---|
| 已废弃 | `governance/distributed_sync.py`(166) — 此前已确认主链路未使用 |
| 评测/基准 | `benchmarks/**`（**105 个 `.py`**，含 `STATE-Bench/` 与 `LongMemEval/` 两棵子树） |
| 脚本工具 | `scripts/mock_conflict_test.py`、`scripts/omni_dashboard.py`、`scripts/check_dependency_sync.py`、`scripts/omni_import.py`、`doctor.py` |
| ↳ **例外** | `scripts/scan_test_coverage.py` 虽在 `scripts/` 下，但它是**本文全部数字的可复现来源**，属分析工具而非产品脚本，**建议提交**（见 §3.1） |
| 示例 | `examples/**`（5 个文件，约 1400 行） |
| 一次性验证脚本 | 仓库根 20 个：`_verify_m4/m6/m8`、`_verify_phase1/2/3`、`_verify_pool`、`_patch1/2a/2b/3/4`、`_repro2`、`_repro_eval`、`_list_api`、`_t_gate`、`_t_re`、`async_sdk.py`、`doctor.py`、`langchain_memory.py` |
| 遗留 | `core/test_engram_bridge.py`(318) — 放在 core/ 下的旧测试脚本 |
| 空壳 | 各 `__init__.py` 聚合导出（`facades`/`associative`/`handlers`/`services`/`context`/`deep`/`perception`/`compat`/`compression`/`internalize`） |

---

## 3. 提交与仓库状态（本轮产出已提交；既有 WIP 仍未动）

当前 `git status` 混杂了 **既有 WIP** 与 **本轮新增**。本轮新增已按 7 个**可独立 revert** 的
commit 入库；既有 WIP 一律未碰。

### 3.1 本轮产出（已提交，7 个 commit）

> 下表**最多滞后一个 commit**：记录本次更新的 commit 无法引用自己的哈希，
> 所以"7 个"这个计数天然可能少算它自己一次。**权威来源是 `git log --oneline`，不是本表。**
> 本表的价值在于解释每个 commit 为什么存在，而非充当计数台账。

提交前按 `AGENTS.md` 要求跑了 `node .gitnexus/run.cjs detect-changes --scope all`，结果
**84 文件 / 266 符号 / 105 流程 / risk=critical**。该 critical **全部来自既有 WIP**，与本轮
无关，已核实两点：

- 本轮 3 个新文件在变更图中**完全不出现**（`grep -E "pending_work|run_tests.sh|scan_test_coverage"` 无命中）
- `scripts/scan_test_coverage.py` **零 importer** —— 一个 markdown 文档、一个 bash 脚本、
  一个无人 import 的独立分析脚本，结构上不可能影响代码图

| commit | 内容 | 规模 |
|---|---|---|
| `cf7388a` | `docs/pending_work.md` + `scripts/scan_test_coverage.py` + `run_tests.sh` | 3 文件 |
| `ac799fb` | 本轮 62 个补测文件（2026-09-20） | 14796 行 |
| `69efd86` | 收编 4 个自 2026-08-07 游离的测试 + ruff 清零 | 761 行 |
| `500c7c2` | 文档同步（§0/§1/§3/§5 收口） | 1 文件 |
| `37c5f59` | **补交 22 个未入库源模块** + `.gitignore` 修复3 | 23 文件 / 2899 行 |
| `09b4adf` | **撤回 8 个依赖未提交 WIP 的测试** | 8 文件 / −2742 行 |
| `3f55027` | 文档收口：记录 clone 复验结论 + 双状态覆盖率口径 | 1 文件 / +231 −50 |

> `3f55027` 就是"写下这张表"的 commit，所以它**无法在表里准确引用自己的哈希**
> （提交前哈希未知）。它是**纯文档改动**，不含任何 `.py`，因此不可能影响
> `09b4adf` 上已验证的 2462 passed —— 这也是本文所有覆盖率数字标注为
> 「代码状态 = `09b4adf`」而非「HEAD」的原因。

后两个代码 commit（`37c5f59`/`09b4adf`）是被一次 **fresh clone 复验**逼出来的，
详见下面的「教训」小节。

入库前的核查（均实测，非推断）：

- **暂存集精确比对**：`git add --pathspec-from-file` 后 `diff` 暂存清单与预期清单 → EXACT MATCH；
  全程**未用 `git add -A`**，故 63 个 WIP modified 文件一个都没被卷进来
- **`.bak` 已排除**：`tests/*.bak.20260730_155405` 两个文件**未提交**（见下）
- **凭据扫描**：对 68 个未跟踪文件跑硬编码密钥正则，仅命中
  `tests/test_security.py.bak.*` 两行 —— 经核实是脱敏测试的**夹具字面量**
  （`text = 'api_key="supersecret12345678"'`、假 Fernet token `gAAAAAB1234567890abcdef`），
  **非真实凭据**，误报
- **ruff**：62 个新文件 `All checks passed`；4 个游离文件有 10 处 I001/F401，已 `--fix`
  并复跑 **68 passed** 确认未破坏行为
- **提交后复验**：3 个 commit 全部落地后重跑全量 suite → **2680 passed / 0 failed（255.63s）**，
  即入库状态本身是绿的（`/tmp/post_commit_suite.log`）

#### 教训：上一条"提交后复验"是**不充分**的，它掩盖了一个真实缺陷

那次复验跑在**带 63 个 WIP modified 的工作树**里，所以"绿"只证明了
「WIP + 本轮测试」能跑通，**没有证明「已提交内容」能跑通**。补做 fresh clone 复验后
立刻暴露问题：

```bash
git clone --no-hardlinks <repo> /path/on/ext4 && cd $_ && pytest tests/
# → 32 failed, 2590 passed, 8 errors        ← HEAD 不自洽
```

两类根因，分别对应两个修复 commit：

**(a) 已提交的测试 import 了从未入库的源模块** → `37c5f59`

`ac799fb`/`69efd86` 里的测试引用了 22 个只存在于磁盘、git 从未跟踪的源文件，
clone 后必然 `ImportError`。补交这 22 个（4 组）：

| 组 | 文件 |
|---|---|
| `retrieval/` | `planner.py` `verifier.py` `agentic.py` `organizer.py` `contradiction.py` `canonical_entity.py` `self_healing.py` `session_local.py` `temporal_separation.py` |
| `multimodal/` | `__init__.py` `types.py` `parse.py` `embedders.py` `store.py` |
| `importers/` | `__init__.py` `_base.py` `sources.py` |
| `deep/kg/backends/` | `__init__.py` `protocol.py` `factory.py` `sqlite_backend.py` `neo4j_backend.py` |

入库前对这 22 个做了 ruff 清理（114 处 `--fix` 自动修 + 手工把
`retrieval/verifier.py` 的 `_count_filled(intent=…)` 形参改名 `_intent` 清掉最后一处
ARG004；唯一调用点是 `verifier.py:89` 的位置传参，改名安全）。

> 🔴 **`.gitignore` 修复3 —— 同一误伤第三次发生。** `importers/_base.py` 被
> `_[!_]*.py` 这条**未锚定到仓库根**的规则吞掉，而 `importers/__init__.py:17`
> 在模块级 `from . import _base`，漏掉它 clone 后必然 ImportError。文件里原有注释
> 显示 `config/_config.py`、`handlers/_compat.py` 已是前两次的受害者。
> 已加 `!importers/_base.py`，并把"根因是规则未锚定、今后新增此类文件都要补一行
> `!path/to/_name.py`（或干脆把规则改成 `/_[!_]*.py`）"写进注释。
> **这个坑还会第四次发生**，除非把规则本身锚定。

**(b) 已提交的测试调用了只存在于 WIP 的 API** → `09b4adf`（撤回 8 个）

这 8 个测试是**照着未提交的 WIP 工作树写的**，依赖 WIP 才有的接口，无法随源码一起提交
（源码属 §3.2「不要动」）：

| 撤回的测试 | 行数 | 依赖的未提交 WIP |
|---|---|---|
| `test_fusion_mixin.py` | 808 | `FusionMixin.rrf_fuse(planner_weights=…)` 新形参 |
| `test_forgetting_stages.py` | 556 | `governance/forgetting_stages._ACCESS_RETRY_COUNT` |
| `test_hybrid_orchestrator.py` | 493 | `HybridOrchestrator._planner_enabled` |
| `test_query_planner.py` | 373 | 同上链路（WIP 版 orchestrator） |
| `test_privacy_audit.py` | 186 | `AuditLogger.log()` 已改签名 |
| `test_dashboard.py` | 184 | 未跟踪的 `scripts/omni_dashboard.py` |
| `test_multimodal_ingest.py` | 91 | `RecallService` 新增属性 |
| `test_recall_agentic_wiring.py` | 51 | `RecallService` 新增属性 |

用 `git rm --cached`（**不是** `git rm`）撤回，8 个文件**全部仍在磁盘上**、退回未跟踪状态，
等 WIP 提交后即可原样 `git add` 回来 —— 已逐个 `[ -f ]` 核实存在。
**未做任何 history rewrite**，`ac799fb` 保持原样。其余 58 个测试确认不依赖 WIP，保留。

**最终复验（决定性）**：

```
fresh clone @ HEAD=09b4adf，ext4，TMPDIR 也在 ext4
→ 2462 passed / 0 failed / 0 skipped（collected 2462），36.93s
```

即 **HEAD 现在是自洽的**：任何人 clone 下来都能全绿，不需要这台机器的 WIP。

> 🟠 **`/tmp` 陷阱（复验时踩到，会伪造失败）**：本机 `/tmp` 是带 `usrquota` 的 tmpfs
> （7.6G）。把 2.6G 的 clone 放进去会**打爆用户配额**，症状是
> `bash: pwd: 写入错误: 超出磁盘配额` + 大面积 `sqlite3.OperationalError: disk I/O error`
> + 跑得异常快（47s 而非 ~260s，因为测试在 setup 阶段就炸了）。
> **这些失败与仓库无关**，纯粹是环境。复现方法：clone 放 ext4、并显式
> `TMPDIR=<ext4 路径>`（pytest 的 `tmp_path` fixture 默认落在 `/tmp`）。
> 第一次 clone 复验得出的「2 failed（`test_sdk.py`）」同样就是这个假象 ——
> 已用 `git log -1 -- tests/test_sdk.py`（属 `a3e5419`，非本轮）+ 真实树 5/5 通过 双重排除。

**已决定保留的：**

- ✅ `run_tests.sh` **不删**。它只有 36 行、已入库（删除反而是净损失），且是 venv 缺失/损坏时
  唯一不依赖 venv 的复现路径 —— 而"venv 损坏"正是本轮真实发生过的事。已在其头部注释里
  写明"首选 `.venv/bin/python -m pytest tests/`，本脚本为兜底"。
  硬编码绝对路径属**有意为之**：该问题是这台机器特有的，CI 不经过此脚本。
- ✅ `.venv.broken.bak`（1.4G）**已删除**，释放 1.4G（磁盘 76G/116G）。删除前已核实三件事：
  1. 新 venv 健康 —— `pytest 9.1.1 / cryptography 50.0.1 / torch 2.14.0+cpu`
  2. 该备份**从来不是可用回滚点** —— 其 `pyvenv.cfg` 写死 `executable = /usr/bin/python3.13`，
     而该文件**已不存在**，正是它损坏的原因
  3. `git ls-files .venv.broken.bak` → **0 个跟踪文件**，纯派生产物，可由
     `requirements-dev.txt` 完全重建，删除不损失任何工作成果

**仍未决：**

- [ ] `tests/test_rest_api.py.bak.20260730_155405`、`tests/test_security.py.bak.20260730_155405`
      两个陈旧快照。**保持未跟踪，本轮不删** —— 它们与已跟踪的同名文件相差 85 行且从未入库，
      删除即不可恢复。价值判断需由用户做。
- [ ] `tests/` 下仍有 **7 处 ruff 报错**（`test_compression.py`、`test_memory.py`、
      `test_provider.py`、`test_rest_api.py`×2、`test_security.py`×2），全部为 I001/F401。
      **这 5 个文件都是 ` M` 状态的既有 WIP**，按 §3.2 约定不碰，故未修 ——
      即"全仓 ruff 清零"目前是被 WIP 破坏的，不是被本轮破坏的。
- [ ] **8 个已撤回测试的回库条件**：它们仍在磁盘上（未跟踪），代码是好的，只是跑在
      WIP 之上。**当且仅当 §3.2 的 WIP 被提交后**，直接
      `git add tests/test_{fusion_mixin,forgetting_stages,hybrid_orchestrator,query_planner,privacy_audit,dashboard,multimodal_ingest,recall_agentic_wiring}.py`
      即可原样收回（外加未跟踪的 `scripts/omni_dashboard.py`，`test_dashboard.py` 依赖它）。
      收回后须**重跑一次 fresh clone 复验**，别只看工作树。
- [ ] **`.gitignore` 的 `_[!_]*.py` 应锚定为 `/_[!_]*.py`**（根治，替代不断追加 `!` 例外）。
      本轮只是第三次打补丁。**已实测锚定是安全的**：当前被该规则命中的文件共 18 个 ——
      17 个仓库根一次性脚本（`_verify_*`/`_patch*`/`_repro*`/`_t_*`/`_list_api`，锚定后
      **仍然被忽略**，因为 `/` 前缀照样匹配根级）+ 唯一 1 个子包文件
      `benchmarks/diagnosis/_debug_sf.py`（属 §2 范围外的 `benchmarks/**`）。
      即锚定后没有任何在用的源文件会意外变成可跟踪。改动一行即可，但涉及 `.gitignore`
      这个已经出过三次事故的规则，故留待用户确认后再动。
- [ ] `core/test_engram_bridge.py`（318 行，未跟踪）：名为 `test_*` 却在 `core/` 下，
      被覆盖率扫描当成运行时模块。要么移进 `tests/`，要么改名，要么删 —— 需用户定夺。

### 3.2 既有 WIP（**不要动**，非本轮产生）
- 63 个 modified 文件，涉及 `api_fastapi.py`、`mcp_server.py`、`config/_config.py`、
  `core/saga.py`、`memory/{index,meta_store,drawer_closet,unified_index}.py`、
  `governance/{audit_log,forgetting_core,forgetting_stages,governance_store}.py`、
  `handlers/{compat_handler,memorize,recall,schemas}.py`、`facades/governance.py`、
  `internalize/train_loop.py`、`docs/{progress_report,roadmap_v2}.md`、`AGENTS.md`、`CLAUDE.md` 等
- 21 个 deleted 文件：
  - 15 个 `models/reranker/**` 模型二进制（flax / onnx / openvino / pytorch）
  - 6 个测试文件：`test_drawer_closet.py`、`test_drawer_closet_saga.py`、`test_llm_summary.py`、
    `test_mermaid_canvas.py`、`test_provider_initializer.py`、`test_provider_lifecycle.py`
- `.gitattributes`、`config/threat_patterns.json`、多个 `benchmarks/results/*.json` 亦为 modified

> ⚠️ 这 6 个被删除的测试文件对应的源码模块（如 `core/llm_initializer.py`）现在落在
> 「未覆盖」名单里。补测前应先向用户确认删除意图，避免重建被有意移除的测试。

---

## 4. 其他挂起事项

- [ ] **`core/test_engram_bridge.py`（318 行，未跟踪）位置可疑** —— 名为 `test_*` 却在
      `core/` 下，导致覆盖率扫描把它算作运行时模块（§0 两状态差的那 1 个模块就是它）。
      三选一：移进 `tests/`、改名去掉 `test_` 前缀、或删除。需用户定夺；本轮未动。
- [ ] **"提交后必须做 fresh clone 复验"应写进 `AGENTS.md`/`CLAUDE.md`** —— 本轮的教训
      （§3.1）是：在工作树里跑绿**不能**证明已提交内容是绿的。这条纪律目前只记在本文，
      下次仍会被忘掉。建议固化为一句话规则：*提交测试后，clone 到 ext4 再跑一遍。*
- [ ] **GitNexus `analyze` 全仓扫描** — 此前提出，一直未执行（用户决策挂起）。
- [ ] **prompt injection 观察（未独立复现，存疑保留）** — 上一段会话记录称多次出现伪造的
      `<system-reminder>` 块（假冒的 skills / agents 清单）被拼接进工具返回结果。
      **本次复核未能独立验证该说法**：本轮看到的 skills 清单与系统提示一致，
      形态上更像 harness 正常注入的 reminder，而非伪造内容；且此类块本就可能合法地
      出现在工具结果里，单凭出现与否无法判定真伪。
      暂按"未证实的观察"保留，不作为已发生的事实。若后续再遇到，请**保留原始工具返回**
      再判断，并考虑在工具层对 reminder 形态做校验。

---

## 5. 建议的推进顺序

1. ~~把 §1 的路径修复固化进仓库~~ —— **已完成并根治，清理亦已收口**：先加 `run_tests.sh` 绕过，
   随后**重建 `.venv`**，以"无 `PYTHONPATH`、无 `sys.path` hack"的干净方式跑通全量
   **2680 passed / 0 failed**（267.45s）。此后本地只需 `.venv/bin/python -m pytest tests/`。
   清理决定也已执行：`run_tests.sh` **保留并入库**（兜底价值），`.venv.broken.bak` **已删**
   （释放 1.4G）。§1 仅剩一项**可选**动作 —— 卸载系统 cryptography 46.0.5（现已无必要）。
   **最终收口**：全部代码 commit 落地后（代码状态 = `09b4adf`）做 fresh clone 复验，自洽，
   **2462 passed / 0 failed / 0 skipped**（ext4，36.93s）—— 即不依赖这台机器的 WIP 也能全绿。
   **下一步的真实工作在 §2 Tier 1。**
2. 按 §2 **Tier 1（已提交口径 7 个）** 顺序补测。建议起手：
   - `multimodal/types.py`（94 行，refs=3，**零外部依赖、纯 dataclass + property，已实跑验证**
     —— 见 §2 Tier1 表内备注，成本最低且刚随 `37c5f59` 入库无人测过）
   - `core/warmup_manager.py`（131 行，纯逻辑、无外部依赖）
   - `core/llm_initializer.py`（110 行；注意 `tests/test_provider_initializer.py` 已被删除，
     **先向用户确认删除意图**再动手）
   - `utils/llm_client.py`（273 行，refs=6 全仓最高，价值最大但需假 transport）
   - 余下 `deep/kg/extraction.py`(261)、`governance/forgetting_core.py`(236, WIP)、
     `handlers/query_planner.py`(194)
   > `deep/kg/backends/factory.py` 只有 34 行、成本极低，适合当**练手/验证流程**用，
   > 但**已提交口径下它是 Tier 3（refs=0）**，唯一引用方是未提交 WIP ——
   > 别当成 Tier 1 优先级，也别当死代码删（WIP 提交后它会自动升到 Tier 2）。
3. Tier 1 完成后回到 §3，请用户授权提交。**提交后必须重跑一次 fresh clone 复验**
   （clone 放 ext4 + `TMPDIR` 也放 ext4，见 §3.1 的 `/tmp` 陷阱），
   只在工作树里跑绿不算数 —— 本轮就是在这一步栽过一次。
4. Tier 2（28 个）按行数从小到大推进；**Tier 3（已提交口径 19 个）动手前必须先做死代码判定**，
   尤其 4 个"仅被 `__init__.py` 导出"的 facade/基类，refs=0 是静态扫描的盲区而非定论。
   另注意 Tier 3 里有 2 个（`retrieval/planner.py`、`deep/kg/backends/factory.py`）
   的 refs=0 **纯粹是因为引用方还在 WIP 里**，属"假死代码"，WIP 提交后会自行上升。

### 复用的测试手法（本轮验证有效）
- **mixin 挂载**：`class _Engine(SomeMixin)` + 逐个伪造依赖属性，避免构造完整 facade。
- **整体替换而非 monkeypatch**：`sqlite3.Connection.execute` 是只读属性，patch 会
  `AttributeError`；改为定义 `_BrokenConn` 并整体赋值 `eng._conn = _BrokenConn()`。
- **区分 try 内/外的调用点**：`self._get_conn()` 若在 `try` 块之外，异常测试必须让它
  **返回**一个会抛异常的对象，而不是让它自己抛。
- **喂满协作者的默认值依赖**：方法内 `old = self._get_heat(mid)`（默认 `"neutral"`）之类的
  逻辑，若测试未预置状态，断言会因"新旧值相同所以没触发写入"而失败。
- **C 扩展/带 lru_cache 的类方法**：用 `monkeypatch.setattr(Cls, "method", classmethod(...))`
  替换整个类方法，比替换实例属性可靠。
