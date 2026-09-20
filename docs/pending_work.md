# OmniMem 未完成工作清单

> 生成时间：2026-09-20（16:16 补入全量回归数字；16:40 复核 AST 扫描并修正若干条目；
> 17:30 重建 `.venv`，§1 由"绕过"升级为"根治"）
> 仓库：`/home/xxh/.hermes/plugins/omnimem`（git 分支 `master`，**全部改动均未提交**）
> 数据来源：AST 静态扫描运行时模块 × `tests/` 全部 import 的真实引用图，非凭记忆
> 回归基线：全量 suite **2680 passed / 0 failed**，经三次独立运行确认
> （`272.30s` 系统解释器 + `sys.path` 修复 → `/tmp/full_suite_fixed.log`；
> `275.03s` 经 `run_tests.sh` → `/tmp/run_tests_sh_full.log`；
> `267.45s` **重建后的 `.venv`，无任何路径 hack** → `/tmp/venv_rebuild_suite.log`）

---

## 0. 现状快照

| 指标 | 数值 |
|---|---|
| 仓库内 `.py` 文件总数（排除隐藏目录与 `*.bak` 备份目录） | 489 |
| 其中运行时模块（**本文口径**） | 197 |
| 已被至少一个测试文件 import | 145（73%） |
| 未被任何测试 import | 52（Tier1 6 + Tier2 28 + Tier3 18） |
| `tests/` 下 `.py` 文件数 | 133（131 个 `test_*.py` + `__init__.py` + `conftest.py`） |
| `tests/` 下未跟踪（新增、未提交）文件 | 68（62 个为 2026-09-20 当天产出） |
| 全量 suite 结果 | 2680 passed / 0 failed |
| 本地跑测试的推荐命令 | `.venv/bin/python -m pytest tests/`（**无需任何路径 hack**，见 §1） |
| `.venv` 状态 | 2026-09-20 重建：uv + CPython 3.13.13，136 包 / 1.5G，`torch 2.14.0+cpu`，0 个 CUDA 包 |

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
> 复核后的「145 已覆盖 / 52 未覆盖 / Tier1+Tier2 = 34 值得补」为准。
> 两者结论方向一致（旧 ≈45 vs 复核 34），但**逐项 tier 归属有出入**，详见 §2。
>
> 复核中还发现 `tests/` 内**混用两种 import 风格**：既有 `from omnimem.retrieval.fusion import …`
> （走 editable install 的 `.pth` → `/home/xxh/.hermes/plugins`；旧环境是
> `__editable__.omnimem-1.0.0.pth`，重建后的 `.venv` 是 `__editable__.omnimem-1.2.0.pth`，
> 指向同一目录。版本号差异来自 `pyproject.toml` 已升到 1.2.0，而该文件处于 modified/WIP 状态），
> 也有 `from governance import triple_extractor as …`（走 repo 根）。只匹配其中一种风格
> 会大幅高估未覆盖数（本次复核中途就因此得出过 8% 和 199 个未覆盖的错误结果），
> 扫描脚本必须先归一化掉 `omnimem.` 前缀。

本轮（最近一段会话）新增的 5 个测试文件，均已 pytest + ruff 双绿：

| 文件 | 用例数 | 被测模块 |
|---|---|---|
| `tests/test_fusion_mixin.py` | 66 | `retrieval/fusion.py` |
| `tests/test_forgetting_stages.py` | 49 | `governance/forgetting_stages.py` |
| `tests/test_screening_engine.py` | 53 | `governance/screening_engine.py` |
| `tests/test_hybrid_orchestrator.py` | 43 | `retrieval/hybrid_orchestrator.py` |
| `tests/test_context_manager.py` | 138 | `context/manager.py` |

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
- [x] (a) 在仓库根加 `run_tests.sh` —— **已完成并实测通过，但已降级为历史兜底**。它针对的是
      "用系统解释器 `/home/xxh/.local/bin/python3.13` 跑测试"这一旧路径，通过在解释器内
      `sys.path.append` 绕过遮蔽（注意不能用 `PYTHONPATH` 表达本修复 —— PYTHONPATH 必然排在
      site-packages 之前，正是病因）。重建 venv 后该前提消失，脚本可以删除；保留它的唯一价值
      是不依赖 venv 也能复现全量测试。**是否删除待用户决定。**
- [ ] (c) 卸载 `/usr/lib/python3/dist-packages/cryptography` 46.0.5 —— **动系统包，需用户授权**，
      未执行。重建 venv 后已无必要：venv 内 `include-system-site-packages = false`，
      系统包不再参与解析。仅在"有人继续用系统解释器裸跑 pytest"时才需要。
- [ ] (e) 清理 `.venv.broken.bak`（1.4G）—— **删除操作，需用户授权**，未执行。
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
下文一律为严格口径结果 **Tier1=6 / Tier2=28 / Tier3=18**（合计仍 52）。
跨会话比对时若看到 17/22/13，说明对方用的是宽口径，不是数据变了。

### Tier 1 — 主链引用 ≥ 2，建议优先补（**6 个**）

| 行数 | refs | 模块 | 备注 |
|---|---|---|---|
| 273 | 6 | `utils/llm_client.py` | 全仓引用最多的未测模块；LLM 调用封装，可用假 transport 离线测 |
| 261 | 3 | `deep/kg/extraction.py` | 见上 (D)：别被同名的 `test_triple_extractor.py` 误导 |
| 242 | 2 | `governance/forgetting_core.py` | **modified（既有 WIP）**，补测前先确认改动意图 |
| 194 | 2 | `handlers/query_planner.py` | **同名陷阱（已核实）**：`tests/test_query_planner.py` 存在，但它 import 的是 `retrieval.planner.QueryPlanner`，与本文件无关。refs=2 来自 `core/provider_lifecycle.py:282` 与 `services/recall_service.py:196` |
| 131 | 2 | `core/warmup_manager.py` | 冷启动预热，纯逻辑可测 |
| 110 | 2 | `core/llm_initializer.py` | 注意：`tests/test_provider_initializer.py` 已被删除 |

> Tier 1 从宽口径的 17 个收缩到 6 个 —— 这正是严格口径的价值：
> 其余 11 个此前看着"有 2 处依赖"，实际有一处是 `__init__.py`，真实耦合只有 1。

### Tier 2 — 主链引用 = 1（**28 个**）

| 行数 | 模块 | 备注 |
|---|---|---|
| 554 | `memory/unified_index.py` | 最大的未测运行时模块；**modified（既有 WIP）** |
| 482 | `deep/kg/query.py` | 见上 (D)：`test_kg_query.py` 只 import 了 `entity` + `builder` |
| 457 | `memory/migration_tool.py` | 见上 (D)：`test_migration.py` 测的是 `utils/migration.py` |
| 404 | `core/provider_middleware.py` | |
| 307 | `core/llm_memory_manager.py` | |
| 304 | `governance/auditor.py` | |
| 295 | `governance/forgetting_ops.py` | |
| 257 | `core/reflection_trigger.py` | |
| 228 | `deep/kg/backends/neo4j_backend.py` | 需假 driver，离线可测 |
| 217 | `core/llm_client_manager.py` | |
| 201 | `facades/governance.py` | **modified（既有 WIP）** |
| 200 | `handlers/priming.py` | |
| 197 | `utils/async_llm.py` | |
| 191 | `core/async_provider.py` | |
| 169 | `internalize/train_loop.py` | **modified（既有 WIP）** |
| 167 | `core/system_prompt_builder.py` | |
| 157 | `core/plur_config.py` | |
| 145 | `handlers/compat_handler.py` | **modified（既有 WIP）** |
| 138 | `facades/retrieval.py` | |
| 114 | `compression/priority.py` | 宽口径 refs=2 的另一处是 `compression/__init__.py`；真实引用方只有 `compression/pipeline.py` |
| 108 | `facades/storage.py` | |
| 108 | `memory/markdown_store.py` | |
| 101 | `compat/provider_proxy.py` | |
| 100 | `deep/kg/relationships.py` | |
| 93 | `deep/kg/temporal.py` | |
| 63 | `core/backup_manager.py` | |
| 47 | `deep/reflect/writer.py` | |
| 34 | `deep/kg/backends/factory.py` | **最小、成本最低，适合第一个动手验证流程** |

### Tier 3 — 主链引用 = 0（**18 个**，补测前必须先做死代码判定）

| 行数 | 模块 | 说明 |
|---|---|---|
| 417 | `deep/reflect/pipeline.py` | **仅被 `deep/reflect/__init__.py` 导出**，无真实调用方；宽口径下曾被列进 Tier 1 |
| 318 | `core/test_engram_bridge.py` | 放在 `core/` 下的旧测试脚本，**不是运行时模块**，建议移出或删除 |
| 286 | `core/plur_client.py` | PLUR 客户端 |
| 214 | `core/plur_server.py` | PLUR 服务端 |
| 192 | `storage/milvus_store.py` | 需外部服务；考虑只测参数校验/异常分支 |
| 166 | `governance/distributed_sync.py` | **已确认废弃**，主链路未使用 —— 不要补测 |
| 147 | `utils/llm_backend.py` | |
| 131 | `embedding/onnx_provider.py` | |
| 118 | `protocols.py` | |
| 110 | `multimodal/parse.py` | |
| 110 | `multimodal/embedders.py` | |
| 98 | `multimodal/store.py` | |
| 95 | `facades/sync_facade.py` | 仅被 `facades/__init__.py` 导出 |
| 86 | `core/cross_db_coordinator.py` | |
| 53 | `facades/deep_memory.py` | 仅被 `facades/__init__.py` 导出 |
| 51 | `services/base.py` | 仅被 `services/__init__.py` 导出；基类，可能被动态继承 |
| 20 | `handlers/_compat.py` | 私有兼容 shim |
| 11 | `deep/knowledge_graph.py` | 极小，疑似仅 re-export |

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

## 3. 提交与仓库状态（需用户决策，未执行）

当前 `git status` 混杂了 **既有 WIP** 与 **本轮新增**，两者必须分开处理。

### 3.1 本轮产出（纯新增，安全）

**(a) `tests/` 下 68 个未跟踪文件。** 按 mtime 拆分：62 个为 2026-09-20 当天产出，4 个为 2026-08-07
（`test_query_planner.py`、`test_hms_p0/p1/p2*.py`），另 2 个为 `.bak.20260730_155405` 备份文件。

**(b) 本次复核新增的 3 个文件（`tests/` 之外）：**

| 文件 | 用途 | 重建 venv 后的状态 |
|---|---|---|
| `run_tests.sh` | §1 路径修复的落地入口，已实测全量 2680 passed | **降级为历史兜底**，可删（见 §1 选项 (a)） |
| `scripts/scan_test_coverage.py` | §0/§2 全部覆盖率与 tier 数字的可复现来源 | 仍需要，是本文所有数字的唯一出处 |
| `docs/pending_work.md` | 本文 | 仍需要 |

> `scripts/` 目录本身在覆盖率口径里属"范围外"，但 `scan_test_coverage.py` 是**分析工具**
> 而非产品脚本，与 `scripts/mock_conflict_test.py` 等性质不同，建议一并提交以便复现。

> ✅ **已核实：`.venv.broken.bak`（1.4G）不会污染 `git add -A`。** `git check-ignore` 对它
> 返回 rc=1（`.gitignore` 第 48 行只匹配 `.venv/`），乍看像是个隐患；但 `git status --porcelain
> --ignored=matching` 显示其内容全为 `!!`。原因是 venv 自带一个内容为 `*` 的内层
> `.gitignore`，随目录一起被搬了过去，把自身全部内容忽略掉，因此该目录对 git 完全不可见。
> 新建的 `.venv` 同理。**结论：无需改 `.gitignore`，但删除该备份仍需用户授权。**

- [ ] **待用户授权后**再 commit。建议逐个点名 `git add`，**不用 `git add -A`**：
      `git add tests/test_*.py scripts/scan_test_coverage.py docs/pending_work.md`
      （`run_tests.sh` 是否入列取决于 §1 选项 (a) 的删除决定）
- [ ] 决定 2 个 `.bak` 文件是删除还是保留。
- [ ] 决定 `.venv.broken.bak`（1.4G）删除还是保留 —— 新 venv 已验证可用，回滚价值已很低。
- [ ] `run_tests.sh` 硬编码了本机绝对路径，若要进主干需先确认是否改为可配置
      （或只作为本地未跟踪工具、加进 `.gitignore`）。

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

1. ~~把 §1 的路径修复固化进仓库~~ —— **已完成并根治**：先加 `run_tests.sh` 绕过，随后按用户选择
   **重建 `.venv`**，并以"无 `PYTHONPATH`、无 `sys.path` hack"的干净方式第三次跑通全量
   **2680 passed / 0 failed**（267.45s）。§1 从此不再是阻塞项，本地只需
   `.venv/bin/python -m pytest tests/`。剩余的全是**清理决策，需用户授权**：
   删 `run_tests.sh`？删 `.venv.broken.bak`（1.4G）？卸载系统 cryptography 46.0.5（现已无必要）？
2. 按 §2 **Tier 1（6 个）** 顺序补测。建议起手：
   - `core/warmup_manager.py`（131 行，纯逻辑、无外部依赖，成本最低）
   - `core/llm_initializer.py`（110 行；注意 `tests/test_provider_initializer.py` 已被删除，
     **先向用户确认删除意图**再动手）
   - `utils/llm_client.py`（273 行，refs=6 全仓最高，价值最大但需假 transport）
   - 余下 `deep/kg/extraction.py`(261)、`governance/forgetting_core.py`(242, WIP)、
     `handlers/query_planner.py`(194)
   > `deep/kg/backends/factory.py` 只有 34 行、成本极低，适合当**练手/验证流程**用，
   > 但严格口径下它是 Tier 2（唯一引用方非 `__init__.py` 的只有 1 处），别当成 Tier 1 优先级。
3. Tier 1 完成后回到 §3，请用户授权提交。
4. Tier 2（28 个）按行数从小到大推进；**Tier 3（18 个）动手前必须先做死代码判定**，
   尤其 4 个"仅被 `__init__.py` 导出"的 facade/基类，refs=0 是静态扫描的盲区而非定论。

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
