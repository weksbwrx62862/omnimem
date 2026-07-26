# OmniMem 优化进度台账（Progress Report）

> 更新时间：2026-07-26 ｜ 当前版本：1.1.0 ｜ 分支：master（HEAD `7d7c9cf`）
> 配套文档：[roadmap_v2.md](roadmap_v2.md)（任务定义与验收标准）

---

## 一、总体进度

| 里程碑 | 状态 | 说明 |
|:---|:---:|:---|
| Phase 0 止血与对齐 | ✅ | Qdrant 真向量、REST fail-closed、依赖对齐、文档去虚、LRU 泄漏 |
| Phase 1 抽取质量 | ✅ | LLM hybrid 抽取、威胁模式外置、2 个正则 bug |
| Phase 2 存储并发 | ✅ | UnifiedMemoryIndex 灰度、共享线程池、embedding 缓存 SQLite |
| Phase 3 功能补全 | ✅ | AES-256-GCM V2、检索通道对齐 |
| M4 L4 闭环 | ✅ | LoRA 三段式（导出/回注）+ govern 接线 |
| M5 验证与入库 | ◐ 2/4 | 3/4 ✅；**1/2 待 Linux 环境** |
| M6 存储层深化 | ✅ 5/5 | 迁移工具、消除双写、FTS5、治理合库、大文件拆分 |
| M7 抽取与检索质量 | ✅ 5/5 | ADD/UPDATE/DELETE、KG 收口、评测集、模型收敛、Reranker 设备 |
| M8 L4 与安全补全 | ◐ 3/5 | 17/18/19 ✅；**15/16 待 GPU 环境** |
| M9 服务化与工程化 | ◐ 2/4 | 22/23 ✅；**20/21 发布前启动** |

**结论：本地环境（Windows/SMB、无 GPU、无 LLM 凭证）可交付项已 100% 完成。**

---

## 二、提交台账（自 1.1.0 基线起）

| 提交 | 主题 | 对应任务 |
|:---|:---|:---|
| `cfa4825` | Phase 0-M4 基础设施新模块 | Phase 0–M4 |
| `64d491d` | Phase 0-M4 增量修复与文档对齐 | Phase 0–M4 |
| `4470b79` | LLM 驱动 ADD/UPDATE/DELETE 记忆决策 | M7-10 |
| `0878fa4` | FTS5Retriever 灰度替代 BM25 | M6-7 |
| `385968b` | 索引迁移工具 + 消除 MetaStore 双写 | M6-5/6 |
| `7e44121` | GovernanceStore 治理合库 + forgetting 拆分 | M6-8/9 |
| `dba15be` | 抽取质量评测集 + rule 基线入库 | M7-12 |
| `94c5afe` | roadmap 进度标注 | 文档 |
| `73d94d2` | DEFAULT_LLM_MODEL 单一来源 + Reranker 设备 | M7-13/14 |
| `35acbf7` | secret 解密审计 + reencrypt + MCP 安全对齐 | M8-17/18/19 |
| `f3c8acb` | 依赖单一来源校验 + jieba 钉版 + 行尾统一 | M9-22/23 |
| `ef1c9cd` | .gitignore 误伤 `__init__.py` 修复 | 缺陷修复 |
| `7d7c9cf` | 补录 22 个子包 `__init__.py` | 缺陷修复 |

---

## 三、关键量化指标

| 指标 | 数值 |
|:---|:---|
| 修复真实 bug | **13 个**（Qdrant 全零向量、admin_token 越权、倒排索引泄漏、LLM 缓存清理、KG 抽取正则、工具注入正则、unified_index 缩进+FTS5 建表、embedding 迁移误判、fts5 列名、闭包死连接、评测脚本 GBK、`.gitignore` 误伤包结构 ×23 文件） |
| 抽取评测 rule 基线 | P=85.71% / R=51.50% / **F1=64.34%**（refusal 维度 F1=0） |
| 新增 govern 动作 | 4 个：`export_training_data` / `register_adapter` / `reencrypt` / （评测 CLI） |
| 新增配置项 | `extraction_mode` / `use_unified_index` / `use_fts5` / `llm_model` / `reranker_device` / `mcp_rate_limit_per_minute` |
| 本地回归测试 | 累计 300+ 项通过（分批执行，覆盖全部改动模块） |
| 灰度开关（默认关闭） | `use_unified_index`、`use_fts5`、GovernanceStore 注入 |

---

## 四、剩余待办（均需外部环境）

### M5-1/2 — Linux 侧验证（收官 1.1.x 的唯一阻塞）
```bash
# 1. 全量回归
pytest tests/ -m "not slow"
# 2. 基准 A/B（需 LLM 凭证）
bash benchmarks/run_longmemeval.sh          # rule vs hybrid 各跑一轮
python benchmarks/run_extraction_eval.py    # hybrid 模式对比 rule 基线 F1=64.34%
```
验收：410+ 用例全绿；hybrid 分数 ≥ rule。

### M8-15/16 — GPU 窗口
- `_real_train` 最小训练循环（HF Trainer + QLoRA，Qwen2.5 系，24G 显存验收）
- 适配器推理侧加载 hook（shade_switch 通知宿主）

### M9-20/21 — 发布前工程
- FastAPI 替换 http.server（`omnimem[api]` extra）
- CI：Windows runner + coverage ≥75% 门禁 + ruff/mypy 全量

### 决策点（待 M5-2 数据）
1. `use_unified_index` / `use_fts5` 是否转默认开启（存量走 `omni-doctor migrate-index`）
2. `extraction_mode: hybrid` 保持默认还是回退 rule

---

## 五、仓库卫生遗留（用户决策项）

| 项 | 说明 | 建议 |
|:---|:---|:---|
| 5 个 " M" 幻影文件 | 根因为 SMB 挂载下 filemode 位变化（`100755→100644`），非行尾 | Linux 侧执行 `git config core.fileMode false`（一次性） |
| `benchmarks/results/longmemeval_v7~v9/` 等未跟踪产物 | 2026-07-14 的历史基准结果 | 提交存档或加 .gitignore 二选一 |

---

## 六、验证方法备忘

- 每批改动均经「功能断言脚本（临时、验证后删除）+ pytest 回归」双重验证后提交
- 模型加载类集成测试在 Windows/SMB 环境必超时，属环境限制而非代码问题，统一留待 M5-1
- 抽取质量以 `benchmarks/extraction_quality_eval.json`（100 条中文标注）为回归基准，
  rule 基线成绩存于 `benchmarks/results/extraction_rule_baseline.json`
