# OmniMem v2 后续优化计划（Roadmap）

> 基线：Phase 0–M4 已交付（2026-07）——8 个真实 bug 修复、LLM hybrid 事实抽取、AES-256-GCM V2 加密、
> 共享检索线程池、embedding 缓存 SQLite 化、UnifiedMemoryIndex 灰度接入、LoRA 三段式闭环、文档全面对齐。
> 本文档为剩余债务 + 执行过程中新发现问题的完整清单，按优先级排列，工作量按 1 人全职估算。

> **进度更新（2026-07-26，详细台账见 [progress_report.md](progress_report.md)）**：
> ✅ 已完成：M5-3/4、M6 全部（5/6/7/8/9）、M7 全部（10/11/12/13/14）、
> M8-17/18/19（解密审计 / reencrypt / MCP 安全对齐）、M9-22/23（依赖单一来源校验 / jieba 钉版）。
> 另修复发布级缺陷：.gitignore `_*.py` 误伤全部 `__init__.py`（23 个包结构文件从未入库，
> clone 后无法 import），已补录。
> ✅ 补充完成（2026-07-26 晚,按本机实际环境重评后执行）：M6-9 拆分补做（fusion/cache/index_admin,主文件 429 行）、
> M8-15/16（LoRA 最小训练循环已在本机 GPU（RX 7900 XTX 24G/ROCm）tiny 模型冒烟跑通,QLoRA 4bit 依赖 bnb 可用性自动降级；适配器推理 hook + active_adapter.json）、
> M9-20（FastAPI 版 REST + /docs,19 项安全语义测试对齐）、M9-21（CI 三平台矩阵 + ruff 全量阻断 + coverage 45% 棘轮门禁,绿灯验证待配置 git remote）。
> ⬜ 待办：M5-1/2（Linux 回归 + LongMemEval/hybrid A/B,需 Linux SSH 授权与 LLM 凭证）；
> mypy 全量 791 处历史类型债（CI 中为信息性检查,逐步偿还）；coverage 52%→75% 爬坡。

---

## M5 — 验证与入库（P0，最先执行，约 1 周）

| # | 任务 | 要点 | 验收标准 |
|:--|:---|:---|:---|
| 1 | Linux 侧全量回归 | `pytest tests/ -m "not slow"`（Windows 网络盘无法跑模型类用例） | 410+ 用例全绿，失败项归因清单 |
| 2 | 基准 A/B 对比 | 用已有 `benchmarks/run_longmemeval.sh`，对比 `extraction_mode: rule` vs `hybrid` | hybrid 分数 ≥ rule；若更低，回查 prompt 与解析逻辑 |
| 3 | 分离提交 | 本轮改动（约 25 个文件）与仓库既有未提交修改（services/handlers/benchmarks 等）分开 review、分开 commit | git 历史可追溯，两批改动互不混淆 |
| 4 | CHANGELOG + 版本号 | 记录 V2 加密格式、新 govern 动作（export_training_data / register_adapter）、灰度开关等新增/破坏性变更 | CHANGELOG.md 更新，版本升至 1.1.0 |

---

## M6 — 存储层深化（P1，约 4 周，风险最高，全程灰度）

| # | 任务 | 要点 | 验收标准 |
|:--|:---|:---|:---|
| 5 | UnifiedMemoryIndex 迁移工具 | `omni-doctor migrate-index`：ThreeLevelIndex + MetaStore → unified_index.db，校验行数/抽样内容一致后原库改名备份 | 迁移前后 recall 结果一致（自动 diff 脚本） |
| 6 | 消除双写 | 启用 unified 后，DrawerClosetStore 的 MetaStore 双写与对应 Saga 步骤下线 | 写入路径 Saga 步骤 4 → 3；写延迟 -20% |
| 7 | BM25 → SQLite FTS5 | 复用 unified_index 的 FTS5 表替换 rank-bm25（全量重建 O(n) → 增量）；中文用 jieba 预分词列或 trigram tokenizer | 增量写入 O(1)；中文召回不低于现 BM25（基准集验证）；删除 bm25.py 缓冲/重建逻辑约 300 行 |
| 8 | 治理库合库 + 单写者 | forgetting.db / reflect.db / kv_cache.db / consolidation.db → 单 db + 后台单写线程；删除 `_FORGETTING_DB_LOCK`、类级共享连接、引用计数等约 500 行防御代码 | 并发压测（4 线程 × 1000 写）零 `database is locked` |
| 9 | 大文件拆分 | forgetting.py（1037 行）→ stage/heat/scheduler；hybrid_orchestrator.py（943 行）→ fusion/cache 模块 | 单文件 < 500 行，测试全绿 |

---

## M7 — 抽取与检索质量（P1，约 3 周，可与 M6 并行）

| # | 任务 | 要点 | 验收标准 |
|:--|:---|:---|:---|
| 10 | LLM 抽取升级 ADD/UPDATE/DELETE 决策 | mem0 范式：抽取输出 `{facts, action, target_id}`，与现有 superseded 机制对接，替代纯相似度去重 | 冲突更新类场景（"改用 PostgreSQL"）正确标记旧记忆 |
| 11 | KG LLM 抽取统一收口（新发现问题） | `deep/kg/extraction.py::_extract_triples_llm` 绕过 LLMBackend 直连 OpenAI、自读 config.yaml、硬编码模型名 `deepseek-v4-flash` —— 改走 AsyncLLMClient/LLMBackend，模型可配 | 无任何模块自建 LLM 连接；grep 无硬编码模型名 |
| 12 | 抽取质量评测集 | 人工标注 100 条中文对话（该记/不该记/事实文本），入 benchmarks/，CI 输出准确率 | rule 基线成绩入库；hybrid ≥ 85% |
| 13 | LLM 默认模型收敛 | `AsyncLLMClient` 默认 `glm-5.1` 硬编码 → 统一走 `llm_model` 配置 | 模型配置单一来源 |
| 14 | Reranker 设备可配 | 当前强制 CPU（`CUDA_VISIBLE_DEVICES=""`），加 `reranker_device` 配置项 | GPU 环境重排延迟 -80% |

---

## M8 — L4 与安全补全（P2，约 2.5 周）

| # | 任务 | 要点 | 验收标准 |
|:--|:---|:---|:---|
| 15 | `_real_train` 最小训练循环 | HF Trainer + peft SFT，限定 Qwen2.5 系；显存不足自动 QLoRA(4bit)；产物落 `adapter.path` | 24G 显存跑通 7B QLoRA，回注后 `status=ready` |
| 16 | 适配器推理侧集成 | 回注的 adapter 当前无消费方 —— 增加推理加载 hook（active_shade 变更时通知宿主/输出加载指令） | shade_switch 后适配器路径可被宿主获取 |
| 17 | secret 解密审计 | omni_detail 解锁 secret 记忆时强制写 AuditLogger | 审计日志含解密事件（actor / memory_id / 时间） |
| 18 | V1 → V2 重加密工具 | `omni_govern action=reencrypt`：批量把 Fernet 密文升级为 AESGCM | 迁移后全部密文 `OMNI_ENC_V2` 前缀且可解密 |
| 19 | MCP 入口安全对齐 | REST 已有速率限制/审计，MCP 入口补齐同等策略 | MCP 工具调用进审计日志 |

---

## M9 — 服务化与工程化（P2-P3，约 2 周，可裁剪）

| # | 任务 | 要点 | 验收标准 |
|:--|:---|:---|:---|
| 20 | FastAPI 替换 http.server | `omnimem[api]` extra；Auth/RateLimit/Admin 中间件平移；OpenAPI 自动文档 | /docs 可用；现有 REST 测试平移通过 |
| 21 | CI 增强 | GitHub Actions 加 Windows runner（覆盖 sync.py fcntl 降级路径）+ coverage 门禁 ≥ 75% + ruff/mypy 全量 | CI 三平台绿 |
| 22 | 依赖单一来源 | 脚本从 pyproject.toml 生成 plugin.yaml / requirements.txt，防再度漂移 | 三文件由 pre-commit 钩子校验一致 |
| 23 | jieba pkg_resources 告警 | jieba 依赖将随 setuptools ≥ 81 失效（测试输出已见警告）→ 锁 setuptools 或换 jieba-fast | 测试无 DeprecationWarning |

---

## 执行顺序与里程碑

```
第 1 周      M5 验证入库（阻塞后续，必须先做）
第 2-5 周    M6 存储深化 ────┐ 可并行
第 2-4 周    M7 抽取质量 ────┘（M7-12 评测集先行，为 M6-7 FTS5 召回验收提供基准）
第 6-8 周    M8 L4 + 安全（15/16 需 GPU 环境窗口）
第 9-10 周   M9 服务化（可裁剪）
```

**关键依赖**：
- 12（评测集）→ 7（FTS5 召回验收）与 10（抽取决策验收）
- 5（迁移工具）→ 6/7/8（合库系列）

**风险预案**：
- M6 全程 feature flag 灰度（`use_unified_index` 模式已验证可行），旧路径保留一个版本周期
- M7-10 的 UPDATE/DELETE 决策有误删风险 → 首版仅 ADD/UPDATE（标记不删除），DELETE 默认 dry-run
- M8-15 若 GPU 资源不可得，降级为只交付 16（推理集成），训练继续走三段式外部路径

**预期终态**：单一存储引擎（unified db + FTS5）、全 LLM 化抽取管线（含 KG）、L4 训练-推理闭环、
三平台 CI —— 届时版本推进到 2.0 并可考虑对外发布。

---

## 附：Phase 0–M4 已交付基线（本计划的起点）

| 阶段 | 交付内容 |
|:---|:---|
| Phase 0 止血 | QdrantStore 真向量修复、REST fail-closed + 默认 127.0.0.1、依赖对齐、文档去虚、倒排索引 LRU 泄漏修复 |
| Phase 1 抽取质量 | LLM hybrid 事实抽取（extraction_mode）、威胁模式外置热更新（threat_patterns.json）、KG/工具注入 2 个正则 bug 修复 |
| Phase 2 存储并发 | UnifiedMemoryIndex 可用化 + use_unified_index 灰度、全进程共享检索线程池、embedding 缓存 SQLite 化 |
| Phase 3 功能补全 | AES-256-GCM V2 加密（V1/legacy 三代解密兼容 + GCM 篡改检测）、检索通道现状文档对齐 |
| M4 L4 闭环 | LoRA 三段式（export_training_data 导出 alpaca JSONL / register_adapter 回注）+ govern 动作接线 |
