# OmniMem 配置参考（按功能域分组）

本文档基于 `config/_config.py` 中的 `_CONFIG_SCHEMA` 自动整理，覆盖全部 83 个配置项。按功能域分组（非字母顺序），便于按场景查阅。

---

## 配置加载优先级

配置按以下优先级加载（高到低）：

1. **代码显式设置**：`OmniMemConfig.set(key, value)` 运行时直接写入内存
2. **环境变量**：`OMNIMEM_<KEY>`（大小写不敏感，`-` 转 `_`；点分键 `embedding.provider` 对应 `OMNIMEM_EMBEDDING_PROVIDER`）
3. **配置文件**：`<storage_dir>/omnimem/config.yaml`（或 `~/.config/omnimem/config.yaml`），支持嵌套 YAML（如 `embedding.provider`）
4. **默认值**：`_CONFIG_SCHEMA` 中 `default` 字段

加载时通过 `_validate()` 进行类型、范围（`min`/`max`）和枚举（`choices`）校验；非法值会被跳过并记录 warning，使用默认值兜底。

> **特例**：`api_key` 禁止空字符串。未配置或显式置空时，初始化阶段会强制生成 32 字节随机 hex（`secrets.token_hex(32)`）并回写配置文件，避免生产服务无认证启动。

### 热重载机制

- 每 10 轮对话由 `OmniMemConfig.reload()` 检测 `config.yaml` 的 `mtime`
- 文件变更时自动重新加载并校验，无需重启
- 可通过 `config.reload(force=True)` 强制重载

### 配置文件示例（YAML 嵌套形式）

```yaml
save_interval: 15
retrieval_mode: rag

embedding:
  provider: sentence_transformers
  model_name: all-MiniLM-L6-v2

vector_store:
  provider: chroma
  collection_name: omnimem
  persist_dir: /tmp/omnimem/storage/chroma
```

---

## 1. 检索配置（Retrieval）

控制记忆检索的行为、策略、超时与降级。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `retrieval_mode` | string | `"rag"` | `rag` / `hybrid` / `vector` / `bm25` | 默认检索模式。`rag` 为快速向量+BM25 混合检索（毫秒级）；`hybrid` 为深度检索 |
| `recall_strategy` | string | `"hybrid"` | `hybrid` / `keyword` / `embedding` | 召回策略（与 `retrieval_mode` 互补，控制召回通道组合） |
| `recall_timeout_ms` | int | `5000` | [100, 30000] | 检索整体超时（毫秒），超时后降级到 fallback 策略 |
| `max_prefetch_tokens` | int | `300` | [10, 100000] | prefetch（预取注入）阶段的 token 预算，控制每轮自动注入的记忆量 |
| `budget_tokens` | int | `4000` | [100, 100000] | 工作记忆的 token 预算上限，影响 `omni_compact` 与上下文注入总量 |
| `enable_reranker` | bool | `false` | — | 是否启用 Cross-Encoder 重排序（需 `pip install omnimem[embeddings]`） |
| `enable_catalog` | bool | `true` | — | 是否启用 OpenViking 目录递归检索 |
| `catalog_weight` | float | `2.0` | [0.1, 10.0] | 目录项检索的权重加权系数 |
| `max_overview_chars` | int | `200` | [50, 1000] | 三层渐进式披露的概览字符数上限 |
| `enable_trace_by_default` | bool | `false` | — | 默认是否启用可视化检索轨迹 |
| `prefetch_record_access` | bool | `true` | — | prefetch 是否记录访问到遗忘曲线（驱动热度分类） |
| `rrf_k` | int | `60` | [1, 1000] | RRF（倒数排名融合）参数 k |
| `rrf_min_score` | float | `0.035` | [0.0, 1.0] | RRF 最小分数阈值，低于此值的结果被丢弃 |
| `circuit_breaker_threshold` | int | `3` | [1, 100] | 熔断器触发阈值（连续失败次数） |
| `circuit_breaker_cooldown_seconds` | int / float | `60` | [1, 3600] | 熔断器冷却时间（秒） |
| `query_cache_ttl` | int / float | `60` | [0, 3600] | 查询缓存 TTL（秒）；`0` 表示禁用缓存 |
| `mermaid_tool_log_patterns` | list | `[]` | — | Mermaid 符号化压缩的工具日志模式列表 |
| `max_refs_age_days` | int | `30` | [1, 365] | 引用条目最大有效期（天），超期清理 |

---

## 2. 工作记忆配置（L1 Working Memory）

控制 L1 工作记忆的存档节奏、Pipeline 调度、人格触发与 LLM 蒸馏。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `save_interval` | int / float | `15` | [1, 3600] | 每 N 轮对话自动执行一次存档检查点。值越小越频繁但开销越大 |
| `auto_memorize` | bool | `true` | — | 是否启用感知驱动的自动记忆写入。关闭后仅通过 `omni_memorize` 工具手动存储 |
| `max_sync_turn_entries` | int | `1000` | [10, 100000] | 单轮同步写入的最大条目数，防止突发流量打爆下游 |
| `write_buffer_threshold` | int | `20` | [1, 200] | 写缓冲刷盘阈值（条目数达到后批量落盘） |
| `pipeline_every_n_conversations` | int | `5` | [1, 100] | Pipeline 调度器触发 L2/L3 处理的对话间隔 |
| `pipeline_enable_warmup` | bool | `true` | — | 是否启用 Pipeline 预热 |
| `pipeline_l2_delay_after_l1_seconds` | int | `90` | [10, 3600] | L1 完成后触发 L2 的延迟（秒），避免与 L1 IO 抢锁 |
| `persona_trigger_every_n` | int | `15` | [5, 1000] | Persona（人格切片）触发的对话间隔 |
| `persona_min_interval_seconds` | int | `300` | [60, 86400] | Persona 最小触发间隔（秒），防止短时间频繁切换 |
| `distill_enabled` | bool | `true` | — | 是否启用 LLM 蒸馏引擎（定期将 auto-captured raw facts 喂给 LLM 提炼） |
| `distill_model` | string | `""` | — | 蒸馏使用的模型名；空字符串表示使用主模型（`llm_backend` 指定） |
| `distill_interval` | int | `15` | [5, 1000] | 蒸馏执行间隔（轮） |

---

## 3. 结构化记忆配置（L2 Storage）

控制向量库后端、连接参数、索引重建并行度。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `vector_backend` | string | `"chromadb"` | `chromadb` / `qdrant` / `faiss` | 向量存储后端（旧字段，与 `vector_store_backend` 保持一致） |
| `vector_store_backend` | string | `"chromadb"` | `chromadb` / `qdrant` / `faiss` | 向量存储后端（`vector_backend` 别名） |
| `qdrant_url` | string | `"localhost:6333"` | — | Qdrant 服务器地址（仅当 `vector_backend=qdrant` 生效） |
| `vector_store.provider` | string | `"chroma"` | `chroma` / `milvus` | 向量库提供方（点分键，对应嵌套 YAML `vector_store.provider`） |
| `vector_store.collection_name` | string | `"omnimem"` | — | 向量库集合名 |
| `vector_store.persist_dir` | string | `"/tmp/omnimem/storage/chroma"` | — | Chroma 持久化目录 |
| `vector_store.uri` | string | `"http://localhost:19530"` | — | Milvus 服务 URI |
| `vector_store.token` | string | `""` | — | Milvus 访问令牌 |
| `vector_store.embedding_dimension` | int | `384` | [1, 10000] | 嵌入向量维度（需与 `embedding.model_name` 输出维度一致） |
| `vector_store.metric_type` | string | `"COSINE"` | — | 距离度量类型（Milvus 用，如 `COSINE` / `L2` / `IP`） |
| `vector_store.consistency_level` | string | `"Bounded"` | — | Milvus 一致性级别（`Strong` / `Bounded` / `Eventually`） |
| `rebuild_batch_size` | int | `32` | [1, 1024] | 索引重建的批大小 |
| `rebuild_max_workers` | int | `4` | [1, 64] | 索引重建的并发 worker 数 |

### 向量库后端对比

| 后端 | 部署方式 | 适用场景 |
|:---|:---|:---|
| `chromadb` / `chroma` | 本地嵌入式 | 单实例开发/轻量部署，零外部依赖 |
| `qdrant` | 独立服务 | 大规模生产环境，支持分布式 |
| `faiss` | 本地库 | 高性能批量检索，无持久化 |
| `milvus` | 独立服务 | 超大规模向量检索，支持分布式与多副本 |

---

## 4. 深层记忆配置（L3 Deep Memory）

控制 Consolidation（事实→观察→心智模型升华）触发条件。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `fact_threshold` | int | `10` | [1, 1000] | Consolidation 触发阈值。待升华的记忆数达到此值时自动执行 Consolidation |

### Consolidation 流程

```
事实(fact) → 观察(observation) → 心智模型(mental_model)
```

- **事实**：原始记忆条目
- **观察**：多条事实的综合归纳
- **心智模型**：深层模式识别和抽象

---

## 5. 内化层配置（L4 Internalization）

控制 KV Cache（高频记忆预填充）与 LoRA（深层人格微调）。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `kv_cache_threshold` | int | `10` | [1, 1000] | KV Cache 自动预填充的访问次数阈值。记忆被访问达到此次数后自动预填充到 KV Cache |
| `kv_cache_max` | int | `100` | [10, 10000] | KV Cache 最大条目数。超过时按 LRU 策略淘汰 |
| `lora_base_model` | string | `"Qwen2.5-7B"` | — | LoRA 微调的基座模型名称 |
| `lora_rank` | int | `16` | [1, 256] | LoRA 秩（rank），影响微调参数量。值越大表达力越强但显存占用越多 |
| `lora_alpha` | int | `32` | [1, 512] | LoRA alpha 参数，控制微调强度。通常设为 rank 的 2 倍 |

### LoRA 依赖

LoRA 功能需额外安装：

```bash
pip install omnimem[lora]
```

依赖包括：`peft>=0.8.0`、`transformers>=4.35.0`、`torch>=2.0.0`

### Shade（人格切片）

LoRA 支持多 shade 切换，每个 shade 代表一种人格/风格模式：

- 通过 `omni_govern(shade_switch)` 切换
- 通过 `omni_govern(shade_list)` 列出所有可用 shade
- 未知 shade 名称会自动创建

---

## 6. 治理配置（Governance）

### 6.1 冲突与遗忘

控制冲突解决策略与遗忘曲线三阶段。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `conflict_strategy` | string | `"latest"` | `latest` / `merge` / `reject` | 冲突解决策略。`latest`：保留最新；`merge`：合并；`reject`：拒绝并报错 |
| `conflict_scan_max_group_size` | int | `50` | [2, 1000] | 冲突扫描时的最大分组大小，控制单次扫描成本 |
| `forgetting_active_days` | int / float | `7` | [1, 365] | 活跃记忆保留天数。在此期间记忆处于 active 状态，可被正常检索 |
| `forgetting_consolidating_days` | int / float | `30` | [1, 365] | 巩固记忆保留天数。超过活跃期后进入 consolidating 状态，仍可检索但优先级降低 |
| `forgetting_archived_days` | int / float | `90` | [1, 3650] | 归档记忆保留天数。超过巩固期后归档，不再参与常规检索，可通过 `omni_govern(reactivate)` 恢复 |

#### 遗忘曲线阶段

```
active → consolidating → archived → (自动清理)
  7天       30天           90天
```

### 6.2 隐私与加密

控制记忆的隐私级别与加密参数。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `default_privacy` | string | `"personal"` | `public` / `team` / `personal` / `secret` | 新记忆的默认隐私级别 |
| `enable_encryption` | bool | `true` | — | 是否启用静态加密 |
| `export_key` | string | `""` | — | 导出/备份加密密钥；未配置时默认拒绝无密钥导出 |

### 6.3 审计与溯源

控制审计日志的容量与执行节奏。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `audit_log_max_rows` | int | `100000` | [1000, 10000000] | 审计日志最大行数。达到后按 FIFO 淘汰 |
| `audit_log_retention_days` | int | `90` | [1, 3650] | 审计日志保留天数 |
| `audit_interval_turns` | int | `50` | [5, 500] | 审计执行间隔（轮） |

---

## 7. 同步配置（Sync）

控制多实例间的记忆同步与分布式锁。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `sync_mode` | string | `"none"` | `none` / `file_lock` / `changelog` | 同步模式。`none`：禁用；`file_lock`：文件锁互斥；`changelog`：变更日志+向量时钟 |
| `sync_interval` | int / float | `30` | [1, 3600] | `changelog` 模式下的同步间隔（秒） |
| `sync_conflict_resolution` | string | `"latest_wins"` | `latest_wins` / `manual` / `merge` | 同步冲突解决策略 |
| `lock.backend` | string | `"file"` | `file` / `redis` | 分布式锁后端 |
| `lock.redis_url` | string | `"redis://localhost:6379/0"` | — | Redis 连接 URL（仅当 `lock.backend=redis` 生效） |

### 同步模式对比

| 模式 | 机制 | 适用场景 |
|:---|:---|:---|
| `none` | 无同步 | 单实例部署 |
| `file_lock` | 文件锁互斥 | 同机器多进程 |
| `changelog` | 变更日志 + 向量时钟 | 多机器分布式部署 |

---

## 8. LLM 配置

控制 Reflect 与 Consolidation 使用的 LLM 客户端后端。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `llm_backend` | string | `"openai"` | `openai` / `ollama` / `anthropic` | LLM 后端类型 |
| `ollama_model` | string | `"llama3"` | — | Ollama 模型名（仅当 `llm_backend=ollama` 生效） |
| `ollama_base_url` | string | `"http://localhost:11434"` | — | Ollama 服务地址 |
| `anthropic_model` | string | `"claude-3-haiku-20240307"` | — | Anthropic 模型名（仅当 `llm_backend=anthropic` 生效） |

### LLM 凭证优先级

LLM API Key 通过环境变量加载，优先级从高到低：

1. 环境变量（`OPENAI_API_KEY`、`OPENAI_BASE_URL` 等）
2. Hermes 环境变量
3. Hermes 配置文件

| 环境变量 | 说明 |
|:---|:---|
| `OPENAI_API_KEY` | LLM API Key |
| `OPENAI_BASE_URL` | LLM API Base URL |
| `HERMES_LLM_MODEL` | 默认模型名称 |

未配置 LLM 时，Reflect 和 Consolidation 功能将降级运行（返回缓存结果或跳过 LLM 调用）。

---

## 9. 嵌入配置（Embedding）

控制 Embedding 模型与服务的可配置化（点分键，对应嵌套 YAML `embedding.*`）。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `embedding.provider` | string | `"sentence_transformers"` | `sentence_transformers` / `openai` / `onnx` | 嵌入提供方 |
| `embedding.model_name` | string | `"all-MiniLM-L6-v2"` | — | 嵌入模型名（需与 `vector_store.embedding_dimension` 匹配） |
| `embedding.api_key` | string | `""` | — | 嵌入服务 API Key（如 `provider=openai` 时使用） |
| `embedding.base_url` | string | `""` | — | 嵌入服务 Base URL |

---

## 10. 安全配置（Security）

控制 REST API / MCP 服务的认证与限流。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `api_key` | string | （运行时生成 32 字节随机 hex） | — | REST/MCP API Key；禁止空字符串，未配置时自动生成 |
| `admin_token` | string | `""` | — | 管理员令牌，用于高危管理接口 |
| `mcp_require_api_key` | bool | `false` | — | MCP 服务是否强制要求 API Key |
| `api_rate_limit_per_minute` | int | `60` | [1, 10000] | API 每分钟速率限制 |
| `cors_allowed_origins` | list | `[]` | — | CORS 允许的来源列表；空列表表示禁用跨域 |

---

## 11. 压缩配置（Compression）

控制上下文压缩管线。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `enable_compression` | bool | `false` | — | 是否启用 5 层压缩管线。在 `on_pre_compress` 钩子中执行：collapse → priority → micro → line_compress → llm_summary |

### 压缩管线阶段

```
collapse      — 块级折叠（合并重复段落）
priority      — 优先级过滤（移除低价值内容）
micro         — 微观压缩（缩写、去冗余）
line_compress — 行级压缩（精简每行表达）
llm_summary   — LLM 摘要（语义级浓缩）
```

---

## 12. 备份与导出

控制自动备份节奏与副本保留策略。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `backup_interval_hours` | int | `24` | [1, 720] | 自动备份间隔（小时） |
| `backup_max_copies` | int | `3` | [1, 100] | 最大备份副本数。超过时按最旧优先淘汰 |

> **导出加密**：导出操作受 `export_key`（见 §6.2）控制，未配置 `export_key` 时默认拒绝无密钥导出。

---

## 13. MCP / REST API 服务配置

控制服务运行时行为与诊断。

| 键名 | 类型 | 默认值 | 取值范围 | 说明 |
|:---|:---|:---|:---|:---|
| `health_check_interval` | int | `10` | [1, 1000] | 健康检查执行间隔（轮） |
| `debug_mode` | bool | `false` | — | 调试模式，开启后输出详细日志与诊断信息 |

> **服务认证相关配置**（`api_key`、`admin_token`、`mcp_require_api_key`、`api_rate_limit_per_minute`、`cors_allowed_origins`）见 §10 安全配置。

---

## 验证

### 配置项覆盖核对

- **`_CONFIG_SCHEMA` 定义总数**：83 项
- **本文档覆盖**：83 项（100%）
- **任务描述预期**：90+ 项

实际 schema 中定义为 83 项（与任务描述中的 90+ 存在差异），本文档已覆盖 schema 中全部配置项，无遗漏。

### 分组覆盖核对

| 功能域 | 章节编号 | 配置项数 |
|:---|:---|:---|
| 检索（Retrieval） | §1 | 18 |
| L1 工作记忆（Working Memory） | §2 | 12 |
| L2 结构化存储（Storage） | §3 | 13 |
| L3 深层记忆（Deep Memory） | §4 | 1 |
| L4 内化层（Internalization） | §5 | 5 |
| 治理-冲突与遗忘 | §6.1 | 5 |
| 治理-隐私与加密 | §6.2 | 3 |
| 治理-审计与溯源 | §6.3 | 3 |
| 同步（Sync） | §7 | 5 |
| LLM | §8 | 4 |
| 嵌入（Embedding） | §9 | 4 |
| 安全（Security） | §10 | 5 |
| 压缩（Compression） | §11 | 1 |
| 备份与导出 | §12 | 2 |
| MCP / REST API 服务 | §13 | 2 |
| **合计** | — | **83** |

### 加载优先级链路核对

- [x] 代码显式设置（`OmniMemConfig.set`）
- [x] 环境变量（`OMNIMEM_<KEY>`，大小写不敏感，`-` 转 `_`）
- [x] 配置文件（`config.yaml`，支持嵌套 YAML，由 `_flatten_dict` 展平为点分键）
- [x] 默认值（`_CONFIG_SCHEMA` 中 `default`）
- [x] 特例处理（`api_key` 空值自动生成）
- [x] 校验机制（`_validate`：类型 / `min` / `max` / `choices`）
- [x] 热重载（`reload()`，每 10 轮检测 `mtime`）

---

## 相关文档

- [配置参考（旧版）](config_reference.md) — 按章节组织的配置说明（仅覆盖部分核心项）
- [快速入门](quickstart.md) — 5 分钟上手指南
- [API 参考](api_reference.md) — 7 种工具接口的详细参数说明
- [架构设计](architecture.md) — L1-L4 分层架构与数据流
