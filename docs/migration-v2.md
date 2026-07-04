# OmniMem v1.0.x → v2.0.0 迁移指南

> 本文档说明从 OmniMem v1.0.x 升级到 v2.0.0 时需要关注的破坏性变更、新增配置项以及推荐迁移步骤。

---

## 1. 概述

OmniMem v2.0.0 是一次以**可维护性、可扩展性和稳定性**为主题的重大重构：

- 将 `provider.py` 和 `retrieval/engine.py` 两个超千行的大文件按职责拆分；
- 引入统一的抽象接口层：`BaseRetriever`、`EmbeddingProvider`、`VectorStore`、`LockProvider`；
- 新增配置驱动的 Provider / VectorStore 选择机制；
- 为向量检索增加熔断器、读写锁、同义词扩展、垃圾查询检测等治理能力；
- 将 `cryptography` 提升为默认依赖，并加固 REST API / MCP Server 安全。

**好消息**：对外公开 API（`omni_memorize`、`omni_recall`、`omni_reflect`、`omni_govern` 等工具接口，以及 `OmniMemProvider` 类的主要方法）保持兼容。大多数用户只需调整配置文件即可升级。

---

## 2. 破坏性变更

### 2.1 内部文件路径变更

以下模块的职责已迁移到新文件，旧导入路径可能仍然保留兼容别名，但建议迁移：

| v1.0.x 位置 | v2.0.0 新位置 | 说明 |
|:---|:---|:---|
| `provider.py` 中 `__init__` / 初始化逻辑 | `core/provider_initializer.py` | `ProviderInitializerMixin` |
| `provider.py` 中生命周期/开关逻辑 | `core/provider_lifecycle.py` | `ProviderLifecycleMixin` |
| `provider.py` 中中间件/包装逻辑 | `core/provider_middleware.py` | `ProviderMiddlewareMixin` |
| `provider.py` 中兼容代理逻辑 | `compat/provider_proxy.py` | `ProviderProxyMixin` |
| `retrieval/engine.py` 中熔断逻辑 | `retrieval/circuit_breaker.py` | `CircuitBreaker` |
| `retrieval/engine.py` 中读写锁逻辑 | `retrieval/rw_lock.py` | `FairReadWriteLock` |
| `retrieval/engine.py` 中垃圾查询检测 | `retrieval/query_quality.py` | `is_garbage_query` / `trim_to_budget` |
| `retrieval/engine.py` 中同义词扩展 | `retrieval/synonym_expander.py` | `SynonymExpander` |
| `retrieval/engine.py` 中检索编排 | `retrieval/hybrid_orchestrator.py` | `HybridOrchestrator` |

**影响**：如果你直接引用上述内部类或函数，需要更新导入路径。`HybridRetriever` 仍作为统一入口保留在 `retrieval/engine.py`。

### 2.2 抽象接口层取代部分旧类

v2.0.0 新增以下抽象基类，用于解耦具体实现：

| 抽象接口 | 文件 | 备注 |
|:---|:---|:---|
| `BaseRetriever` | `retrieval/base.py` | 检索通道统一接口 |
| `EmbeddingProvider` | `embedding/base.py` | 嵌入服务统一接口 |
| `VectorStore` | `storage/base.py` | 向量存储统一接口 |
| `LockProvider` | `utils/lock.py` | 分布式锁统一接口 |

**影响**：自定义检索通道、嵌入模型或向量后端时，应优先继承上述抽象基类。

### 2.3 向量存储接口调整

新的 `storage/base.py` 中的 `VectorStore` 接口与旧 `retrieval/vector_store.py` 不完全一致：

- 调用方需要**自行计算 embeddings** 后再调用 `add()`；
- 方法签名从 `upsert()` 变为 `add()`；
- 查询方法从 `query()` 变为 `search()`；
- 同步方法优先，子类可自行提供异步包装。

旧接口仍保留在 `retrieval/vector_store.py` 以维持兼容，但新后端建议按 `storage/base.py` 实现。

### 2.4 配置项变更

#### 新增配置项

| 配置项 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `embedding.provider` | string | `sentence_transformers` | 嵌入后端：`sentence_transformers` / `openai` / `onnx` |
| `embedding.model_name` | string | `all-MiniLM-L6-v2` | 模型名称 |
| `embedding.api_key` | string | `""` | OpenAI 兼容服务密钥 |
| `embedding.base_url` | string | `""` | OpenAI 兼容服务基础 URL |
| `vector_store.provider` | string | `chroma` | 向量存储后端：`chroma` / `milvus` |
| `vector_store.collection_name` | string | `omnimem` | Collection 名称 |
| `vector_store.persist_dir` | string | `/tmp/omnimem/storage/chroma` | Chroma 持久化目录 |
| `vector_store.uri` | string | `http://localhost:19530` | Milvus 服务地址 |
| `vector_store.token` | string | `""` | Milvus 认证令牌 |
| `vector_store.embedding_dimension` | int | `384` | 向量维度 |
| `vector_store.metric_type` | string | `COSINE` | Milvus 度量类型 |
| `vector_store.consistency_level` | string | `Bounded` | Milvus 一致性级别 |
| `lock.backend` | string | `file` | 锁后端：`file` / `redis` |
| `lock.redis_url` | string | `redis://localhost:6379/0` | Redis 锁地址 |
| `recall_timeout_ms` | int | `5000` | 单通道检索超时 |
| `recall_strategy` | string | `hybrid` | 召回策略：`hybrid` / `keyword` / `embedding` |
| `rrf_k` | int | `60` | RRF 平滑常数 |
| `circuit_breaker_threshold` | int | `3` | 熔断失败阈值 |
| `circuit_breaker_cooldown_seconds` | int/float | `60` | 熔断冷却时间 |
| `rebuild_batch_size` | int | `32` | 向量重建批次大小 |
| `rebuild_max_workers` | int | `4` | 向量重建并行 workers |

#### 已废弃但仍兼容的配置项

| 配置项 | 说明 |
|:---|:---|
| `vector_backend` | 旧向量后端选择，现由 `vector_store.provider` 接管 |

---

## 3. 公开 API 变化

### 3.1 保持不变

以下公开接口在 v2.0.0 中保持行为一致：

- `OmniMemProvider` 类构造方法与主要方法：`initialize()`、`shutdown()`、`is_available()`、`omni_memorize()`、`omni_recall()`、`omni_reflect()`、`omni_govern()` 等；
- 工具函数名称与参数（通过 `ToolRouter` 暴露）；
- `MemoryRecord` 数据模型；
- REST API 端点路径与请求/响应格式（仅在安全中间件上有增强）。

### 3.2 内部属性访问路径变化

虽然 `OmniMemProvider` 实例仍通过显式属性暴露子系统，但部分属性的初始化顺序和所属 Mixin 发生变化：

- `_init_l1()`、`_init_store()`、`_init_retrieval()`、`_init_governance_sync_services()` 等方法现在定义在 `core/provider_initializer.py`；
- `initialize()`、`shutdown()`、`_degraded_mode` 等生命周期相关逻辑在 `core/provider_lifecycle.py`；
- `_wrap_model_call()`、`_build_system_prompt()` 等中间件逻辑在 `core/provider_middleware.py`；
- `__getattr__` 兼容代理已替换为显式属性赋值，位于 `compat/provider_proxy.py`。

**建议**：不要在业务代码中直接访问 `provider._xxx` 私有属性；如需扩展，请通过 Facade 或新公开的 Manager 方法。

---

## 4. 新配置项示例

### 4.1 最小改动配置（默认 sentence-transformers + Chroma）

```yaml
# config.yaml
save_interval: 15
auto_memorize: true
default_privacy: personal

retrieval_mode: rag
recall_strategy: hybrid
recall_timeout_ms: 5000

# v2.0.0 推荐写法
embedding:
  provider: sentence_transformers
  model_name: all-MiniLM-L6-v2

vector_store:
  provider: chroma
  collection_name: omnimem
  persist_dir: /tmp/omnimem/storage/chroma
  embedding_dimension: 384

lock:
  backend: file
```

### 4.2 使用 OpenAI Embedding + Chroma

```yaml
embedding:
  provider: openai
  model_name: text-embedding-3-small
  api_key: ${OPENAI_API_KEY}
  base_url: ""  # 留空使用官方端点；使用兼容服务时填写

vector_store:
  provider: chroma
  collection_name: omnimem
  persist_dir: /tmp/omnimem/storage/chroma
  embedding_dimension: 1536
```

### 4.3 使用 Milvus 向量存储

```yaml
embedding:
  provider: sentence_transformers
  model_name: all-MiniLM-L6-v2

vector_store:
  provider: milvus
  collection_name: omnimem
  uri: http://localhost:19530
  token: ""
  embedding_dimension: 384
  metric_type: COSINE
  consistency_level: Bounded
```

### 4.4 使用 Redis 锁

```yaml
lock:
  backend: redis
  redis_url: redis://localhost:6379/0
```

> **注意**：配置支持嵌套 YAML 写法，OmniMemConfig 会在加载时自动展平为点分键（如 `embedding.provider`）。

---

## 5. 迁移步骤

1. **备份数据目录**：在升级前备份 `~/.omnimem/` 或项目数据目录。
2. **更新依赖**：安装 v2.0.0 包，`cryptography` 已成为默认依赖。
3. **更新配置文件**：
   - 将 `vector_backend: chromadb` 迁移为 `vector_store.provider: chroma`；
   - 新增 `embedding` 段落；
   - 如需要分布式锁，新增 `lock` 段落。
4. **检查自定义代码**：
   - 搜索 `from omnimem.provider import` 以外的内部导入；
   - 若直接引用 `provider._vector_breaker`、`provider._query_quality` 等，请改为使用公开 Facade 方法。
5. **运行测试**：执行 `python3 -m pytest tests/ -q --tb=short`，确认全量通过。
6. **验证功能**：执行一次 `omni_memorize` + `omni_recall` 端到端验证。

---

## 6. 常见问题

### Q1: 升级后向量索引是否需要重建？

不需要。v2.0.0 保持 ChromaDB/SQLite 数据格式兼容。但如果你在 v1.0.x 中使用了非默认向量后端，请先确认该后端在 v2.0.0 中的适配状态。

### Q2: 自定义的 Embedding 模型如何接入？

继承 `embedding/base.py` 中的 `EmbeddingProvider`，实现 `dimension`、`model_name`、`embed()`、`aembed()`，然后在 `embedding/__init__.py` 的 `create_embedding_provider()` 中注册识别名。

### Q3: `retrieval/vector_store.py` 中的旧 `VectorStore` 还能用吗？

可以，v2.0.0 保留该文件作为兼容层。但建议新实现迁移到 `storage/base.py` 的新接口。

### Q4: 是否需要修改 REST API/MCP Server 的调用方式？

不需要。v2.0.0 仅增加默认 API Key 认证、请求体大小限制等安全增强；客户端调用方式不变。如果启用了 `api_key`，请在请求头中携带 `X-API-Key`。

---

## 7. 参考文档

- [架构设计](architecture.md)
- [配置参考](config_reference.md)
- [API 参考](api_reference.md)
- [CHANGELOG](../CHANGELOG.md)
