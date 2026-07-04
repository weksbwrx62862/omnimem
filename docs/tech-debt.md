# OmniMem 技术债务清单

> 扫描范围：`/home/xxh/.hermes/plugins/omnimem` 下所有 `.py` 文件。  
> 标记类型：严格匹配 `TODO/FIXME/XXX/HACK`，并扩展纳入 `Task N`、`Phase/Stage`、`P0/P1/P3 方案`、`OPT:`、`fallback`、`硬编码`、`占位`、`mock` 等同类债务注释。  
> 统计：共梳理 **53** 处标记；其中严格 `TODO/FIXME/XXX/HACK` 代码注释仅 **1** 处，其余为计划项、兼容兜底或已完成的修复标注。

---

## 已解决（OmniMem v2.0.0 第二轮修复）

| 债务项 | 原位置 | 解决方式 | 备注 |
|---|---|---|---|
| `provider.py` 职责过重（1,322 行） | `provider.py` | 拆分为 `core/provider_initializer.py`、`core/provider_lifecycle.py`、`core/provider_middleware.py`、`compat/provider_proxy.py` | 单文件核心逻辑 < 500 行，公开接口保持不变 |
| `retrieval/engine.py` 职责过重（1,647 行） | `retrieval/engine.py` | 拆分为 `retrieval/circuit_breaker.py`、`retrieval/rw_lock.py`、`retrieval/query_quality.py`、`retrieval/synonym_expander.py`、`retrieval/hybrid_orchestrator.py` | `HybridRetriever` 仍作为统一入口 |
| 缺乏统一检索抽象 | 多处 | 新增 `retrieval/base.py` 定义 `BaseRetriever` | 所有检索通道实现统一 `search/asearch` 契约 |
| 嵌入后端硬编码 | `retrieval/vector.py` | 新增 `embedding/base.py` + `SentenceTransformersProvider` / `OpenAIEmbeddingProvider` / `ONNXEmbeddingProvider` | 通过 `embedding.provider` 配置切换 |
| 向量存储硬编码 ChromaDB | `retrieval/vector.py` | 新增 `storage/base.py` + `ChromaVectorStore` / `MilvusVectorStore` | 通过 `vector_store.provider` 配置切换 |
| 熔断器仅骨架 | `retrieval/engine.py` | `CircuitBreaker` 三态转换完整实现并接入向量检索 | CLOSED → OPEN → HALF_OPEN → CLOSED |
| 同义词映射为类属性污染 | `retrieval/engine.py` | 迁移到 `retrieval/synonym_expander.py` 并成为实例属性 | 多实例间不再相互污染 |
| 检索参数硬编码 | `retrieval/engine.py` | `rrf_k`、`rrf_min_score`、`circuit_breaker_threshold` 等接入配置 | 已完成 Task 3 |
| 新增检索通道需改 engine.py | `retrieval/engine.py` | 新增 `retrieval/registry.py` 插件化注册表 | 新通道注册后 `HybridRetriever` 自动加载 |
| 文件锁未抽象 | `governance/sync.py` 等 | 新增 `utils/lock.py` 定义 `LockProvider` + `FileLockProvider` / `RedisLockProvider` | 为分布式同步做准备 |
| `.bak` 备份文件堆积 | 多处 | 删除 7 个 `.bak` 文件 | 清理临时/备份文件 |

---

## 安全相关

| 位置 | 标记 | 内容 | 优先级 | 备注 |
|---|---|---|---|---|
| `handlers/memorize.py:647` | Task 2 | `★ Task 2: secret 级记忆透明化加密状态` | 已完成 Task 2 | 加密状态字段已透传到 SDK memorize 返回结果 |
| `governance/kms.py:35` | P0 | `★ P0修复：密钥缓存，避免频繁磁盘 IO` | 已完成（2026-07-03） | KMS LRU 密钥缓存已实现，含环境变量优先级回退 |
| `governance/kms.py:70` | P0 | `★ P0修复：优先从环境变量读取密钥` | 已完成（2026-07-03） | 4 种 provider（local/aws/azure/gcp）全部实现 |

---

## 性能相关

| 位置 | 标记 | 内容 | 优先级 | 备注 |
|---|---|---|---|---|
| `handlers/recall.py:22` | P1 | `★ P1修复：模块级共享 executor，避免每次 recall 调用创建新线程池` | 已完成 | 线程池复用，降低 recall 调用开销 |
| `handlers/recall.py:36` | R26 | `★ R26优化：提取公共正则常量，避免4处硬编码重复` | 已完成 | 代码去重，提升可维护性 |
| `handlers/recall.py:132` | OPT | `★ OPT: 检索超时保护 — 使用模块级共享 executor（P1修复）` | 已完成 | recall 超时熔断已接入共享线程池 |
| `handlers/memorize.py:545` | 异步化 | `★ 异步化：非关键路径提交到后台线程，降低主路径延迟` | 已完成 | 后台线程池处理 LLM 决策、KG 抽取等 |
| `handlers/memorize.py:619` | 嵌入缓存 | `★ 嵌入缓存持久化（确保新记忆的 embedding 写入磁盘）` | 已完成 | 缓存持久化减少重复嵌入计算 |
| `governance/forgetting.py:187` | Phase 1 | `★ Phase 1 优化：添加复合索引提升查询性能` | 已完成（2026-07-03） | 3 个复合索引（stage+created_at, heat+updated_at, heat+recall_count）已添加 |
| `governance/forgetting.py:633` | Phase 1 | `★ Phase 1 优化：自动升级检查` | 已完成 | 高频访问记忆自动升级回 active 已实现 |
| `retrieval/bm25.py:428` | P3 | `★ P3 LRU淘汰：超出上限时删除最旧文档（使用切片替代 pop(0) 避免 O(n) 开销）` | 计划内（Phase 3） | BM25 索引内存上限控制 |
| `retrieval/engine.py:414` | OPT | `★ OPT: 向量检索熔断器 — 连续故障时自动降级纯BM25` | 计划内 | 检索可用性兜底，依赖 `CircuitBreaker` 已有骨架 |
| `retrieval/engine.py:394` | fallback | `初始化失败时降级为原有 dict 缓存，保持向后兼容` | 暂不处理 | 兼容旧缓存行为的兜底 |
| `retrieval/faiss_store.py:128` | Fallback | `Fallback: 空索引` | 暂不处理 | 空索引降级，避免启动失败 |

---

## 功能缺失 / Phase 3

| 位置 | 标记 | 内容 | 优先级 | 备注 |
|---|---|---|---|---|
| `config/_config.py:81` | Task 3 | `Task 3: 检索参数可配置化` | 已完成 Task 3 | rrf_k、rrf_min_score、circuit_breaker 等已入配置 |
| `retrieval/engine.py:364` | Task 3 | `Task 3: 从配置读取检索参数，无配置时使用当前硬编码默认值` | 已完成 Task 3 | `HybridRetriever.__init__` 已接入配置 |
| `retrieval/engine.py:376` | Task 3 | `Task 3: 同义词映射改为实例属性，避免多实例间相互污染` | 已完成 Task 3 | `_SYNONYM_MAP` 已改为实例属性 |
| `context/manager.py:78` | P1 | `★ P1方案三：结构化压缩模板 — 将常见长句模式压缩为固定格式短摘要` | 计划内（Phase 3） | context 压缩策略增强 |
| `context/manager.py:173` | P1 | `★ P1方案三：策略2 — 结构化模板压缩` | 计划内（Phase 3） | 同上，策略实现 |
| `deep/knowledge_graph.py:907` | P1 | `★ P1方案四：增量局部推理（替代全表扫描）` | 已完成（2026-07-03） | 增量 2-hop 邻居查询 + 局部子图推理已实现 |
| `retrieval/engine.py:410` | P1 | `★ P1方案四：动态来源权重（由 FeedbackCollector 驱动）` | 计划内（Phase 3） | 多路检索权重动态调整 |
| `retrieval/engine.py:1187` | P1 | `★ P1方案四：应用动态来源权重（基于 FeedbackCollector 的 CTR 统计）` | 计划内（Phase 3） | 同上，在线应用侧 |
| `handlers/memorize.py:488` | P0 | `★ P0方案二：Saga 协调派生数据写入` | 计划内 | 派生数据（向量、BM25、KG）写入一致性 |
| `memory/drawer_closet.py:84` | P0 | `★ P0方案一：MetaStore SQLite 元数据存储（并行双写）` | 已完成（2026-07-03） | buffer flush + Saga 双写全链路落地，补偿语义修复 |
| `memory/drawer_closet.py:87` | P0 | `★ P0修复：Saga 协调器，保证 Drawer/MetaStore 双写事务一致性` | 已完成（2026-07-03） | buffer flush 前移至 Saga 之前，补偿可真正删除已落盘文件 |
| `memory/drawer_closet.py:208` | P0 | `★ P0修复：使用 Saga 协调 Drawer/MetaStore 双写，保证事务一致性` | 已完成（2026-07-03） | 同上 |
| `memory/drawer_closet.py:448` | P0 | `★ P0方案一：同步预热 MetaStore` | 已完成 | WarmupManager 已实现 MetaStore 预热 |
| `memory/drawer_closet.py:476` | P0 | `★ P0方案一：同步更新 MetaStore` | 已完成 | Saga 双写已实现同步更新 MetaStore |
| `governance/conflict.py:129` | Stage 2 | `★ 否定词检测必须配合 Stage 2 语义验证，否则 "纠正: xxx" 类内容会误判为冲突` | 计划内 | 当前 Stage 2 语义验证未完全启用，存在误判风险 |

---

## 代码质量

| 位置 | 标记 | 内容 | 优先级 | 备注 |
|---|---|---|---|---|
| `core/tool_names.py:4` | 硬编码 | `避免硬编码字符串分散在多处。` | 已完成 Task 10 | 工具名已集中为常量 |
| `retrieval/engine.py:361` | 硬编码 | `config: OmniMemConfig 实例或配置字典，None 时使用硬编码默认值` | 已完成 Task 3 | 默认值已迁移到配置 |
| `handlers/recall.py:36` | 硬编码 | `★ R26优化：提取公共正则常量，避免4处硬编码重复` | 已完成 | 同“性能相关”条目，兼具代码质量属性 |
| `compression/mermaid_canvas.py:288` | 硬编码 | `支持从 config 读取可配 pattern 列表，硬编码正则作为 fallback。` | 计划内 | Mermaid 解析 pattern 应走配置 |
| `compression/mermaid_canvas.py:301` | 硬编码 | `硬编码正则作为 fallback` | 计划内 | 同上，具体 fallback 点 |
| `retrieval/vector.py:172` | 占位 | `ChromaDB 要求 metadata 非空，添加占位字段` | 计划内 / 低优先级 | metadata 占位字段可改为更语义化内容 |
| `memory/index.py:112` | 硬编码 | `列名来自硬编码常量，非用户输入，安全使用 f-string` | 暂不处理 | 列名为内部常量，风险可控 |
| `memory/meta_store.py:117` | 硬编码 | `列名来自硬编码常量，非用户输入，安全使用 f-string` | 暂不处理 | 同上 |
| `governance/forgetting.py:165` | 硬编码 | `列名来自硬编码常量，非用户输入，安全使用 f-string` | 暂不处理 | 同上 |
| `__init__.py:29` | Mock | `Mock agent.memory_provider 模块（Hermes 框架依赖）` | 暂不处理 | Hermes 框架兼容 hack，非运行时债务 |
| `provider.py:213` | Fallback | `Fallback 动态代理：仅在显式赋值完成前生效。` | 暂不处理 | Provider 懒加载兼容设计 |
| `provider.py:290` | 降级模式 | `降级模式：跳过向量检索和 ChromaDB，仅 BM25 检索` | 暂不处理 | 依赖缺失时的优雅降级 |
| `provider.py:1025` | Fallback | `Fallback: 直接执行（测试兼容）` | 暂不处理 | 测试兼容分支 |
| `handlers/govern.py:82` | 兼容 | `兼容 Provider 实例的 config/_config 两种属性名；优先 _config，再回退 config` | 暂不处理 | 历史属性名兼容 |
| `handlers/recall.py:767` | fallback | `★ 结果不足时 fallback 到 FTS / store 全量扫描` | 暂不处理 | 检索兜底策略 |
| `handlers/recall.py:244` | Fallback | `Fallback: 原始三元组搜索` | 暂不处理 | 三元组检索兜底 |
| `handlers/recall.py:668` | Fallback | `Fallback: 原始三元组搜索` | 暂不处理 | 同上 |
| `handlers/govern.py:231` | Fallback | `Fallback：conflict_warning 无记录时，走语义搜索路径` | 暂不处理 | 冲突治理兜底 |
| `handlers/govern.py:280` | fallback | `fallback: 全局否定词扫描（即使语义不矛盾，也检查否定词重叠）` | 暂不处理 | 冲突扫描兜底 |
| `compression/priority.py:85` | 临时字段 | `4. 清理临时字段` | 已完成 / 低优先级 | 压缩中间态清理 |

---

## 测试相关

| 位置 | 标记 | 内容 | 优先级 | 备注 |
|---|---|---|---|---|
| `tests/test_optimizations.py:384` | 硬编码 | `同义词映射应从 synonyms.json 加载，而非硬编码。` | 计划内 | 测试已指出同义词映射应外部化 |
| `tests/test_handlers.py:32` | MagicMock | `config — 使用真实 dict 避免 .get() 返回 MagicMock 导致数值比较失败` | 暂不处理 | 测试桩代码兼容处理 |
| `core/prompt_builder.py:71` | MagicMock | `确保 stable_cache_turn 是整数（处理测试中的 MagicMock）` | 暂不处理 | 生产代码中为测试场景做类型兜底 |
| `tests/conftest.py:13` | Mock | `Mock agent.memory_provider 模块（Hermes 框架依赖）` | 暂不处理 | 测试 fixtures 兼容框架依赖 |

---

## 其他

| 位置 | 标记 | 内容 | 优先级 | 备注 |
|---|---|---|---|---|
| `compression/micro.py:25` | TODO | `"TODO:",` | 已废弃 / 非实际待办 | 该 `TODO:` 仅作为微压缩保留的关键字标记，并非代码待办事项 |

---

## 附录：扫描说明

- **严格标记结果**：在全部 `.py` 文件中，`TODO/FIXME/XXX/HACK` 仅出现 1 处（`compression/micro.py:25`），且为保留关键字而非实际待办。
- **扩展标记口径**：为了建立可见的技术债务清单，本次将 `Task N`、`Phase/Stage`、`P0/P1/P3 方案`、`OPT:`、`fallback`、`硬编码`、`占位`、`mock` 等注释一并纳入。
- **不修改源码**：本任务不产生业务代码变更，所有 `TODO` 等注释保持原样。
