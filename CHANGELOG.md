# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-07-26

### Added
- **LLM Hybrid 事实抽取**：新增 `extraction_mode` 配置项（`rule`/`hybrid`/`llm`），支持 LLM 精炼规则抽取的事实内容与类型，失败时自动回退规则结果
- **外置威胁模式**：`config/threat_patterns.json` 支持 21 条中英文威胁检测模式（prompt 注入、角色劫持、数据泄露等），支持热重载
- **AES-256-GCM V2 加密**：新写入默认使用 `OMNI_ENC_V2:` 前缀的 AEAD 认证加密，密钥派生经 PBKDF2；保留 V1 Fernet (`OMNI_ENC_V1:`) 与 legacy 纯 Fernet token 解密兼容；GCM 认证标签防篡改
- **UnifiedMemoryIndex**：合并 ThreeLevelIndex + MetaStore 为单一 SQLite 数据库，读写连接分离（WAL 模式）；`use_unified_index` 灰度开关
- **LoRA 三段式闭环**：`export_training_data` 导出 Alpaca JSONL 训练数据 → 外部微调 → `register_adapter` 回注适配器；Shade 角色分身系统
- **治理动作扩展**：新增 34 个 govern 动作（`lora_train`、`export_training_data`、`register_adapter`、`shade_switch` 等）；MinHash/LSH 粗筛冲突扫描
- **共享检索线程池**：模块级共享检索线程池 + 引用计数管理，解决多实例线程数线性膨胀
- **Embedding 缓存 SQLite 化**：`_CachedEmbeddingFunction` 从 JSON 文件迁移为 SQLite 持久化存储
- **检索增强**：六通道编排（向量 + BM25 + 目录 + 图谱 + 时间 + 实体）；时序感知检索；检索熔断器；查询质量评估；同义扩展
- **LongMemEval + STATE-Bench 评测体系**：完整适配器、LLM-as-Judge 集成、性能基准测试、对比报告生成
- **Provider 架构拆分**：`ProviderInitializerMixin` / `ProviderLifecycleMixin` 分离初始化与生命周期管理；工具 schema 独立模块

### Fixed
- **QdrantStore 真向量修复**：修复向量搜索结果为空的问题
- **REST API 安全加固**：默认绑定 127.0.0.1（fail-closed），添加请求体大小限制
- **倒排索引 LRU 泄漏修复**：修复缓存条目不被释放导致的内存泄漏
- **KG 正则匹配 bug**：修复知识图谱实体抽取中 2 个正则表达式问题
- **工具注入检测修复**：`is_tool_injection()` 匹配逻辑修正
- **ChromaDB Telemetry 噪音**：过滤 ChromaDB 内部 telemetry 日志噪音

### Changed
- **依赖对齐**：消除文档与代码之间的依赖声明差异
- **检索通道文档对齐**：检索管道现状文档与实际代码一致
- **配置文件文档**：按功能域分组的 90+ 配置项参考文档
- **加密密钥优先级**：KMS > master_key > session_seed > 环境变量
- **加密失败策略**：加密不可用时 fail-closed（不降级为明文）
- **向量时钟优化**：`sync_mode=none` 时跳过 VectorClock 初始化

## [1.0.0] - 2026-04-30

### Added
- Initial release of OmniMem
- Five-layer memory architecture (L0 Perception → L1 Working → L2 Structured → L3 Deep → L4 Internalized)
- Hybrid retrieval engine (Vector + BM25 + RRF fusion + Knowledge Graph)
- Complete governance engine (Conflict resolution, Temporal decay, Forgetting curve, Privacy levels, Provenance tracking)
- Saga transaction coordination for derived data consistency
- Multi-instance synchronization with vector clocks
- Built-in memory tool compatibility layer
- Context Manager with semantic deduplication and token budget control
- Security features (anti-recursion, input sanitization, Unicode normalization)
- Comprehensive test suite covering all core modules
- `pyproject.toml` for modern Python packaging and dependency management
- `requirements.txt` and `requirements-dev.txt` for dependency installation
- GitHub Actions CI workflow for automated testing and linting
- Restructured tests into `tests/` directory with proper pytest configuration
- Pre-commit hooks configuration (ruff, mypy, trailing-whitespace)
- GitHub Issue templates (bug report, feature request)
- GitHub Pull Request template
- Social preview generator

### Changed
- Tests now use `omnimem.*` imports instead of `plugins.memory.omnimem.*`

### Fixed
- Unified import paths from `plugins.memory.omnimem` to `omnimem`
- Upgraded GitHub Actions to Node.js 24 (checkout v5, setup-python v6, codecov-action v5)
