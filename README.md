# OmniMem

> 五层混合记忆系统：感知 → 工作 → 结构化 → 深层 → 内化

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

OmniMem 是一个为 AI Agent 设计的多层混合记忆系统，采用**五层架构**模拟人类记忆机制，并配备完整的**治理引擎**，实现高效、可靠、可溯源的记忆管理。

## 核心特性

- **五层记忆架构** — 从感知到内化，分层管理不同生命周期和抽象程度的记忆
- **混合检索引擎** — 向量语义检索 + BM25 关键词检索 + RRF 融合排序 + 知识图谱补充
- **完整治理引擎** — 冲突仲裁、时间衰减、遗忘曲线、隐私分级、溯源追踪、多实例同步
- **智能上下文管理** — 自适应预算、语义去重、懒加载细节，精准控制 Token 消耗
- **Saga 事务协调** — 派生数据（索引/检索/图谱）最终一致性保证
- **内置安全机制** — 反递归防护、输入净化、安全扫描，防止自我污染
- **兼容内置 memory 工具** — 无缝替换现有记忆系统，零迁移成本

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                        治理引擎 (Governance)                  │
│  冲突仲裁 │ 时间衰减 │ 遗忘曲线 │ 隐私分级 │ 溯源追踪 │ 同步  │
└─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────┬─────────┬─────────┬─────────┬─────────┐
    ▼         ▼         ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌─────────┐ ┌───────┐ ┌─────────┐
│  L0   │ │  L1   │ │   L2    │ │  L3   │ │   L4    │
│ 感知层 │ │ 工作  │ │ 结构化  │ │ 深层  │ │  内化   │
│       │ │ 记忆  │ │  记忆   │ │ 记忆  │ │  记忆   │
└───────┘ └───────┘ └─────────┘ └───────┘ └─────────┘
```

### 五层记忆架构

| 层级 | 名称 | 功能描述 | 数据特征 |
|:---:|:---:|:---|:---|
| L0 | **感知层** | 主动监控、信号检测、意图预测 | 实时流数据，无需持久化 |
| L1 | **工作记忆** | CoreBlock(常驻上下文) + Attachment(压缩后状态) | 高频访问，内存驻留 |
| L2 | **结构化记忆** | Wing/Room 宫殿导航 + Drawer/Closet 双存储 | 持久化，三级索引 |
| L3 | **深层记忆** | Consolidation(事实→观察→心智模型) + 知识图谱 | 周期性升华，图结构 |
| L4 | **内化记忆** | KV Cache(高频) + LoRA(深层) [可选] | 模型参数级，训练获得 |

### 治理引擎（横切面）

治理引擎贯穿所有记忆层级，提供横切面能力：

- **冲突仲裁** — 多源记忆冲突自动检测与解决，支持语义级矛盾识别
- **时间衰减** — 基于时间的记忆重要性动态调整，久未访问自动降权
- **遗忘曲线** — 模拟艾宾浩斯遗忘曲线的智能清理，支持软删除和硬删除
- **隐私分级** — `public` / `team` / `personal` / `secret` 四级隐私控制
- **溯源追踪** — 完整记录记忆来源、变更历史、操作者身份
- **同步机制** — 多实例间的记忆同步与冲突解决，支持向量时钟

## 安装

### 环境要求

- Python 3.10+
- 依赖包见 `plugin.yaml`

### 方式一：作为 Hermes 插件安装

1. 将本目录放入 `plugins/memory/omnimem/`
2. 在 `config.yaml` 中配置：

```yaml
memory:
  provider: omnimem
```

### 方式二：独立使用

```bash
# 克隆仓库
git clone <repository-url>
cd omnimem

# 安装核心依赖
pip install chromadb>=0.4.0,<0.7.0
pip install rank-bm25>=0.2.0,<0.3.0
pip install tiktoken>=0.7.0
pip install pyyaml>=6.0

# 可选依赖
pip install sentence-transformers>=2.2.0  # 语义嵌入缓存
pip install cryptography>=42.0.0          # 加密隐私数据
```

### 方式三：使用 requirements.txt

```bash
pip install -r requirements.txt
```

## 快速开始

### 基础用法

```python
from plugins.memory.omnimem.provider import OmniMemProvider

# 初始化记忆系统
provider = OmniMemProvider()
provider.initialize("session_001", hermes_home="~/.hermes")

# 存储记忆
provider.memorize("用户喜欢Python编程", tags=["preference", "tech"])

# 检索记忆
results = provider.recall("用户的编程偏好", mode="rag")
```

### 完整工作流示例

```python
# 1. 初始化
provider = OmniMemProvider()
provider.initialize("session_001", hermes_home="~/.hermes")

# 2. 存储不同类型的记忆
provider.handle_tool_call("omni_memorize", {
    "content": "用户偏好使用 FastAPI 而非 Flask",
    "memory_type": "preference",
    "confidence": 5,
    "privacy": "personal"
})

provider.handle_tool_call("omni_memorize", {
    "content": "项目使用 PostgreSQL 作为主数据库",
    "memory_type": "fact",
    "confidence": 4,
    "scope": "project"
})

# 3. 主动检索
result = provider.handle_tool_call("omni_recall", {
    "query": "数据库选型",
    "mode": "rag",
    "max_tokens": 1500
})

# 4. 深层反思
result = provider.handle_tool_call("omni_reflect", {
    "query": "用户的技术栈偏好",
    "disposition": {
        "skepticism": 3,
        "literalness": 4,
        "empathy": 3
    }
})

# 5. 治理操作
provider.handle_tool_call("omni_govern", {
    "action": "scan_conflicts"
})

# 6. 会话结束自动归档
provider.on_session_end(messages)
```

## 核心功能

### 7 个工具接口

OmniMem 向 Agent 暴露 7 个工具接口：

| 工具 | 功能 | 典型使用场景 |
|:---:|:---|:---|
| `omni_memorize` | 主动存储记忆 | 用户纠正、偏好声明、重要决策 |
| `omni_recall` | 主动检索记忆 | 回答需要历史上下文的问题 |
| `omni_compact` | 压缩前准备 | 上下文过长时触发压缩 |
| `omni_reflect` | L3 深层反思 | 从原始事实提炼心智模型 |
| `omni_govern` | 治理操作 | 冲突解决、隐私设置、归档 |
| `omni_detail` | 按需拉取记忆细节 | 需要查看记忆完整内容时 |
| `memory` | 兼容内置 memory 工具 | 无缝替换现有记忆系统 |

### 记忆类型

| 类型 | 说明 | 示例 |
|:---:|:---|:---|
| `fact` | 客观事实 | "项目使用 Docker 部署" |
| `preference` | 用户偏好 | "用户喜欢暗色主题" |
| `correction` | 纠正信息 | "之前说的不对，实际是..." |
| `skill` | 技能/能力 | "用户擅长 React 开发" |
| `procedural` | 流程/步骤 | "部署流程：1. git pull 2. ..." |
| `event` | 事件记录 | "[Turn 15] 用户切换了分支" |

### 检索模式

| 模式 | 速度 | 适用场景 | 技术实现 |
|:---:|:---:|:---|:---|
| `rag` | 毫秒级 | 快速检索、常规查询 | 向量+BM25+RRF融合 |
| `llm` | 秒级 | 深度推理、意图预测 | 扩展同义词+语义推理 |

## 项目结构

```
omnimem/
├── compression/          # 数据压缩与优化
│   ├── collapse.py       # 记忆折叠压缩
│   ├── line_compress.py  # 行级压缩
│   ├── llm_summary.py    # LLM 摘要生成
│   ├── micro.py          # 微压缩
│   └── priority.py       # 优先级压缩策略
├── context/              # 上下文管理
│   └── manager.py        # ContextManager：精炼/去重/预算控制
├── core/                 # 核心组件
│   ├── block.py          # CoreBlock：常驻上下文块
│   ├── attachment.py     # CompactAttachment：压缩状态附件
│   ├── soul.py           # SoulSystem：身份与性格管理
│   ├── budget.py         # BudgetManager：Token 预算控制
│   ├── store_service.py  # MemoryStoreService：存储服务层
│   ├── background.py     # BackgroundTaskExecutor：后台任务
│   ├── saga.py           # SagaCoordinator：分布式事务协调
│   └── async_provider.py # AsyncOmniMemProvider：异步包装器
├── deep/                 # 深层记忆处理（L3）
│   ├── consolidation.py  # ConsolidationEngine：知识升华
│   ├── knowledge_graph.py# KnowledgeGraph：知识图谱
│   └── reflect.py        # ReflectEngine：反思引擎
├── governance/           # 治理引擎
│   ├── conflict.py       # ConflictResolver：冲突仲裁
│   ├── decay.py          # TemporalDecay：时间衰减
│   ├── forgetting.py     # ForgettingCurve：遗忘曲线
│   ├── privacy.py        # PrivacyManager：隐私管理
│   ├── provenance.py     # ProvenanceTracker：溯源追踪
│   ├── sync.py           # SyncEngine：多实例同步
│   ├── auditor.py        # GovernanceAuditor：治理巡检
│   ├── feedback.py       # FeedbackCollector：反馈收集
│   └── vector_clock.py   # VectorClock：分布式向量时钟
├── handlers/             # API 处理器
│   ├── memorize.py       # 记忆存储处理
│   ├── recall.py         # 检索处理
│   ├── govern.py         # 治理处理
│   ├── schemas.py        # 工具模式定义
│   └── _compat.py        # 兼容层工具
├── internalize/          # 内化记忆（L4）
│   ├── kv_cache.py       # KVCacheManager：高频缓存
│   └── lora_train.py     # LoRATrainer：LoRA 训练器
├── memory/               # 记忆存储（L2）
│   ├── wing_room.py      # WingRoomManager：宫殿导航
│   ├── drawer_closet.py  # DrawerClosetStore：双存储
│   ├── index.py          # ThreeLevelIndex：三级索引
│   ├── markdown_store.py # MarkdownStore：Markdown 存储
│   ├── meta_store.py     # MetaStore：元数据存储
│   └── types.py          # 类型定义
├── perception/           # 感知层（L0）
│   └── engine.py         # PerceptionEngine：感知引擎
├── retrieval/            # 检索引擎
│   ├── engine.py         # HybridRetriever：混合检索器
│   ├── vector.py         # VectorRetriever：向量检索
│   ├── bm25.py           # BM25Retriever：关键词检索
│   ├── rrf.py            # RRF：倒数排序融合
│   └── reranker.py       # Cross-Encoder 重排序
├── utils/                # 工具函数
│   ├── llm_client.py     # AsyncLLMClient：异步 LLM 客户端
│   └── security.py       # SecurityValidator：安全验证
├── config.py             # OmniMemConfig：配置管理（支持热重载）
├── provider.py           # OmniMemProvider：主入口
├── plugin.yaml           # 插件配置
└── __init__.py           # 插件注册
```

## 配置选项

OmniMem 支持通过 `config.yaml` 或代码配置，所有配置均支持**热重载**（每 10 轮自动检查）。

```yaml
# 默认配置
save_interval: 15              # 自动保存间隔（轮数）
retrieval_mode: rag            # 默认检索模式: rag / llm
vector_backend: chromadb       # 向量数据库后端: chromadb / qdrant / pgvector
max_prefetch_tokens: 300       # 最大预取 token 数
budget_tokens: 4000            # 工作记忆预算 token 数
fact_threshold: 10             # L3 Consolidation 触发阈值
enable_reranker: false         # 是否启用 Cross-Encoder 重排序
conflict_strategy: latest      # 冲突解决策略: latest / confidence / manual
default_privacy: personal      # 默认隐私级别
auto_memorize: true            # 自动记忆开关

# L4 内化记忆配置
kv_cache_threshold: 10         # KV Cache 自动预填充阈值（访问次数）
kv_cache_max: 100              # KV Cache 最大条目数
lora_base_model: Qwen2.5-7B    # LoRA 基座模型
lora_rank: 16                  # LoRA 秩
lora_alpha: 32                 # LoRA alpha

# 同步配置
sync_mode: none                # 同步模式: none / file_lock / changelog
sync_interval: 30              # 同步间隔（秒）
sync_conflict_resolution: latest_wins  # 同步冲突解决策略
```

### 配置热重载

```python
# 手动触发重载
provider._config.reload()

# 获取当前配置
all_config = provider._config.values

# 修改配置
provider._config.set("budget_tokens", 6000)
provider._config.save()
```

## 高级用法

### 自定义检索权重

```python
# 基于反馈统计动态调整检索来源权重
weights = provider._feedback.get_source_weights(window=100)
provider._retriever.set_source_weights(weights)
```

### 隐私级别管理

```python
# 设置记忆隐私级别
provider.handle_tool_call("omni_govern", {
    "action": "set_privacy",
    "target": "memory_id_xxx",
    "params": {"level": "secret"}
})
```

### 多实例同步

```python
# 启用 changelog 同步模式
provider._config.set("sync_mode", "changelog")
provider._config.set("sync_interval", 30)

# 查看同步状态
provider.handle_tool_call("omni_govern", {
    "action": "sync_status"
})
```

### 深层反思（L3）

```python
# 触发反思，指定性格倾向
result = provider.handle_tool_call("omni_reflect", {
    "query": "用户项目管理风格",
    "disposition": {
        "skepticism": 4,    # 高怀疑度，更谨慎
        "literalness": 3,   # 中等字面度
        "empathy": 2        # 低共情，更客观
    }
})
```

## 性能优化

### 检索性能

- **向量检索**：ChromaDB 持久化，首次加载后内存缓存
- **BM25 重建**：会话启动时从 SQLite 索引批量重建（2000 条/秒）
- **预热机制**：前 500 条记忆自动预热到内存
- **异步预取**：后台线程预检索下一轮可能需要的记忆

### 存储优化

- **Saga 事务**：派生数据（索引/检索/图谱）异步写入，避免阻塞主流程
- **批量刷新**：存储缓冲批量落盘，减少 IO 次数
- **KV Cache**：高频访问记忆自动缓存，毫秒级响应

### Token 优化

- **ContextManager 精炼**：所有注入记忆自动压缩为 ≤60 字摘要
- **自适应预算**：根据查询复杂度动态调整上下文预算
- **语义去重**：注入前自动去重，避免重复信息浪费 Token

## 测试

```bash
# 运行综合测试
python test_omnimem_comprehensive.py

# 运行质量修复测试
python test_qual2_fix.py

# 运行特定模块测试
python -m pytest tests/ -v
```

## 依赖说明

### 核心依赖

| 包名 | 版本 | 用途 |
|:---:|:---:|:---|
| chromadb | >=0.4.0,<0.7.0 | 向量数据库存储 |
| rank-bm25 | >=0.2.0,<0.3.0 | BM25 关键词检索 |
| tiktoken | >=0.7.0 | Token 计数 |
| pyyaml | >=6.0 | YAML 配置解析 |

### 可选依赖

| 包名 | 版本 | 用途 | 未安装时的行为 |
|:---:|:---:|:---|:---|
| sentence-transformers | >=2.2.0 | 语义嵌入缓存 | 使用默认嵌入模型 |
| cryptography | >=42.0.0 | 加密隐私数据 | 自动降级为明文标记 |

## 设计理念

### 人类记忆模拟

OmniMem 的设计灵感来源于人类记忆系统：

- **感知层 (L0)**：类似感觉记忆，瞬时捕获环境信号
- **工作记忆 (L1)**：类似短期记忆，容量有限（4±1 组块），持续激活
- **结构化记忆 (L2)**：类似长期记忆的海马体索引，支持快速检索
- **深层记忆 (L3)**：类似语义记忆，通过 Consolidation 从经验中提取抽象知识
- **内化记忆 (L4)**：类似程序性记忆，通过反复训练成为"本能"

### 三层分离设计

受 Anthropic managed-agents 启发，OmniMem 采用存储层/上下文管理层/上下文窗口的三层分离：

1. **存储层**：全量检索，不丢弃任何信息
2. **上下文管理层**：精炼、去重、预算控制
3. **上下文窗口**：只注入精炼摘要，细节通过 `omni_detail` 按需拉取

### 安全设计

- **反递归防护**：拒绝存储系统注入内容，防止 `prefetch→store→prefetch` 循环
- **输入净化**：剥离系统注入内容后再做感知分析
- **安全扫描**：Unicode 归一化、编码绕过检测
- **Saga 补偿**：会话结束前重试未完成的派生数据写入

## 路线图

- [x] 五层记忆架构
- [x] 混合检索引擎（向量+BM25+RRF）
- [x] 完整治理引擎
- [x] Saga 事务协调
- [x] 多实例同步
- [x] 内置 memory 工具兼容
- [ ] 分布式部署支持
- [ ] Web 管理界面
- [ ] 记忆可视化（知识图谱渲染）
- [ ] 自动 LoRA 训练流水线

## 贡献

欢迎提交 Issue 和 Pull Request！

### 开发流程

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 代码规范

- 使用 Black 格式化代码
- 遵循 PEP 8 规范
- 新增功能需附带测试
- 更新相关文档

## 许可证

MIT License

---

> **提示**：如果你从现有记忆系统迁移到 OmniMem，只需修改 `config.yaml` 中的 `memory.provider` 为 `omnimem`，所有现有 `memory` 工具调用将自动兼容。
