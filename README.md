# OmniMem

> 五层混合记忆系统：感知 → 工作 → 结构化 → 深层 → 内化

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

OmniMem 是一个为 AI Agent 设计的多层混合记忆系统，采用五层架构模拟人类记忆机制，并配备完整的治理引擎，实现高效、可靠、可溯源的记忆管理。

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

| 层级 | 名称 | 功能描述 |
|:---:|:---:|:---|
| L0 | **感知层** | 主动监控、信号检测、意图预测 |
| L1 | **工作记忆** | CoreBlock(常驻上下文) + Attachment(压缩后状态) |
| L2 | **结构化记忆** | Wing/Room 宫殿导航 + Drawer/Closet 双存储 |
| L3 | **深层记忆** | Consolidation(事实→观察→心智模型) + 知识图谱 |
| L4 | **内化记忆** | KV Cache(高频) + LoRA(深层) [可选] |

### 治理引擎（横切面）

- **冲突仲裁** — 多源记忆冲突自动检测与解决
- **时间衰减** — 基于时间的记忆重要性动态调整
- **遗忘曲线** — 模拟艾宾浩斯遗忘曲线的智能清理
- **隐私分级** — 个人/敏感/公开三级隐私控制
- **溯源追踪** — 完整记录记忆来源与变更历史
- **同步机制** — 多实例间的记忆同步与冲突解决

## 安装

### 环境要求

- Python 3.10+
- 依赖包见 `plugin.yaml`

### 安装步骤

1. 将本目录放入 `plugins/memory/omnimem/`
2. 安装依赖：

```bash
pip install chromadb>=0.4.0,<0.7.0
pip install rank-bm25>=0.2.0,<0.3.0
pip install tiktoken>=0.7.0
pip install pyyaml>=6.0
```

3. 在 `config.yaml` 中配置：

```yaml
memory:
  provider: omnimem
```

## 快速开始

```python
from plugins.memory.omnimem.provider import OmniMemProvider

# 初始化记忆系统
provider = OmniMemProvider()

# 存储记忆
provider.memorize("用户喜欢Python编程", tags=["preference", "tech"])

# 检索记忆
results = provider.recall("用户的编程偏好", mode="rag")
```

## 核心功能

### 7 个工具接口

| 工具 | 功能 |
|:---:|:---|
| `omni_memorize` | 主动存储记忆 |
| `omni_recall` | 主动检索记忆（RAG/LLM/关键词三种模式） |
| `omni_compact` | 压缩前准备 |
| `omni_reflect` | L3 深层反思（四步循环 + Disposition 性格修饰） |
| `omni_govern` | 治理操作（shade/conflict/kv_cache/stats等） |
| `omni_detail` | 按需拉取记忆细节（lazy provisioning） |
| `memory` | 兼容内置 memory 工具（add/replace/remove） |

## 项目结构

```
omnimem/
├── compression/          # 数据压缩与优化
│   ├── collapse.py
│   └── line_compress.py
├── context/              # 上下文管理
│   └── manager.py
├── core/                 # 核心组件
│   ├── block.py          # CoreBlock 常驻上下文
│   ├── attachment.py     # 压缩状态附件
│   ├── soul.py           # SoulSystem 灵魂系统
│   ├── budget.py         # BudgetManager 预算管理
│   ├── store_service.py  # 存储服务
│   ├── background.py     # 后台任务执行器
│   └── saga.py           # Saga 协调器
├── deep/                 # 深层记忆处理
│   └── consolidation.py  # 知识巩固
├── governance/           # 治理引擎
│   ├── conflict.py       # 冲突仲裁
│   ├── decay.py          # 时间衰减
│   ├── forgetting.py     # 遗忘曲线
│   ├── privacy.py        # 隐私管理
│   ├── provenance.py     # 溯源追踪
│   ├── sync.py           # 同步引擎
│   ├── auditor.py        # 审计器
│   ├── feedback.py       # 反馈收集
│   └── vector_clock.py   # 向量时钟
├── handlers/             # API 处理器
│   ├── memorize.py       # 记忆处理
│   ├── recall.py         # 检索处理
│   ├── govern.py         # 治理处理
│   └── schemas.py        # 工具模式定义
├── internalize/          # 内化记忆
│   ├── kv_cache.py       # KV 缓存
│   └── lora.py           # LoRA 训练
├── memory/               # 记忆存储
│   ├── wing_room.py      # Wing/Room 宫殿导航
│   ├── drawer_closet.py  # Drawer/Closet 双存储
│   ├── index.py          # 三级索引
│   └── markdown_store.py # Markdown 存储
├── perception/           # 感知层
│   └── engine.py         # 感知引擎
├── retrieval/            # 检索引擎
│   └── engine.py         # 混合检索器（向量+BM25+重排序）
├── utils/                # 工具函数
│   ├── llm_client.py     # LLM 客户端
│   └── security.py       # 安全验证
├── config.py             # 配置管理
├── provider.py           # 主入口：OmniMemProvider
├── plugin.yaml           # 插件配置
└── __init__.py           # 插件注册
```

## 配置选项

```python
# 默认配置
{
    "save_interval": 15,              # 自动保存间隔（秒）
    "retrieval_mode": "rag",          # 检索模式: rag / llm / keyword
    "vector_backend": "chromadb",     # 向量数据库后端
    "max_prefetch_tokens": 300,       # 最大预取token数
    "budget_tokens": 4000,            # 预算token数
    "fact_threshold": 10,             # Consolidation 触发阈值
    "enable_reranker": False,         # 是否启用重排序
    "conflict_strategy": "latest",    # 冲突解决策略
    "default_privacy": "personal",    # 默认隐私级别
    "auto_memorize": True,            # 自动记忆开关
    "kv_cache_threshold": 10,         # KV Cache 自动预填充阈值
    "kv_cache_max": 100,              # KV Cache 最大条目数
    "lora_base_model": "Qwen2.5-7B",  # LoRA 基座模型
    "lora_rank": 16,                  # LoRA 秩
    "lora_alpha": 32,                 # LoRA alpha
    "sync_mode": "none",              # 同步模式: none / file_lock / changelog
    "sync_interval": 30,              # 同步间隔（秒）
    "sync_conflict_resolution": "latest_wins",  # 同步冲突解决策略
}
```

## 测试

```bash
# 运行综合测试
python test_omnimem_comprehensive.py

# 运行质量修复测试
python test_qual2_fix.py
```

## 依赖说明

### 核心依赖
- **chromadb** — 向量数据库存储
- **rank-bm25** — BM25 关键词检索
- **tiktoken** — Token 计数
- **pyyaml** — YAML 配置解析

### 可选依赖
- **sentence-transformers** — 语义嵌入缓存（体积大，按需安装）
- **cryptography** — 加密隐私数据（未安装时自动降级为明文标记）

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
