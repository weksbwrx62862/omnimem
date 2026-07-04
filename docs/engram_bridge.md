# EngramBridge - Plur 共享记忆集成到 OmniMem

## 概述

EngramBridge 是一个桥接层，将 Plur 的共享记忆能力集成到 OmniMem 系统中，解决跨实例记忆同步问题。

### 核心特性

- **记忆格式转换**: 将 OmniMem 记忆转换为 Plur Engram 格式
- **跨实例同步**: 支持多个 OmniMem 实例之间的记忆同步
- **联邦查询**: 聚合多个实例的记忆进行统一查询
- **冲突解决**: 支持多种冲突解决策略
- **配置管理**: 灵活的配置管理

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Federation                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Instance 1  │  │  Instance 2  │  │  Instance 3  │      │
│  │  (OmniMem)   │  │  (OmniMem)   │  │  (OmniMem)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         ▼                 ▼                 ▼               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              EngramBridge Layer                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   Bridge 1  │  │   Bridge 2  │  │   Bridge 3  │  │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │  │
│  └─────────┼────────────────┼────────────────┼──────────┘  │
│            │                │                │              │
│            ▼                ▼                ▼              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Shared Memory Sync                       │  │
│  │  ┌─────────────────────────────────────────────┐     │  │
│  │  │          Plur Shared Memory                  │     │  │
│  │  │         (Central Storage)                    │     │  │
│  │  └─────────────────────────────────────────────┘     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 安装

EngramBridge 已集成到 OmniMem 核心模块中，无需额外安装。

### 2. 基本使用

```python
import asyncio
from omnimem.core import (
    create_engram_bridge,
    create_shared_memory_sync,
    create_memory_federation
)

async def main():
    # 创建 EngramBridge
    bridge = create_engram_bridge(
        instance_id="my-instance",
        plur_endpoint="http://localhost:8080"
    )
    
    # 模拟 OmniMem 记忆
    memories = [
        {
            "memory_id": "mem-001",
            "content": "用户偏好简洁回复",
            "type": "preference",
            "confidence": 5
        }
    ]
    
    # 同步到 Plur
    result = await bridge.sync_to_plur(memories)
    print(f"同步结果: {result}")

asyncio.run(main())
```

### 3. 同步管理

```python
# 创建同步管理器
sync = create_shared_memory_sync(
    bridge=bridge,
    auto_sync_interval=300,  # 5分钟
    conflict_strategy="merge"
)

# 启动自动同步
await sync.start()

# 手动同步
result = await sync.perform_sync()

# 停止同步
await sync.stop()
```

### 4. 联邦查询

```python
# 创建联邦
federation = create_memory_federation()

# 注册实例
federation.register_instance("main", bridge1)
federation.register_instance("mobile", bridge2)

# 联邦查询
results = await federation.federated_query("用户偏好")

# 聚合记忆
aggregated = await federation.aggregate_memories("技术知识")
```

## 核心组件

### EngramBridge

桥接层，负责：
- 将 OmniMem 记忆转换为 Plur Engram 格式
- 从 Plur 共享记忆获取外部 Engram
- 处理记忆冲突和合并
- 维护本地-远程记忆映射

```python
bridge = create_engram_bridge(
    instance_id="my-instance",      # 实例 ID
    plur_endpoint="http://...",     # Plur 端点
    sync_interval=300,              # 同步间隔
    auto_sync=True                  # 自动同步
)
```

### SharedMemorySync

同步管理器，负责：
- 定期同步本地记忆到 Plur
- 从 Plur 拉取新记忆
- 处理记忆冲突
- 维护同步状态

```python
sync = create_shared_memory_sync(
    bridge=bridge,
    auto_sync_interval=300,
    conflict_strategy="merge"  # local, remote, newest, highest_confidence, merge
)
```

### MemoryFederation

联邦查询，负责：
- 跨多个 OmniMem 实例查询记忆
- 聚合多实例记忆
- 提供统一的查询接口

```python
federation = create_memory_federation()
federation.register_instance("instance-1", bridge1)
federation.register_instance("instance-2", bridge2)

# 联邦查询
results = await federation.federated_query("query")

# 聚合记忆
aggregated = await federation.aggregate_memories("query")
```

## 配置管理

### 配置文件位置

```
~/.hermes/plugins/omnimem/plur_config.json
```

### 配置示例

```json
{
  "plur": {
    "endpoint": "http://localhost:8080",
    "api_version": "v1",
    "timeout": 30,
    "retry_count": 3,
    "retry_delay": 5
  },
  "sync": {
    "auto_sync_enabled": true,
    "sync_interval_seconds": 300,
    "batch_size": 50,
    "conflict_strategy": "merge",
    "max_conflict_queue_size": 100
  },
  "federation": {
    "enabled": true,
    "max_instances": 10,
    "query_timeout": 60,
    "aggregation_strategy": "confidence_weighted"
  },
  "cache": {
    "local_cache_size": 1000,
    "remote_cache_size": 5000,
    "cache_ttl_seconds": 3600
  }
}
```

### 配置管理代码

```python
from omnimem.core import get_config, reload_config

# 获取配置
config = get_config()

# 修改配置
config.plur_endpoint = "http://new-endpoint:8080"
config.sync_interval = 600
config.auto_sync_enabled = False

# 保存配置
config.save_config()

# 重新加载配置
reloaded_config = reload_config()
```

## 冲突解决策略

EngramBridge 支持多种冲突解决策略：

| 策略 | 说明 |
|------|------|
| `local` | 保留本地版本 |
| `remote` | 保留远程版本 |
| `newest` | 保留最新版本 |
| `highest_confidence` | 保留置信度最高的版本 |
| `merge` | 合并两个版本（默认） |

```python
# 使用不同策略解决冲突
resolved = await bridge.resolve_conflict(
    local_engram,
    remote_engram,
    strategy="merge"  # 或 local, remote, newest, highest_confidence
)
```

## API 参考

### EngramBridge

#### 方法

- `convert_to_engram(omni_memory)`: 转换 OmniMem 记忆为 Engram
- `sync_to_plur(memories)`: 同步记忆到 Plur
- `fetch_from_plur(query, tags, since, limit)`: 从 Plur 获取记忆
- `resolve_conflict(local, remote, strategy)`: 解决记忆冲突
- `get_sync_status()`: 获取同步状态

### SharedMemorySync

#### 方法

- `start()`: 启动自动同步
- `stop()`: 停止自动同步
- `perform_sync()`: 执行一次同步

### MemoryFederation

#### 方法

- `register_instance(instance_id, bridge)`: 注册实例
- `federated_query(query, instances, limit)`: 联邦查询
- `aggregate_memories(query, strategy)`: 聚合记忆
- `get_federation_status()`: 获取联邦状态

## 示例

### 基本使用

```python
# examples/engram_bridge_example.py
python3 ~/.hermes/plugins/omnimem/examples/engram_bridge_example.py
```

### 测试

```python
# 运行测试
cd ~/.hermes/plugins/omnimem
pytest core/test_engram_bridge.py -v
```

## 部署指南

### 1. 部署 Plur 服务器

```bash
# 安装 Plur
pip install plur

# 启动 Plur 服务器
plur-server start --port 8080
```

### 2. 配置 OmniMem

```bash
# 编辑配置文件
vim ~/.hermes/plugins/omnimem/plur_config.json
```

### 3. 启动同步

```python
# 在应用中启动同步
from omnimem.core import create_engram_bridge, create_shared_memory_sync

bridge = create_engram_bridge("my-instance")
sync = create_shared_memory_sync(bridge)
await sync.start()
```

## 故障排除

### 常见问题

1. **连接失败**
   - 检查 Plur 服务器是否运行
   - 检查网络连接
   - 检查防火墙设置

2. **同步失败**
   - 检查 Plur 服务器日志
   - 检查 OmniMem 日志
   - 检查配置文件

3. **冲突解决失败**
   - 检查冲突策略配置
   - 检查记忆格式
   - 查看冲突队列

### 日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能优化

1. **批量同步**: 使用 `batch_size` 配置批量同步
2. **缓存优化**: 调整缓存大小和 TTL
3. **并行查询**: 使用联邦查询并行获取记忆
4. **增量同步**: 使用 `since` 参数进行增量同步

## 安全考虑

1. **认证**: 配置 Plur 服务器的认证
2. **加密**: 使用 HTTPS 连接
3. **权限**: 配置记忆的访问权限
4. **审计**: 启用操作日志

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

## 联系方式

- 项目地址: https://github.com/your-org/omnimem
- 问题反馈: https://github.com/your-org/omnimem/issues
- 文档: https://omnimem.readthedocs.io
