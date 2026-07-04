"""
EngramBridge 使用示例
演示如何将 Plur 共享记忆集成到 OmniMem
"""

import asyncio
import os
import sys

# 添加路径
sys.path.insert(0, os.path.expanduser("~/.hermes/plugins/omnimem"))

from core.engram_bridge import (
    Engram,
    create_engram_bridge,
    create_memory_federation,
    create_shared_memory_sync,
)
from core.plur_config import get_config, reload_config


async def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("示例 1: 基本使用")
    print("=" * 60)

    # 1. 创建 EngramBridge
    bridge = create_engram_bridge(
        instance_id="hermes-main",
        plur_endpoint="http://localhost:8080"
    )

    print(f"✅ 创建 EngramBridge: {bridge.instance_id}")

    # 2. 模拟 OmniMem 记忆
    omni_memories = [
        {
            "memory_id": "mem-001",
            "content": "用户偏好简洁直接的中文回复",
            "type": "preference",
            "confidence": 5,
            "stored_at": "2026-06-13T10:00:00",
            "wing": "personal",
            "privacy": "personal"
        },
        {
            "memory_id": "mem-002",
            "content": "RTK Token 压缩插件已安装，版本 0.42.4",
            "type": "fact",
            "confidence": 5,
            "stored_at": "2026-06-13T11:00:00",
            "wing": "memory",
            "privacy": "personal"
        }
    ]

    # 3. 转换为 Engram 格式
    print("\n📝 转换 OmniMem 记忆为 Engram 格式:")
    engrams = []
    for memory in omni_memories:
        engram = bridge.convert_to_engram(memory)
        engrams.append(engram)
        print(f"  - {engram.id}: {engram.content[:30]}... (置信度: {engram.confidence})")

    # 4. 同步到 Plur
    print("\n🔄 同步到 Plur:")
    result = await bridge.sync_to_plur(omni_memories)
    print(f"  - 成功: {result.success}")
    print(f"  - 同步数量: {result.synced_count}")
    print(f"  - 失败数量: {result.failed_count}")
    print(f"  - 耗时: {result.duration_ms:.2f}ms")

    return bridge


async def example_sync_management(bridge):
    """同步管理示例"""
    print("\n" + "=" * 60)
    print("示例 2: 同步管理")
    print("=" * 60)

    # 1. 创建同步管理器
    sync = create_shared_memory_sync(
        bridge=bridge,
        auto_sync_interval=60,  # 1分钟同步一次
        conflict_strategy="merge"
    )

    print("✅ 创建同步管理器")
    print(f"  - 同步间隔: {sync.auto_sync_interval}秒")
    print(f"  - 冲突策略: {sync.conflict_strategy}")

    # 2. 执行一次同步
    print("\n🔄 执行同步:")
    result = await sync.perform_sync()
    print(f"  - 同步结果: {'成功' if result.success else '失败'}")
    print(f"  - 同步数量: {result.synced_count}")
    print(f"  - 冲突数量: {result.conflict_count}")

    # 3. 获取同步状态
    print("\n📊 同步状态:")
    status = bridge.get_sync_status()
    print(f"  - 实例ID: {status['instance_id']}")
    print(f"  - 本地缓存: {status['local_cache_size']} 条")
    print(f"  - 远程缓存: {status['remote_cache_size']} 条")

    return sync


async def example_federation():
    """联邦查询示例"""
    print("\n" + "=" * 60)
    print("示例 3: 联邦查询")
    print("=" * 60)

    # 1. 创建多个实例
    bridge1 = create_engram_bridge("hermes-main")
    bridge2 = create_engram_bridge("hermes-mobile")
    bridge3 = create_engram_bridge("hermes-desktop")

    # 2. 为每个实例添加一些记忆
    memories = [
        {"content": "主实例的知识", "type": "fact", "confidence": 4},
        {"content": "移动端的偏好", "type": "preference", "confidence": 5},
        {"content": "桌面端的配置", "type": "fact", "confidence": 3}
    ]

    bridges = [bridge1, bridge2, bridge3]
    for i, (bridge, memory) in enumerate(zip(bridges, memories)):
        memory["memory_id"] = f"mem-{i:03d}"
        memory["stored_at"] = "2026-06-13T10:00:00"
        await bridge.sync_to_plur([memory])

    # 3. 创建联邦
    federation = create_memory_federation()
    federation.register_instance("main", bridge1)
    federation.register_instance("mobile", bridge2)
    federation.register_instance("desktop", bridge3)

    print(f"✅ 创建联邦，包含 {len(federation._bridges)} 个实例")

    # 4. 联邦查询
    print("\n🔍 联邦查询 '知识':")
    results = await federation.federated_query("知识")

    for instance_id, engrams in results.items():
        print(f"  - {instance_id}: {len(engrams)} 条记忆")
        for engram in engrams:
            print(f"    • {engram.content[:30]}...")

    # 5. 聚合记忆
    print("\n📚 聚合记忆:")
    aggregated = await federation.aggregate_memories("偏好")
    print(f"  - 共找到 {len(aggregated)} 条相关记忆")

    # 6. 获取联邦状态
    print("\n📊 联邦状态:")
    status = federation.get_federation_status()
    print(f"  - 实例数量: {status['instance_count']}")
    print(f"  - 注册实例: {', '.join(status['registered_instances'])}")

    return federation


async def example_conflict_resolution():
    """冲突解决示例"""
    print("\n" + "=" * 60)
    print("示例 4: 冲突解决")
    print("=" * 60)

    bridge = create_engram_bridge("hermes-main")

    # 创建冲突的记忆
    local_engram = Engram(
        id="conflict-001",
        content="本地版本：用户喜欢简洁回复",
        memory_type="preference",
        confidence=4,
        source_instance="hermes-main",
        created_at="2026-06-13T10:00:00",
        updated_at="2026-06-13T11:00:00",
        metadata={"version": 1},
        tags=["preference"],
        relationships=[]
    )

    remote_engram = Engram(
        id="conflict-001",
        content="远程版本：用户偏好详细解释",
        memory_type="preference",
        confidence=5,
        source_instance="hermes-mobile",
        created_at="2026-06-13T09:00:00",
        updated_at="2026-06-13T12:00:00",
        metadata={"version": 2},
        tags=["preference", "detail"],
        relationships=[]
    )

    print("🔄 冲突记忆:")
    print(f"  - 本地: {local_engram.content}")
    print(f"  - 远程: {remote_engram.content}")

    # 测试不同解决策略
    strategies = ["local", "remote", "newest", "highest_confidence", "merge"]

    print("\n📊 不同策略的解决结果:")
    for strategy in strategies:
        resolved = await bridge.resolve_conflict(
            local_engram, remote_engram, strategy=strategy
        )
        print(f"  - {strategy:20s}: {resolved.content[:40]}... (置信度: {resolved.confidence})")

    return bridge


async def example_config_management():
    """配置管理示例"""
    print("\n" + "=" * 60)
    print("示例 5: 配置管理")
    print("=" * 60)

    # 1. 获取配置
    config = get_config()

    print("📋 当前配置:")
    print(f"  - Plur 端点: {config.plur_endpoint}")
    print(f"  - 同步间隔: {config.sync_interval}秒")
    print(f"  - 自动同步: {'启用' if config.auto_sync_enabled else '禁用'}")
    print(f"  - 冲突策略: {config.conflict_strategy}")

    # 2. 修改配置
    print("\n🔧 修改配置:")
    config.plur_endpoint = "http://plur.example.com:8080"
    config.sync_interval = 600
    config.auto_sync_enabled = False

    print(f"  - 新端点: {config.plur_endpoint}")
    print(f"  - 新间隔: {config.sync_interval}秒")
    print(f"  - 自动同步: {'启用' if config.auto_sync_enabled else '禁用'}")

    # 3. 保存配置
    print("\n💾 保存配置...")
    config.save_config()
    print("  ✅ 配置已保存")

    # 4. 重新加载
    print("\n🔄 重新加载配置...")
    reloaded_config = reload_config()
    print(f"  - 端点: {reloaded_config.plur_endpoint}")

    return config


async def main():
    """主函数"""
    print("🚀 EngramBridge 集成示例")
    print("=" * 60)

    try:
        # 运行所有示例
        bridge = await example_basic_usage()
        await example_sync_management(bridge)
        await example_federation()
        bridge = await example_conflict_resolution()
        await example_config_management()

        print("\n" + "=" * 60)
        print("✅ 所有示例运行完成！")
        print("=" * 60)

        print("\n📝 总结:")
        print("1. ✅ EngramBridge 已集成到 OmniMem")
        print("2. ✅ 支持记忆格式转换")
        print("3. ✅ 支持跨实例同步")
        print("4. ✅ 支持联邦查询")
        print("5. ✅ 支持冲突解决")
        print("6. ✅ 支持配置管理")

        print("\n🎯 下一步:")
        print("1. 部署 Plur 服务器")
        print("2. 配置多实例连接")
        print("3. 启用自动同步")
        print("4. 测试联邦查询")

    except Exception as e:
        print(f"\n❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
