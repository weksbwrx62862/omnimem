"""
OmniMem Hermes 插件集成示例

演示如何在 Hermes 插件系统中使用 OmniMem，包括工具注册和调用流程。

前置条件：
    - Hermes 框架已安装
    - OmniMem 已作为插件安装到 ~/.hermes/plugins/omnimem
    - ~/.hermes/config.yaml 中已启用 omnimem 插件

运行方式：
    cd ~/.hermes/plugins/omnimem
    python -m examples.hermes_integration
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_tool_registration() -> None:
    """示例：OmniMem 在 Hermes 中的工具注册流程"""
    print("=" * 60)
    print("1. 工具注册流程")
    print("=" * 60)

    # OmniMem 在 Hermes 中注册的 7 个工具
    tools = [
        {
            "name": "omni_memorize",
            "description": "主动存储记忆，支持 fact/preference/correction/skill/procedural/event 类型",
            "required_params": ["content"],
        },
        {
            "name": "omni_recall",
            "description": "检索相关记忆，支持 rag（快速）和 llm（深度）两种模式",
            "required_params": ["query"],
        },
        {
            "name": "omni_compact",
            "description": "手动触发上下文压缩，控制 Token 预算",
            "required_params": [],
        },
        {
            "name": "omni_reflect",
            "description": "深层反思，将原始事实整合为观察和心智模型",
            "required_params": ["query"],
        },
        {
            "name": "omni_govern",
            "description": "治理操作：冲突仲裁、隐私分级、遗忘曲线、溯源追踪等",
            "required_params": ["action"],
        },
        {
            "name": "omni_detail",
            "description": "按 ID 获取记忆完整详情，支持 list/get/events 三种操作",
            "required_params": [],
        },
        {
            "name": "memory",
            "description": "兼容内置 memory 工具的简化接口，路由到 omni_memorize",
            "required_params": ["action", "target"],
        },
    ]

    print("  OmniMem 注册的工具列表：")
    for tool in tools:
        params = ", ".join(tool["required_params"]) if tool["required_params"] else "无必填参数"
        print(f"    - {tool['name']}: {tool['description']}")
        print(f"      必填参数: {params}")


def example_tool_call_flow() -> None:
    """示例：Hermes 中 Agent 调用 OmniMem 工具的完整流程"""
    print("\n" + "=" * 60)
    print("2. 工具调用流程")
    print("=" * 60)

    # 模拟 Agent 调用 omni_memorize 的完整流程
    print("\n  场景 A：Agent 存储用户偏好")
    print("  " + "-" * 50)

    # Step 1: Agent 决定存储记忆
    tool_call = {
        "tool": "omni_memorize",
        "arguments": {
            "content": "用户喜欢使用 TypeScript 进行前端开发",
            "memory_type": "preference",
            "confidence": 4,
            "scope": "personal",
            "privacy": "personal",
        },
    }
    print("  1. Agent 发起工具调用:")
    print(f"     {json.dumps(tool_call, ensure_ascii=False, indent=6)}")

    # Step 2: Hermes 路由到 OmniMem handler
    print("  2. Hermes 路由到 OmniMem → handlers/memorize.py")
    print("     - 安全扫描 (SecurityValidator)")
    print("     - 反递归检查")
    print("     - 语义去重检查")
    print("     - 冲突检测")
    print("     - 写入 L2 结构化记忆")
    print("     - 更新三级索引 (向量 + BM25 + 元数据)")
    print("     - Saga 异步协调")

    # Step 3: 返回结果
    mock_result = {
        "status": "stored",
        "memory_id": "mem-ts001",
        "wing": "personal",
        "room": "preferences-dev",
        "type": "preference",
        "privacy": "personal",
        "kv_cached": False,
    }
    print("  3. 返回结果:")
    print(f"     {json.dumps(mock_result, ensure_ascii=False, indent=6)}")

    # 模拟 Agent 调用 omni_recall
    print("\n  场景 B：Agent 检索用户偏好")
    print("  " + "-" * 50)

    tool_call = {
        "tool": "omni_recall",
        "arguments": {
            "query": "用户的前端开发偏好",
            "mode": "rag",
            "max_tokens": 1500,
        },
    }
    print("  1. Agent 发起工具调用:")
    print(f"     {json.dumps(tool_call, ensure_ascii=False, indent=6)}")

    print("  2. Hermes 路由到 OmniMem → handlers/recall.py")
    print("     - 查询扩展 (同义词)")
    print("     - 混合检索 (向量 + BM25 + RRF 融合)")
    print("     - 隐私过滤 + 时间衰减")
    print("     - ContextManager 精炼压缩")

    mock_result = {
        "status": "found",
        "query": "用户的前端开发偏好",
        "count": 1,
        "memories": [
            {
                "memory_id": "mem-ts001",
                "content": "用户喜欢使用 TypeScript 进行前端开发",
                "type": "preference",
                "score": 0.92,
                "summary": "偏好 TypeScript 前端开发",
            }
        ],
        "hint": "Use omni_detail with a memory_id to fetch full content.",
    }
    print("  3. 返回结果:")
    print(f"     {json.dumps(mock_result, ensure_ascii=False, indent=6)}")

    # 模拟 Agent 调用 omni_govern
    print("\n  场景 C：Agent 执行治理操作")
    print("  " + "-" * 50)

    tool_call = {
        "tool": "omni_govern",
        "arguments": {
            "action": "set_privacy",
            "target": "mem-ts001",
            "params": {"level": "team"},
        },
    }
    print("  1. Agent 发起工具调用:")
    print(f"     {json.dumps(tool_call, ensure_ascii=False, indent=6)}")

    print("  2. Hermes 路由到 OmniMem → handlers/govern.py")
    print("     - RBAC 权限校验")
    print("     - PrivacyManager 更新隐私级别")
    print("     - 同步更新 index/store/wing")

    mock_result = {
        "status": "updated",
        "memory_id": "mem-ts001",
        "privacy": "team",
        "wing": "team",
    }
    print("  3. 返回结果:")
    print(f"     {json.dumps(mock_result, ensure_ascii=False, indent=6)}")


def example_memory_compat_layer() -> None:
    """示例：兼容内置 memory 工具的使用方式"""
    print("\n" + "=" * 60)
    print("3. 兼容内置 memory 工具")
    print("=" * 60)

    # memory 工具是 omni_memorize 的兼容层
    # 适用于从内置 memory 工具迁移到 OmniMem 的场景

    print("  memory 工具的 action/target 映射：")
    print("    - action=add, target=memory  → omni_memorize(memory_type='fact')")
    print("    - action=add, target=user    → omni_memorize(memory_type='preference')")
    print("    - action=replace             → 归档旧条目 + 写入新内容")
    print("    - action=remove              → 软删除（走遗忘曲线归档）")

    # 示例调用
    add_call = {
        "tool": "memory",
        "arguments": {
            "action": "add",
            "target": "user",
            "content": "用户习惯在晚上进行代码审查",
        },
    }
    print("\n  新增记忆调用:")
    print(f"    {json.dumps(add_call, ensure_ascii=False)}")

    replace_call = {
        "tool": "memory",
        "arguments": {
            "action": "replace",
            "target": "user",
            "old_text": "晚上进行代码审查",
            "content": "用户习惯在上午进行代码审查（时间调整）",
        },
    }
    print("\n  替换记忆调用:")
    print(f"    {json.dumps(replace_call, ensure_ascii=False)}")


def example_prefetch_injection() -> None:
    """示例：Hermes 中 OmniMem 的预取注入机制"""
    print("\n" + "=" * 60)
    print("4. 预取注入机制")
    print("=" * 60)

    print("  OmniMem 在 Hermes 中的自动注入流程：")
    print()
    print("  每个 Turn 开始时：")
    print("    1. PerceptionEngine (L0) 分析当前对话上下文")
    print("    2. 预测可能需要的记忆，触发预取")
    print("    3. HybridRetriever 执行混合检索")
    print("    4. ContextManager 精炼压缩结果")
    print("    5. 将精炼后的记忆摘要注入 Agent 上下文")
    print()
    print("  注入格式（示例）：")
    injection_example = """  [OmniMem Context]
  - mem-abc123: 用户偏好深色主题 (preference, score=0.89)
  - mem-def456: 项目使用 FastAPI 框架 (fact, score=0.85)
  Use omni_detail to fetch full content."""
    print(injection_example)
    print()
    print("  Agent 可通过 omni_detail 按需拉取完整内容，避免上下文膨胀。")


def main() -> None:
    """主函数"""
    print("OmniMem Hermes 插件集成示例")
    print("=" * 60)
    print()
    print("注意：本示例展示 Hermes 框架中的集成流程，")
    print("不需要实际运行 Hermes，仅演示调用模式和返回结构。")
    print()

    example_tool_registration()
    example_tool_call_flow()
    example_memory_compat_layer()
    example_prefetch_injection()

    print("\n" + "=" * 60)
    print("所有示例展示完成")
    print("=" * 60)
    print()
    print("在 Hermes 中启用 OmniMem：")
    print("  1. 确保 ~/.hermes/plugins/omnimem 目录存在")
    print("  2. 在 ~/.hermes/config.yaml 中添加：")
    print("     plugins:")
    print("       enabled:")
    print("         - omnimem")
    print("  3. 重启 Hermes，Agent 即可使用 7 个 OmniMem 工具")


if __name__ == "__main__":
    main()
