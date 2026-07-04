"""
OmniMem SDK 基本使用示例

演示如何通过 OmniMemSDK 进行记忆的写入、检索、治理和反思操作。

前置条件：
    pip install -r requirements.txt

运行方式：
    cd ~/.hermes/plugins/omnimem
    python -m examples.sdk_basic
"""

from __future__ import annotations

import os
import sys

# 添加项目根目录到 sys.path，确保可以直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnimem.sdk import OmniMemSDK


def example_memorize(sdk: OmniMemSDK) -> None:
    """示例：存储不同类型的记忆"""
    print("=" * 60)
    print("1. 存储记忆 (memorize)")
    print("=" * 60)

    # 存储事实型记忆
    result = sdk.memorize(
        content="项目使用 Python 3.12 和 FastAPI 框架",
        memory_type="fact",
        confidence=5,
        scope="project",
        privacy="team",
    )
    print(f"  事实记忆: status={result['status']}, id={result.get('memory_id', 'N/A')}")

    # 存储偏好型记忆
    result = sdk.memorize(
        content="用户偏好深色主题，在所有应用中启用 dark mode",
        memory_type="preference",
        confidence=4,
        scope="personal",
        privacy="personal",
    )
    print(f"  偏好记忆: status={result['status']}, id={result.get('memory_id', 'N/A')}")

    # 存储纠正型记忆（覆盖旧信息）
    result = sdk.memorize(
        content="数据库已从 PostgreSQL 迁移到 MySQL 8.0",
        memory_type="correction",
        confidence=5,
        scope="project",
        privacy="team",
    )
    print(f"  纠正记忆: status={result['status']}, id={result.get('memory_id', 'N/A')}")

    # 存储技能型记忆
    result = sdk.memorize(
        content="部署流程：git push origin main → GitHub Actions 自动构建 → Docker 镜像推送到 Registry",
        memory_type="skill",
        confidence=4,
        scope="project",
        privacy="team",
    )
    print(f"  技能记忆: status={result['status']}, id={result.get('memory_id', 'N/A')}")

    # 语义去重演示：存储与已有记忆高度相似的内容
    result = sdk.memorize(
        content="用户偏好深色主题",
        memory_type="preference",
        confidence=4,
    )
    print(f"  重复记忆: status={result['status']} (预期为 duplicate_skipped)")


def example_recall(sdk: OmniMemSDK) -> None:
    """示例：检索记忆"""
    print("\n" + "=" * 60)
    print("2. 检索记忆 (recall)")
    print("=" * 60)

    # RAG 模式：快速混合检索（毫秒级）
    result = sdk.recall("用户的技术偏好", mode="rag", max_tokens=1500)
    print(f"  RAG 模式: status={result['status']}, count={result.get('count', 0)}")
    for mem in result.get("memories", []):
        print(f"    - [{mem.get('type', '?')}] {mem.get('summary', mem.get('content', '')[:50])}")

    # LLM 模式：深度推理检索（秒级）
    result = sdk.recall("项目的技术栈和部署方式", mode="llm")
    print(f"  LLM 模式: status={result['status']}, count={result.get('count', 0)}")
    for mem in result.get("memories", []):
        print(f"    - [{mem.get('type', '?')}] {mem.get('summary', mem.get('content', '')[:50])}")

    # 无结果查询
    result = sdk.recall("不存在的量子计算项目")
    print(f"  无结果查询: status={result['status']}")


def example_govern(sdk: OmniMemSDK) -> None:
    """示例：治理操作"""
    print("\n" + "=" * 60)
    print("3. 治理操作 (govern)")
    print("=" * 60)

    # 查看遗忘曲线状态
    result = sdk.govern(action="forgetting_status")
    forgetting = result.get("forgetting", {})
    print(f"  遗忘状态: active={forgetting.get('active_count', 0)}, "
          f"consolidating={forgetting.get('consolidating_count', 0)}, "
          f"archived={forgetting.get('archived_count', 0)}")

    # 扫描冲突
    result = sdk.govern(action="resolve_conflict")
    print(f"  冲突扫描: status={result.get('status', 'N/A')}")

    # 导出记忆
    result = sdk.govern(
        action="export_memories",
        params={"output_path": "/tmp/omnimem_backup.json", "format": "json"},
    )
    print(f"  导出记忆: status={result.get('status', 'N/A')}, "
          f"count={result.get('count', 0)}")

    # 查看同步状态
    result = sdk.govern(action="sync_status")
    print(f"  同步状态: status={result.get('status', 'N/A')}")


def example_reflect(sdk: OmniMemSDK) -> None:
    """示例：深层反思"""
    print("\n" + "=" * 60)
    print("4. 深层反思 (reflect)")
    print("=" * 60)

    # 基本反思
    result = sdk.reflect(query="用户的技术偏好模式")
    print(f"  反思状态: status={result.get('status', 'N/A')}")
    if "observation" in result:
        print(f"  观察结果: {result['observation'][:80]}...")
    if "mental_model" in result:
        print(f"  心智模型: {result['mental_model'][:80]}...")

    # 带性格参数的反思
    result = sdk.reflect(
        query="项目的技术选型决策",
        disposition={
            "skepticism": 4,   # 高怀疑度，谨慎推理
            "literalness": 3,  # 适中字面程度
            "empathy": 2,      # 低共情，事实导向
        },
    )
    print(f"  带性格反思: status={result.get('status', 'N/A')}")
    if "observation" in result:
        print(f"  观察结果: {result['observation'][:80]}...")


def example_detail(sdk: OmniMemSDK) -> None:
    """示例：记忆详情查询"""
    print("\n" + "=" * 60)
    print("5. 记忆详情 (detail)")
    print("=" * 60)

    # 列出当前 turn 注入的记忆
    result = sdk.detail_list()
    print(f"  当前记忆列表: count={result.get('count', 0)}")
    for mem in result.get("memories", [])[:3]:
        print(f"    - {mem.get('memory_id', 'N/A')}: {mem.get('content', '')[:50]}")

    # 获取单条记忆详情（如果存在）
    if result.get("memories"):
        first_id = result["memories"][0].get("memory_id")
        if first_id:
            detail = sdk.detail(memory_id=first_id)
            print(f"  记忆详情: id={detail.get('memory_id', 'N/A')}, "
                  f"type={detail.get('type', 'N/A')}, "
                  f"privacy={detail.get('privacy', 'N/A')}")

    # 查询事件日志
    events = sdk.detail_events(from_turn=0, to_turn=10)
    print(f"  事件日志: count={events.get('count', 0)}")


def example_health_check(sdk: OmniMemSDK) -> None:
    """示例：健康检查"""
    print("\n" + "=" * 60)
    print("6. 健康检查")
    print("=" * 60)

    result = sdk.health_check()
    print(f"  状态: {result['status']}")
    print(f"  会话 ID: {result['session_id']}")
    print(f"  存储目录: {result['data_dir']}")
    print(f"  存储可访问: {result.get('store_accessible', 'N/A')}")


def main() -> None:
    """主函数：依次运行所有示例"""
    print("OmniMem SDK 基本使用示例")
    print("=" * 60)

    # 使用上下文管理器初始化 SDK，确保资源正确释放
    try:
        with OmniMemSDK() as sdk:
            example_memorize(sdk)
            example_recall(sdk)
            example_govern(sdk)
            example_reflect(sdk)
            example_detail(sdk)
            example_health_check(sdk)

        print("\n" + "=" * 60)
        print("所有示例运行完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
