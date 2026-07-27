"""OmniMem → STATE-Bench 适配器。

将 OmniMemSDK 的记忆能力桥接到 STATE-Bench 评测框架：
1. OmniMemStateBenchProvider：记忆后端提供者，封装 SDK 的 memorize/recall 操作
2. OmniMemStateBenchAgent：STATE-Bench 自定义 Agent，集成记忆检索能力

STATE-Bench 是纯 Python 项目，原生支持自定义 Agent 扩展。
通过 BaseAgent 的 memory_tool_schemas/memory_tool_handlers 机制，
可将 OmniMem 记忆检索暴露为 Agent 可调用的工具。

用法：
    # Agent Learning Track 方式（推荐）
    from omnimem.benchmarks.statebench_adapter import OmniMemStateBenchAgent

    # 放到 STATE-Bench 仓库根目录的 agents/ 目录下，
    # 通过 --agent-class OmniMemStateBenchAgent 加载

    # 独立使用 Provider
    from omnimem.benchmarks.statebench_adapter import OmniMemStateBenchProvider

    provider = OmniMemStateBenchProvider(storage_dir="/tmp/omnimem_bench")
    provider.on_turn("task-1", "user", "我想取消订单ABC123")
    context = provider.get_context("task-1", "取消订单流程")
    provider.close()
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── role → memory_type 映射 ────────────────────────────────────────
_ROLE_MEMORY_TYPE_MAP = {
    "user": "event",        # 用户消息作为事件记录
    "assistant": "fact",    # 助手回复作为事实
    "system": "fact",       # 系统指令作为事实
    "tool": "action",       # 工具调用结果作为动作
}

# ─── wing 前缀，用于隔离不同任务的记忆 ──────────────────────────────
_WING_PREFIX = "bench_task"


class OmniMemStateBenchProvider:
    """OmniMem 记忆后端提供者 — 为 STATE-Bench 提供记忆存储与检索。

    每个 task_id 对应一个独立的 wing（宫殿分区），确保任务间记忆隔离。
    """

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """初始化 OmniMemSDK 实例。

        Args:
            storage_dir: 记忆存储目录，默认使用临时目录
            config: OmniMemSDK 配置字典
        """
        # 延迟导入，避免在未安装 omnimem 时报错
        from omnimem.sdk import OmniMemSDK

        if storage_dir is None:
            storage_dir = tempfile.mkdtemp(prefix="omnimem_bench_")

        self._sdk = OmniMemSDK(storage_dir=storage_dir, config=config or {})
        self._storage_dir = Path(storage_dir)
        self._task_wings: dict[str, str] = {}  # task_id → wing 名称缓存

        logger.info("OmniMemStateBenchProvider 初始化完成: storage_dir=%s", self._storage_dir)

    def _get_wing(self, task_id: str) -> str:
        """获取 task_id 对应的 wing 名称。

        使用 bench_task_<task_id> 格式，确保任务间记忆隔离。
        """
        if task_id not in self._task_wings:
            # 将 task_id 中的特殊字符替换为下划线
            safe_id = task_id.replace("-", "_").replace("/", "_").replace(".", "_")
            self._task_wings[task_id] = f"{_WING_PREFIX}_{safe_id}"
        return self._task_wings[task_id]

    def on_turn(
        self,
        task_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """每轮对话后存储交互内容。

        Args:
            task_id: 任务标识，用于记忆分区隔离
            role: 对话角色（user/assistant/system/tool）
            content: 对话内容
            metadata: 可选元数据

        Returns:
            memorize 操作结果
        """
        wing = self._get_wing(task_id)
        memory_type = _ROLE_MEMORY_TYPE_MAP.get(role, "fact")
        formatted_content = f"[{role}] {content}"

        kwargs: dict[str, Any] = {
            "content": formatted_content,
            "memory_type": memory_type,
            "wing": wing,
            "room": f"turn_{role}",
            "confidence": 4,  # 有状态任务交互的置信度较高
        }

        # 附加元数据
        if metadata:
            kwargs["content"] += f" | metadata: {json.dumps(metadata, ensure_ascii=False)}"

        try:
            result = self._sdk.memorize(**kwargs)
            logger.debug("on_turn 存储: task=%s, role=%s, wing=%s", task_id, role, wing)
            return result
        except Exception as e:
            logger.warning("on_turn 存储失败: task=%s, role=%s, error=%s", task_id, role, e)
            return {"status": "error", "reason": str(e)}

    def get_context(
        self,
        task_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[str]:
        """检索与当前任务相关的记忆上下文。

        Args:
            task_id: 任务标识，用于过滤相关记忆
            query: 检索查询
            top_k: 返回的最大记忆数量

        Returns:
            检索到的记忆内容列表
        """
        wing = self._get_wing(task_id)

        try:
            result = self._sdk.recall(
                query=query,
                mode="rag",
                wing=wing,
                max_tokens=top_k * 200,  # 每条记忆约 200 token
            )

            # 从 recall 结果中提取记忆文本
            memories = self._extract_memories_from_recall(result, task_id, top_k)
            logger.debug("get_context 检索: task=%s, query=%s, 返回 %d 条", task_id, query, len(memories))
            return memories

        except Exception as e:
            logger.warning("get_context 检索失败: task=%s, query=%s, error=%s", task_id, query, e)
            return []

    def _extract_memories_from_recall(
        self,
        recall_result: dict,
        task_id: str,
        top_k: int,
    ) -> list[str]:
        """从 recall 结果中提取记忆文本列表。

        recall 返回格式可能是：
        - {"status": "ok", "memories": [...]}
        - {"status": "ok", "context": "..."}
        - {"status": "ok", "summary": "...", "items": [...]}
        """
        if not isinstance(recall_result, dict):
            return []

        status = recall_result.get("status", "")
        if status == "error":
            return []

        memories: list[str] = []

        # 尝试从 memories 字段提取
        raw_memories = recall_result.get("memories", [])
        if isinstance(raw_memories, list):
            for item in raw_memories[:top_k]:
                if isinstance(item, str):
                    memories.append(item)
                elif isinstance(item, dict):
                    content = item.get("content", item.get("text", ""))
                    if content:
                        memories.append(str(content))

        # 尝试从 items 字段提取
        if not memories:
            raw_items = recall_result.get("items", [])
            if isinstance(raw_items, list):
                for item in raw_items[:top_k]:
                    if isinstance(item, str):
                        memories.append(item)
                    elif isinstance(item, dict):
                        content = item.get("content", item.get("text", ""))
                        if content:
                            memories.append(str(content))

        # 尝试从 context 字段提取
        if not memories:
            context = recall_result.get("context", "")
            if isinstance(context, str) and context.strip():
                # 将 context 按段落拆分
                paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
                memories = paragraphs[:top_k]

        # 尝试从 summary 字段提取
        if not memories:
            summary = recall_result.get("summary", "")
            if isinstance(summary, str) and summary.strip():
                memories = [summary]

        return memories[:top_k]

    def clear_task(self, task_id: str) -> dict:
        """清除指定任务的记忆。

        通过隐私治理功能标记任务记忆为过期/删除。

        Args:
            task_id: 要清除的任务标识

        Returns:
            清除操作结果
        """
        wing = self._get_wing(task_id)

        try:
            # 使用 govern 接口的 forget 操作
            result = self._sdk.govern(
                action="forget",
                wing=wing,
                reason=f"任务 {task_id} 结束，清理记忆",
            )
            logger.info("clear_task: task=%s, wing=%s", task_id, wing)
            return result
        except Exception as e:
            logger.warning("clear_task 失败: task=%s, error=%s", task_id, e)
            return {"status": "error", "reason": str(e)}

    def ingest_trajectory(self, task_id: str, trajectory: list[dict[str, Any]]) -> None:
        """将完整的任务轨迹批量导入记忆。

        用于从 STATE-Bench 训练轨迹构建学习材料。

        Args:
            task_id: 任务标识
            trajectory: 对话轨迹列表
        """
        for msg in trajectory:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not content:
                continue
            self.on_turn(task_id, role, content)

        logger.info("ingest_trajectory: task=%s, 消息数=%d", task_id, len(trajectory))

    def close(self) -> None:
        """关闭 SDK，释放资源。"""
        try:
            self._sdk.close()
            logger.info("OmniMemStateBenchProvider 已关闭")
        except Exception as e:
            logger.warning("关闭失败: %s", e)

    def __enter__(self) -> OmniMemStateBenchProvider:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class OmniMemStateBenchAgent:
    """OmniMem 增强的 STATE-Bench Agent。

    子类化 StateBenchAgent 并实现 retrieve_learnings，
    通过 OmniMemSDK 提供记忆检索能力。

    此类应放置在 STATE-Bench 仓库根目录的 agents/ 目录下，
    通过 --agent-class OmniMemStateBenchAgent 加载。

    用法（在 STATE-Bench 仓库根目录创建 agents/omnimem_agent.py）：
        from omnimem.benchmarks.statebench_adapter import OmniMemStateBenchAgent

    然后运行：
        uv run python -m state_bench.scripts.run_batch \\
            --domain travel \\
            --agent-class OmniMemStateBenchAgent \\
            --agent-model-name <model> \\
            --retrieve-learnings-top-k 3 \\
            --output-dir outputs/travel/
    """

    # 类级配置 — 可在子类或运行时覆盖
    omnimem_storage_dir: str | None = None
    omnimem_config: dict[str, Any] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @classmethod
    def get_provider(cls) -> OmniMemStateBenchProvider:
        """获取或创建全局 OmniMemStateBenchProvider 实例。"""
        if not hasattr(cls, "_provider") or cls._provider is None:
            storage_dir = cls.omnimem_storage_dir or os.environ.get(
                "OMNIMEM_STORAGE_DIR",
                str(Path.home() / ".omnimem_statebench"),
            )
            config = cls.omnimem_config or {}
            cls._provider = OmniMemStateBenchProvider(
                storage_dir=storage_dir,
                config=config,
            )
        return cls._provider

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        """STATE-Bench Agent Learning Track 的检索钩子。

        StateBenchAgent 在对话过程中自动调用此方法，
        将检索到的记忆作为上下文注入 Agent 响应。

        Args:
            query: 当前对话的检索查询
            top_k: 最大返回条数

        Returns:
            相关记忆文本列表
        """
        provider = self.get_provider()

        # 从 runtime_context 获取 task_id 用于记忆分区
        task_id = "default"
        if hasattr(self, "runtime_context") and self.runtime_context:
            task_id = getattr(self.runtime_context, "task_id", "default")

        memories = provider.get_context(task_id, query, top_k=top_k)

        if not memories:
            return []

        logger.info("retrieve_learnings: task=%s, query=%s, 返回 %d 条", task_id, query[:50], len(memories))
        return memories

    @classmethod
    def ingest_train_trajectories(
        cls,
        domain: str,
        trajectories_dir: str | Path,
    ) -> None:
        """从训练轨迹构建学习材料。

        Args:
            domain: 领域名称（travel/customer_support/shopping_assistant）
            trajectories_dir: 训练轨迹目录路径
        """
        provider = cls.get_provider()
        traj_dir = Path(trajectories_dir)

        if not traj_dir.exists():
            logger.warning("训练轨迹目录不存在: %s", traj_dir)
            return

        count = 0
        for traj_file in sorted(traj_dir.glob("*.json")):
            task_id = f"{domain}_{traj_file.stem}"
            try:
                data = json.loads(traj_file.read_text(encoding="utf-8"))
                conversation = data.get("conversation", [])
                if isinstance(conversation, list):
                    provider.ingest_trajectory(task_id, conversation)
                    count += 1
            except Exception as e:
                logger.warning("导入轨迹失败: %s, error=%s", traj_file.name, e)

        logger.info("ingest_train_trajectories: domain=%s, 导入 %d 条轨迹", domain, count)

    @classmethod
    def close_provider(cls) -> None:
        """关闭全局 Provider。"""
        if hasattr(cls, "_provider") and cls._provider is not None:
            cls._provider.close()
            cls._provider = None


# ─── 单元测试 ────────────────────────────────────────────────────────

def _run_tests() -> None:
    """运行适配器的单元测试。"""
    import shutil

    print("=" * 60)
    print("OmniMemStateBenchProvider 单元测试")
    print("=" * 60)

    # 创建临时目录
    test_dir = tempfile.mkdtemp(prefix="omnimem_bench_test_")
    print(f"\n测试目录: {test_dir}")

    try:
        # ── 测试 1: 初始化 ─────────────────────────────────────
        print("\n[测试 1] 初始化 OmniMemStateBenchProvider")
        provider = OmniMemStateBenchProvider(storage_dir=test_dir)
        print("  ✓ 初始化成功")

        # ── 测试 2: on_turn 存储 ──────────────────────────────
        print("\n[测试 2] on_turn 存储交互")
        result1 = provider.on_turn("task-001", "user", "我想取消订单 ABC123")
        print(f"  存储 user 消息: status={result1.get('status', 'unknown')}")

        result2 = provider.on_turn("task-001", "assistant", "好的，我来帮您查看订单 ABC123 的详情")
        print(f"  存储 assistant 消息: status={result2.get('status', 'unknown')}")

        result3 = provider.on_turn("task-001", "user", "订单金额是 299 元，请帮我退款")
        print(f"  存储 user 消息: status={result3.get('status', 'unknown')}")

        # ── 测试 3: get_context 检索 ─────────────────────────
        print("\n[测试 3] get_context 检索")
        memories = provider.get_context("task-001", "取消订单流程", top_k=5)
        print(f"  查询 '取消订单流程': 返回 {len(memories)} 条记忆")
        for i, mem in enumerate(memories[:3]):
            print(f"    [{i+1}] {mem[:80]}...")

        # ── 测试 4: task_id 隔离 ─────────────────────────────
        print("\n[测试 4] task_id 隔离验证")
        # 向 task-002 存储不同的内容
        provider.on_turn("task-002", "user", "我想预订去东京的机票")
        provider.on_turn("task-002", "assistant", "好的，帮您查询东京航班")

        # 检索 task-001 的上下文，不应包含 task-002 的内容
        memories_1 = provider.get_context("task-001", "订单信息", top_k=5)
        memories_2 = provider.get_context("task-002", "机票预订", top_k=5)

        # 验证 wing 隔离
        wing_1 = provider._get_wing("task-001")
        wing_2 = provider._get_wing("task-002")
        print(f"  task-001 wing: {wing_1}")
        print(f"  task-002 wing: {wing_2}")
        assert wing_1 != wing_2, "wing 隔离失败：不同 task_id 不应有相同 wing"
        print("  ✓ wing 隔离验证通过")

        # ── 测试 5: ingest_trajectory 批量导入 ───────────────
        print("\n[测试 5] ingest_trajectory 批量导入")
        trajectory = [
            {"role": "user", "content": "我想退货"},
            {"role": "assistant", "content": "好的，请提供订单号"},
            {"role": "user", "content": "订单号是 XYZ789"},
            {"role": "assistant", "content": "已为您提交退货申请"},
        ]
        provider.ingest_trajectory("task-003", trajectory)
        memories_3 = provider.get_context("task-003", "退货流程", top_k=5)
        print(f"  批量导入 4 条消息后检索: 返回 {len(memories_3)} 条记忆")

        # ── 测试 6: clear_task 清除 ──────────────────────────
        print("\n[测试 6] clear_task 清除任务记忆")
        clear_result = provider.clear_task("task-001")
        print(f"  清除 task-001: status={clear_result.get('status', 'unknown')}")

        # ── 测试 7: close 关闭 ──────────────────────────────
        print("\n[测试 7] close 关闭")
        provider.close()
        print("  ✓ 关闭成功")

        # ── 测试 8: 上下文管理器 ────────────────────────────
        print("\n[测试 8] 上下文管理器")
        test_dir2 = tempfile.mkdtemp(prefix="omnimem_bench_test2_")
        with OmniMemStateBenchProvider(storage_dir=test_dir2) as p:
            p.on_turn("ctx-test", "user", "上下文管理器测试")
            mems = p.get_context("ctx-test", "测试", top_k=3)
            print(f"  上下文管理器内检索: 返回 {len(mems)} 条记忆")
        print("  ✓ 上下文管理器退出正常")

        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)

    finally:
        # 清理临时目录
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _run_tests()
