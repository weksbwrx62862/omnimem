"""SagaCoordinator — Saga 事务协调器。

解决 OmniMem 四数据源（Store/Index/Vector/BM25/KG）写入不一致问题。
设计原则：
  1. 主存储（Store）作为唯一事实来源，必须先成功
  2. 索引/检索/图谱作为派生数据，允许最终一致
  3. 失败时记录到 pending queue，由后台任务重试补偿

不实现跨服务分布式事务，而是本地 Saga 模式：
  - 正向操作：按序执行各步骤
  - 反向补偿：目前阶段只记录失败，后台重试（而非回滚主存储）
    原因：记忆写入是追加操作，回滚意义不大，重试补索引更有价值。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SagaStep:
    """Saga 单步定义。"""

    name: str
    action: Callable[[], Any]


@dataclass
class SagaResult:
    """Saga 执行结果。"""

    success: bool
    memory_id: str
    completed_steps: list[str] = field(default_factory=list)
    failed_step: str = ""
    error: str = ""
    step_results: dict[str, Any] = field(default_factory=dict)


class SagaCoordinator:
    """Saga 事务协调器。

    职责：
      1. 编排多步骤写入（store → index → retriever → kg）
      2. 记录失败到 pending queue
      3. 提供 retry_pending() 供后台任务批量补偿
    """

    def __init__(self, pending_path: Path | None = None):
        """初始化 Saga 协调器。

        Args:
            pending_path: pending 队列持久化文件路径。
                          若提供，进程重启后可恢复未完成的任务。
        """
        self._pending: list[dict[str, Any]] = []
        self._pending_path = pending_path
        if pending_path and pending_path.exists():
            self._load_pending()

    def execute(self, memory_id: str, steps: list[SagaStep]) -> SagaResult:
        """执行 Saga 事务。

        按顺序执行 steps，任一失败即停止，记录已完成的步骤和失败步骤。
        成功执行的步骤返回值会被收集到 SagaResult.step_results 中。

        Args:
            memory_id: 记忆 ID，用于追踪和重试
            steps: Saga 步骤列表

        Returns:
            SagaResult，包含成功/失败状态、步骤详情和各步骤返回值
        """
        completed: list[str] = []
        step_results: dict[str, Any] = {}
        for step in steps:
            try:
                result = step.action()
                completed.append(step.name)
                step_results[step.name] = result
                logger.debug("Saga step '%s' OK for %s", step.name, memory_id)
            except Exception as e:
                logger.warning(
                    "Saga step '%s' failed for %s: %s",
                    step.name,
                    memory_id,
                    e,
                )
                record = {
                    "memory_id": memory_id,
                    "failed_step": step.name,
                    "completed_steps": completed,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._pending.append(record)
                self._persist_pending()
                return SagaResult(
                    success=False,
                    memory_id=memory_id,
                    completed_steps=completed,
                    failed_step=step.name,
                    error=str(e),
                    step_results=step_results,
                )
        return SagaResult(
            success=True,
            memory_id=memory_id,
            completed_steps=completed,
            step_results=step_results,
        )

    def get_pending(self) -> list[dict[str, Any]]:
        """获取所有待重试的 pending 记录。"""
        return list(self._pending)

    def retry_pending(
        self,
        step_actions: dict[str, Callable[[str], Any]],
    ) -> int:
        """批量重试 pending 任务。

        Args:
            step_actions: 步骤名 → (memory_id) -> Any 的映射。
                          调用方需要提供每个失败步骤的重试逻辑。

        Returns:
            成功修复的条目数
        """
        if not self._pending:
            return 0

        fixed = 0
        still_pending: list[dict[str, Any]] = []

        for record in self._pending:
            memory_id = record.get("memory_id", "")
            failed_step = record.get("failed_step", "")
            action = step_actions.get(failed_step)

            if not action:
                logger.debug("No retry action for step '%s', keeping pending", failed_step)
                still_pending.append(record)
                continue

            try:
                action(memory_id)
                logger.info("Saga retry OK: %s step '%s'", memory_id, failed_step)
                fixed += 1
            except Exception as e:
                logger.warning("Saga retry failed: %s step '%s': %s", memory_id, failed_step, e)
                still_pending.append(record)

        self._pending = still_pending
        self._persist_pending()
        return fixed

    def clear_pending(self) -> int:
        """清空 pending 队列（谨慎使用）。返回清空的条目数。"""
        count = len(self._pending)
        self._pending.clear()
        self._persist_pending()
        return count

    # ─── 持久化 ─────────────────────────────────────────────

    def _persist_pending(self) -> None:
        """将 pending 队列持久化到磁盘。"""
        if not self._pending_path:
            return
        try:
            self._pending_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._pending_path, "w", encoding="utf-8") as f:
                json.dump(self._pending, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug("Saga pending persist failed: %s", e)

    def _load_pending(self) -> None:
        """从磁盘加载 pending 队列。"""
        if not self._pending_path or not self._pending_path.exists():
            return
        try:
            with open(self._pending_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._pending = data
                logger.info("Loaded %d pending saga records", len(self._pending))
        except Exception as e:
            logger.debug("Saga pending load failed: %s", e)
