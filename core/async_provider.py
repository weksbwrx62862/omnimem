"""AsyncOmniMemProvider — OmniMem 的异步包装器。

P2方案五：为 Hermes asyncio 事件循环提供非阻塞接口。
设计原则：
  1. 零侵入：不修改 OmniMemProvider 的任何同步代码
  2. 镜像接口：所有公共方法提供 async 版本
  3. 线程池隔离：使用独立 ThreadPoolExecutor，避免阻塞事件循环

所有耗时操作（检索、存储、Saga 执行、LLM 调用）都委托到线程池，
在 asyncio 事件循环中通过 run_in_executor 异步执行。
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


class AsyncOmniMemProvider:
    """OmniMemProvider 的异步包装器。

    使用方式：
        provider = OmniMemProvider(...)
        async_provider = AsyncOmniMemProvider(provider)
        result = await async_provider.prefetch("用户查询")
    """

    def __init__(self, provider, max_workers: int = 4):
        """初始化异步包装器。

        Args:
            provider: OmniMemProvider 实例
            max_workers: 线程池大小。默认 4：
                1-2 用于检索，1 用于存储，1 用于工具调用
        """
        self._provider = provider
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="omnimem_async",
        )

    # ─── 核心异步接口 ─────────────────────────────────────────

    async def prefetch(self, query: str, *, session_id: str = "") -> str:
        """异步预检索。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._provider.prefetch,
            query,
            session_id=session_id,
        )

    async def system_prompt_block(self) -> str:
        """异步获取系统提示块。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._provider.system_prompt_block,
        )

    async def handle_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs,
    ) -> str:
        """异步处理工具调用。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._provider.handle_tool_call,
            tool_name,
            args,
            **kwargs,
        )

    async def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """异步同步单轮对话。"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._provider.sync_turn,
            user_content,
            assistant_content,
            session_id=session_id,
        )

    async def on_turn_start(
        self,
        turn_number: int,
        message: str,
        **kwargs,
    ) -> None:
        """异步 turn 开始钩子。"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._provider.on_turn_start,
            turn_number,
            message,
            **kwargs,
        )

    async def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """异步会话结束钩子。"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._provider.on_session_end,
            messages,
        )

    async def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """异步压缩前钩子。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._provider.on_pre_compress,
            messages,
        )

    # ─── 治理/诊断接口 ────────────────────────────────────────

    async def run_governance_audit(self) -> dict[str, Any]:
        """异步运行治理审计。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._provider._auditor.run_full_audit(limit=1000)
            if hasattr(self._provider, "_auditor") and self._provider._auditor
            else {"error": "auditor not available"},
        )

    async def get_health_status(self) -> dict[str, Any]:
        """异步获取健康状态。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._provider._auditor.quick_health_check()
            if hasattr(self._provider, "_auditor") and self._provider._auditor
            else {"error": "auditor not available"},
        )

    # ─── 生命周期 ─────────────────────────────────────────────

    def shutdown(self, wait: bool = True) -> None:
        """关闭异步线程池。"""
        logger.info("AsyncOmniMemProvider shutting down")
        self._executor.shutdown(wait=wait)

    @property
    def sync_provider(self):
        """返回底层的同步 Provider（用于需要同步操作的场景）。"""
        return self._provider
