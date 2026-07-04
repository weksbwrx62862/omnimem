"""OmniMem 异步 LLM 包装器 — 使用 asyncio.to_thread 包装同步 LLM 客户端。

提供异步 LLM 调用能力，避免阻塞事件循环。
支持并发调用和批处理。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class AsyncLLMWrapper:
    """异步 LLM 包装器 — 将同步 LLM 客户端包装为异步接口。

    使用 asyncio.to_thread 将同步调用转移到独立线程，
    避免阻塞事件循环。支持单次调用、批量并发调用和带重试的调用。

    线程安全说明：
      - 同步客户端本身需线程安全（AsyncLLMClient.call_sync 已处理嵌套事件循环）
      - 本包装器不引入额外共享状态，可安全并发调用
    """

    def __init__(self, sync_client: Any) -> None:
        """初始化异步包装器。

        Args:
            sync_client: 同步 LLM 客户端，需提供 call_sync(prompt, system, max_tokens, temperature) 方法
        """
        self._sync_client = sync_client

    async def call(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str | None:
        """异步 LLM 调用 — 使用 asyncio.to_thread 包装同步调用。

        Args:
            prompt: 用户提示文本
            system: 系统提示（可选）
            max_tokens: 最大输出 token 数
            temperature: 采样温度

        Returns:
            LLM 响应文本，失败时返回 None
        """
        if not self._sync_client:
            return None
        try:
            result = await asyncio.to_thread(
                self._sync_client.call_sync,
                prompt=prompt,
                system=system or "",
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return result.content if result and result.content else None
        except Exception as e:
            logger.warning("AsyncLLMWrapper.call 失败: %s", e)
            return None

    async def batch_call(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[str | None]:
        """异步批量 LLM 调用 — 使用 asyncio.gather 并发执行。

        Args:
            prompts: 提示文本列表
            **kwargs: 传递给 call() 的额外参数（system/max_tokens/temperature）

        Returns:
            响应文本列表，与 prompts 一一对应，失败项为 None
        """
        if not prompts:
            return []
        tasks = [self.call(p, **kwargs) for p in prompts]
        return await asyncio.gather(*tasks)

    async def call_with_retry(
        self,
        prompt: str,
        retries: int = 3,
        delay: float = 1.0,
        **kwargs: Any,
    ) -> str | None:
        """带重试的异步 LLM 调用。

        失败后按指数退避等待并重试，直到成功或耗尽重试次数。

        Args:
            prompt: 用户提示文本
            retries: 最大重试次数
            delay: 初始退避间隔（秒），每次翻倍
            **kwargs: 传递给 call() 的额外参数

        Returns:
            LLM 响应文本，全部失败后返回 None
        """
        for attempt in range(retries + 1):
            result = await self.call(prompt, **kwargs)
            if result is not None:
                return result
            if attempt < retries:
                wait = delay * (2 ** attempt)
                logger.debug(
                    "AsyncLLMWrapper.call_with_retry 第 %d 次重试，等待 %.1fs",
                    attempt + 1, wait,
                )
                await asyncio.sleep(wait)
        logger.warning(
            "AsyncLLMWrapper.call_with_retry 耗尽 %d 次重试: prompt=%s",
            retries, prompt[:80],
        )
        return None


class AsyncBatchProcessor:
    """异步批处理器 — 使用 Semaphore 控制最大并发数。

    适用于大批量 LLM 调用或检索任务，避免一次性提交过多任务
    导致线程池耗尽或触发服务端限流。

    支持失败项重试：首次执行失败的项会按 max_retries 再次尝试。
    """

    def __init__(self, max_concurrency: int = 5) -> None:
        """初始化批处理器。

        Args:
            max_concurrency: 最大并发数，默认 5
        """
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_concurrency = max_concurrency

    async def process(
        self,
        items: list[Any],
        fn: Callable[..., Any],
        max_retries: int = 1,
        **kwargs: Any,
    ) -> list[Any]:
        """并发处理批量任务，使用 Semaphore 控制并发度。

        Args:
            items: 待处理项列表
            fn: 处理函数（同步或异步均可）。
                若为同步函数，会自动用 asyncio.to_thread 包装。
                接受 item 作为首个位置参数，以及 **kwargs。
            max_retries: 失败项重试次数，默认 1 次
            **kwargs: 传递给 fn 的额外参数

        Returns:
            处理结果列表，与 items 一一对应。
            失败项（重试后仍失败）对应位置为 None。
        """
        if not items:
            return []

        async def _run_one(item: Any) -> Any:
            """执行单个任务，带信号量控制和重试。"""
            async with self._semaphore:
                last_error: Exception | None = None
                for attempt in range(max_retries + 1):
                    try:
                        if asyncio.iscoroutinefunction(fn):
                            return await fn(item, **kwargs)
                        # 同步函数包装为异步
                        return await asyncio.to_thread(fn, item, **kwargs)
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries:
                            logger.debug(
                                "AsyncBatchProcessor 任务失败（第 %d 次重试）: %s",
                                attempt + 1, e,
                            )
                logger.warning(
                    "AsyncBatchProcessor 任务最终失败（重试 %d 次）: %s",
                    max_retries, last_error,
                )
                return None

        tasks = [_run_one(item) for item in items]
        return await asyncio.gather(*tasks)

    @property
    def max_concurrency(self) -> int:
        """返回最大并发数。"""
        return self._max_concurrency
