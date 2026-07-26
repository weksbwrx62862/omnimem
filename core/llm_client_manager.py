"""LLM 客户端管理器 — 负责 LLM 客户端初始化、Reflect 调用和 Distill 调用。"""

from __future__ import annotations

import logging
from typing import Any

from omnimem.core.llm_initializer import init_llm_client
from omnimem.core.tool_router import (
    async_call_llm_for_reflect,
    call_llm_for_reflect,
    make_llm_call_fn,
)
from omnimem.utils.async_llm import AsyncLLMWrapper

logger = logging.getLogger(__name__)

_REFLECT_CACHE_TTL = 60.0


class LLMClientManager:
    """管理 OmniMem 的 LLM 客户端生命周期和调用。

    职责:
      1. 初始化主 LLM 客户端（含 Hermes 凭证回退）
      2. 初始化蒸馏专用 LLM 客户端
      3. 提供 Reflect LLM 调用入口
      4. 提供 Distill LLM 调用入口
      5. 生成 LLM 调用函数（供 CompressionPipeline 等使用）
    """

    def __init__(self, config: Any, reflect_cache: dict[str, tuple[str, float]] | None = None) -> None:
        self._config = config
        self._reflect_cache = reflect_cache or {}
        self._llm_client: Any = None
        self._distill_llm_client: Any = None

    @property
    def llm_client(self) -> Any:
        return self._llm_client

    @property
    def distill_llm_client(self) -> Any:
        return self._distill_llm_client

    def init_llm_client(self) -> None:
        """初始化主 LLM 客户端。"""
        self._llm_client = init_llm_client(self._config)
        # 若 LLM 客户端凭证为空，尝试从 Hermes 主配置获取
        if self._llm_client and not getattr(self._llm_client, "_api_key", "").strip():
            try:
                from omnimem.utils.llm_client import DEFAULT_LLM_MODEL, AsyncLLMClient
                hermes_creds = AsyncLLMClient.load_credentials_from_hermes_config()
                if hermes_creds.get("api_key") and hermes_creds.get("base_url"):
                    logger.info("OmniMem: using Hermes main config LLM credentials for Reflect")
                    self._llm_client = AsyncLLMClient(
                        api_key=hermes_creds["api_key"],
                        base_url=hermes_creds["base_url"],
                        model=hermes_creds.get("model", DEFAULT_LLM_MODEL),
                        max_concurrent=3,
                        timeout=30.0,
                        cache_ttl=_REFLECT_CACHE_TTL,
                    )
            except Exception as e:
                logger.warning("OmniMem: failed to load Hermes main config LLM credentials: %s", e)

    def make_llm_call_fn(self):
        """生成 LLM 调用函数（供 CompressionPipeline 等使用）。"""
        return make_llm_call_fn(self._llm_client)

    def call_llm_for_reflect(self, prompt: str, system: str, max_tokens: int = 800) -> str | None:
        """Reflect 引擎的 LLM 调用入口。"""
        return call_llm_for_reflect(
            prompt, system,
            llm_client=self._llm_client,
            reflect_cache=self._reflect_cache,
            max_tokens=max_tokens,
        )

    def call_llm_for_distill(
        self, prompt: str, system: str, max_tokens: int = 600, model: str | None = None
    ) -> str | None:
        """蒸馏引擎的 LLM 调用入口，支持自定义模型。

        Args:
            prompt: 蒸馏 prompt
            system: 系统提示
            max_tokens: 最大输出 token
            model: 自定义模型名（None=使用主模型）
        """
        client = None
        if model:
            # 为蒸馏模型创建/复用独立客户端
            if self._distill_llm_client is None:
                self._init_distill_llm_client(model)
            client = self._distill_llm_client or self._llm_client
        else:
            client = self._llm_client

        if not client:
            return None

        try:
            result = client.call_sync(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return result.content if result and result.content else None
        except Exception as e:
            logger.warning("Distill LLM call failed: %s", e)
            return None

    # ─── 异步方法（向后兼容：同步方法保持不变） ───────────────

    def _get_async_wrapper(self) -> AsyncLLMWrapper | None:
        """获取异步 LLM 包装器（懒初始化，复用同步客户端）。"""
        if not self._llm_client:
            return None
        return AsyncLLMWrapper(self._llm_client)

    async def async_call_llm_for_reflect(
        self, prompt: str, system: str | None = None, max_tokens: int = 800
    ) -> str | None:
        """异步 Reflect 引擎的 LLM 调用入口。

        委托给 tool_router.async_call_llm_for_reflect，复用相同的缓存逻辑
        （模块级锁 + LRU），LLM 调用使用 asyncio.to_thread 包装，不持锁。

        Args:
            prompt: 反思 prompt
            system: 系统提示（可选，None 时使用空字符串）
            max_tokens: 最大输出 token

        Returns:
            LLM 响应文本，失败时返回 None
        """
        return await async_call_llm_for_reflect(
            prompt,
            system or "",
            llm_client=self._llm_client,
            reflect_cache=self._reflect_cache,
            max_tokens=max_tokens,
        )

    async def async_call_llm_for_distill(
        self, prompt: str, system: str | None = None, max_tokens: int = 600, model: str | None = None
    ) -> str | None:
        """异步蒸馏引擎的 LLM 调用入口，支持自定义模型。

        使用 AsyncLLMWrapper 包装同步客户端，通过 asyncio.to_thread
        在独立线程执行 LLM 调用，避免阻塞事件循环。

        Args:
            prompt: 蒸馏 prompt
            system: 系统提示（可选）
            max_tokens: 最大输出 token
            model: 自定义模型名（None=使用主模型）

        Returns:
            LLM 响应文本，失败时返回 None
        """
        client = None
        if model:
            # 为蒸馏模型创建/复用独立客户端
            if self._distill_llm_client is None:
                self._init_distill_llm_client(model)
            client = self._distill_llm_client or self._llm_client
        else:
            client = self._llm_client

        if not client:
            return None

        wrapper = AsyncLLMWrapper(client)
        try:
            return await wrapper.call(
                prompt=prompt,
                system=system or "",
                max_tokens=max_tokens,
                temperature=0.3,
            )
        except Exception as e:
            logger.warning("Async distill LLM call failed: %s", e)
            return None

    def _init_distill_llm_client(self, model: str) -> None:
        """为蒸馏任务创建专用 LLM 客户端。"""
        from omnimem.utils.llm_client import AsyncLLMClient

        # 使用与主客户端相同的凭证
        if self._llm_client:
            creds_key = getattr(self._llm_client, "_api_key", "")
            creds_url = getattr(self._llm_client, "_base_url", "")
        else:
            creds = AsyncLLMClient.load_credentials_from_hermes_config()
            creds_key = creds.get("api_key", "")
            creds_url = creds.get("base_url", "")

        if creds_key and creds_url:
            self._distill_llm_client = AsyncLLMClient(
                api_key=creds_key,
                base_url=creds_url,
                model=model,
                max_concurrent=1,  # 蒸馏是低频后台任务，1并发足够
                timeout=30.0,
                cache_ttl=0.0,  # 蒸馏不缓存
            )
            logger.info("Distill LLM client created: model=%s", model)

    def close(self) -> None:
        """关闭所有 LLM 客户端。"""
        if self._llm_client:
            self._llm_client.close()
        if self._distill_llm_client:
            self._distill_llm_client.close()
