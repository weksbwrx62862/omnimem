"""OpenAI EmbeddingProvider 实现。"""

from __future__ import annotations

import logging
import os
from typing import Any

from omnimem.embedding.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """基于 OpenAI API 的远程嵌入服务。"""

    # 常见模型维度映射
    _DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
    ):
        self._model_name = model_name
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._base_url = base_url
        self._dimensions = dimensions
        self._sync_client: Any = None
        self._async_client: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        """返回模型维度，优先使用用户指定值，其次查表。"""
        if self._dimensions is not None:
            return self._dimensions
        return self._DIMENSIONS.get(self._model_name, 1536)

    def _get_sync_client(self) -> Any:
        if self._sync_client is None:
            try:
                import openai
            except ImportError as e:
                raise RuntimeError("openai package not installed") from e
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._sync_client = openai.OpenAI(**kwargs)
        return self._sync_client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            try:
                import openai
            except ImportError as e:
                raise RuntimeError("openai package not installed") from e
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._async_client = openai.AsyncOpenAI(**kwargs)
        return self._async_client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """同步调用 OpenAI Embedding API。"""
        if not texts:
            return []
        client = self._get_sync_client()
        try:
            response = client.embeddings.create(input=texts, model=self._model_name)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.warning("OpenAI embed failed: %s", e)
            raise

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """异步调用 OpenAI Embedding API。"""
        if not texts:
            return []
        client = self._get_async_client()
        try:
            response = await client.embeddings.create(input=texts, model=self._model_name)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.warning("OpenAI aembed failed: %s", e)
            raise
