"""Embedding Provider 统一抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """文本嵌入服务抽象基类，支持同步与异步调用。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """嵌入向量维度。"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称。"""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """同步嵌入文本列表。"""
        ...

    @abstractmethod
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """异步嵌入文本列表。"""
        ...
