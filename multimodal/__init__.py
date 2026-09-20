"""多模态记忆子系统包（改进项 #12）。

对外暴露：媒体类型/协议、内容寻址存储、消息媒体解析、媒体嵌入器工厂。
ClipEmbedder 走懒加载，避免默认导入拉入 open_clip/torch/PIL 等可选依赖。
本包刻意自足：不改动 retrieval/handlers，可独立离线使用与测试。
"""
from __future__ import annotations

from omnimem.multimodal.parse import parse_content, parse_message
from omnimem.multimodal.store import MediaStore
from omnimem.multimodal.types import (
    MediaEmbedder,
    MediaPart,
    MediaRef,
)

__all__ = [
    "MediaStore",
    "MediaPart",
    "MediaRef",
    "MediaEmbedder",
    "parse_content",
    "parse_message",
    "HashEmbedder",
    "ClipEmbedder",
    "create_media_embedder",
    "SUPPORTED_MEDIA_BACKENDS",
]


def __getattr__(name: str):
    # 懒加载嵌入器，保持包导入轻量、可选项不前置（对齐 kg.backends 风格）。
    if name in ("HashEmbedder", "ClipEmbedder", "create_media_embedder", "SUPPORTED_MEDIA_BACKENDS"):
        from omnimem.multimodal import embedders

        return getattr(embedders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
