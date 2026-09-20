"""多模态抽象与数据类型（改进项 #12）。

对标 embedding/base.py 的 EmbeddingProvider 抽象，为图片/音频/视频定义统一的
「媒体嵌入器」协议与内容寻址引用。刻意与文本主链路解耦：仅新增类型，不改动
retrieval/handlers，任何带 .embed(blob)->list[float] 的对象都满足 MediaEmbedder。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# 支持的内容/嵌入模态标签（与消息 content-part 的 type 对齐）。
IMAGE_KINDS = {"image", "image_url"}
AUDIO_KINDS = {"audio", "input_audio"}
VIDEO_KINDS = {"video"}


def _kind_from_mime(mime: str) -> str:
    m = (mime or "").lower()
    if m.startswith("image/"):
        return "image"
    if m.startswith("audio/"):
        return "audio"
    if m.startswith("video/"):
        return "video"
    return "file"


@runtime_checkable
class MediaEmbedder(Protocol):
    """媒体嵌入器协议：把一段字节流转成定长向量。

    结构性类型（无需显式继承），HashEmbedder / ClipEmbedder 均满足。
    """

    modality: str
    dimension: int

    def embed(self, blob: bytes) -> list[float]:
        ...


@dataclass
class MediaPart:
    """从消息 content 解析出的一个媒体片段（可能尚未落盘）。

    inline base64 → data 非空；远程 URL → url 非空、data 为空。
    """

    kind: str
    mime: str
    data: bytes = b""
    url: str = ""
    name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def has_inline(self) -> bool:
        return bool(self.data)


@dataclass
class MediaRef:
    """落盘后的媒体引用（内容寻址，sha256 唯一）。"""

    sha256: str
    kind: str
    mime: str
    size: int
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "kind": self.kind,
            "mime": self.mime,
            "size": self.size,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MediaRef:
        return cls(
            sha256=str(d.get("sha256", "")),
            kind=str(d.get("kind", "file")),
            mime=str(d.get("mime", "")),
            size=int(d.get("size", 0) or 0),
            path=str(d.get("path", "")),
        )
