"""从消息 content 解析媒体片段（改进项 #12，纯函数、可离线测）。

兼容三种主流 content-part 形态：
  - OpenAI:  {"type":"image_url","image_url":{"url":"data:image/png;base64,..|http://.."}}
             {"type":"input_audio","input_audio":{"data":"<b64>","format":"mp3"}}
  - Anthropic:{"type":"image","source":{"type":"base64","media_type":"image/png","data":"<b64>"}}
             {"type":"document"/"audio" 同理带 source}
字符串 content 无媒体，返回 []。仅做“抽取”，不落盘、不调模型。
"""
from __future__ import annotations

import base64
import binascii
import re
from typing import Any

from omnimem.multimodal.types import (
    AUDIO_KINDS,
    IMAGE_KINDS,
    VIDEO_KINDS,
    MediaPart,
    _kind_from_mime,
)

_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;,]+)(?:;base64)?,(?P<data>.*)$", re.DOTALL)

_ALL_MEDIA_KINDS = IMAGE_KINDS | AUDIO_KINDS | VIDEO_KINDS


def _decode_data_uri(url: str) -> tuple[str, bytes] | None:
    """解析 data URI → (mime, bytes)；非 data URI 返回 None。"""
    m = _DATA_URI_RE.match(url or "")
    if not m:
        return None
    mime = m.group("mime") or "application/octet-stream"
    raw = m.group("data") or ""
    try:
        blob = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        # 允许非 base64 的 URL 编码 data URI（少见），退化为按文本字节存
        blob = raw.encode("utf-8", "ignore")
    return mime, blob


def _b64_to_bytes(s: str) -> bytes:
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        return base64.b64decode(s + "===")  # 容忍缺失 padding


def _part_from_dict(part: dict[str, Any]) -> MediaPart | None:
    ptype = str(part.get("type", "")).lower()

    # OpenAI image_url
    if ptype in IMAGE_KINDS and "image_url" in part:
        iu = part["image_url"] or {}
        url = str(iu.get("url", ""))
        parsed = _decode_data_uri(url)
        if parsed:
            mime, blob = parsed
            return MediaPart(kind="image", mime=mime, data=blob, name=str(iu.get("detail", "") or ""))
        if url:
            return MediaPart(kind="image", mime="", url=url)
        return None

    # OpenAI input_audio
    if ptype in AUDIO_KINDS and "input_audio" in part:
        ia = part["input_audio"] or {}
        data = str(ia.get("data", ""))
        fmt = str(ia.get("format", "") or "wav")
        if data:
            return MediaPart(kind="audio", mime=f"audio/{fmt}", data=_b64_to_bytes(data))
        return None

    # Anthropic: {"type": "image"|"audio"|..., "source": {...}}
    src = part.get("source")
    if isinstance(src, dict) and ptype in _ALL_MEDIA_KINDS:
        stype = str(src.get("type", "")).lower()
        data = str(src.get("data", ""))
        mime = str(src.get("media_type", "") or src.get("mime_type", ""))
        if stype == "base64" and data:
            kind = _kind_from_mime(mime) if mime else ("audio" if ptype in AUDIO_KINDS else "image")
            return MediaPart(kind=kind, mime=mime, data=_b64_to_bytes(data), name=str(src.get("filename", "") or ""))
        if stype == "url" and data:
            kind = _kind_from_mime(mime) if mime else ("audio" if ptype in AUDIO_KINDS else "image")
            return MediaPart(kind=kind, mime=mime, url=data)
        return None

    return None


def parse_content(content: Any) -> list[MediaPart]:
    """把消息 content（str 或 parts 列表）解析为媒体片段列表。"""
    if isinstance(content, str) or content is None:
        return []
    if isinstance(content, dict):
        content = [content]
    parts: list[MediaPart] = []
    for item in content:
        if isinstance(item, dict):
            mp = _part_from_dict(item)
            if mp is not None:
                parts.append(mp)
    return parts


def parse_message(message: dict[str, Any]) -> list[MediaPart]:
    """从一条 {"role","content"} 消息解析媒体片段。"""
    return parse_content(message.get("content"))
