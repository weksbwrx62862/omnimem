"""改进项 #12 多模态子系统单测（离线，无需 open_clip/torch/PIL）。

覆盖：消息 content 解析（OpenAI/Anthropic/dataURI/远程URL/纯文本）、
内容寻址存储（落盘+去重+取回+列举）、HashEmbedder 确定性/归一/维度、
工厂分派与未知后端报错、MediaEmbedder 协议一致性。
"""
from __future__ import annotations

import base64
import hashlib
import math

import pytest
from omnimem.multimodal import (
    SUPPORTED_MEDIA_BACKENDS,
    HashEmbedder,
    MediaEmbedder,
    MediaPart,
    MediaStore,
    create_media_embedder,
    parse_content,
    parse_message,
)

_PNG = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + b"fakepngdata"
_B64 = base64.b64encode(_PNG).decode()


# ── 解析 ──────────────────────────────────────────────
def test_parse_str_content_no_media():
    assert parse_content("just text") == []
    assert parse_content(None) == []


def test_parse_text_part_list_ignored():
    content = [{"type": "text", "text": "hello"}]
    assert parse_content(content) == []


def test_parse_openai_datauri_image():
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_B64}"}}]
    parts = parse_content(content)
    assert len(parts) == 1
    assert parts[0].kind == "image"
    assert parts[0].mime == "image/png"
    assert parts[0].data == _PNG
    assert parts[0].sha256 == hashlib.sha256(_PNG).hexdigest()


def test_parse_openai_remote_image_url():
    content = [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]
    parts = parse_content(content)
    assert len(parts) == 1
    assert parts[0].url == "https://x.test/a.png"
    assert parts[0].has_inline is False


def test_parse_openai_input_audio():
    content = [{"type": "input_audio", "input_audio": {"data": _B64, "format": "mp3"}}]
    parts = parse_content(content)
    assert len(parts) == 1
    assert parts[0].kind == "audio"
    assert parts[0].mime == "audio/mp3"
    assert parts[0].data == _PNG


def test_parse_anthropic_base64_image():
    content = [{
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": _B64},
    }]
    parts = parse_content(content)
    assert len(parts) == 1
    assert parts[0].kind == "image"
    assert parts[0].mime == "image/jpeg"
    assert parts[0].data == _PNG


def test_parse_mixed_and_message_helper():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_B64}"}},
        ],
    }
    parts = parse_message(message)
    assert len(parts) == 1
    assert parts[0].kind == "image"


def test_parse_tolerates_unpadded_base64():
    stripped = _B64.rstrip("=")
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": stripped}}]
    parts = parse_content(content)
    assert len(parts) == 1
    assert parts[0].data == _PNG


# ── 存储 ──────────────────────────────────────────────
def test_store_put_get_roundtrip(tmp_path):
    store = MediaStore(tmp_path)
    ref = store.put(MediaPart(kind="image", mime="image/png", data=_PNG))
    assert ref is not None
    assert store.exists(ref.sha256)
    assert store.get(ref) == _PNG
    assert ref.size == len(_PNG)


def test_store_dedup_identical_bytes(tmp_path):
    store = MediaStore(tmp_path)
    r1 = store.put(MediaPart(kind="image", mime="image/png", data=_PNG))
    r2 = store.put(MediaPart(kind="image", mime="image/png", data=_PNG))
    assert r1.sha256 == r2.sha256
    assert store.list_refs().__len__() == 1  # 内容寻址去重，只有一份


def test_store_skips_url_only_parts(tmp_path):
    store = MediaStore(tmp_path)
    ref = store.put(MediaPart(kind="image", mime="", url="https://x.test/a.png"))
    assert ref is None
    assert store.list_refs() == []


def test_store_list_and_stat(tmp_path):
    store = MediaStore(tmp_path)
    refs = store.put_all([
        MediaPart(kind="image", mime="image/png", data=_PNG),
        MediaPart(kind="audio", mime="audio/wav", data=b"RIFFwavebytes"),
    ])
    assert len(refs) == 2
    listed = {r.sha256 for r in store.list_refs()}
    assert listed == {r.sha256 for r in refs}
    one = store.stat(refs[0].sha256)
    assert one is not None and one.kind == refs[0].kind


# ── 嵌入器 ────────────────────────────────────────────
def test_hash_embedder_deterministic_and_normalized():
    emb = HashEmbedder(dimension=64)
    v1 = emb.embed(_PNG)
    v2 = emb.embed(_PNG)
    assert v1 == v2                       # 确定性
    assert len(v1) == 64
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-6)  # L2 归一
    assert emb.embed(b"") != v1           # 不同输入 → 不同向量


def test_hash_embedder_satisfies_protocol():
    assert isinstance(HashEmbedder(), MediaEmbedder)


def test_factory_hash_and_unknown():
    assert isinstance(create_media_embedder("hash"), MediaEmbedder)
    assert "hash" in SUPPORTED_MEDIA_BACKENDS
    with pytest.raises(ValueError):
        create_media_embedder("does-not-exist")


def test_factory_clip_without_deps_raises(monkeypatch):
    # 离线环境通常缺 open_clip/torch/PIL → 惰性导入应抛 RuntimeError。
    # 若宿主恰好装了依赖，则跳过（不假设必须失败）。
    try:
        import open_clip  # noqa: F401
        import PIL  # noqa: F401
        import torch  # noqa: F401
        pytest.skip("multimodal deps present; lazy-import failure path not reachable")
    except ImportError:
        with pytest.raises(RuntimeError):
            create_media_embedder("clip")
