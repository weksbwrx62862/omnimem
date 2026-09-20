"""改进项 #12 多模态摄入接线单测（离线）。

以轻量 stub 承载 OmniMemSDK.memorize_media 的自引用，验证：媒体落盘去重、
可检索正文拼装、embed 可选路径、URL-only 引用不落盘、返回结构。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omnimem.multimodal.types import MediaPart
from omnimem.sdk import OmniMemSDK


class _Stub:
    """仅暴露 memorize_media 依赖的成员：_data_dir 与 memorize()。"""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self.captured: dict[str, Any] = {}

    def memorize(self, content: str, memory_type: str = "fact", **kwargs: Any) -> dict:
        self.captured = {"content": content, "memory_type": memory_type, **kwargs}
        return {"status": "stored", "memory_id": "m-1"}


def _run(stub: _Stub, **kw) -> dict:
    return OmniMemSDK.memorize_media(stub, **kw)


def test_inline_media_persisted_and_caption_searchable(tmp_path):
    stub = _Stub(tmp_path)
    res = _run(stub, content="a chart snapshot",
               media=[MediaPart(kind="image", mime="image/png", data=b"PNGDATA")],
               memory_type="event")
    assert res["status"] == "stored"
    assert res["media"] and res["media"][0]["kind"] == "image"
    # 正文包含原始说明 + 媒体引用行（走文本检索通道）
    assert stub.captured["content"].startswith("a chart snapshot")
    assert "sha256=" in stub.captured["content"]
    assert stub.captured["memory_type"] == "event"
    # 内容寻址：blob 落盘可回读
    from omnimem.multimodal import MediaStore
    got = MediaStore(tmp_path).get(res["media"][0]["sha256"])
    assert got == b"PNGDATA"


def test_dedup_identical_blobs(tmp_path):
    stub = _Stub(tmp_path)
    p = MediaPart(kind="image", mime="image/png", data=b"SAME")
    res = _run(stub, content="two identical", media=[p, MediaPart(kind="image", mime="image/png", data=b"SAME")])
    # put_all 去重 -> 单条引用
    assert len(res["media"]) == 1


def test_url_only_not_stored(tmp_path):
    stub = _Stub(tmp_path)
    res = _run(stub, content="remote", media=[MediaPart(kind="image", mime="image/png", url="https://x/y.png")])
    assert res["media"] == []  # 无内联字节 -> 不落盘
    assert stub.captured["content"] == "remote"


def test_embed_optional_hash(tmp_path):
    stub = _Stub(tmp_path)
    res = _run(stub, content="emb", media=[MediaPart(kind="image", mime="image/png", data=b"DATA")],
               embed=True, backend="hash")
    emb = res["media_embeddings"]
    assert emb["backend"] == "hash"
    assert emb["dimension"] > 0
    assert emb["vectors"]  # 至少一个向量


def test_multimodal_content_autoparsed(tmp_path):
    stub = _Stub(tmp_path)
    content = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
            + __import__("base64").b64encode(b"BLOB").decode()}},
    ]
    res = _run(stub, content=content)
    assert len(res["media"]) == 1
    assert "look" in stub.captured["content"]


def test_unknown_media_type_nonfatal(tmp_path):
    stub = _Stub(tmp_path)
    res = _run(stub, content="bad backend", media=[MediaPart(kind="image", mime="image/png", data=b"D")],
               embed=True, backend="nope")
    # 工厂报错被吞掉为 error 字段，记忆仍写入
    assert res["status"] == "stored"
    assert "error" in res["media_embeddings"]
