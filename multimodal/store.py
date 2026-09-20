"""内容寻址媒体存储（改进项 #12）。

把 MediaPart 的字节流按 sha256 落盘到 data_dir/media/<前2位>/<sha>，并写一个
同名 .meta.json 侧车记录 kind/mime/size/path。相同内容天然去重；无需数据库、
完全离线，风格对齐 TemporalKnowledgeGraph 的本地存储层。
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from omnimem.multimodal.types import MediaPart, MediaRef


class MediaStore:
    """文件系统媒体仓库（内容寻址 + sidecar 元数据）。"""

    def __init__(self, data_dir: str | Path, subdir: str = "media") -> None:
        self._root = Path(data_dir) / subdir
        self._root.mkdir(parents=True, exist_ok=True)

    # ── 路径工具 ──────────────────────────────────────
    def _bucket(self, sha: str) -> Path:
        return self._root / (sha[:2] or "00")

    def _blob_path(self, sha: str) -> Path:
        return self._bucket(sha) / sha

    def _meta_path(self, sha: str) -> Path:
        return self._bucket(sha) / f"{sha}.meta.json"

    # ── 写入 ──────────────────────────────────────────
    def put(self, part: MediaPart) -> MediaRef | None:
        """落盘一个内联媒体片段；无内联数据（仅 URL）或已存在则只回引用。"""
        if not part.has_inline:
            return None
        sha = part.sha256
        ref = MediaRef(
            sha256=sha,
            kind=part.kind,
            mime=part.mime,
            size=len(part.data),
            path=str(self._blob_path(sha)),
        )
        blob = self._blob_path(sha)
        if not blob.exists():
            self._bucket(sha).mkdir(parents=True, exist_ok=True)
            tmp = blob.with_name(blob.name + ".tmp")
            tmp.write_bytes(part.data)
            os.replace(tmp, blob)  # 原子替换，避免半截文件
        meta = self._meta_path(sha)
        if not meta.exists():
            self._atomic_write_json(meta, ref.to_dict())
        return ref

    def put_all(self, parts: Iterable[MediaPart]) -> list[MediaRef]:
        refs: list[MediaRef] = []
        for p in parts:
            ref = self.put(p)
            if ref is not None:
                refs.append(ref)
        return refs

    def _atomic_write_json(self, path: Path, payload: dict) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    # ── 读取 ──────────────────────────────────────────
    def exists(self, sha: str) -> bool:
        return self._blob_path(sha).exists()

    def get(self, ref: MediaRef | str) -> bytes | None:
        """按引用或 sha 取回字节流；不存在返回 None。"""
        sha = ref if isinstance(ref, str) else ref.sha256
        blob = self._blob_path(sha)
        return blob.read_bytes() if blob.exists() else None

    def stat(self, sha: str) -> MediaRef | None:
        meta = self._meta_path(sha)
        if not meta.exists():
            return None
        try:
            return MediaRef.from_dict(json.loads(meta.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return None

    def list_refs(self) -> list[MediaRef]:
        """扫描所有 sidecar，返回媒体引用列表（按 sha 稳定排序）。"""
        refs: list[MediaRef] = []
        for meta in sorted(self._root.rglob("*.meta.json")):
            try:
                refs.append(MediaRef.from_dict(json.loads(meta.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError):
                continue
        return refs
