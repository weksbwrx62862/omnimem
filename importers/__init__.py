"""OmniMem 外部记忆导入器（改进项 #8）。

对外入口：
    convert_source(data, source="mem0", ...)  -> 原生导入信封 dict
    convert_file(path, source=..., output_path=None, ...) -> 原生导入信封 dict
    import_file(sdk, path, source=..., **kw)   -> 落库统计

转换全部为纯函数（不触碰 store/LLM），落库统一委托给
OmniMemSDK.import_memories -> core/import_export.MemoryImporter。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import _base
from .sources import SOURCE_ADAPTERS, SUPPORTED_SOURCES

__all__ = [
    "SUPPORTED_SOURCES",
    "convert",
    "convert_file",
    "convert_source",
    "import_file",
]


def convert_source(
    data: Any,
    *,
    source: str = "auto",
    wing: str = "personal",
    room: str = "imported",
    privacy: str = "personal",
    confidence: int = 3,
) -> dict[str, Any]:
    """按来源适配器拆解结构后，归一为 OmniMem 原生导入信封。"""
    adapter = SOURCE_ADAPTERS.get(source)
    if adapter is None:
        raise ValueError(f"不支持的导入来源: {source}（可选: {', '.join(SUPPORTED_SOURCES)}）")
    flat = adapter(data)
    envelope = _base.convert(
        flat, source=source, wing=wing, room=room, privacy=privacy, confidence=confidence
    )
    envelope["converted_at"] = datetime.now(timezone.utc).isoformat()
    return envelope


def convert_file(
    input_path: str | Path,
    *,
    source: str = "auto",
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """读取上游导出文件 -> 原生信封；给定 output_path 时落盘。"""
    import json

    raw = Path(input_path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # 本函数只服务上游记忆库的导出文件（mem0/letta/zep/graphiti/cognee）。
        # 以前这里抛裸的 JSONDecodeError，看不出"传错文件类型"还是"文件损坏"。
        raise ValueError(
            f"import_file 只接受上游记忆库导出的 JSON，当前文件不是合法 JSON：{input_path}"
            f"（{e.msg}，第 {e.lineno} 行第 {e.colno} 列）。"
            f"支持来源：{', '.join(SUPPORTED_SOURCES)}；"
            f"若要把 Markdown/纯文本变成记忆，请直接用 memorize / import_memories。"
        ) from e
    envelope = convert_source(data, source=source, **kwargs)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    return envelope


def import_file(
    sdk: Any,
    input_path: str | Path,
    *,
    source: str = "auto",
    skip_duplicates: bool = True,
    resolve_conflicts: bool = True,
    workdir: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """转换上游文件并复用 SDK 导入管线落库，返回导入统计。

    临时生成的原生信封写入 workdir（默认系统临时目录），便于问题排查时保留。
    """
    import tempfile

    target_dir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="omni_import_"))
    native_path = target_dir / f"{source}_native.json"
    convert_file(input_path, source=source, output_path=native_path, **kwargs)
    result = sdk.import_memories(
        str(native_path),
        skip_duplicates=skip_duplicates,
        resolve_conflicts=resolve_conflicts,
    )
    return {**result, "native_envelope": str(native_path)}


# 兼容 _base 通用入口的再导出
convert = _base.convert
