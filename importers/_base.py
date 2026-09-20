"""外部记忆项目导入转换层（改进项 #8）。

将 mem0 / Letta / Zep / Graphiti / Cognee 等开源记忆项目的导出 JSON
转换为 OmniMem 原生导入信封，再复用 MemoryImporter.import_json 落库。

设计约束：
- 转换函数保持纯函数（不触碰 store / LLM），可离线单测；
- 上游字段命名不统一，采用候选键兜底提取，避免脆断；
- 落库统一走 core/import_export.MemoryImporter，不复造写入路径。
"""
from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_NATIVE_VERSION = "2.0"

# 上游字段命名各异，按优先级尝试提取正文。
_CONTENT_KEYS = (
    "content", "memory", "text", "value", "fact", "observation",
    "summary", "description", "answer", "message", "data",
)
_ID_KEYS = ("id", "memory_id", "uuid", "passage_id", "fact_id", "node_id", "edge_id")
_TIME_KEYS = (
    "created_at", "timestamp", "time", "valid_at", "updated_at",
    "submitted_for_processing", "last_accessed_at", "date",
)
_TAGS_KEYS = ("tags", "categories", "entities", "metadata", "labels", "keywords")

# 命中以下标签/类型关键字时归类为 preference。
_PREFERENCE_HINTS = ("preference", "prefer", "偏好", "喜好", "喜欢")
_EVENT_HINTS = ("event", "incident", "meeting", "事件", "会议")


def _coerce_text(value: Any) -> str:
    """把任意候选值规整为一段文本。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        # 嵌套结构优先取其中的常见正文字段
        for k in _CONTENT_KEYS:
            if k in value:
                inner = _coerce_text(value[k])
                if inner:
                    return inner
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        parts = [_coerce_text(v) for v in value]
        return " ".join(p for p in parts if p).strip()
    return str(value).strip()


def _first_text(mapping: dict[str, Any], keys: Iterable[str]) -> str:
    for k in keys:
        if k in mapping:
            text = _coerce_text(mapping[k])
            if text:
                return text
    return ""


def _flatten_tags(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for k, v in value.items():
            out.append(str(k))
            if isinstance(v, (str, int, float)):
                out.append(f"{k}:{v}")
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, dict):
                out.extend(str(x) for x in item.values() if isinstance(x, (str, int, float)))
            else:
                out.append(_coerce_text(item))
        return [t for t in out if t]
    return [_coerce_text(value)]


def _infer_type(content: str, memory_type: str, tags: list[str]) -> str:
    """在缺少显式类型时，用启发式给一个稳定的粗分类。"""
    mt = (memory_type or "").strip().lower()
    if mt:
        # 直接沿用可识别的显式类型
        if mt in {"fact", "preference", "event", "entity", "procedure"}:
            return mt
    blob = f"{mt} {' '.join(tags)}".lower()
    hay = f"{content} {blob}".lower()
    if any(h in blob or h in hay for h in _PREFERENCE_HINTS):
        return "preference"
    if any(h in blob or h in hay for h in _EVENT_HINTS):
        return "event"
    return "fact"


def _normalize_time(value: Any) -> str:
    """尽力把上游时间字段转成 ISO 8601；失败则回落到当前 UTC。"""
    if not value:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, (int, float)):
        # epoch 秒或毫秒
        ts = float(value)
        if ts > 1e12:  # 毫秒
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return datetime.now(timezone.utc).isoformat()
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def normalize_record(
    raw: dict[str, Any],
    *,
    source: str,
    wing: str,
    room: str,
    privacy: str = "personal",
    confidence: int = 3,
) -> dict[str, Any] | None:
    """把单条上游记录转换为 OmniMem 原生 record；无正文则返回 None。"""
    if not isinstance(raw, dict):
        # 允许上游直接给纯字符串列表
        text = _coerce_text(raw)
        raw = {"content": text} if text else {}
    content = _first_text(raw, _CONTENT_KEYS)
    if not content:
        return None
    explicit_type = _coerce_text(raw.get("memory_type") or raw.get("type") or raw.get("category") or "")
    tags = _flatten_tags(raw.get("tags") or raw.get("metadata") or raw.get("categories") or raw.get("labels"))
    created = _normalize_time(_first_present(raw, _TIME_KEYS))
    native: dict[str, Any] = {
        "memory_id": _coerce_text(_first_present(raw, _ID_KEYS)) or uuid.uuid4().hex[:12],
        "content": content,
        "summary": content[:200].replace("\n", " "),
        "type": _infer_type(content, explicit_type, tags),
        "wing": wing,
        "room": room or source,
        "privacy": privacy,
        "confidence": confidence,
        "created_at": created,
        "access_count": 0,
        "source": source,
    }
    if tags:
        native["tags"] = tags
    return native


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in mapping and mapping[k] not in (None, "", [], {}):
            return mapping[k]
    return None


def normalize_list(value: Any) -> list[Any]:
    """把上游容器规整成列表：list 原样、单条 dict 包一层、其余丢弃。"""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def extract_records(data: Any) -> list[dict[str, Any]]:
    """从上游导出结构里取出记录列表，兼容裸列表与常见包裹键。"""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, (dict, str)) or _coerce_text(r)]
    if isinstance(data, dict):
        for key in ("results", "memories", "items", "passages", "facts", "data",
                    "nodes", "edges", "episodes", "messages", "records"):
            val = data.get(key)
            if isinstance(val, list):
                return [r for r in val if isinstance(r, (dict, str)) or _coerce_text(r)]
        # 未命中已知键：把顶层 dict 当作单条记录
        return [data]
    return []


def convert(
    data: Any,
    *,
    source: str,
    wing: str = "personal",
    room: str = "imported",
    privacy: str = "personal",
    confidence: int = 3,
) -> dict[str, Any]:
    """把上游导出的原始 JSON 结构转换为 OmniMem 原生导入信封。"""
    raw_records = extract_records(data)
    native: list[dict[str, Any]] = []
    for raw in raw_records:
        rec = normalize_record(
            raw, source=source, wing=wing, room=room, privacy=privacy, confidence=confidence
        )
        if rec:
            native.append(rec)
    return {
        "version": _NATIVE_VERSION,
        "encrypted": False,
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "source_count": len(raw_records),
        "count": len(native),
        "memories": native,
    }


def convert_file(
    input_path: str | Path,
    *,
    source: str,
    output_path: str | Path | None = None,
    **convert_kwargs: Any,
) -> dict[str, Any]:
    """读取上游导出文件并转换为原生信封；给定 output_path 时落盘。"""
    raw_text = Path(input_path).read_text(encoding="utf-8")
    data = json.loads(raw_text)
    envelope = convert(data, source=source, **convert_kwargs)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    return envelope
