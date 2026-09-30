"""OmniMem recall 处理器。

仅保留：schema 校验、依赖注入、调用 RecallService、结果序列化。
核心业务逻辑已下沉到 services.recall_service。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from omnimem.handlers.deps import HandlerDependencies, extract_deps
from omnimem.services.recall_service import RecallService, _extract_query_keywords
from omnimem.utils.logging import sanitize_for_log

logger = logging.getLogger(__name__)

__all__ = ["handle_recall", "async_handle_recall", "_extract_query_keywords"]


def _validate_recall_args(args: dict[str, Any]) -> str | None:
    """对 omni_recall 参数做轻量 schema 校验。"""
    if not isinstance(args, dict):
        return "args must be a dict"
    if "query" not in args or not isinstance(args["query"], str) or not args["query"].strip():
        return "query is required and must be a non-empty string"

    mode = args.get("mode", "rag")
    if mode not in {"rag", "llm", "associative", "agentic"}:
        return f"invalid mode: {mode}"

    max_tokens = args.get("max_tokens", 1500)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        return f"max_tokens must be a positive integer, got {max_tokens}"

    type_filter = args.get("type_filter")
    if type_filter is not None:
        if not isinstance(type_filter, list) or not all(isinstance(t, str) for t in type_filter):
            return "type_filter must be a list of strings"
        valid_types = {
            "fact", "preference", "correction", "skill", "procedural",
            "event", "action", "reasoning", "knowledge", "workflow",
            "project", "convention", "maintenance", "user_profile",
        }
        invalid = [t for t in type_filter if t not in valid_types]
        if invalid:
            return f"invalid type(s) in type_filter: {invalid}"

    return None


def _audit_recall(
    deps: HandlerDependencies,
    args: dict[str, Any],
    result: dict[str, Any],
    started_at: float,
) -> None:
    """★ P2-3f: 为 RecallService 未覆盖的 recall 结局补审计。

    status == "found" 由 RecallService 自身审计（含命中数），这里只写提前返回的路径
    （参数校验失败 / no_results），保证「每一次 omni_recall 调用都留下痕迹」。
    审计写入失败绝不阻断检索。
    """
    if result.get("status") == "found":
        return
    if deps.audit_logger is None:
        return
    try:
        deps.audit_logger.log(
            "recall",
            details={
                "query": sanitize_for_log(args.get("query", ""), 200),
                "mode": args.get("mode", "rag"),
                "status": result.get("status", ""),
                "reason": result.get("reason", result.get("message", "")),
                "hits": len(result.get("memories") or []),
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
            },
            result="error" if result.get("status") == "error" else "no_results",
            instance_id=deps.instance_id,
        )
    except Exception as e:
        logger.warning("OmniMem: recall 审计写入失败（不阻断检索）: %s", e)


def handle_recall(provider: Any, args: dict[str, Any]) -> str:
    """主动检索记忆 — 经 ContextManager 精炼后返回精简摘要。"""
    started_at = time.perf_counter()
    deps = extract_deps(provider)

    validation_error = _validate_recall_args(args)
    if validation_error:
        result = {"status": "error", "reason": validation_error}
        _audit_recall(deps, args, result, started_at)
        return json.dumps(result)

    service = RecallService(deps=deps)
    result = service.handle(args)
    _audit_recall(deps, args, result, started_at)
    return json.dumps(result, ensure_ascii=False)


async def async_handle_recall(provider: Any, args: dict[str, Any]) -> str:
    """异步主动检索记忆。"""
    started_at = time.perf_counter()
    deps = extract_deps(provider)

    validation_error = _validate_recall_args(args)
    if validation_error:
        result = {"status": "error", "reason": validation_error}
        _audit_recall(deps, args, result, started_at)
        return json.dumps(result)

    service = RecallService(deps=deps)
    result = await service.async_handle(args)
    _audit_recall(deps, args, result, started_at)
    return json.dumps(result, ensure_ascii=False)
