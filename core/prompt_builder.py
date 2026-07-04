"""系统提示词构建器 — 双层架构（确定性层+动态层）。

从 core/tool_router.py 拆分而来，解决 SystemPromptBuilder 与 tool_router 之间的循环依赖。

双层架构（DeepAgents 启发）:
  - **确定性层**（始终注入）: CoreBlock 身份 + 稳定记忆（preference/correction 且 confidence >= 阈值）。
    这些是高价值、会话不变的事实。
  - **动态层**（预算控制）: 低置信度的 preference/correction + fact，受字符预算和去重约束。

确定性层缓存一次（``_stable_cache``）并在多轮中复用，只有动态层每轮重新评估。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from omnimem.context.manager import ContextManager

logger = logging.getLogger(__name__)

# 稳定记忆置信度阈值默认值（可通过 config.stable_confidence_threshold 覆盖）
STABLE_CONFIDENCE_THRESHOLD = 4
# 稳定缓存刷新轮次默认值（可通过 config.stable_cache_turns 覆盖）
STABLE_CACHE_TURNS = 10


def build_system_prompt(
    data_dir: str,
    store: Any,
    core_block: Any,
    context_manager: Any,
    config: Any,
    turn_count: int,
    system_prompt_cache_turn: int,
    system_prompt_cache_value: str,
    last_query: str,
) -> tuple[str, int, str]:
    """构建 OmniMem 系统提示词块。

    拆分为两层（DeepAgents 启发）:
      - **确定性层**（始终注入）: CoreBlock 身份 + 稳定记忆（preference/correction 且 confidence >= 阈值）。
        这些是高价值、会话不变的事实。
      - **动态层**（预算控制）: 低置信度的 preference/correction + fact，受字符预算和去重约束。

    确定性层缓存一次（``_stable_cache``）并在多轮中复用，只有动态层每轮重新评估。
    """
    if system_prompt_cache_turn == turn_count:
        return system_prompt_cache_value, system_prompt_cache_turn, system_prompt_cache_value

    stable_confidence_threshold_cfg = config.get("stable_confidence_threshold", STABLE_CONFIDENCE_THRESHOLD)
    stable_cache_turns_cfg = config.get("stable_cache_turns", STABLE_CACHE_TURNS)  # Refresh stable memories every N turns

    parts = [
        "## OmniMem Memory System (Unified)",
        f"Memory directory: {data_dir}",
        "",
    ]

    # ── 确定性层: 稳定记忆（高置信度）──────────────────────────
    # 这些始终注入，不受预算限制 — 它们代表用户已验证的、长期有效的偏好和纠正。
    # 缓存稳定条目以避免重复 store.search() 调用。
    stable_entries = []
    dynamic_boot_entries = []

    # 使用缓存的稳定条目（如果可用且未过期）
    stable_cache = getattr(context_manager, '_stable_cache', None)
    stable_cache_turn = getattr(context_manager, '_stable_cache_turn', -1)

    # 确保 stable_cache_turn 是整数（处理测试中的 MagicMock）
    if not isinstance(stable_cache_turn, int):
        stable_cache_turn = -1

    if (stable_cache is None or
        stable_cache_turn < 0 or
        turn_count - stable_cache_turn >= stable_cache_turns_cfg):

        # 刷新稳定缓存
        for mtype in ("preference", "correction"):
            entries = store.search(memory_type=mtype, limit=10)
            for e in entries:
                e["_mtype"] = mtype
                if (e.get("confidence") or 3) >= stable_confidence_threshold_cfg:
                    stable_entries.append(e)
                else:
                    dynamic_boot_entries.append(e)

        # 缓存稳定条目
        context_manager._stable_cache = stable_entries
        context_manager._stable_cache_turn = turn_count
    else:
        # 使用缓存的稳定条目
        stable_entries = stable_cache
        # 仍需获取动态条目
        for mtype in ("preference", "correction"):
            entries = store.search(memory_type=mtype, limit=10)
            for e in entries:
                e["_mtype"] = mtype
                if (e.get("confidence") or 3) < stable_confidence_threshold_cfg:
                    dynamic_boot_entries.append(e)

    # Facts 始终是动态的（依赖上下文）
    fact_entries = []
    for e in store.search(memory_type="fact", limit=15):
        e["_mtype"] = "fact"
        fact_entries.append(e)

    if not stable_entries and not dynamic_boot_entries and not fact_entries:
        parts.append("### Identity")
        parts.append(core_block.identity_block)
        result = "\n".join(parts)
        return result, turn_count, result

    max_summary = context_manager.max_summary_chars
    seen_fps = set(context_manager.get_injected_fingerprints())

    def _refine_and_add(entries: list[dict[str, Any]], budget_remaining: int) -> tuple[list[str], int]:
        """去重并添加条目到提示词，受字符预算限制。

        使用指纹相似度检测去重，避免重复内容注入。
        """
        lines = []
        used = 0
        for entry in entries:
            raw = entry.get("content", "")
            if not raw:
                continue
            summary = ContextManager.refine_content(raw, max_summary)
            if len(summary) < 3:
                continue
            fp = ContextManager._content_fingerprint(summary)
            if fp:
                is_dup = any(
                    ContextManager._fingerprint_similarity(fp, existing) > 0.7
                    for existing in seen_fps
                )
                if is_dup:
                    continue
                seen_fps.add(fp)
                context_manager.add_persistent_fingerprint(fp)
            line = f"- [{entry.get('_mtype', 'fact')}] {summary}"
            if used + len(line) + 1 > budget_remaining:
                break
            lines.append(line)
            used += len(line) + 1
        return lines, used

    refined_lines = []
    total_chars = 0

    # 1. 稳定层（确定性）— 始终注入，无预算限制
    if stable_entries:
        stable_lines, stable_used = _refine_and_add(stable_entries, 9999)
        refined_lines.extend(stable_lines)
        total_chars += stable_used

    # 2. 动态层（概率性）— 预算控制
    base_budget = config.get("system_prompt_char_limit", 500)
    query_kw_count = (
        len(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", last_query.lower()))
        if last_query
        else 0
    )
    char_budget = base_budget + min(query_kw_count * 40, 300)
    remaining = char_budget - total_chars

    if remaining > 50 and dynamic_boot_entries:
        dyn_lines, dyn_used = _refine_and_add(dynamic_boot_entries, remaining)
        refined_lines.extend(dyn_lines)
        total_chars += dyn_used

    remaining = char_budget - total_chars
    if remaining > 50 and fact_entries:
        fact_lines, fact_used = _refine_and_add(fact_entries, remaining)
        refined_lines.extend(fact_lines)
        total_chars += fact_used

    if refined_lines:
        parts.append("### Core Memories (summaries — use omni_detail for full content)")
        parts.extend(refined_lines)
        parts.append("")

    parts.append("### Identity")
    parts.append(core_block.identity_block)

    result = "\n".join(parts)
    return result, turn_count, result
