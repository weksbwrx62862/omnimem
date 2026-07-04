"""系统提示词构建器 — 负责 system_prompt_block 生成和 AGENTS.md 集成。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omnimem.core.prompt_builder import build_system_prompt

logger = logging.getLogger(__name__)


class SystemPromptBuilder:
    """构建 OmniMem 系统提示词块。

    职责:
      1. 调用 prompt_builder.build_system_prompt 生成基础提示词
      2. 集成 AGENTS.md 项目级确定性经验
      3. 检测 AGENTS.md 与记忆偏好的冲突
      4. 向量检索降级状态提示
    """

    def __init__(
        self,
        data_dir: Path,
        store: Any,
        core_block: Any,
        context_manager: Any,
        config: Any,
        retrieval: Any,
    ) -> None:
        self._data_dir = data_dir
        self._store = store
        self._core_block = core_block
        self._context_manager = context_manager
        self._config = config
        self._retrieval = retrieval
        # AGENTS.md 缓存
        self._agents_md_cache: str | None = None

    def build(
        self,
        turn_count: int,
        system_prompt_cache_turn: int,
        system_prompt_cache_value: str,
        last_query: str,
    ) -> tuple[str, int, str]:
        """构建系统提示词块。

        返回: (提示词文本, 新缓存轮次, 新缓存值)
        """
        result, cache_turn, cache_value = build_system_prompt(
            data_dir=str(self._data_dir),
            store=self._store,
            core_block=self._core_block,
            context_manager=self._context_manager,
            config=self._config,
            turn_count=turn_count,
            system_prompt_cache_turn=system_prompt_cache_turn,
            system_prompt_cache_value=system_prompt_cache_value,
            last_query=last_query,
        )

        # AGENTS.md 集成 — 添加项目级确定性经验
        agents_md_content = self._load_agents_md()
        if agents_md_content:
            try:
                pref_entries = self._store.search(memory_type="preference", limit=5)
                conflicts = self._check_agents_md_conflicts(agents_md_content, pref_entries)
                if conflicts:
                    conflict_note = "\n[⚠ AGENTS.md vs Memory conflicts detected: " + "; ".join(conflicts[:3]) + "]"
                    result += conflict_note
            except Exception as e:
                logger.debug("AGENTS.md conflict check failed: %s", e)

            result += f"\n\n### Project Context (from AGENTS.md)\n{agents_md_content}"

        # 向量检索降级状态提示
        if hasattr(self._retrieval, 'retriever'):
            try:
                health = self._retrieval.retriever._check_vector_health()
                if health.get("vector_count", -1) <= 0 or health.get("breaker_state") == "open":
                    result += "\n[⚠ OmniMem: 向量检索不可用，已降级到关键词模式]"
            except Exception:
                logger.debug("OmniMem vector health check failed", exc_info=True)

        return result, cache_turn, cache_value

    def _load_agents_md(self) -> str:
        """从当前工作目录加载 AGENTS.md 内容。

        返回内容字符串，未找到则返回空字符串。
        """
        if self._agents_md_cache is not None:
            return self._agents_md_cache

        try:
            candidates = [
                Path.cwd() / "AGENTS.md",
                Path.cwd() / "agents.md",
                Path.home() / ".hermes" / "AGENTS.md",
            ]

            for candidate in candidates:
                if candidate.exists():
                    try:
                        content = candidate.read_text(encoding="utf-8").strip()
                        if content:
                            if len(content) > 2000:
                                content = content[:2000] + "\n...(truncated)"
                            self._agents_md_cache = content
                            return content
                    except Exception as e:
                        logger.debug("Could not read %s: %s", candidate, e)

            self._agents_md_cache = ""
            return ""
        except Exception as e:
            logger.debug("AGENTS.md loading failed: %s", e)
            self._agents_md_cache = ""
            return ""

    def _extract_agents_md_instructions(self, content: str) -> list[str]:
        """从 AGENTS.md 内容中提取关键指令。

        返回指令字符串列表，用于冲突检测。
        """
        if not content:
            return []

        instructions = []
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                instructions.append(line[2:].strip())
            elif line.startswith('## ') or line.startswith('### '):
                header = line.lstrip('#').strip()
                if any(keyword in header.lower() for keyword in ['must', 'should', 'always', 'never', 'do', "don't"]):
                    instructions.append(header)

        return instructions[:10]

    def _check_agents_md_conflicts(self, agents_md_content: str, memory_preferences: list) -> list[str]:
        """检测 AGENTS.md 指令与记忆偏好之间的冲突。

        返回冲突描述列表。
        """
        if not agents_md_content or not memory_preferences:
            return []

        conflicts = []
        agents_instructions = self._extract_agents_md_instructions(agents_md_content)

        for pref in memory_preferences:
            pref_content = pref.get("content", "").lower()
            for instruction in agents_instructions:
                instruction_lower = instruction.lower()
                if ("never" in instruction_lower and "always" in pref_content) or \
                   ("always" in instruction_lower and "never" in pref_content) or \
                   ("don't" in instruction_lower and "do" in pref_content) or \
                   ("do" in instruction_lower and "don't" in pref_content):
                    conflicts.append(f"AGENTS.md says '{instruction}' but memory says '{pref.get('content', '')}'")

        return conflicts[:5]
