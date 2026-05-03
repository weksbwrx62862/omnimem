"""五层压缩管线 — 串联 micro / collapse / line_compress / llm_summary / priority。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from omnimem.compression.collapse import head_tail_collapse
from omnimem.compression.line_compress import structured_line_compress
from omnimem.compression.llm_summary import llm_summarize
from omnimem.compression.micro import microcompact
from omnimem.compression.priority import priority_compress

logger = logging.getLogger(__name__)


class CompressionPipeline:
    def __init__(self, llm_call_fn: Callable[[str], str] | None = None, config: Any = None) -> None:
        self._llm_call_fn = llm_call_fn
        self._config = config

    def compress(self, content: str, memory_type: str = "", priority: int = 2) -> str:
        lines = content.split("\n")
        lines = microcompact(lines)
        lines = head_tail_collapse(lines)
        lines = structured_line_compress(lines)
        result = "\n".join(lines)
        if self._llm_call_fn:
            summary = llm_summarize(result, self._llm_call_fn)
            result = summary.to_text()
        items = [{"content": result, "type": memory_type or "fact", "confidence": 3}]
        items = priority_compress(items)
        return items[0]["content"] if items else result
