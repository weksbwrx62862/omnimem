"""OmniMem 工具名常量。

统一 ToolRouter、Provider 及 Schema 中使用的工具名字符串，
避免硬编码字符串分散在多处。
"""

from __future__ import annotations

# OmniMem 核心工具名
OMNI_MEMORIZE = "omni_memorize"
OMNI_RECALL = "omni_recall"
OMNI_GOVERN = "omni_govern"
OMNI_REFLECT = "omni_reflect"
OMNI_COMPACT = "omni_compact"
OMNI_DETAIL = "omni_detail"
OMNI_RECORD_ACTION = "omni_record_action"

# 兼容内置 memory 工具名（避免与 Hermes 核心工具 'memory' 重名）
MEMORY_COMPAT = "omni_memory_compat"

__all__ = [
    "OMNI_MEMORIZE",
    "OMNI_RECALL",
    "OMNI_GOVERN",
    "OMNI_REFLECT",
    "OMNI_COMPACT",
    "OMNI_DETAIL",
    "OMNI_RECORD_ACTION",
    "MEMORY_COMPAT",
]
