"""OmniMem 工具处理器子模块。

将 provider.py 中的大型方法拆分为独立模块，通过委托调用保持接口不变。
"""

from plugins.memory.omnimem.handlers.schemas import get_tool_schemas
from plugins.memory.omnimem.handlers.memorize import handle_memorize
from plugins.memory.omnimem.handlers.recall import handle_recall
from plugins.memory.omnimem.handlers.govern import handle_govern

__all__ = [
    "get_tool_schemas",
    "handle_memorize",
    "handle_recall",
    "handle_govern",
]
