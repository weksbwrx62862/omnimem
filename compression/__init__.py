"""压缩引擎模块。"""

from plugins.memory.omnimem.compression.micro import microcompact
from plugins.memory.omnimem.compression.collapse import head_tail_collapse
from plugins.memory.omnimem.compression.line_compress import structured_line_compress
from plugins.memory.omnimem.compression.llm_summary import llm_summarize
from plugins.memory.omnimem.compression.priority import priority_compress
