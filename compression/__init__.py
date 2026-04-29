"""压缩引擎模块。"""

from omnimem.compression.collapse import head_tail_collapse as head_tail_collapse
from omnimem.compression.line_compress import structured_line_compress as structured_line_compress
from omnimem.compression.llm_summary import llm_summarize as llm_summarize
from omnimem.compression.micro import microcompact as microcompact
from omnimem.compression.priority import priority_compress as priority_compress
