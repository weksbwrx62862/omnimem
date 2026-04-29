"""OmniMem 测试包。"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock agent.memory_provider 模块（Hermes 框架依赖）
_mock_agent = MagicMock()
_mock_agent.memory_provider = MagicMock()
_mock_agent.memory_provider.MemoryProvider = object
sys.modules.setdefault("agent", _mock_agent)
sys.modules.setdefault("agent.memory_provider", _mock_agent.memory_provider)

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
