"""OmniMem — 五层混合记忆系统，Hermes MemoryProvider 插件。

五层架构:
  L0 感知层  — 主动监控 + 信号检测 + 意图预测
  L1 工作记忆 — CoreBlock(常驻上下文) + Attachment(压缩后状态)
  L2 结构化记忆 — Wing/Room 宫殿导航 + Drawer/Closet 双存储
  L3 深层记忆 — Consolidation(事实→观察→心智模型) + 知识图谱
  L4 内化记忆 — KV Cache(高频) + LoRA(深层) [可选]

治理引擎(横切面):
  冲突仲裁 + 时间衰减 + 遗忘曲线 + 隐私分级 + 溯源追踪

安装: 将本目录放入 plugins/memory/omnimem/
配置: config.yaml → memory.provider: omnimem
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# 当作为独立包运行时，将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Mock agent.memory_provider 模块（Hermes 框架依赖）
_mock_agent = MagicMock()
_mock_agent.memory_provider = MagicMock()
_mock_agent.memory_provider.MemoryProvider = object
sys.modules.setdefault("agent", _mock_agent)
sys.modules.setdefault("agent.memory_provider", _mock_agent.memory_provider)

from omnimem.provider import OmniMemProvider


def register(ctx) -> None:
    """Register OmniMem as a memory provider plugin."""
    ctx.register_memory_provider(OmniMemProvider())
