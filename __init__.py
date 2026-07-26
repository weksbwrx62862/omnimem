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
import warnings as _warnings
from pathlib import Path
from typing import Any  # noqa: F401

# ★ M9-23: jieba 内部使用已弃用的 pkg_resources（setuptools>=81 将移除），
# 在包入口统一屏蔽该 UserWarning（覆盖 bm25/fts5/entity 等全部 import 点）；
# 运行环境另以 setuptools<81 钉版兜底，见 requirements.txt
_warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning
)

# 当作为独立包运行时，将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    # 优先使用真实 Hermes 框架的 MemoryProvider ABC
    from agent.memory_provider import MemoryProvider  # type: ignore[import-not-found]
except ImportError:
    # Hermes 框架不可用时，降级为 object，使插件可独立运行（SDK 模式）
    import sys as _sys
    _sys.modules.setdefault("agent", type("agent", (), {})())
    _sys.modules.setdefault(
        "agent.memory_provider",
        type("agent.memory_provider", (), {"MemoryProvider": object})(),
    )
    MemoryProvider = object  # type: ignore[assignment,misc]

# Register 'omnimem' package alias so absolute imports (e.g. "from omnimem.provider import ...")
# work regardless of how the plugin is loaded (bundled vs user-installed).
# 当插件以非 package 方式加载（如直接 import __init__）时，__path__ 缺失会导致
# 别名模块无法被识别为 package，从而 from omnimem.provider import ... 失败。
# 此处补全 __path__ 以支持子模块绝对导入；真实 package 加载场景下 __path__ 已存在，不会覆盖。
if "__path__" not in globals():
    __path__ = [str(_PROJECT_ROOT)]
sys.modules.setdefault("omnimem", sys.modules[__name__])

# 注意：OmniMemProvider 不在顶层导入，改为在 register() 内部延迟导入，
# 以避免 import 阶段加载完整 provider 链（原耗时 ~858ms，目标 < 200ms）。


def register(ctx: Any) -> None:
    """Register OmniMem as a memory provider plugin."""
    # ctx 校验：容忍 None / 非法类型，避免插件加载器崩溃
    if ctx is None or not hasattr(ctx, "register_memory_provider"):
        import logging
        logging.getLogger(__name__).debug(
            "omnimem: ctx 无效或缺少 register_memory_provider 方法，跳过注册"
        )
        return

    try:
        # 延迟导入：仅在真正注册时加载 provider 模块，加快插件 import 阶段
        from omnimem.provider import OmniMemProvider
        ctx.register_memory_provider(OmniMemProvider())
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception(
            "omnimem: register 失败: %s", e
        )
