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

from plugins.memory.omnimem.provider import OmniMemProvider


def register(ctx) -> None:
    """Register OmniMem as a memory provider plugin."""
    ctx.register_memory_provider(OmniMemProvider())
