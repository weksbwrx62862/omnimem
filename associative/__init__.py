"""OmniMem 联想记忆模块。

联想扩散引擎 + 启动效应缓存：
  - AssociativeSpreader: KG/语义多跳扩散
  - 通过 recall mode="associative" 触发
"""

from omnimem.associative.spreader import AssociativeSpreader

__all__ = ["AssociativeSpreader"]
