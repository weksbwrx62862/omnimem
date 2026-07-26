"""OmniMem 配置包 — 同时提供配置管理器和外部化词典。"""

from omnimem.config._config import DEFAULTS as DEFAULTS
from omnimem.config._config import OmniMemConfig as OmniMemConfig

__all__ = ["OmniMemConfig", "DEFAULTS"]
