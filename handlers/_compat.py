"""内置记忆安全扫描兼容层 (从 provider.py 移植)。

供 handlers 和 provider 共享，避免循环导入。

本模块已委托到 utils.security.SecurityValidator，保留此文件以维持
向后兼容的导入路径。
"""

from typing import Optional

from plugins.memory.omnimem.utils.security import SecurityValidator


def compat_scan_memory_content(content: str) -> Optional[str]:
    """扫描内容是否包含注入/外泄模式。

    委托 SecurityValidator.scan_threats() 实现，支持 Unicode 归一化、
    同形字符检测和零宽字符过滤。
    """
    return SecurityValidator.scan_threats(content)
