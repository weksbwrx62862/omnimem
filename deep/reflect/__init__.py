"""deep/reflect 子包 —— L3 深层反思引擎。

向后兼容：原 omnimem.deep.reflect 的公共 API 已全部迁移至此，
旧导入路径通过 deep/reflect.py shim 继续可用。
"""

from __future__ import annotations

from omnimem.deep.reflect.disposition import (
    Disposition,
    ReflectionContext,
    ReflectResult,
    _apply_disposition,
)
from omnimem.deep.reflect.pipeline import ReflectEngine

__all__ = [
    "Disposition",
    "ReflectEngine",
    "ReflectResult",
    "ReflectionContext",
    "_apply_disposition",
]
