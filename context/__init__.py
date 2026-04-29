"""上下文管理层：精炼/去重/预算控制/按需加载。"""
from .manager import ContextManager, ContextBudget, RefinedItem

__all__ = ["ContextManager", "ContextBudget", "RefinedItem"]
