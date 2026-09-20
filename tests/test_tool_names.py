"""core/tool_names.py 常量与 __all__ 一致性单测。"""
from __future__ import annotations

import omnimem.core.tool_names as tn


def test_values_are_stable():
    assert tn.OMNI_MEMORIZE == "omni_memorize"
    assert tn.OMNI_RECALL == "omni_recall"
    assert tn.MEMORY_COMPAT == "omni_memory_compat"


def test_all_lists_only_existing_names():
    for name in tn.__all__:
        assert hasattr(tn, name), name


def test_core_tool_names_prefix():
    names = [
        tn.OMNI_MEMORIZE, tn.OMNI_RECALL, tn.OMNI_GOVERN,
        tn.OMNI_REFLECT, tn.OMNI_COMPACT, tn.OMNI_DETAIL, tn.OMNI_RECORD_ACTION,
    ]
    assert all(n.startswith("omni_") for n in names)
    assert len(set(names)) == len(names)  # 无重名
