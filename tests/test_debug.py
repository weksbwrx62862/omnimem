"""utils/debug.py is_debug_mode 单测（离线）。"""
from __future__ import annotations

import pytest
from omnimem.utils.debug import is_debug_mode


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, False),   # 未设置
        ("", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("  true  ", True),  # 两端空白被 strip
        ("0", False),
        ("no", False),
        ("TRUE", False),     # 大小写敏感：仅接受小写 true/yes
    ],
)
def test_is_debug_mode(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("OMNIMEM_DEBUG", raising=False)
    else:
        monkeypatch.setenv("OMNIMEM_DEBUG", raw)
    assert is_debug_mode() is expected
