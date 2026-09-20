"""head_tail_collapse 单测（改进项 #2 覆盖，纯函数，离线）。

覆盖：短输入原样返回（阈值 head+tail+2）、折叠计数标记、head/tail 边界、
自定义 marker、空/零 head 等极端参数。
"""
from __future__ import annotations

from omnimem.compression.collapse import head_tail_collapse


def _lines(n: int) -> list[str]:
    return [f"L{i}" for i in range(n)]


def test_short_input_unchanged():
    lines = _lines(17)  # <= 5+10+2 = 17
    assert head_tail_collapse(lines) == lines


def test_boundary_equal_threshold_unchanged():
    lines = _lines(17)  # 恰好 == head+tail+2 → 不折叠
    assert head_tail_collapse(lines) == lines


def test_collapses_when_over_threshold():
    lines = _lines(18)  # > 17 → 折叠
    out = head_tail_collapse(lines)
    assert out[:5] == lines[:5]                 # 头部保留
    assert out[-10:] == lines[-10:]             # 尾部保留
    assert len(out) == 5 + 1 + 10
    marker = out[5]
    assert "middle section collapsed" in marker
    assert "(3 lines)" in marker                # 18-5-10 = 3


def test_custom_marker_text():
    out = head_tail_collapse(_lines(20), collapse_marker="<<cut>>")
    assert out[5].startswith("<<cut>>")
    assert "(5 lines)" in out[5]                # 20-5-10 = 5


def test_zero_head_positive_tail():
    out = head_tail_collapse(_lines(10), head_lines=0, tail_lines=3)
    assert out == ["[... middle section collapsed ...] (7 lines)", "L7", "L8", "L9"]


def test_empty_input():
    assert head_tail_collapse([]) == []


def test_large_fold_count():
    out = head_tail_collapse(_lines(100))
    assert "(85 lines)" in out[5]               # 100-5-10
    assert len(out) == 16
