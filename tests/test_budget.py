"""core/budget.py BudgetManager 单测（离线）。

强制走无 tiktoken 的兜底估算（len//4），使 token 计数可预测。
"""
from __future__ import annotations

import sys

import pytest
from omnimem.core.budget import _CHARS_PER_TOKEN, BudgetManager


@pytest.fixture(autouse=True)
def _force_fallback(monkeypatch):
    # sys.modules 置 None -> 函数内 `import tiktoken` 抛 ImportError -> 走 len//4
    monkeypatch.setitem(sys.modules, "tiktoken", None)


def test_max_tokens_property():
    assert BudgetManager(2048).max_tokens == 2048


def test_estimate_tokens_fallback_floor():
    b = BudgetManager()
    assert b.estimate_tokens("") == 1            # max(1, 0)
    assert b.estimate_tokens("a" * 8) == 2        # 8 // 4
    assert b.estimate_tokens("a" * 7) == 1        # 7 // 4 = 1


def test_fits():
    b = BudgetManager(max_tokens=10)
    assert b.fits("a" * 40)                       # 40//4=10 <=10
    assert not b.fits("a" * 44)                   # 11 > 10
    assert not b.fits("a" * 40, extra_tokens=1)   # 10+1>10


def test_trim_keeps_within_budget():
    b = BudgetManager()
    items = [{"content": "x" * 40}, {"content": "y" * 40}]  # 各 10 token
    out = b.trim_to_budget(items, max_tokens=10)
    assert len(out) == 1                          # 第二条放不下


def test_trim_truncates_last_when_room():
    b = BudgetManager()
    items = [{"content": "a" * 100}, {"content": "b" * 4000}]  # 25 + 1000 token
    out = b.trim_to_budget(items, max_tokens=1000)
    assert len(out) == 2
    remaining_chars = (1000 - 25) * _CHARS_PER_TOKEN
    assert out[1]["content"] == "b" * remaining_chars


def test_trim_all_fit():
    b = BudgetManager()
    items = [{"content": "s"}, {"content": "m"}]
    assert b.trim_to_budget(items, max_tokens=1000) == items
