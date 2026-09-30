"""★ P1-5 回归：``on_memory_write`` 必须带 ``metadata`` 形参。

Hermes 框架不看你读不读 metadata，它**按签名决定怎么传**：
``agent/memory_manager.py`` 的 ``_provider_memory_write_metadata_mode`` 用
``inspect.signature`` 探测 —— 有 ``metadata`` 形参（或有 ``**kwargs``）走 keyword，
没有形参但位置参数 ≥4 走 positional，否则 legacy 即一个都不传。

我们原先的签名是 ``(self, action, target, content)`` → 3 个位置参数 → legacy，
写入来源（write_origin / session_id / tool_name / old_text）静默丢失。
"""

from __future__ import annotations

import inspect
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from omnimem.core.provider_middleware import ProviderMiddlewareMixin


def _metadata_mode(fn: Any) -> str:
    """复刻框架的探测规则（agent/memory_manager.py:712-719）。"""
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()) or "metadata" in params:
        return "keyword"
    accepted = sum(p.kind is not inspect.Parameter.VAR_POSITIONAL for p in params.values())
    return "positional" if accepted >= 4 else "legacy"


class _Conflict:
    def __init__(self, has_conflict: bool, existing_memory: str = ""):
        self.has_conflict = has_conflict
        self.existing_memory = existing_memory


class _Provider(ProviderMiddlewareMixin):
    """只装 on_memory_write 需要的依赖，其余走 mixin 的默认实现。"""

    def __init__(self, conflict: _Conflict):
        self._conflict_resolver = SimpleNamespace(check=lambda content: conflict)


class TestSignatureContract:
    def test_provider_lands_on_keyword_mode(self):
        """关键断言：不能被判定成 legacy（那等于溯源直接丢）。"""
        provider = _Provider(_Conflict(False))

        assert _metadata_mode(provider.on_memory_write) == "keyword"

    def test_legacy_three_positional_call_still_works(self):
        """向后兼容：老框架按位置传 3 个参数不能炸。"""
        provider = _Provider(_Conflict(False))

        provider.on_memory_write("add", "memory", "用户喜欢 Python")

    def test_keyword_call_accepts_metadata(self):
        provider = _Provider(_Conflict(False))

        provider.on_memory_write(
            "add", "memory", "用户喜欢 Python",
            metadata={"write_origin": "assistant_tool", "session_id": "s1"},
        )

    def test_metadata_defaults_to_none(self):
        sig = inspect.signature(ProviderMiddlewareMixin.on_memory_write)
        assert sig.parameters["metadata"].default is None


class TestProvenanceIsUsed:
    def test_conflict_warning_carries_write_origin(self, caplog):
        provider = _Provider(_Conflict(True, "已有的记忆"))

        with caplog.at_level(logging.WARNING, logger="omnimem.core.provider_middleware"):
            provider.on_memory_write(
                "add", "memory", "新的记忆",
                metadata={"write_origin": "background_review", "session_id": "sess-42",
                          "old_text": "旧文本"},
            )

        message = caplog.text
        assert "background_review" in message
        assert "sess-42" in message

    def test_conflict_warning_without_metadata_is_unchanged(self, caplog):
        provider = _Provider(_Conflict(True, "已有的记忆"))

        with caplog.at_level(logging.WARNING, logger="omnimem.core.provider_middleware"):
            provider.on_memory_write("add", "memory", "新的记忆")

        assert "已有的记忆" in caplog.text
        assert "来源" not in caplog.text

    def test_non_add_action_does_not_trigger_conflict_check(self):
        calls: list[str] = []
        provider = _Provider(_Conflict(True, "已有的记忆"))
        provider._conflict_resolver = SimpleNamespace(check=lambda content: calls.append(content) or _Conflict(False))

        provider.on_memory_write("remove", "memory", "删掉的文本", metadata={"session_id": "s"})

        assert calls == []

    @pytest.mark.parametrize("action", ["add", "replace", "remove"])
    def test_framework_call_shape_never_raises(self, action):
        """框架真实调用形态：metadata 永远是 dict（可能为空）。"""
        provider = _Provider(_Conflict(False))

        provider.on_memory_write(action, "user", "内容", metadata={})
