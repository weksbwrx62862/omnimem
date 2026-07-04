"""event_publisher 模块测试。

覆盖 PluginOrchestrator 未安装、已安装（mock）以及发布失败三种场景。
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from omnimem.utils.event_publisher import (
    NoOpPublisher,
    PluginOrchestratorPublisher,
    get_event_publisher,
)


@pytest.fixture
def _clear_orchestrator_module():
    """测试前后清理 `sys.modules` 中模拟的 orchestrator 模块。"""
    original = sys.modules.pop("plugin_orchestrator.context", None)
    yield
    sys.modules.pop("plugin_orchestrator.context", None)
    if original is not None:
        sys.modules["plugin_orchestrator.context"] = original


def test_no_orchestrator_returns_noop_publisher(_clear_orchestrator_module) -> None:
    """PluginOrchestrator 未安装时，工厂返回 NoOpPublisher 且调用不报错。"""
    publisher = get_event_publisher("session-001")
    assert isinstance(publisher, NoOpPublisher)
    publisher.publish("memory_stored", source_plugin="omnimem")


def test_orchestrator_publisher_publishes_event(_clear_orchestrator_module) -> None:
    """PluginOrchestrator 已加载时，工厂返回 PluginOrchestratorPublisher 并发布事件。"""
    ctx = MagicMock()
    ctx.event_bus.publish = MagicMock()
    ctx.shared_set = MagicMock()

    ctx_mod = ModuleType("plugin_orchestrator.context")
    ctx_mod.get_context = MagicMock(return_value=ctx)
    sys.modules["plugin_orchestrator.context"] = ctx_mod

    publisher = get_event_publisher("session-002")
    assert isinstance(publisher, PluginOrchestratorPublisher)

    publisher.publish(
        "memory_stored",
        source_plugin="omnimem",
        session_id="session-002",
        memory_id="mem-002",
        memory_type="fact",
        wing="personal",
        room="python",
        privacy="personal",
        confidence=4,
        content_preview="preview",
    )

    ctx.shared_set.assert_called_once_with("last_memory_write", {
        "memory_id": "mem-002",
        "type": "fact",
        "wing": "personal",
        "room": "python",
        "privacy": "personal",
        "confidence": 4,
    })
    ctx.event_bus.publish.assert_called_once_with(
        "memory_stored",
        source_plugin="omnimem",
        session_id="session-002",
        memory_id="mem-002",
        memory_type="fact",
        wing="personal",
        room="python",
        privacy="personal",
        confidence=4,
        content_preview="preview",
    )


def test_publish_failure_is_silent(_clear_orchestrator_module) -> None:
    """事件发布失败时不应抛异常。"""
    ctx = MagicMock()
    ctx.event_bus.publish.side_effect = RuntimeError("event bus down")

    ctx_mod = ModuleType("plugin_orchestrator.context")
    ctx_mod.get_context = MagicMock(return_value=ctx)
    sys.modules["plugin_orchestrator.context"] = ctx_mod

    publisher = get_event_publisher("session-003")
    publisher.publish(
        "session_memory_processed",
        source_plugin="omnimem",
        session_id="session-003",
    )

    ctx.event_bus.publish.assert_called_once()


