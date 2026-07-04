"""PluginOrchestrator 事件发布解耦模块。

将 `sys.modules` 中查找 `plugin_orchestrator.context` 的逻辑集中到这里，
使业务代码不再直接依赖 `sys.modules`。
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class EventPublisher(Protocol):
    """事件发布器协议。"""

    def publish(self, event_name: str, **kwargs: Any) -> None:
        """发布事件；实现者应保证失败时不抛异常。"""
        ...


class PluginOrchestratorPublisher:
    """通过 PluginOrchestrator 的 context 发布事件。"""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    def publish(self, event_name: str, **kwargs: Any) -> None:
        """尝试获取会话上下文并发布事件；任何异常都静默降级。"""
        if not self.session_id:
            return
        try:
            ctx_mod = sys.modules.get("plugin_orchestrator.context")
            if ctx_mod is None:
                return
            get_ctx = getattr(ctx_mod, "get_context", None)
            if get_ctx is None:
                return
            ctx = get_ctx(self.session_id)
            if ctx is None:
                return
            # 保持原有行为：memory_stored 事件同时写入共享状态
            if event_name == "memory_stored":
                ctx.shared_set("last_memory_write", {
                    "memory_id": kwargs.get("memory_id", ""),
                    "type": kwargs.get("memory_type", "fact"),
                    "wing": kwargs.get("wing", ""),
                    "room": kwargs.get("room", ""),
                    "privacy": kwargs.get("privacy", ""),
                    "confidence": kwargs.get("confidence", 3),
                })
            ctx.event_bus.publish(event_name, **kwargs)
        except Exception as e:
            logger.debug("PluginOrchestrator 事件发布失败: %s", e)


class NoOpPublisher:
    """空实现发布器，用于 PluginOrchestrator 未安装时的回退。"""

    def publish(self, event_name: str, **kwargs: Any) -> None:
        """什么都不做。"""
        return


def get_event_publisher(session_id: str) -> EventPublisher:
    """根据 `plugin_orchestrator.context` 是否已加载返回对应发布器。

    - 已加载：返回 `PluginOrchestratorPublisher`
    - 未加载：返回 `NoOpPublisher`，避免运行时依赖
    """
    if "plugin_orchestrator.context" in sys.modules:
        return PluginOrchestratorPublisher(session_id)
    return NoOpPublisher()
