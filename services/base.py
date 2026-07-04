"""Service 层抽象接口 / Protocol。

定义 memorize、recall、govern 三个核心 Service 的公共接口，
供 handler 通过依赖注入调用，便于后续替换实现或单元测试 mock。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from omnimem.handlers.deps import MemoryWriteResult, RecallResult


@runtime_checkable
class MemoryWriteServiceProtocol(Protocol):
    """记忆写入服务协议。

    负责安全校验、去重、冲突检测、多写、Saga 协调及后台任务提交。
    """

    def handle(self, args: dict[str, Any]) -> MemoryWriteResult:
        """处理一次记忆写入请求，返回结构化结果（非 JSON 字符串）。"""
        ...


@runtime_checkable
class RecallServiceProtocol(Protocol):
    """记忆召回服务协议。

    负责检索编排、过滤、ContextManager 精炼及异步召回。
    """

    def handle(self, args: dict[str, Any]) -> RecallResult:
        """同步召回，返回结构化结果。"""
        ...

    async def async_handle(self, args: dict[str, Any]) -> RecallResult:
        """异步召回，返回结构化结果。"""
        ...


@runtime_checkable
class GovernanceServiceProtocol(Protocol):
    """记忆治理服务协议。

    负责冲突扫描、遗忘调度、隐私审计、导入导出等治理动作。
    """

    def handle(self, args: dict[str, Any]) -> str:
        """处理一次治理请求，直接返回 JSON 字符串。"""
        ...
