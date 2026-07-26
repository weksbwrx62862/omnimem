"""OmniMem Service 层。

将 handler 中的核心业务逻辑下沉到 Service，handler 只负责：
- schema 校验
- 依赖注入（deps 解包）
- 调用 Service
- 结果序列化
"""

from __future__ import annotations

from omnimem.services.base import (
    GovernanceServiceProtocol,
    MemoryWriteServiceProtocol,
    RecallServiceProtocol,
)
from omnimem.services.governance_service import GovernanceService
from omnimem.services.memory_write_service import (
    MemoryWriteService,
    get_background_executor,
    shutdown_background_executor,
)
from omnimem.services.recall_service import RecallService

__all__ = [
    "MemoryWriteServiceProtocol",
    "RecallServiceProtocol",
    "GovernanceServiceProtocol",
    "MemoryWriteService",
    "RecallService",
    "GovernanceService",
    "get_background_executor",
    "shutdown_background_executor",
]
