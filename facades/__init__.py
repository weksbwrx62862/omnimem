"""OmniMem Facades — 将 provider 的 30 子系统按关注点分组。

每个 Facade 封装一组内聚的子系统，对外暴露简化接口。
Provider 通过 Facade 访问子系统，Handler 通过 Facade 解耦。
"""

from omnimem.facades.deep_memory import DeepMemoryFacade
from omnimem.facades.governance import GovernanceFacade
from omnimem.facades.retrieval import RetrievalFacade
from omnimem.facades.storage import StorageFacade
from omnimem.facades.sync_facade import SyncFacade

__all__ = [
    "StorageFacade",
    "RetrievalFacade",
    "GovernanceFacade",
    "DeepMemoryFacade",
    "SyncFacade",
]
