"""L1 工作记忆核心模块。"""

from omnimem.core.attachment import CompactAttachment as CompactAttachment
from omnimem.core.attachment import build_attachments as build_attachments
from omnimem.core.block import CoreBlock as CoreBlock
from omnimem.core.budget import BudgetManager as BudgetManager
from omnimem.core.dedup import SemanticDedupService as SemanticDedupService
from omnimem.core.engram_bridge import Engram as Engram

# Plur 共享记忆集成
from omnimem.core.engram_bridge import EngramBridge as EngramBridge
from omnimem.core.engram_bridge import MemoryFederation as MemoryFederation
from omnimem.core.engram_bridge import SharedMemorySync as SharedMemorySync
from omnimem.core.engram_bridge import create_engram_bridge as create_engram_bridge
from omnimem.core.engram_bridge import create_memory_federation as create_memory_federation
from omnimem.core.engram_bridge import create_shared_memory_sync as create_shared_memory_sync
from omnimem.core.plur_client import PlurClient as PlurClient
from omnimem.core.plur_client import create_plur_client as create_plur_client
from omnimem.core.plur_config import PlurConfig as PlurConfig
from omnimem.core.plur_config import get_config as get_config
from omnimem.core.soul import SoulSystem as SoulSystem
