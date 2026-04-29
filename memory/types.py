"""记忆数据模型。"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


class MemoryType(str, enum.Enum):
    """记忆类型枚举。"""
    FACT = "fact"
    PREFERENCE = "preference"
    CORRECTION = "correction"
    SKILL = "skill"
    PROCEDURAL = "procedural"
    EVENT = "event"


class PrivacyLevel(str, enum.Enum):
    """隐私级别枚举。"""
    PUBLIC = "public"
    TEAM = "team"
    PERSONAL = "personal"
    SECRET = "secret"


@dataclass
class MemoryEntry:
    """一条记忆的完整数据模型。"""
    memory_id: str
    content: str
    memory_type: MemoryType = MemoryType.FACT
    confidence: int = 3
    privacy: PrivacyLevel = PrivacyLevel.PERSONAL
    scope: str = "personal"
    wing: str = ""
    room: str = ""
    provenance: Optional[Dict[str, Any]] = None
    stored_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）。"""
        result = {
            "memory_id": self.memory_id,
            "content": self.content,
            "type": self.memory_type.value,
            "confidence": self.confidence,
            "privacy": self.privacy.value,
            "scope": self.scope,
            "wing": self.wing,
            "room": self.room,
            "stored_at": self.stored_at.isoformat() if self.stored_at else None,
        }
        if self.provenance:
            result["provenance"] = self.provenance
        result.update(self.metadata)
        return result
