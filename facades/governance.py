"""GovernanceFacade — 治理引擎横切面。

封装: ConflictResolver, TemporalDecay, ForgettingCurve, PrivacyManager,
      ProvenanceTracker, SyncEngine, VectorClock, GovernanceAuditor
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omnimem.governance.conflict import ConflictResolver
from omnimem.governance.decay import TemporalDecay
from omnimem.governance.forgetting import ForgettingCurve
from omnimem.governance.privacy import PrivacyManager
from omnimem.governance.provenance import ProvenanceTracker
from omnimem.governance.sync import SyncConfig, SyncEngine
from omnimem.governance.vector_clock import VectorClock
from omnimem.governance.audit_log import AuditLogger
from omnimem.governance.auditor import GovernanceAuditor


class GovernanceFacade:
    """治理门面：冲突仲裁 + 时间衰减 + 遗忘 + 隐私 + 溯源 + 同步。"""

    def __init__(
        self,
        data_dir: Path,
        config: Any,
        session_id: str,
        storage_facade: Any,
        retriever: Any,
    ):
        gov_dir = data_dir / "governance"

        self._conflict_resolver = ConflictResolver(
            strategy=config.get("conflict_strategy", "latest")
        )
        self._temporal_decay = TemporalDecay()
        self._forgetting = ForgettingCurve(gov_dir, config)
        self._privacy = PrivacyManager(
            default_level=config.get("default_privacy", "personal"),
            session_id=session_id,
        )
        self._privacy.bind_store(storage_facade.store)
        storage_facade.store.bind_privacy_manager(self._privacy)

        self._provenance = ProvenanceTracker(data_dir=gov_dir)

        # 同步引擎
        self._sync_engine = SyncEngine(
            data_dir,
            SyncConfig(
                mode=config.get("sync_mode", "none"),
                instance_name=f"omnimem-{session_id[:8]}",
                sync_interval=config.get("sync_interval", 30),
                conflict_resolution=config.get("sync_conflict_resolution", "latest_wins"),
            ),
        )

        # 向量时钟
        self._vector_clock = VectorClock()
        self._instance_id = self._sync_engine._config.instance_id

        # 审计器
        self._auditor = GovernanceAuditor(
            store=storage_facade.store,
            index=storage_facade.index,
            retriever=retriever,
            forgetting=self._forgetting,
        )

        # 操作审计日志
        self._audit_logger = AuditLogger(gov_dir)

    @property
    def conflict_resolver(self) -> ConflictResolver:
        return self._conflict_resolver

    @property
    def temporal_decay(self) -> TemporalDecay:
        return self._temporal_decay

    @property
    def forgetting(self) -> ForgettingCurve:
        return self._forgetting

    @property
    def privacy(self) -> PrivacyManager:
        return self._privacy

    @property
    def provenance(self) -> ProvenanceTracker:
        return self._provenance

    @property
    def sync_engine(self) -> SyncEngine:
        return self._sync_engine

    @property
    def auditor(self) -> GovernanceAuditor:
        return self._auditor

    @property
    def audit_logger(self) -> AuditLogger:
        return self._audit_logger

    @property
    def vector_clock(self) -> VectorClock:
        return self._vector_clock

    @property
    def instance_id(self) -> str:
        return self._instance_id

    def close(self) -> None:
        """关闭治理资源。"""
        self.forgetting.close()
        self.provenance.close()
        self.sync_engine.close()
        self._audit_logger.close()
