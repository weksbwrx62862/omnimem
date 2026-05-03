"""OmniMemSDK — 独立 SDK 模式，不依赖 Hermes MemoryProvider 接口。

提供轻量级 API，可直接创建 OmniMem 实例并调用记忆操作，
无需 Hermes 框架注册机制。

用法:
    from omnimem.sdk import OmniMemSDK

    sdk = OmniMemSDK(storage_dir="~/.omnimem")
    sdk.memorize("用户喜欢Python", memory_type="preference")
    result = sdk.recall("用户喜欢什么")
    sdk.close()
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

from omnimem.config import OmniMemConfig
from omnimem.provider import OmniMemProvider


def _ensure_memory_provider_mock() -> None:
    if "agent" not in sys.modules:
        from unittest.mock import MagicMock

        _mock_agent = MagicMock()
        _mock_agent.memory_provider = MagicMock()
        _mock_agent.memory_provider.MemoryProvider = object
        sys.modules["agent"] = _mock_agent
        sys.modules["agent.memory_provider"] = _mock_agent.memory_provider


class OmniMemSDK:
    def __init__(
        self,
        storage_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        _ensure_memory_provider_mock()

        if storage_dir is None:
            storage_dir = Path.home() / ".omnimem"
        self._data_dir = Path(storage_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._session_id = f"sdk-{uuid.uuid4().hex[:12]}"

        self._provider = OmniMemProvider()
        self._provider.initialize(
            session_id=self._session_id,
            hermes_home=str(self._data_dir.parent),
            platform="sdk",
            agent_context="primary",
        )

        if config:
            for key, value in config.items():
                self._provider._config.set(key, value)

    def memorize(self, content: str, memory_type: str = "fact", **kwargs: Any) -> dict:
        args = {"content": content, "memory_type": memory_type, **kwargs}
        raw = self._provider._handle_memorize(args)
        return json.loads(raw)

    def recall(self, query: str, mode: str = "rag", **kwargs: Any) -> dict:
        args = {"query": query, "mode": mode, **kwargs}
        raw = self._provider._handle_recall(args)
        return json.loads(raw)

    def reflect(self, query: str, **kwargs: Any) -> dict:
        args = {"query": query, **kwargs}
        raw = self._provider._handle_reflect(args)
        return json.loads(raw)

    def govern(self, action: str, **kwargs: Any) -> dict:
        args = {"action": action, **kwargs}
        raw = self._provider._handle_govern(args)
        return json.loads(raw)

    def compact(self, **kwargs: Any) -> dict:
        args = {**kwargs}
        raw = self._provider._handle_compact(args)
        return json.loads(raw)

    def detail(self, memory_id: str, **kwargs: Any) -> dict:
        args = {"action": "get", "memory_id": memory_id, **kwargs}
        raw = self._provider._handle_detail(args)
        return json.loads(raw)

    def detail_list(self, **kwargs: Any) -> dict:
        args = {"action": "list", **kwargs}
        raw = self._provider._handle_detail(args)
        return json.loads(raw)

    def detail_events(self, from_turn: int = 0, to_turn: int | None = None, **kwargs: Any) -> dict:
        args: dict[str, Any] = {"action": "events", "from_turn": from_turn, **kwargs}
        if to_turn is not None:
            args["to_turn"] = to_turn
        raw = self._provider._handle_detail(args)
        return json.loads(raw)

    def health_check(self) -> dict:
        result: dict[str, Any] = {
            "status": "healthy",
            "session_id": self._session_id,
            "data_dir": str(self._data_dir),
        }

        try:
            available = self._provider.is_available()
            result["dependencies"] = available
        except Exception as e:
            result["dependencies"] = False
            result["dependency_error"] = str(e)

        if hasattr(self._provider, "_auditor") and self._provider._auditor:
            try:
                health = self._provider._auditor.quick_health_check()
                result["audit"] = health
                if not health.get("healthy", True):
                    result["status"] = "degraded"
            except Exception as e:
                result["audit_error"] = str(e)

        try:
            store_count = len(self._provider._store.search(limit=1))
            result["store_accessible"] = True
        except Exception as e:
            result["store_accessible"] = False
            result["store_error"] = str(e)
            result["status"] = "unhealthy"

        return result

    def export_memories(
        self, output_path: str, format: str = "json", **kwargs: Any
    ) -> dict:
        from omnimem.core.import_export import MemoryExporter

        exporter = MemoryExporter(
            self._provider._store,
            self._provider._index,
            self._provider._store._meta_store,
        )
        if format == "markdown":
            count = exporter.export_markdown(output_path, wing=kwargs.get("wing"))
        else:
            count = exporter.export_json(
                output_path,
                wing=kwargs.get("wing"),
                memory_type=kwargs.get("memory_type"),
            )
        return {"status": "exported", "count": count, "path": str(output_path)}

    def import_memories(self, input_path: str, **kwargs: Any) -> dict:
        from omnimem.core.import_export import MemoryImporter

        importer = MemoryImporter(
            self._provider._store,
            self._provider._index,
            self._provider._retriever,
            self._provider._dedup,
            self._provider._conflict_resolver,
            self._provider._forgetting,
        )
        result = importer.import_json(
            input_path,
            skip_duplicates=kwargs.get("skip_duplicates", True),
            resolve_conflicts=kwargs.get("resolve_conflicts", True),
        )
        return {"status": "imported", **result}

    def close(self) -> None:
        self._provider.shutdown()

    def __enter__(self) -> OmniMemSDK:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
