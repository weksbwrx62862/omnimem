"""GovernanceService 导出/导入安全加固测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from cryptography.fernet import Fernet
from omnimem.handlers.deps import HandlerDependencies
from omnimem.services.governance_service import GovernanceService


class TestGovernanceExportImportSecurity(unittest.TestCase):
    """治理动作 export_memories / import_memories 必须使用加密路径。"""

    def setUp(self) -> None:
        self.key = Fernet.generate_key().decode("utf-8")
        self.tmpdir = tempfile.mkdtemp()

        self.store = MagicMock()
        self.index = MagicMock()
        self.retriever = MagicMock()
        self.dedup = MagicMock()
        self.conflict = MagicMock()
        self.forgetting = MagicMock()

        self.deps = HandlerDependencies(
            store=self.store,
            index=self.index,
            retriever=self.retriever,
            dedup=self.dedup,
            conflict_resolver=self.conflict,
            forgetting=self.forgetting,
            config={"export_key": self.key},
        )
        self.service = GovernanceService(self.deps)

    def _make_record(self) -> dict:
        return {
            "memory_id": "m1",
            "content": "secret content",
            "summary": "summary",
            "type": "fact",
            "wing": "personal",
            "room": "test",
            "privacy": "personal",
            "confidence": 3,
            "stored_at": "2026-01-01T00:00:00Z",
        }

    def test_export_rejected_without_key(self) -> None:
        """未配置 export_key 时，export_memories 应拒绝导出。"""
        deps = HandlerDependencies(
            store=self.store,
            index=self.index,
            retriever=self.retriever,
            dedup=self.dedup,
            conflict_resolver=self.conflict,
            forgetting=self.forgetting,
            config={},
        )
        service = GovernanceService(deps)
        self.store.search.return_value = [self._make_record()]
        self.store.get.return_value = self._make_record()

        result = service.handle(
            {"action": "export_memories", "params": {"output_path": "/tmp/x.json"}}
        )
        data = json.loads(result)
        self.assertIn("error", data)
        self.assertIn("未配置导出密钥", data["error"])

    def test_export_encrypted_with_config_key(self) -> None:
        """配置 export_key 后，导出文件应为加密信封格式。"""
        self.store.search.return_value = [self._make_record()]
        self.store.get.return_value = self._make_record()
        self.store.meta_store = MagicMock()

        output_path = Path(self.tmpdir) / "export.json"
        result = self.service.handle(
            {
                "action": "export_memories",
                "params": {"output_path": str(output_path)},
            }
        )
        data = json.loads(result)
        self.assertEqual(data.get("status"), "exported")
        self.assertTrue(output_path.exists())
        envelope = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(envelope.get("version"), "2.0")
        self.assertTrue(envelope.get("encrypted"))
        self.assertIn("checksum", envelope)
        self.assertIn("payload", envelope)
        self.assertNotIn("secret content", envelope.get("payload", ""))

    def test_import_roundtrip_with_key(self) -> None:
        """使用正确密钥可解密并导入加密导出文件。"""
        self.store.search.return_value = [self._make_record()]
        self.store.get.return_value = self._make_record()
        self.store.meta_store = MagicMock()
        self.dedup.semantic_dedup.return_value = {"action": "create"}
        conflict_result = MagicMock()
        conflict_result.has_conflict = False
        self.conflict.check.return_value = conflict_result

        output_path = Path(self.tmpdir) / "roundtrip.json"
        self.service.handle(
            {
                "action": "export_memories",
                "params": {"output_path": str(output_path)},
            }
        )

        result = self.service.handle(
            {
                "action": "import_memories",
                "params": {"input_path": str(output_path)},
            }
        )
        data = json.loads(result)
        self.assertEqual(data.get("status"), "imported")
        self.assertEqual(data.get("imported"), 1)

    def test_import_rejects_checksum_tampering(self) -> None:
        """HMAC 校验失败时，import_memories 应拒绝导入。"""
        self.store.search.return_value = [self._make_record()]
        self.store.get.return_value = self._make_record()
        self.store.meta_store = MagicMock()

        output_path = Path(self.tmpdir) / "tampered.json"
        self.service.handle(
            {
                "action": "export_memories",
                "params": {"output_path": str(output_path)},
            }
        )

        envelope = json.loads(output_path.read_text(encoding="utf-8"))
        envelope["checksum"] = "0" * 64
        output_path.write_text(json.dumps(envelope), encoding="utf-8")

        result = self.service.handle(
            {
                "action": "import_memories",
                "params": {"input_path": str(output_path)},
            }
        )
        data = json.loads(result)
        self.assertIn("error", data)
        self.assertIn("校验和不匹配", data["error"])


if __name__ == "__main__":
    unittest.main()
