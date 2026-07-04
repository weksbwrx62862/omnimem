"""安全加固相关测试。

覆盖：加密失败策略、导出加密与校验、REST API 默认认证与权限、
      MCP Server API Key、日志脱敏。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from omnimem.core.import_export import MemoryExporter, MemoryImporter
from omnimem.governance.encryption import EncryptionUnavailableError, MemoryEncryption
from omnimem.mcp_server import OmniMemMCPServer
from omnimem.rest_api import (
    AdminAuthMiddleware,
    AuthMiddleware,
    OmniMemAPIHandler,
    RateLimiter,
    _generate_default_key,
)
from omnimem.utils.logging import sanitize_for_log


class TestMemoryEncryptionSecurity(unittest.TestCase):
    """加密不可用时应明确拒绝，而非降级为明文。"""

    def test_disabled_encrypt_raises(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OMNIMEM_ENCRYPTION_KEY", None)
            enc = MemoryEncryption(session_seed="")
            self.assertFalse(enc.is_available())
            with self.assertRaises(EncryptionUnavailableError):
                enc.encrypt("secret")

    def test_disabled_encrypt_with_status_raises(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OMNIMEM_ENCRYPTION_KEY", None)
            enc = MemoryEncryption(session_seed="")
            with self.assertRaises(EncryptionUnavailableError):
                enc.encrypt_with_status("secret")


class TestLogSanitization(unittest.TestCase):
    """日志脱敏规则测试。"""

    def test_masks_api_key(self):
        text = "用户密钥是 sk-abc123def456ghi789"
        self.assertNotIn("sk-abc", sanitize_for_log(text))
        self.assertIn("sk-***", sanitize_for_log(text))

    def test_masks_token_assignment(self):
        text = 'api_key="supersecret12345678"'
        self.assertNotIn("supersecret", sanitize_for_log(text))

    def test_masks_fernet_token(self):
        token = "gAAAAAB1234567890abcdef"
        self.assertEqual(sanitize_for_log(token), "[加密令牌]")

    def test_truncates_long_text(self):
        text = "x" * 1000
        result = sanitize_for_log(text, max_length=100)
        self.assertTrue(result.endswith("...[已截断]"))
        self.assertLessEqual(len(result), 120)


class TestExportEncryption(unittest.TestCase):
    """导出文件默认加密并附带 SHA-256 校验和。"""

    def setUp(self) -> None:
        from cryptography.fernet import Fernet

        self.key = Fernet.generate_key().decode("utf-8")
        self.store = MagicMock()
        self.index = MagicMock()
        self.meta = MagicMock()
        self.exporter = MemoryExporter(self.store, self.index, self.meta)

    def _make_record(self) -> dict:
        return {
            "memory_id": "m1",
            "content": "my secret content",
            "type": "fact",
            "wing": "personal",
            "room": "test",
            "privacy": "secret",
            "confidence": 3,
            "stored_at": "2026-01-01T00:00:00Z",
        }

    def test_export_is_encrypted_envelope(self):
        self.store.search.return_value = [self._make_record()]
        self.store.get.return_value = self._make_record()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "export.json"
            count = self.exporter.export_json(out, encryption_key=self.key)
            self.assertEqual(count, 1)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["version"], "2.0")
            self.assertTrue(data["encrypted"])
            self.assertIn("checksum", data)
            self.assertIn("payload", data)
            # 确保 payload 不是明文 JSON
            self.assertNotIn("my secret content", data["payload"])

    def test_import_roundtrip(self):
        self.store.search.return_value = [self._make_record()]
        self.store.get.return_value = self._make_record()

        dedup = MagicMock()
        dedup.semantic_dedup.return_value = {"action": "create"}
        conflict = MagicMock()
        conflict_result = MagicMock()
        conflict_result.has_conflict = False
        conflict.check.return_value = conflict_result
        forgetting = MagicMock()
        importer = MemoryImporter(self.store, self.index, MagicMock(), dedup, conflict, forgetting)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "export.json"
            self.exporter.export_json(out, encryption_key=self.key)
            result = importer.import_json(out, encryption_key=self.key)
            self.assertEqual(result["imported"], 1)

    def test_import_checksum_mismatch(self):
        self.store.search.return_value = [self._make_record()]
        self.store.get.return_value = self._make_record()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "export.json"
            self.exporter.export_json(out, encryption_key=self.key)
            data = json.loads(out.read_text(encoding="utf-8"))
            data["checksum"] = "0" * 64
            out.write_text(json.dumps(data), encoding="utf-8")

            importer = MemoryImporter(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
            with self.assertRaises(ValueError) as ctx:
                importer.import_json(out, encryption_key=self.key)
            self.assertIn("校验和不匹配", str(ctx.exception))

    def test_import_wrong_key(self):
        from cryptography.fernet import Fernet

        wrong_key = Fernet.generate_key().decode("utf-8")
        self.store.search.return_value = [self._make_record()]
        self.store.get.return_value = self._make_record()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "export.json"
            self.exporter.export_json(out, encryption_key=self.key)
            importer = MemoryImporter(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
            with self.assertRaises(ValueError):
                importer.import_json(out, encryption_key=wrong_key)


class TestAdminAuthMiddleware(unittest.TestCase):
    def test_disabled_when_admin_token_empty(self):
        m = AdminAuthMiddleware(admin_token="")
        self.assertTrue(m.validate({}))

    def test_valid_admin_token(self):
        m = AdminAuthMiddleware(admin_token="admin123")
        self.assertTrue(m.validate({"X-Admin-Token": "admin123"}))

    def test_invalid_admin_token_returns_403(self):
        m = AdminAuthMiddleware(admin_token="admin123")
        result = m.validate({"X-Admin-Token": "wrong"})
        self.assertEqual(result[0], 403)


class TestDefaultKeyGeneration(unittest.TestCase):
    def test_generate_default_key_is_hex(self):
        key = _generate_default_key()
        self.assertEqual(len(bytes.fromhex(key)), 32)


class _RunningServerMixin:
    """测试 HTTP 服务器辅助类。"""

    def _start_server(
        self,
        api_key: str = "",
        admin_token: str = "",
        rate_limit: int = 60,
    ):
        OmniMemAPIHandler._sdk = MagicMock()
        OmniMemAPIHandler._sdk.health_check.return_value = {"status": "ok"}
        OmniMemAPIHandler._sdk.export_memories.return_value = {"status": "exported", "count": 0}
        OmniMemAPIHandler._sdk.import_memories.return_value = {"status": "imported", "total": 0, "imported": 0}
        OmniMemAPIHandler._auth_middleware = AuthMiddleware(api_key=api_key)
        OmniMemAPIHandler._admin_auth_middleware = AdminAuthMiddleware(admin_token=admin_token)
        OmniMemAPIHandler._rate_limiter = RateLimiter(limit_per_minute=rate_limit)
        OmniMemAPIHandler._cors_allowed_origins = []

        from http.server import HTTPServer

        self.server = HTTPServer(("127.0.0.1", 0), OmniMemAPIHandler)
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def _stop_server(self):
        if hasattr(self, "server"):
            self.server.shutdown()
            self.server.server_close()
            self.server_thread.join(timeout=2)

    def _request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
        url = self.base_url + path
        req = Request(url, data=body, method=method)
        headers = headers or {}
        for key, value in headers.items():
            req.add_header(key, value)
        with urlopen(req, timeout=5) as resp:
            return resp.status, resp.read(), resp.headers


class TestRestApiSecurityHardening(_RunningServerMixin, unittest.TestCase):
    def tearDown(self):
        self._stop_server()

    def test_export_requires_admin_token(self):
        self._start_server(api_key="api123", admin_token="admin123")
        # 仅 API Key 应返回 403
        with self.assertRaises(HTTPError) as ctx:
            self._request(
                "POST",
                "/api/export",
                body=json.dumps({"output_path": "/tmp/x.json"}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer api123",
                },
            )
        self.assertEqual(ctx.exception.code, 403)

        # 同时提供 Admin Token 才能成功
        status, body, _ = self._request(
            "POST",
            "/api/export",
            body=json.dumps({"output_path": "/tmp/x.json"}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer api123",
                "X-Admin-Token": "admin123",
            },
        )
        self.assertEqual(status, 200)

    def test_import_requires_admin_token(self):
        self._start_server(api_key="api123", admin_token="admin123")
        with self.assertRaises(HTTPError) as ctx:
            self._request(
                "POST",
                "/api/import",
                body=json.dumps({"input_path": "/tmp/x.json"}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer api123",
                },
            )
        self.assertEqual(ctx.exception.code, 403)

    def test_body_size_limit(self):
        self._start_server(api_key="")
        huge_body = b'{"content": "' + b"x" * (11 * 1024 * 1024) + b'"}'
        with self.assertRaises(HTTPError) as ctx:
            self._request(
                "POST",
                "/api/memorize",
                body=huge_body,
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(ctx.exception.code, 413)


class TestMCPApiKeyMiddleware(unittest.TestCase):
    """MCP Server 在配置 OMNIMEM_API_KEY 后要求工具调用携带 _api_key。"""

    @patch("omnimem.mcp_server.OmniMemMCPServer.__init__", lambda _self, **_kwargs: None)
    def _make_server(self, api_key: str = "", require_api_key: bool = False):
        server = OmniMemMCPServer.__new__(OmniMemMCPServer)
        server._api_key = api_key
        server._require_api_key = require_api_key
        server._auth_middleware = MagicMock()
        server._auth_middleware.validate.return_value = True
        server._sdk = MagicMock()
        server._sdk.memorize.return_value = {"status": "stored"}
        return server

    def test_no_auth_when_env_not_set(self):
        server = self._make_server(api_key="")
        result = server.call_tool("omni_memorize", {"content": "hello"})
        self.assertIn("stored", result)

    def test_missing_api_key_returns_error(self):
        server = self._make_server(api_key="secret123")
        result = server.call_tool("omni_memorize", {"content": "hello"})
        data = json.loads(result)
        self.assertIn("error", data)
        self.assertIn("API Key", data["error"])

    def test_valid_api_key_allowed(self):
        server = self._make_server(api_key="secret123")
        result = server.call_tool("omni_memorize", {"content": "hello", "_api_key": "secret123"})
        data = json.loads(result)
        self.assertEqual(data["status"], "stored")

    def test_valid_authorization_bearer_allowed(self):
        server = self._make_server(api_key="secret123")
        result = server.call_tool(
            "omni_memorize",
            {"content": "hello", "Authorization": "Bearer secret123"},
        )
        data = json.loads(result)
        self.assertEqual(data["status"], "stored")

    def test_invalid_authorization_bearer_returns_error(self):
        server = self._make_server(api_key="secret123")
        result = server.call_tool(
            "omni_memorize",
            {"content": "hello", "Authorization": "Bearer wrong"},
        )
        data = json.loads(result)
        self.assertIn("error", data)
        self.assertIn("API Key", data["error"])

    def test_mcp_auth_middleware_validates_header(self):
        from omnimem.mcp_server import MCPAuthMiddleware

        middleware = MCPAuthMiddleware(api_key="secret123")
        self.assertTrue(middleware.validate({"Authorization": "Bearer secret123"}))
        self.assertEqual(
            middleware.validate({"Authorization": "Bearer wrong"}),
            "API Key 校验失败",
        )
        self.assertTrue(middleware.validate({}))  # 未配置 api_key 时跳过

    def test_require_api_key_config_blocks_without_key(self):
        server = self._make_server(api_key="secret123", require_api_key=True)
        result = server.call_tool("omni_memorize", {"content": "hello"})
        data = json.loads(result)
        self.assertIn("error", data)
        self.assertIn("API Key", data["error"])


if __name__ == "__main__":
    unittest.main()
