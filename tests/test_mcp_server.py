"""OmniMemMCPServer 安全中间件测试。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from omnimem.mcp_server import MCPAuthMiddleware, OmniMemMCPServer


class TestMCPAuthMiddleware(unittest.TestCase):
    """MCPAuthMiddleware 校验 Authorization Bearer Token。"""

    def test_disabled_when_api_key_empty(self):
        middleware = MCPAuthMiddleware(api_key="")
        self.assertTrue(middleware.validate({}))
        self.assertTrue(middleware.validate({"Authorization": "Bearer xxx"}))

    def test_valid_bearer_token(self):
        middleware = MCPAuthMiddleware(api_key="secret123")
        self.assertTrue(middleware.validate({"Authorization": "Bearer secret123"}))

    def test_invalid_bearer_token_returns_error(self):
        middleware = MCPAuthMiddleware(api_key="secret123")
        result = middleware.validate({"Authorization": "Bearer wrong"})
        self.assertEqual(result, "API Key 校验失败")

    def test_missing_auth_header_returns_error(self):
        middleware = MCPAuthMiddleware(api_key="secret123")
        result = middleware.validate({})
        self.assertEqual(result, "API Key 校验失败")


class TestOmniMemMCPServerAuth(unittest.TestCase):
    """OmniMemMCPServer 在配置 api_key 后要求工具调用携带凭证。"""

    @patch("omnimem.mcp_server.OmniMemMCPServer.__init__", lambda _self, **_kwargs: None)
    def _make_server(self, api_key: str = "", require_api_key: bool = False):
        server = OmniMemMCPServer.__new__(OmniMemMCPServer)
        server._api_key = api_key
        server._require_api_key = require_api_key
        server._auth_middleware = MCPAuthMiddleware(api_key=api_key)
        server._sdk = MagicMock()
        server._sdk.memorize.return_value = {"status": "stored"}
        return server

    def test_call_tool_without_key_when_not_required(self):
        server = self._make_server(api_key="")
        result = server.call_tool("omni_memorize", {"content": "hello"})
        self.assertIn("stored", result)

    def test_call_tool_with_valid_api_key_argument(self):
        server = self._make_server(api_key="secret123")
        result = server.call_tool(
            "omni_memorize", {"content": "hello", "_api_key": "secret123"}
        )
        self.assertIn("stored", result)

    def test_call_tool_with_valid_authorization_header(self):
        server = self._make_server(api_key="secret123")
        result = server.call_tool(
            "omni_memorize",
            {"content": "hello", "Authorization": "Bearer secret123"},
        )
        self.assertIn("stored", result)

    def test_call_tool_with_invalid_authorization_header(self):
        server = self._make_server(api_key="secret123")
        result = server.call_tool(
            "omni_memorize",
            {"content": "hello", "Authorization": "Bearer wrong"},
        )
        self.assertIn("error", result)
        self.assertIn("API Key", result)

    def test_require_api_key_blocks_missing_key(self):
        server = self._make_server(api_key="secret123", require_api_key=True)
        result = server.call_tool("omni_memorize", {"content": "hello"})
        self.assertIn("error", result)
        self.assertIn("API Key", result)


if __name__ == "__main__":
    unittest.main()
