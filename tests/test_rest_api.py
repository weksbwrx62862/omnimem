"""REST API 安全加固测试。"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from omnimem.rest_api import AuthMiddleware, OmniMemAPIHandler, RateLimiter


class RunningServerMixin:
    """在随机端口启动真实 HTTP 服务器的测试辅助类。"""

    def _start_server(self, api_key: str = "", rate_limit: int = 60, cors_origins: list[str] | None = None):
        """启动测试服务器并返回 base URL。"""
        if cors_origins is None:
            cors_origins = []

        # 模拟 SDK，避免真实初始化存储与检索组件
        OmniMemAPIHandler._sdk = MagicMock()
        OmniMemAPIHandler._sdk.health_check.return_value = {"status": "ok"}
        OmniMemAPIHandler._auth_middleware = AuthMiddleware(api_key=api_key)
        OmniMemAPIHandler._rate_limiter = RateLimiter(limit_per_minute=rate_limit)
        OmniMemAPIHandler._cors_allowed_origins = cors_origins

        from http.server import HTTPServer

        self.server = HTTPServer(("127.0.0.1", 0), OmniMemAPIHandler)
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def _stop_server(self):
        """关闭测试服务器。"""
        if hasattr(self, "server"):
            self.server.shutdown()
            self.server.server_close()
            self.server_thread.join(timeout=2)

    def _request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
        """发送 HTTP 请求并返回 (status, body, headers) 三元组。"""
        url = self.base_url + path
        req = Request(url, data=body, method=method)
        headers = headers or {}
        for key, value in headers.items():
            req.add_header(key, value)
        with urlopen(req, timeout=5) as resp:
            return resp.status, resp.read(), resp.headers


class TestAuthMiddleware(unittest.TestCase):
    """AuthMiddleware 单元测试。"""

    def test_disabled_when_api_key_empty(self):
        middleware = AuthMiddleware(api_key="")
        self.assertTrue(middleware.validate({}))
        self.assertTrue(middleware.validate({"Authorization": "Bearer xxx"}))

    def test_valid_bearer_token(self):
        middleware = AuthMiddleware(api_key="secret123")
        self.assertTrue(middleware.validate({"Authorization": "Bearer secret123"}))

    def test_invalid_bearer_token_returns_401(self):
        middleware = AuthMiddleware(api_key="secret123")
        result = middleware.validate({"Authorization": "Bearer wrong"})
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], 401)
        self.assertEqual(result[1], {"error": "Unauthorized"})

    def test_missing_auth_header_returns_401(self):
        middleware = AuthMiddleware(api_key="secret123")
        result = middleware.validate({})
        self.assertEqual(result[0], 401)


class TestRateLimiter(unittest.TestCase):
    """RateLimiter 单元测试。"""

    def test_allow_requests_within_limit(self):
        limiter = RateLimiter(limit_per_minute=2)
        self.assertTrue(limiter.is_allowed("127.0.0.1", "/api/health"))
        self.assertTrue(limiter.is_allowed("127.0.0.1", "/api/health"))

    def test_block_requests_over_limit(self):
        limiter = RateLimiter(limit_per_minute=2)
        limiter.is_allowed("127.0.0.1", "/api/health")
        limiter.is_allowed("127.0.0.1", "/api/health")
        self.assertFalse(limiter.is_allowed("127.0.0.1", "/api/health"))

    def test_limit_is_per_ip_and_path(self):
        limiter = RateLimiter(limit_per_minute=1)
        self.assertTrue(limiter.is_allowed("127.0.0.1", "/api/health"))
        self.assertFalse(limiter.is_allowed("127.0.0.1", "/api/health"))
        self.assertTrue(limiter.is_allowed("127.0.0.2", "/api/health"))
        self.assertTrue(limiter.is_allowed("127.0.0.1", "/api/tools"))

    def test_window_slides_over_time(self):
        limiter = RateLimiter(limit_per_minute=1)
        limiter.is_allowed("127.0.0.1", "/api/health")
        self.assertFalse(limiter.is_allowed("127.0.0.1", "/api/health"))
        # 手动将记录时间戳调整到窗口外，模拟时间流逝
        limiter._records[("127.0.0.1", "/api/health")][0] = time.time() - 61
        self.assertTrue(limiter.is_allowed("127.0.0.1", "/api/health"))


class TestRestApiSecurity(RunningServerMixin, unittest.TestCase):
    """REST API 端到端安全测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        self._stop_server()

    def test_no_api_key_request_succeeds(self):
        self._start_server(api_key="")
        status, body, _ = self._request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body.decode()), {"status": "ok"})

    def test_valid_api_key_request_succeeds(self):
        self._start_server(api_key="secret123")
        status, body, _ = self._request(
            "GET", "/api/health", headers={"Authorization": "Bearer secret123"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body.decode()), {"status": "ok"})

    def test_invalid_api_key_returns_401(self):
        self._start_server(api_key="secret123")
        with self.assertRaises(HTTPError) as ctx:
            self._request("GET", "/api/health", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(ctx.exception.code, 401)
        body = json.loads(ctx.exception.read().decode())
        self.assertEqual(body, {"error": "Unauthorized"})

    def test_rate_limit_returns_429(self):
        self._start_server(api_key="", rate_limit=2)
        # 前两次请求应成功
        self.assertEqual(self._request("GET", "/api/health")[0], 200)
        self.assertEqual(self._request("GET", "/api/health")[0], 200)
        # 第三次请求应触发限速
        with self.assertRaises(HTTPError) as ctx:
            self._request("GET", "/api/health")
        self.assertEqual(ctx.exception.code, 429)
        body = json.loads(ctx.exception.read().decode())
        self.assertEqual(body, {"error": "Rate limit exceeded"})

    def test_invalid_json_returns_400(self):
        self._start_server(api_key="")
        with self.assertRaises(HTTPError) as ctx:
            self._request(
                "POST",
                "/api/health",
                body=b"not json",
                headers={"Content-Type": "application/json", "Content-Length": "8"},
            )
        self.assertEqual(ctx.exception.code, 400)
        body = json.loads(ctx.exception.read().decode())
        self.assertEqual(body, {"error": "Invalid JSON"})

    def test_cors_default_no_origin_header(self):
        self._start_server(cors_origins=[])
        req = Request(self.base_url + "/api/health", method="GET")
        with urlopen(req, timeout=5) as resp:
            self.assertIsNone(resp.headers.get("Access-Control-Allow-Origin"))

    def test_cors_configured_matching_origin(self):
        self._start_server(cors_origins=["http://example.com"])
        req = Request(
            self.base_url + "/api/health",
            method="GET",
            headers={"Origin": "http://example.com"},
        )
        with urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "http://example.com")

    def test_cors_configured_non_matching_origin_not_set(self):
        self._start_server(cors_origins=["http://example.com"])
        req = Request(
            self.base_url + "/api/health",
            method="GET",
            headers={"Origin": "http://attacker.com"},
        )
        with urlopen(req, timeout=5) as resp:
            self.assertIsNone(resp.headers.get("Access-Control-Allow-Origin"))

    def test_options_request_returns_cors_headers_when_configured(self):
        self._start_server(cors_origins=["http://example.com"])
        req = Request(
            self.base_url + "/api/health",
            method="OPTIONS",
            headers={"Origin": "http://example.com"},
        )
        with urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 204)
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "http://example.com")
            self.assertIn("POST", resp.headers.get("Access-Control-Allow-Methods", ""))


if __name__ == "__main__":
    unittest.main()
