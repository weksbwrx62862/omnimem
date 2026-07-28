"""FastAPI 版 REST 服务测试（★ M9-20：安全语义与 test_rest_api.py 对齐）。"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from unittest.mock import MagicMock  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from omnimem.api_fastapi import create_app  # noqa: E402


def _make_client(**kwargs) -> TestClient:
    sdk = MagicMock()
    sdk.health_check.return_value = {"status": "ok"}
    sdk.recall.return_value = {"memories": []}
    app = create_app(sdk, **kwargs)
    return TestClient(app)


class TestAuth:
    def test_no_api_key_request_succeeds(self):
        c = _make_client(api_key="")
        r = c.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_valid_api_key_request_succeeds(self):
        c = _make_client(api_key="secret123")
        r = c.get("/api/health", headers={"Authorization": "Bearer secret123"})
        assert r.status_code == 200

    def test_invalid_api_key_returns_401(self):
        c = _make_client(api_key="secret123")
        r = c.get("/api/health", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
        assert r.json() == {"error": "Unauthorized"}

    def test_missing_auth_returns_401(self):
        c = _make_client(api_key="secret123")
        assert c.get("/api/health").status_code == 401


class TestRateLimit:
    def test_rate_limit_returns_429(self):
        c = _make_client(api_key="", rate_limit_per_minute=2)
        assert c.get("/api/health").status_code == 200
        assert c.get("/api/health").status_code == 200
        r = c.get("/api/health")
        assert r.status_code == 429
        assert r.json() == {"error": "Rate limit exceeded"}


class TestBodyAndRouting:
    def test_invalid_json_returns_400(self):
        c = _make_client()
        r = c.post("/api/health", content=b"not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 400
        assert r.json() == {"error": "Invalid JSON"}

    def test_unknown_endpoint_returns_404(self):
        c = _make_client()
        assert c.post("/api/nonexistent", json={}).status_code == 404

    def test_oversized_body_returns_413(self):
        c = _make_client()
        r = c.post("/api/recall", content=b"x", headers={"Content-Length": str(11 * 1024 * 1024)})
        assert r.status_code == 413

    def test_recall_dispatches_to_sdk(self):
        c = _make_client()
        r = c.post("/api/recall", json={"query": "hello"})
        assert r.status_code == 200
        assert r.json() == {"memories": []}

    def test_tools_list(self):
        c = _make_client()
        r = c.get("/api/tools")
        assert r.status_code == 200
        assert "memorize" in r.json()["tools"]

    def test_sdk_exception_returns_500(self):
        c = _make_client()
        c.app.state.sdk.govern.side_effect = RuntimeError("boom")
        r = c.post("/api/govern", json={})
        assert r.status_code == 500
        assert "boom" in r.json()["error"]


class TestAdminGate:
    def test_export_without_admin_token_returns_403(self):
        c = _make_client(admin_token="")
        r = c.post("/api/export", json={})
        assert r.status_code == 403

    def test_export_with_wrong_admin_token_returns_403(self):
        c = _make_client(admin_token="admintok")
        r = c.post("/api/export", json={}, headers={"X-Admin-Token": "wrong"})
        assert r.status_code == 403

    def test_export_with_valid_admin_token_succeeds(self):
        c = _make_client(admin_token="admintok")
        c.app.state.sdk.export_memories.return_value = {"exported": 0}
        r = c.post("/api/export", json={}, headers={"X-Admin-Token": "admintok"})
        assert r.status_code == 200


class TestCorsAndDocs:
    def test_cors_matching_origin(self):
        c = _make_client(cors_allowed_origins=["http://example.com"])
        r = c.get("/api/health", headers={"Origin": "http://example.com"})
        assert r.headers.get("Access-Control-Allow-Origin") == "http://example.com"

    def test_cors_non_matching_origin_not_set(self):
        c = _make_client(cors_allowed_origins=["http://example.com"])
        r = c.get("/api/health", headers={"Origin": "http://attacker.com"})
        assert r.headers.get("Access-Control-Allow-Origin") is None

    def test_options_returns_204_with_cors(self):
        c = _make_client(cors_allowed_origins=["http://example.com"])
        r = c.options("/api/health", headers={"Origin": "http://example.com"})
        assert r.status_code == 204
        assert "POST" in r.headers.get("Access-Control-Allow-Methods", "")

    def test_docs_and_openapi_available_without_auth(self):
        c = _make_client(api_key="secret123")
        assert c.get("/openapi.json").status_code == 200
        assert c.get("/docs").status_code == 200

    def test_metrics_open_endpoint(self):
        c = _make_client(api_key="secret123")
        r = c.get("/metrics")
        assert r.status_code == 200
