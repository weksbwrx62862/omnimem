"""改进项 #5 面板启动器单测（离线，纯标准库，无需 omnimem 运行时）。

用一个 mock 上游 HTTP 服务验证反向代理的转发/鉴权头透传/错误中继，
以及静态托管与目录穿越防护。
"""
from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

MOD_PATH = Path(__file__).resolve().parent.parent / "scripts" / "omni_dashboard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("omni_dashboard", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


class _Upstream(BaseHTTPRequestHandler):
    captured: dict = {}

    def log_message(self, *a):  # silence
        pass

    def _read_body(self):
        n = int(self.headers.get("content-length", 0) or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):  # noqa: N802
        type(self).captured = {"method": "GET", "path": self.path, "auth": self.headers.get("authorization")}
        payload = json.dumps({"status": "healthy", "vector_count": 7}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802
        body = self._read_body()
        type(self).captured = {
            "method": "POST", "path": self.path, "auth": self.headers.get("authorization"),
            "admin": self.headers.get("x-admin-token"), "body": body.decode(),
        }
        if self.path == "/api/boom":
            payload = json.dumps({"error": "kaboom"}).encode()
            self.send_response(500)
        else:
            payload = json.dumps({"status": "ok", "echo": json.loads(body or b"{}")}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, r.read().decode()


def _post(url, obj, headers=None):
    data = json.dumps(obj).encode()
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, method="POST", headers=h)
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, r.read().decode()


@pytest.fixture
def dash(monkeypatch):
    mod = _load_module()
    up_srv, up_url = _serve(_Upstream)
    monkeypatch.setattr(mod, "API_BASE", up_url)
    dash_srv, dash_url = _serve(mod.Handler)
    yield mod, dash_url
    dash_srv.shutdown()
    dash_srv.server_close()
    up_srv.shutdown()
    up_srv.server_close()


def _free_upstream_url():
    # 指向一个未监听端口以触发 502
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    port = srv.server_address[1]
    srv.server_close()  # 注意：未跑 serve_forever，绝不可调用 shutdown()（会永久阻塞）
    return f"http://127.0.0.1:{port}"


def test_static_index_served(dash):
    mod, url = dash
    status, body = _get(url + "/")
    assert status == 200
    assert "OmniMem" in body and "<html" in body.lower()


def test_static_assets_content_type(dash):
    _, url = dash
    req = urllib.request.Request(url + "/app.js")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200
        assert "javascript" in r.headers.get("content-type", "").lower()


def test_path_traversal_blocked(dash):
    _, url = dash
    try:
        _get(url + "/..%2f..%2f..%2fetc%2fpasswd")
        raise AssertionError("expected 404 for traversal")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_proxy_get_forwards_auth(dash):
    mod, url = dash
    status, body = _get(url + "/api/health", headers={"Authorization": "Bearer tok123"})
    assert status == 200
    assert json.loads(body)["status"] == "healthy"
    assert _Upstream.captured["auth"] == "Bearer tok123"
    assert _Upstream.captured["path"] == "/api/health"


def test_proxy_post_forwards_body_and_admin(dash):
    _, url = dash
    status, body = _post(
        url + "/api/recall", {"query": "db", "mode": "hybrid"},
        headers={"Authorization": "Bearer k", "X-Admin-Token": "admin-1"},
    )
    assert status == 200
    cap = _Upstream.captured
    assert cap["method"] == "POST"
    assert cap["admin"] == "admin-1"
    assert json.loads(cap["body"]) == {"query": "db", "mode": "hybrid"}
    assert json.loads(body)["echo"]["query"] == "db"


def test_proxy_relays_upstream_error(dash):
    _, url = dash
    try:
        _post(url + "/api/boom", {"x": 1})
        raise AssertionError("expected HTTP 500 relayed")
    except urllib.error.HTTPError as e:
        assert e.code == 500
        assert json.loads(e.read().decode())["error"] == "kaboom"


def test_proxy_metrics_path(dash):
    _, url = dash
    # 上游 GET 统一返回 JSON，/metrics 也应被代理
    status, body = _get(url + "/metrics", headers={"Authorization": "Bearer z"})
    assert status == 200
    assert _Upstream.captured["path"] == "/metrics"


def test_unreachable_upstream_returns_502(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "API_BASE", _free_upstream_url())
    srv, url = _serve(mod.Handler)
    try:
        try:
            _get(url + "/api/health")
            raise AssertionError("expected 502")
        except urllib.error.HTTPError as e:
            assert e.code == 502
    finally:
        srv.shutdown()
        srv.server_close()
