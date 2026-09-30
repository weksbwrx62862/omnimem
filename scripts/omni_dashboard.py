#!/usr/bin/env python3
"""omni_dashboard.py — OmniMem 可视化面板启动器（改进项 #5）。

零依赖：仅用标准库 http.server。做两件事：
  1) 静态托管 dashboard/ 目录（index.html / app.js / styles.css）；
  2) 反向代理 /api/* 与 /metrics 到 OmniMem REST 后端，
     使浏览器与面板同源，规避 CORS（REST 服务本身无需任何改动）。

默认只绑定 127.0.0.1（fail-closed，与 REST/服务层安全基调一致），
并原样转发 Authorization / X-Admin-Token 请求头。

用法：
  # 先按需启动 REST 后端： python -m omnimem.api_fastapi
  python scripts/omni_dashboard.py                 # 面板 http://127.0.0.1:8790
  python scripts/omni_dashboard.py --port 9000 --api http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DASH_DIR = Path(__file__).resolve().parent.parent / "dashboard"
STATIC_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}
# 转发给上游的客户端请求头白名单。
RELAY_HEADERS = ("authorization", "x-admin-token", "content-type", "accept")

API_BASE = "http://127.0.0.1:8765"


class Handler(BaseHTTPRequestHandler):
    server_version = "OmniMemDashboard/1.0"

    def log_message(self, fmt, *args):  # noqa: A002 - 覆写默认噪声日志
        sys.stderr.write("[dashboard] %s\n" % (fmt % args))

    # ── 入口分发 ──────────────────────────────────────
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/") or self.path == "/metrics":
            self._proxy("GET")
        else:
            self._static()

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/api/") or self.path == "/metrics":
            self._proxy("POST")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    # ── 静态文件 ──────────────────────────────────────
    def _static(self):
        rel = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        target = (DASH_DIR / rel).resolve()
        # 目录穿越防护
        if not str(target).startswith(str(DASH_DIR.resolve())) or not target.is_file():
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        mime = STATIC_MIME.get(target.suffix.lower(), "application/octet-stream")
        self._send(200, target.read_bytes(), mime)

    # ── 反向代理 ──────────────────────────────────────
    def _proxy(self, method: str):
        url = API_BASE.rstrip("/") + self.path
        body = None
        length = int(self.headers.get("content-length", 0) or 0)
        if length:
            body = self.rfile.read(length)
        req = urllib.request.Request(url, data=body, method=method)
        for h in RELAY_HEADERS:
            val = self.headers.get(h)
            if val:
                req.add_header(h, val)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                self._send(resp.status, resp.read(), resp.headers.get("content-type", "application/json"))
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read(), e.headers.get("content-type", "application/json") if e.headers else "application/json")
        except (urllib.error.URLError, OSError) as e:
            payload = json.dumps({"error": f"上游不可达 {API_BASE}: {e}"}, ensure_ascii=False).encode()
            self._send(502, payload, "application/json; charset=utf-8")

    # ── 写出 ──────────────────────────────────────────
    def _send(self, status: int, payload: bytes, ctype: str):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main(argv: list[str] | None = None) -> int:
    global API_BASE
    parser = argparse.ArgumentParser(description="启动 OmniMem 可视化面板（反向代理到 REST 后端）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--port", type=int, default=8790, help="监听端口（默认 8790）")
    parser.add_argument("--api", default=API_BASE, help="OmniMem REST 后端地址")
    args = parser.parse_args(argv)
    API_BASE = args.api

    if not DASH_DIR.is_dir():
        print(f"[error] 未找到面板目录: {DASH_DIR}", file=sys.stderr)
        return 1

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[dashboard] http://{args.host}:{args.port}  →  代理 {API_BASE}  (Ctrl-C 停止)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] 已停止")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
