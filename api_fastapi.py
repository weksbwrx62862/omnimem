"""FastAPI 版 REST 服务（★ M9-20：omnimem[api] extra，替代 http.server 实现）。

与 rest_api.py 共用 Auth/AdminAuth/RateLimiter 中间件，安全语义完全一致；
自动提供 /docs OpenAPI 文档。fastapi/uvicorn 未安装时本模块不可导入（omnimem[api]）。
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from omnimem.rest_api import (
    _MAX_REQUEST_BODY_BYTES,
    AdminAuthMiddleware,
    AuthMiddleware,
    RateLimiter,
    _generate_default_key,
)
from omnimem.utils.metrics import (
    get_metrics_collector,
    record_memorize_duration,
    record_recall_duration,
    record_reflect_duration,
)

logger = logging.getLogger(__name__)

# from __future__ import annotations makes hints strings; FastAPI resolves
# Request via module globals. Fallback to Any when fastapi is not installed.
try:
    from fastapi import Request
except ImportError:
    Request = Any  # type: ignore

_OPEN_PATHS = {"/metrics", "/docs", "/redoc", "/openapi.json"}
_ADMIN_PATHS = {"/api/export", "/api/import"}
_TOOLS = ["memorize", "recall", "reflect", "govern", "compact", "detail", "export", "import", "health"]


def create_app(
    sdk: Any = None,
    *,
    api_key: str = "",
    admin_token: str = "",
    rate_limit_per_minute: int = 60,
    cors_allowed_origins: list[str] | None = None,
):
    """构建 FastAPI 应用（sdk 可注入 mock 便于测试）。"""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, PlainTextResponse

    app = FastAPI(title="OmniMem REST API", version="1.1.0")
    auth = AuthMiddleware(api_key=api_key)
    admin_auth = AdminAuthMiddleware(admin_token=admin_token)
    limiter = RateLimiter(limit_per_minute=rate_limit_per_minute)
    cors_origins = cors_allowed_origins or []
    app.state.sdk = sdk

    def _cors_headers(request: Request) -> dict[str, str]:
        origin = (request.headers.get("origin") or "").strip()
        if cors_origins and origin in cors_origins:
            return {"Access-Control-Allow-Origin": origin}
        return {}

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        path = request.url.path
        if path in _OPEN_PATHS or request.method == "OPTIONS":
            return await call_next(request)
        headers = {"Authorization": request.headers.get("authorization", "")}
        result = auth.validate(headers)
        if result is not True:
            code, body = result
            return JSONResponse(body, status_code=code, headers=_cors_headers(request))
        client_ip = request.client.host if request.client else "unknown"
        if not limiter.is_allowed(client_ip, path):
            return JSONResponse({"error": "Rate limit exceeded"}, status_code=429, headers=_cors_headers(request))
        if path in _ADMIN_PATHS:
            admin_headers = {"X-Admin-Token": request.headers.get("x-admin-token", "")}
            admin_result = admin_auth.validate(admin_headers)
            if admin_result is not True:
                code, body = admin_result
                return JSONResponse(body, status_code=code, headers=_cors_headers(request))
        cl = request.headers.get("content-length")
        if cl and int(cl) > _MAX_REQUEST_BODY_BYTES:
            return JSONResponse({"error": "请求体超过 10MB 限制"}, status_code=413, headers=_cors_headers(request))
        response = await call_next(request)
        for k, v in _cors_headers(request).items():
            response.headers[k] = v
        return response

    @app.options("/{full_path:path}")
    async def options_handler(request: Request, full_path: str):  # noqa: ARG001
        headers = {
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
        headers.update(_cors_headers(request))
        return JSONResponse(None, status_code=204, headers=headers)

    async def _read_json(request: Request) -> dict | JSONResponse:
        raw = await request.body()
        if len(raw) > _MAX_REQUEST_BODY_BYTES:
            return JSONResponse({"error": "请求体超过 10MB 限制"}, status_code=413)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    def _audit(operation: str, details: dict | None = None) -> None:
        sdk_obj = app.state.sdk
        governance = getattr(sdk_obj, "_governance", None)
        audit_logger = getattr(governance, "audit_logger", None) if governance else None
        if audit_logger is not None:
            try:
                audit_logger.log(operation, details=details, result="success",
                                 instance_id=getattr(sdk_obj, "_instance_id", None))
            except Exception as e:
                logger.warning("审计日志写入失败: %s", e)

    def _dispatch(path_name: str, body: dict) -> Any:
        """按端点名分发到 SDK,与 rest_api.py 路由语义一致。"""
        sdk_obj = app.state.sdk
        if path_name == "memorize":
            start = time.time()
            try:
                return sdk_obj.memorize(**body)
            finally:
                record_memorize_duration(time.time() - start)
        if path_name == "recall":
            start = time.time()
            try:
                return sdk_obj.recall(**body)
            finally:
                record_recall_duration(time.time() - start)
        if path_name == "reflect":
            start = time.time()
            try:
                return sdk_obj.reflect(**body)
            finally:
                record_reflect_duration(time.time() - start)
        if path_name == "govern":
            return sdk_obj.govern(**body)
        if path_name == "compact":
            return sdk_obj.compact(**body)
        if path_name == "detail":
            action = body.get("action", "list")
            if action == "list":
                return sdk_obj.detail_list(**body)
            if action == "events":
                return sdk_obj.detail_events(**body)
            return sdk_obj.detail(**body)
        if path_name == "export":
            _audit("export", details={"path": body.get("output_path"), "format": body.get("format", "json")})
            return sdk_obj.export_memories(**body)
        if path_name == "import":
            _audit("import", details={"path": body.get("input_path")})
            return sdk_obj.import_memories(**body)
        if path_name == "health":
            return sdk_obj.health_check()
        return None

    @app.post("/api/{endpoint}")
    async def post_endpoint(endpoint: str, request: Request):
        if endpoint not in _TOOLS:
            return JSONResponse({"error": f"Not found: /api/{endpoint}"}, status_code=404)
        body = await _read_json(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            return JSONResponse(_dispatch(endpoint, body), status_code=200)
        except Exception as e:
            logger.exception("REST API 处理失败: path=/api/%s", endpoint)
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/health")
    async def get_health():
        sdk_obj = app.state.sdk
        return JSONResponse(sdk_obj.health_check() if sdk_obj else {"status": "not initialized"})

    @app.get("/api/tools")
    async def get_tools():
        return JSONResponse({"tools": list(_TOOLS)})

    @app.get("/metrics")
    async def get_metrics():
        try:
            body = get_metrics_collector().collect_all()
        except Exception as e:
            body = f"# 指标收集失败: {e}\n"
        return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")

    return app


def run_api(host: str = "127.0.0.1", port: int = 8765, storage_dir: str | None = None, config: dict | None = None):
    """启动 FastAPI 版 REST 服务（默认仅绑定 127.0.0.1,fail-closed 与 rest_api.run_api 一致）。"""
    import uvicorn

    from omnimem.sdk import OmniMemSDK

    sdk = OmniMemSDK(storage_dir=storage_dir, config=config)
    api_key = sdk._config.get("api_key", "")
    if not api_key:
        api_key = _generate_default_key()
        logger.warning("REST API 未配置 api_key,已自动生成默认密钥（请查看控制台输出）")
    admin_token = sdk._config.get("admin_token", "") or os.environ.get("OMNIMEM_ADMIN_TOKEN", "")
    if not admin_token:
        admin_token = _generate_default_key()
        logger.warning("REST API 未配置 admin_token,已自动生成默认管理令牌（请查看控制台输出）")

    app = create_app(
        sdk,
        api_key=api_key,
        admin_token=admin_token,
        rate_limit_per_minute=sdk._config.get("api_rate_limit_per_minute", 60),
        cors_allowed_origins=sdk._config.get("cors_allowed_origins", []),
    )
    print(f"OmniMem REST API (FastAPI) running on http://{host}:{port} — docs at /docs")
    print(f"API Key: {api_key}")
    print(f"Admin Token: {admin_token}")
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        sdk.close()


if __name__ == "__main__":
    import sys

    _port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    _host = os.environ.get("OMNIMEM_API_HOST", "127.0.0.1")
    run_api(host=_host, port=_port)
