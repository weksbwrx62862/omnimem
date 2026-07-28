"""Plur 服务器模拟器(测试/开发用)。

.. deprecated::
    实验特性, 全仓无导入者。保留供本地联邦调试, 不在任何主链路或包入口引用。
"""

import asyncio
import json
import logging
from datetime import datetime

from aiohttp import web

logger = logging.getLogger(__name__)


class PlurServer:
    """Plur 服务器模拟器"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.engrams: dict[str, dict] = {}

        # 设置路由
        self._setup_routes()

        logger.info(f"PlurServer initialized on {host}:{port}")

    def _setup_routes(self):
        """设置路由"""
        self.app.router.add_get("/health", self.health_check)
        self.app.router.add_get("/stats", self.get_stats)
        self.app.router.add_get("/engrams", self.query_engrams)
        self.app.router.add_get("/engrams/{engram_id}", self.get_engram)
        self.app.router.add_post("/engrams", self.store_engram)
        self.app.router.add_put("/engrams/{engram_id}", self.update_engram)
        self.app.router.add_delete("/engrams/{engram_id}", self.delete_engram)

    async def health_check(self, request: web.Request) -> web.Response:
        """健康检查"""
        return web.json_response({"status": "healthy", "timestamp": datetime.now().isoformat()})

    async def get_stats(self, request: web.Request) -> web.Response:
        """获取统计信息"""
        stats = {
            "total_engrams": len(self.engrams),
            "memory_usage": f"{len(json.dumps(self.engrams))} bytes",
            "uptime": "running",
            "timestamp": datetime.now().isoformat()
        }
        return web.json_response(stats)

    async def store_engram(self, request: web.Request) -> web.Response:
        """存储 Engram"""
        try:
            data = await request.json()
            engram_id = data.get("id")

            if not engram_id:
                return web.json_response({"error": "Missing engram ID"}, status=400)

            # 添加时间戳
            data["created_at"] = data.get("created_at", datetime.now().isoformat())
            data["updated_at"] = datetime.now().isoformat()

            # 存储
            self.engrams[engram_id] = data

            logger.info(f"Stored engram: {engram_id}")
            return web.json_response({"success": True, "id": engram_id})

        except Exception as e:
            logger.error(f"Store engram error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def get_engram(self, request: web.Request) -> web.Response:
        """获取单个 Engram"""
        try:
            engram_id = request.match_info["engram_id"]

            if engram_id not in self.engrams:
                return web.json_response({"error": "Engram not found"}, status=404)

            return web.json_response(self.engrams[engram_id])

        except Exception as e:
            logger.error(f"Get engram error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def query_engrams(self, request: web.Request) -> web.Response:
        """查询 Engram"""
        try:
            # 获取查询参数
            query = request.query.get("query")
            tags = request.query.get("tags")
            since = request.query.get("since")
            limit = int(request.query.get("limit", 100))
            offset = int(request.query.get("offset", 0))

            # 过滤
            engrams = list(self.engrams.values())

            if query:
                engrams = [e for e in engrams if query.lower() in e.get("content", "").lower()]

            if tags:
                tag_list = tags.split(",")
                engrams = [e for e in engrams if any(t in e.get("tags", []) for t in tag_list)]

            if since:
                since_dt = datetime.fromisoformat(since)
                engrams = [e for e in engrams if datetime.fromisoformat(e.get("updated_at", "2000-01-01")) >= since_dt]

            # 排序（按更新时间倒序）
            engrams.sort(key=lambda e: e.get("updated_at", ""), reverse=True)

            # 分页
            engrams = engrams[offset:offset + limit]

            return web.json_response({
                "engrams": engrams,
                "total": len(engrams),
                "offset": offset,
                "limit": limit
            })

        except Exception as e:
            logger.error(f"Query engrams error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def update_engram(self, request: web.Request) -> web.Response:
        """更新 Engram"""
        try:
            engram_id = request.match_info["engram_id"]
            data = await request.json()

            if engram_id not in self.engrams:
                return web.json_response({"error": "Engram not found"}, status=404)

            # 更新时间戳
            data["updated_at"] = datetime.now().isoformat()

            # 更新
            self.engrams[engram_id] = data

            logger.info(f"Updated engram: {engram_id}")
            return web.json_response({"success": True, "id": engram_id})

        except Exception as e:
            logger.error(f"Update engram error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def delete_engram(self, request: web.Request) -> web.Response:
        """删除 Engram"""
        try:
            engram_id = request.match_info["engram_id"]

            if engram_id not in self.engrams:
                return web.json_response({"error": "Engram not found"}, status=404)

            # 删除
            del self.engrams[engram_id]

            logger.info(f"Deleted engram: {engram_id}")
            return web.json_response({"success": True, "id": engram_id})

        except Exception as e:
            logger.error(f"Delete engram error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def start(self):
        """启动服务器"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info(f"PlurServer started on http://{self.host}:{self.port}")

        # 保持运行
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await runner.cleanup()

    async def stop(self):
        """停止服务器"""
        logger.info("PlurServer stopped")


async def run_server(host: str = "0.0.0.0", port: int = 8080):
    """运行服务器"""
    server = PlurServer(host=host, port=port)
    await server.start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plur Server Simulator")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    asyncio.run(run_server(args.host, args.port))
