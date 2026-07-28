"""Plur API 客户端。

.. deprecated::
    Plur 联邦为实验特性, 未在主链路(sdk/provider/services)使用。
    保留供未来多实例场景显式导入; 已从 omnimem.core 包入口移除自动导出。
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import aiohttp
from omnimem.core.engram_bridge import Engram
from omnimem.core.plur_config import get_config

logger = logging.getLogger(__name__)


class PlurClient:
    """Plur API 客户端"""

    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint or get_config().plur_endpoint
        self.timeout = get_config().get("plur.timeout", 30)
        self.retry_count = get_config().get("plur.retry_count", 3)
        self.retry_delay = get_config().get("plur.retry_delay", 5)

        self._session: aiohttp.ClientSession | None = None

        logger.info(f"PlurClient initialized with endpoint: {self.endpoint}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        """关闭 HTTP 会话"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        params: dict | None = None
    ) -> dict[str, Any]:
        """发送 HTTP 请求（带重试）"""
        url = f"{self.endpoint}/{path.lstrip('/')}"

        for attempt in range(self.retry_count):
            try:
                session = await self._get_session()

                async with session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 404:
                        return {"error": "Not found", "status": 404}
                    else:
                        error_text = await response.text()
                        logger.warning(f"Request failed: {response.status} - {error_text}")

                        if attempt < self.retry_count - 1:
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                            continue
                        else:
                            return {"error": error_text, "status": response.status}

            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.retry_count})")
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    return {"error": "Timeout", "status": 408}

            except Exception as e:
                logger.error(f"Request error: {e}")
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    return {"error": str(e), "status": 500}

        return {"error": "Max retries exceeded", "status": 500}

    async def store_engram(self, engram: Engram) -> bool:
        """存储 Engram 到 Plur"""
        try:
            data = engram.to_dict()
            result = await self._request("POST", "/engrams", data=data)

            if "error" in result:
                logger.error(f"Failed to store engram: {result['error']}")
                return False

            logger.debug(f"Stored engram {engram.id} to Plur")
            return True

        except Exception as e:
            logger.error(f"Store engram error: {e}")
            return False

    async def get_engram(self, engram_id: str) -> Engram | None:
        """获取单个 Engram"""
        try:
            result = await self._request("GET", f"/engrams/{engram_id}")

            if "error" in result:
                logger.warning(f"Failed to get engram {engram_id}: {result['error']}")
                return None

            return Engram.from_dict(result)

        except Exception as e:
            logger.error(f"Get engram error: {e}")
            return None

    async def query_engrams(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[Engram]:
        """查询 Engram"""
        try:
            params = {
                "limit": limit,
                "offset": offset
            }

            if query:
                params["query"] = query

            if tags:
                params["tags"] = ",".join(tags)

            if since:
                params["since"] = since.isoformat()

            result = await self._request("GET", "/engrams", params=params)

            if "error" in result:
                logger.error(f"Failed to query engrams: {result['error']}")
                return []

            engrams_data = result.get("engrams", [])
            return [Engram.from_dict(data) for data in engrams_data]

        except Exception as e:
            logger.error(f"Query engrams error: {e}")
            return []

    async def update_engram(self, engram: Engram) -> bool:
        """更新 Engram"""
        try:
            data = engram.to_dict()
            result = await self._request("PUT", f"/engrams/{engram.id}", data=data)

            if "error" in result:
                logger.error(f"Failed to update engram: {result['error']}")
                return False

            logger.debug(f"Updated engram {engram.id}")
            return True

        except Exception as e:
            logger.error(f"Update engram error: {e}")
            return False

    async def delete_engram(self, engram_id: str) -> bool:
        """删除 Engram"""
        try:
            result = await self._request("DELETE", f"/engrams/{engram_id}")

            if "error" in result:
                logger.error(f"Failed to delete engram: {result['error']}")
                return False

            logger.debug(f"Deleted engram {engram_id}")
            return True

        except Exception as e:
            logger.error(f"Delete engram error: {e}")
            return False

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            result = await self._request("GET", "/health")
            return "error" not in result
        except Exception:
            return False

    async def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        try:
            result = await self._request("GET", "/stats")
            return result
        except Exception as e:
            logger.error(f"Get stats error: {e}")
            return {"error": str(e)}


# 模拟 Plur 服务器（用于测试）
class MockPlurServer:
    """模拟 Plur 服务器"""

    def __init__(self):
        self.engrams: dict[str, Engram] = {}
        logger.info("MockPlurServer initialized")

    async def store_engram(self, engram: Engram) -> bool:
        """存储 Engram"""
        self.engrams[engram.id] = engram
        return True

    async def get_engram(self, engram_id: str) -> Engram | None:
        """获取 Engram"""
        return self.engrams.get(engram_id)

    async def query_engrams(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[Engram]:
        """查询 Engram"""
        engrams = list(self.engrams.values())

        # 过滤
        if query:
            engrams = [e for e in engrams if query.lower() in e.content.lower()]

        if tags:
            engrams = [e for e in engrams if any(t in e.tags for t in tags)]

        if since:
            engrams = [e for e in engrams if e.updated_at >= since]

        # 排序（按更新时间倒序）
        engrams.sort(key=lambda e: e.updated_at, reverse=True)

        # 分页
        engrams = engrams[offset:offset + limit]

        return engrams

    async def health_check(self) -> bool:
        """健康检查"""
        return True

    async def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        return {
            "total_engrams": len(self.engrams),
            "memory_usage": "N/A",
            "uptime": "N/A"
        }


# 工厂函数
def create_plur_client(endpoint: str | None = None) -> PlurClient:
    """创建 Plur 客户端"""
    return PlurClient(endpoint=endpoint)


def create_mock_server() -> MockPlurServer:
    """创建模拟服务器"""
    return MockPlurServer()
