"""
MemoryAPI — 记忆系统 REST API 模块。

提供:
1. 记忆查询 API
2. 评估 API
3. 统计 API
4. 可视化 API
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryAPI:
    """记忆系统 API"""

    def __init__(self, governance_dir: Path | None = None):
        from omnimem.governance.forgetting import ForgettingCurve
        from omnimem.governance.visualizer import get_visualizer

        self._governance_dir = governance_dir or Path.home() / ".hermes" / "omnimem" / "governance"
        self._curve = ForgettingCurve(self._governance_dir)
        self._visualizer = get_visualizer()

    def get_memory_info(self, memory_id: str) -> dict[str, Any]:
        """获取记忆详细信息

        Args:
            memory_id: 记忆 ID

        Returns:
            记忆信息字典
        """
        try:
            # 基本信息
            self._curve.get_status()

            # FSRS 信息
            fsrs_retention = self._curve.calculate_fsrs_retention(memory_id)
            review_time = self._curve.suggest_review_time(memory_id)

            # 记忆强度
            strength = self._curve.evaluate_memory_strength(memory_id)

            # 语义重要性
            semantic = self._curve.evaluate_semantic_importance(memory_id)

            return {
                "memory_id": memory_id,
                "fsrs": {
                    "retention": fsrs_retention,
                    "suggested_review": review_time.isoformat() if review_time else None,
                },
                "strength": strength,
                "semantic": semantic,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error("get_memory_info failed: %s", e)
            return {"error": str(e)}

    def get_system_stats(self) -> dict[str, Any]:
        """获取系统统计信息

        Returns:
            系统统计字典
        """
        try:
            # 基本状态
            status = self._curve.get_status()

            # FSRS 统计
            fsrs_stats = self._curve.get_fsrs_stats()

            # 记忆强度分布
            strength_dist = self._curve.get_strength_distribution()

            # 语义重要性分布
            semantic_dist = self._curve.get_semantic_importance_distribution()

            return {
                "total_memories": sum(status.get("stages", {}).values()),
                "stages": status.get("stages", {}),
                "heat": status.get("heat", {}),
                "upgrade_candidates": status.get("upgrade_candidates_count", 0),
                "fsrs": fsrs_stats,
                "grades": strength_dist.get("grades", {}),
                "semantic": semantic_dist,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error("get_system_stats failed: %s", e)
            return {"error": str(e)}

    def run_archive_cycle(self) -> dict[str, Any]:
        """运行归档周期

        Returns:
            归档结果
        """
        try:
            archived = self._curve.run_archive_cycle()
            return {
                "archived": archived,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error("run_archive_cycle failed: %s", e)
            return {"error": str(e)}

    def generate_dashboard(self, output_dir: str | None = None) -> dict[str, Any]:
        """生成仪表盘

        Args:
            output_dir: 输出目录

        Returns:
            生成结果
        """
        try:
            stats = self.get_system_stats()
            filepath = self._visualizer.save_dashboard(stats, output_dir=output_dir)

            return {
                "filepath": filepath,
                "stats": stats,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error("generate_dashboard failed: %s", e)
            return {"error": str(e)}

    def evaluate_batch(self, memory_ids: list[str]) -> dict[str, Any]:
        """批量评估记忆

        Args:
            memory_ids: 记忆 ID 列表

        Returns:
            评估结果
        """
        results = []
        for memory_id in memory_ids:
            result = self.get_memory_info(memory_id)
            results.append(result)

        return {
            "evaluated": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

    def close(self):
        """关闭资源"""
        if self._curve:
            self._curve.close()


# 全局实例
_api: MemoryAPI | None = None


def get_api(governance_dir: Path | None = None) -> MemoryAPI:
    """获取全局 API 实例"""
    global _api
    if _api is None:
        _api = MemoryAPI(governance_dir)
    return _api


# 便捷函数
def get_memory_info(memory_id: str) -> dict[str, Any]:
    """获取记忆信息"""
    api = get_api()
    return api.get_memory_info(memory_id)


def get_system_stats() -> dict[str, Any]:
    """获取系统统计"""
    api = get_api()
    return api.get_system_stats()


def run_archive_cycle() -> dict[str, Any]:
    """运行归档周期"""
    api = get_api()
    return api.run_archive_cycle()


def generate_dashboard(output_dir: str | None = None) -> dict[str, Any]:
    """生成仪表盘"""
    api = get_api()
    return api.generate_dashboard(output_dir)
