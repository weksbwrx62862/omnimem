"""
CostTracker — 操作成本追踪器。

记录每条记忆操作（写/读/搜索/更新/删除）的：
  - 耗时 (latency_ms)
  - 操作类型 (operation)
  - 结果数量 (result_count)

用于五维评测的 "操作成本" 维度，
也用于日常监控识别慢操作。
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """单条操作成本记录。"""
    operation: str        # write / read / search / update / delete
    latency_ms: float     # 耗时（毫秒）
    result_count: int     # 返回/写入的记录数
    memory_type: str = "" # 记忆类型（如 fact/preference/correction）
    timestamp: float = 0.0


@dataclass
class CostStats:
    """操作成本统计摘要。"""
    operation: str
    count: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


class CostTracker:
    """轻量操作成本追踪器。

    使用方式：
        tracker = CostTracker()
        with tracker.track("search"):
            results = retriever.search(query)
        # 自动记录耗时

    支持运行时开关（debug mode 关闭追踪以消除开销）。
    """

    def __init__(self, enabled: bool = True, max_records: int = 10000):
        self._enabled = enabled
        self._records: list[CostRecord] = []
        self._max_records = max_records
        self._summary_cache: dict[str, CostStats] | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val

    def record(
        self,
        operation: str,
        latency_ms: float,
        result_count: int = 0,
        memory_type: str = "",
    ) -> None:
        """记录一条操作成本。"""
        if not self._enabled:
            return
        if len(self._records) >= self._max_records:
            return  # 达到上限后静默丢弃
        self._records.append(CostRecord(
            operation=operation,
            latency_ms=latency_ms,
            result_count=result_count,
            memory_type=memory_type,
            timestamp=time.time(),
        ))
        self._summary_cache = None  # 使缓存失效

    def track(self, operation: str, memory_type: str = ""):
        """上下文管理器，自动记录耗时。

        用法：
            with tracker.track("search"):
                ...
        """
        return _CostTrackerContext(self, operation, memory_type)

    def summary(self) -> list[CostStats]:
        """返回按操作类型聚合的统计摘要。"""
        if self._summary_cache is not None:
            return list(self._summary_cache.values())

        by_op: dict[str, list[float]] = defaultdict(list)
        counts: dict[str, int] = defaultdict(int)
        for r in self._records:
            by_op[r.operation].append(r.latency_ms)
            counts[r.operation] += 1

        stats_list = []
        for op, latencies in by_op.items():
            sorted_lat = sorted(latencies)
            stats_list.append(CostStats(
                operation=op,
                count=counts[op],
                total_latency_ms=sum(latencies),
                avg_latency_ms=sum(latencies) / len(latencies),
                max_latency_ms=max(latencies),
                p95_latency_ms=sorted_lat[int(len(sorted_lat) * 0.95)],
            ))

        self._summary_cache = {s.operation: s for s in stats_list}
        return stats_list

    def report(self) -> dict[str, Any]:
        """生成完整的成本报告（用于 benchmark 输出）。"""
        stats = self.summary()
        return {
            "total_operations": len(self._records),
            "by_operation": {
                s.operation: {
                    "count": s.count,
                    "avg_latency_ms": round(s.avg_latency_ms, 2),
                    "max_latency_ms": round(s.max_latency_ms, 2),
                    "p95_latency_ms": round(s.p95_latency_ms, 2),
                    "total_latency_ms": round(s.total_latency_ms, 2),
                }
                for s in stats
            },
        }

    def reset(self) -> None:
        """清空所有记录。"""
        self._records.clear()
        self._summary_cache = None

    def save(self, path: str | Path) -> None:
        """保存成本记录到 JSON 文件。"""
        data = {
            "records": [
                {
                    "operation": r.operation,
                    "latency_ms": r.latency_ms,
                    "result_count": r.result_count,
                    "memory_type": r.memory_type,
                    "timestamp": r.timestamp,
                }
                for r in self._records
            ],
            "summary": self.report(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Cost report saved to %s (%d records)", path, len(self._records))


class _CostTrackerContext:
    """上下文管理器实现。"""

    def __init__(self, tracker: CostTracker, operation: str, memory_type: str = ""):
        self._tracker = tracker
        self._operation = operation
        self._memory_type = memory_type
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._tracker._enabled:
            latency = (time.perf_counter() - self._start) * 1000.0
            self._tracker.record(
                operation=self._operation,
                latency_ms=latency,
                result_count=0,  # 调用方手动更新
                memory_type=self._memory_type,
            )
