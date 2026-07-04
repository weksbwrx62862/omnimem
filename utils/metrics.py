"""OmniMem 监控指标 — 轻量级 Prometheus 兼容指标收集器。

不依赖外部库，自行实现 Counter/Histogram/Gauge 三种指标类型。
支持 Prometheus 文本格式导出。

设计目标：
  1. 零外部依赖（不依赖 prometheus_client）
  2. 线程安全（所有指标操作使用 threading.Lock）
  3. Prometheus 文本格式兼容（version=0.0.4）
  4. 模块级单例，全局可访问
  5. 告警机制可选，处理失败不影响主流程
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

# 默认 Histogram 桶（覆盖 5ms ~ 60s 延迟范围）
_DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60]


def _format_labels(labels: dict[str, Any] | None) -> str:
    """格式化标签为 Prometheus 文本格式。

    示例: {"type": "reflect"} → '{type="reflect"}'
    无标签时返回空字符串。
    """
    if not labels:
        return ""
    parts = []
    for key in sorted(labels.keys()):
        # 转义标签值中的特殊字符（反斜杠、双引号、换行）
        value = str(labels[key]).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        parts.append(f'{key}="{value}"')
    return "{" + ",".join(parts) + "}"


def _merge_labels(base: dict[str, Any] | None, extra: dict[str, Any]) -> dict[str, Any]:
    """合并基础标签与额外标签，extra 优先。"""
    merged: dict[str, Any] = {}
    if base:
        merged.update(base)
    merged.update(extra)
    return merged


class Counter:
    """单调递增计数器。

    适用于：请求次数、错误次数、处理总量等只增不减的指标。
    """

    def __init__(self, name: str, description: str, labels: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[str, float] = {}  # 标签签名 → 值
        self._lock = threading.Lock()

    def _label_key(self, **labels: Any) -> str:
        """生成标签签名键。"""
        if not self.label_names:
            return ""
        # 按声明顺序构造键
        parts = []
        for name in self.label_names:
            parts.append(f"{name}={labels.get(name, '')}")
        return "|".join(parts)

    def inc(self, value: float = 1, **labels: Any) -> None:
        """递增计数器。value 必须非负。"""
        if value < 0:
            raise ValueError(f"Counter increment must be non-negative, got {value}")
        key = self._label_key(**labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def get(self, **labels: Any) -> float:
        """获取当前值。"""
        key = self._label_key(**labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> str:
        """返回 Prometheus 文本格式。"""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            if not self._values:
                # 无数据时输出空标签样本，保证 Prometheus 抓取不报错
                if self.label_names:
                    label_str = _format_labels(dict.fromkeys(self.label_names, ""))
                else:
                    label_str = ""
                lines.append(f"{self.name}{label_str} 0")
            else:
                for key, value in sorted(self._values.items()):
                    if key:
                        # 还原标签字典
                        labels_dict = {}
                        for part in key.split("|"):
                            if "=" in part:
                                k, v = part.split("=", 1)
                                labels_dict[k] = v
                        label_str = _format_labels(labels_dict)
                    else:
                        label_str = ""
                    lines.append(f"{self.name}{label_str} {value}")
        return "\n".join(lines) + "\n"


class Histogram:
    """直方图，统计值分布。

    适用于：延迟分布、响应大小分布等。
    输出 _bucket（各桶累计计数）、_sum（值总和）、_count（观测总数）。
    """

    def __init__(
        self,
        name: str,
        description: str,
        buckets: list[float] | None = None,
        labels: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = sorted(buckets if buckets is not None else _DEFAULT_BUCKETS)
        # 标签签名 → 桶计数列表
        self._bucket_counts: dict[str, list[int]] = {}
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def _label_key(self, **labels: Any) -> str:
        """生成标签签名键。"""
        if not self.label_names:
            return ""
        parts = []
        for name in self.label_names:
            parts.append(f"{name}={labels.get(name, '')}")
        return "|".join(parts)

    def observe(self, value: float, **labels: Any) -> None:
        """记录一次观测值。"""
        key = self._label_key(**labels)
        with self._lock:
            if key not in self._bucket_counts:
                self._bucket_counts[key] = [0] * len(self.buckets)
                self._sums[key] = 0.0
                self._counts[key] = 0
            # 更新桶计数（累计：值 ≤ 桶上界则计入）
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    self._bucket_counts[key][i] += 1
            self._sums[key] += value
            self._counts[key] += 1

    def collect(self) -> str:
        """返回 Prometheus 文本格式（含 _bucket/_sum/_count）。"""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            if not self._bucket_counts:
                # 无数据时输出空标签样本
                if self.label_names:
                    base_labels = dict.fromkeys(self.label_names, "")
                else:
                    base_labels = {}
                for bound in self.buckets:
                    bucket_labels = dict(base_labels)
                    bucket_labels["le"] = _format_le(bound)
                    lines.append(f"{self.name}_bucket{_format_labels(bucket_labels)} 0")
                # +Inf 桶
                inf_labels = dict(base_labels)
                inf_labels["le"] = "+Inf"
                lines.append(f"{self.name}_bucket{_format_labels(inf_labels)} 0")
                lines.append(f"{self.name}_sum{_format_labels(base_labels)} 0")
                lines.append(f"{self.name}_count{_format_labels(base_labels)} 0")
            else:
                for key in sorted(self._bucket_counts.keys()):
                    # 还原标签字典
                    if key:
                        labels_dict = {}
                        for part in key.split("|"):
                            if "=" in part:
                                k, v = part.split("=", 1)
                                labels_dict[k] = v
                    else:
                        labels_dict = {}
                    counts = self._bucket_counts[key]
                    for i, bound in enumerate(self.buckets):
                        bucket_labels = dict(labels_dict)
                        bucket_labels["le"] = _format_le(bound)
                        lines.append(f"{self.name}_bucket{_format_labels(bucket_labels)} {counts[i]}")
                    # +Inf 桶（等于总 count）
                    inf_labels = dict(labels_dict)
                    inf_labels["le"] = "+Inf"
                    lines.append(f"{self.name}_bucket{_format_labels(inf_labels)} {self._counts[key]}")
                    lines.append(f"{self.name}_sum{_format_labels(labels_dict)} {self._sums[key]}")
                    lines.append(f"{self.name}_count{_format_labels(labels_dict)} {self._counts[key]}")
        return "\n".join(lines) + "\n"


def _format_le(bound: float) -> str:
    """格式化 bucket le 标签值。"""
    if bound == int(bound):
        return str(int(bound))
    return str(bound)


class Gauge:
    """可增可减的瞬时值。

    适用于：当前队列长度、活跃连接数、温度等可上下波动的指标。
    """

    def __init__(self, name: str, description: str, labels: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[str, float] = {}
        self._lock = threading.Lock()

    def _label_key(self, **labels: Any) -> str:
        """生成标签签名键。"""
        if not self.label_names:
            return ""
        parts = []
        for name in self.label_names:
            parts.append(f"{name}={labels.get(name, '')}")
        return "|".join(parts)

    def set(self, value: float, **labels: Any) -> None:
        """设置当前值。"""
        key = self._label_key(**labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, value: float = 1, **labels: Any) -> None:
        """递增。"""
        key = self._label_key(**labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def dec(self, value: float = 1, **labels: Any) -> None:
        """递减。"""
        key = self._label_key(**labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) - value

    def get(self, **labels: Any) -> float:
        """获取当前值。"""
        key = self._label_key(**labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> str:
        """返回 Prometheus 文本格式。"""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            if not self._values:
                if self.label_names:
                    label_str = _format_labels(dict.fromkeys(self.label_names, ""))
                else:
                    label_str = ""
                lines.append(f"{self.name}{label_str} 0")
            else:
                for key, value in sorted(self._values.items()):
                    if key:
                        labels_dict = {}
                        for part in key.split("|"):
                            if "=" in part:
                                k, v = part.split("=", 1)
                                labels_dict[k] = v
                        label_str = _format_labels(labels_dict)
                    else:
                        label_str = ""
                    lines.append(f"{self.name}{label_str} {value}")
        return "\n".join(lines) + "\n"


class MetricsCollector:
    """指标收集器单例 — 管理所有注册的指标。

    提供 register/get_metric/collect_all 接口，
    collect_all() 输出完整 Prometheus 文本，可直接作为 /metrics 端点响应。
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Histogram | Gauge] = {}
        self._lock = threading.Lock()

    def register(self, metric: Counter | Histogram | Gauge) -> None:
        """注册指标。同名指标重复注册将被忽略（保持幂等）。"""
        with self._lock:
            if metric.name not in self._metrics:
                self._metrics[metric.name] = metric

    def get_metric(self, name: str) -> Counter | Histogram | Gauge | None:
        """按名称获取指标。"""
        with self._lock:
            return self._metrics.get(name)

    def collect_all(self) -> str:
        """收集所有指标，返回 Prometheus 格式文本。"""
        with self._lock:
            metrics = list(self._metrics.values())
        if not metrics:
            return ""
        parts = [m.collect() for m in metrics]
        return "\n".join(parts)

    @staticmethod
    def format_labels(labels: dict[str, Any] | None) -> str:
        """格式化标签为 Prometheus 格式（静态方法，便于外部调用）。"""
        return _format_labels(labels)


class AlertManager:
    """轻量级告警管理器。

    支持：
      1. 注册多个告警处理函数（handler）
      2. fire() 触发告警，广播到所有 handler
      3. handler 抛异常时静默吞掉，不影响主流程
      4. 保留告警历史，可查询活跃告警
    """

    def __init__(self) -> None:
        self._alerts: list[dict] = []
        self._handlers: list[Callable[[dict], None]] = []
        self._lock = threading.Lock()

    def register_handler(self, handler: Callable[[dict], None]) -> None:
        """注册告警处理函数。"""
        with self._lock:
            self._handlers.append(handler)

    def fire(self, name: str, severity: str, message: str, **context: Any) -> None:
        """触发告警。

        Args:
            name: 告警名称（如 "saga_dead_letter_accumulation"）
            severity: 严重级别 info/warning/critical
            message: 告警消息
            **context: 附加上下文信息
        """
        alert = {
            "name": name,
            "severity": severity,
            "message": message,
            "context": context,
            "timestamp": time.time(),
        }
        with self._lock:
            self._alerts.append(alert)
            handlers = list(self._handlers)
        # 在锁外调用 handler，避免 handler 阻塞其他线程
        for handler in handlers:
            try:
                handler(alert)
            except Exception:
                pass  # 告警处理失败不应影响主流程

    def get_active_alerts(self) -> list[dict]:
        """获取活跃告警列表。"""
        with self._lock:
            return list(self._alerts)

    def clear(self) -> None:
        """清除告警历史。"""
        with self._lock:
            self._alerts.clear()


# ─── 模块级单例 ─────────────────────────────────────────────
_collector = MetricsCollector()
_alert_manager = AlertManager()


# ─── 预定义核心指标 ─────────────────────────────────────────
# 检索/写入/反思延迟分布
recall_duration_seconds = Histogram(
    "omnimem_recall_duration_seconds",
    "检索操作延迟（秒）",
)
memorize_duration_seconds = Histogram(
    "omnimem_memorize_duration_seconds",
    "写入操作延迟（秒）",
)
reflect_duration_seconds = Histogram(
    "omnimem_reflect_duration_seconds",
    "反思操作延迟（秒）",
)

# 缓存命中统计
cache_hits_total = Counter(
    "omnimem_cache_hits_total",
    "缓存命中总次数",
)
cache_misses_total = Counter(
    "omnimem_cache_misses_total",
    "缓存未命中总次数",
)
cache_hit_ratio = Gauge(
    "omnimem_cache_hit_ratio",
    "缓存命中率（0.0~1.0）",
)

# 熔断器状态（0=CLOSED, 1=HALF_OPEN, 2=OPEN）
circuit_breaker_state = Gauge(
    "omnimem_circuit_breaker_state",
    "熔断器状态（0=CLOSED, 1=HALF_OPEN, 2=OPEN）",
)

# Saga 事务统计
saga_pending_count = Gauge(
    "omnimem_saga_pending_count",
    "Saga 待处理事务数",
)
saga_dead_letters_total = Counter(
    "omnimem_saga_dead_letters_total",
    "Saga dead_letter 总数",
)

# LLM 调用统计（按 type 标签区分：reflect/distill/summary/decision）
llm_calls_total = Counter(
    "omnimem_llm_calls_total",
    "LLM 调用总次数",
    labels=["type"],
)
llm_errors_total = Counter(
    "omnimem_llm_errors_total",
    "LLM 调用错误总次数",
    labels=["type"],
)

# 活跃连接数
active_connections = Gauge(
    "omnimem_active_connections",
    "当前活跃连接数",
)

# 注册所有预定义指标到收集器
_collector.register(recall_duration_seconds)
_collector.register(memorize_duration_seconds)
_collector.register(reflect_duration_seconds)
_collector.register(cache_hits_total)
_collector.register(cache_misses_total)
_collector.register(cache_hit_ratio)
_collector.register(circuit_breaker_state)
_collector.register(saga_pending_count)
_collector.register(saga_dead_letters_total)
_collector.register(llm_calls_total)
_collector.register(llm_errors_total)
_collector.register(active_connections)


# ─── 便捷函数 ───────────────────────────────────────────────
def get_metrics_collector() -> MetricsCollector:
    """返回全局指标收集器单例。"""
    return _collector


def get_alert_manager() -> AlertManager:
    """返回全局告警管理器单例。"""
    return _alert_manager


def record_recall_duration(seconds: float) -> None:
    """记录检索操作延迟。"""
    try:
        recall_duration_seconds.observe(seconds)
    except Exception:
        pass  # 指标记录失败不影响主流程


def record_memorize_duration(seconds: float) -> None:
    """记录写入操作延迟。"""
    try:
        memorize_duration_seconds.observe(seconds)
    except Exception:
        pass


def record_reflect_duration(seconds: float) -> None:
    """记录反思操作延迟。"""
    try:
        reflect_duration_seconds.observe(seconds)
    except Exception:
        pass


def record_cache_hit() -> None:
    """记录一次缓存命中。"""
    try:
        cache_hits_total.inc()
        update_cache_hit_ratio()
    except Exception:
        pass


def record_cache_miss() -> None:
    """记录一次缓存未命中。"""
    try:
        cache_misses_total.inc()
        update_cache_hit_ratio()
    except Exception:
        pass


def update_cache_hit_ratio() -> None:
    """根据当前命中/未命中计数更新缓存命中率。"""
    try:
        hits = cache_hits_total.get()
        misses = cache_misses_total.get()
        total = hits + misses
        ratio = hits / total if total > 0 else 0.0
        cache_hit_ratio.set(ratio)
    except Exception:
        pass


def set_circuit_breaker_state(state: str) -> None:
    """设置熔断器状态指标。

    Args:
        state: "closed"/"half_open"/"open" 或对应数字
    """
    try:
        state_map = {
            "closed": 0,
            "half_open": 1,
            "open": 2,
            "CLOSED": 0,
            "HALF_OPEN": 1,
            "OPEN": 2,
        }
        value = state_map.get(state, state) if isinstance(state, str) else state
        circuit_breaker_state.set(float(value))
    except Exception:
        pass


def set_saga_pending_count(count: int) -> None:
    """设置 Saga 待处理事务数。"""
    try:
        saga_pending_count.set(count)
    except Exception:
        pass


def record_saga_dead_letter() -> None:
    """记录一次 Saga dead_letter 事件。"""
    try:
        saga_dead_letters_total.inc()
    except Exception:
        pass


def record_llm_call(call_type: str) -> None:
    """记录一次 LLM 调用。

    Args:
        call_type: 调用类型 reflect/distill/summary/decision
    """
    try:
        llm_calls_total.inc(type=call_type)
    except Exception:
        pass


def record_llm_error(call_type: str) -> None:
    """记录一次 LLM 调用错误。"""
    try:
        llm_errors_total.inc(type=call_type)
    except Exception:
        pass


def inc_active_connections() -> None:
    """活跃连接数 +1。"""
    try:
        active_connections.inc()
    except Exception:
        pass


def dec_active_connections() -> None:
    """活跃连接数 -1。"""
    try:
        active_connections.dec()
    except Exception:
        pass
