"""MemoryMonitor 单测（改进项 #2 覆盖，离线）。

覆盖：get_usage 结构与 psutil 缺失降级、on_warning 注册、_check 超阈值触发回调
并吞掉回调异常、未超阈值不触发、start/stop 运行标志与定时器复位、未启动即 stop 安全。
不依赖真实内存占用——直接以猴子补丁注入 get_usage 结果驱动 _check。
"""
from __future__ import annotations

from omnimem.core.memory_monitor import MemoryMonitor


def test_get_usage_shape():
    m = MemoryMonitor()
    usage = m.get_usage()
    assert set(usage) == {"rss_mb", "objects"}
    assert isinstance(usage["objects"], int)
    assert usage["objects"] >= 0
    assert isinstance(usage["rss_mb"], (int, float))


def test_on_warning_registers_callbacks():
    m = MemoryMonitor()
    m.on_warning(lambda u: None)
    m.on_warning(lambda u: None)
    assert len(m._callbacks) == 2


def test_check_triggers_callback_over_threshold():
    m = MemoryMonitor(warning_mb=1.0)
    seen: list[dict] = []
    m.on_warning(lambda u: seen.append(u))
    m.get_usage = lambda: {"rss_mb": 9999.0, "objects": 5}  # type: ignore[assignment]
    m._check()
    assert seen and seen[0]["rss_mb"] == 9999.0


def test_check_not_triggered_under_threshold():
    m = MemoryMonitor(warning_mb=1_000_000.0)
    seen: list[dict] = []
    m.on_warning(lambda u: seen.append(u))
    m.get_usage = lambda: {"rss_mb": 1.0, "objects": 1}  # type: ignore[assignment]
    m._check()
    assert seen == []


def test_check_swallows_callback_exception():
    m = MemoryMonitor(warning_mb=1.0)

    def boom(_usage):
        raise ValueError("cb down")

    m.on_warning(boom)
    m.get_usage = lambda: {"rss_mb": 9999.0, "objects": 1}  # type: ignore[assignment]
    m._check()  # 不应向外抛异常


def test_start_stop_toggles_running():
    m = MemoryMonitor(interval=3600.0, warning_mb=1e9)  # 巨阈值：启动即检也不触发回调
    m.start()
    assert m._running is True
    assert m._timer is not None
    m.stop()
    assert m._running is False


def test_stop_without_start_is_safe():
    m = MemoryMonitor()
    m.stop()  # _timer 为 None，不应抛错
    assert m._running is False
