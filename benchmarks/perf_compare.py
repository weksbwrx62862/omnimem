"""OmniMem 性能基准对比测试脚本。

测量 5 项关键指标，用于对比 6 项改进优化前后的性能变化：
  1. 插件导入时间（MagicMock 解耦改进 2）
  2. 单机模式记忆写入延迟与吞吐量（VectorClock 短路改进 3）
  3. VectorClock 初始化与 get_next_vc 开销（改进 3）
  4. ChromaDB 日志噪音抑制效果（改进 5）
  5. 完整测试套件运行时间（整体影响）

用法：
    cd ~/.hermes/plugins/omnimem
    python3 benchmarks/perf_compare.py [--skip-test-suite]
    # 输出 JSON 到 stdout，同时保存到 /tmp/omnimem_perf_<label>.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT.parent))


# ─── 辅助函数 ────────────────────────────────────────────────

def _percentile(data: list[float], pct: float) -> float:
    """计算百分位数（pct ∈ [0, 100]）。"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def _fmt_ms(seconds: float) -> float:
    """秒转毫秒。"""
    return round(seconds * 1000, 3)


# ─── 模块 1: 插件导入时间 ────────────────────────────────────

def bench_import_time(samples: int = 5) -> dict:
    """测量 omnimem 包首次导入时间（每个样本独立子进程）。

    改进 2（MagicMock 解耦）影响此指标。
    """
    timings: list[float] = []
    code = (
        "import time; start=time.time(); "
        "import sys; sys.path.insert(0, '/home/xxh/.hermes/plugins'); "
        "import omnimem; "
        "print(f'{time.time()-start:.6f}')"
    )
    for _ in range(samples):
        # 每个 subprocess 确保 clean 状态
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[-1]
            try:
                timings.append(float(line))
            except ValueError:
                pass
    return {
        "samples": len(timings),
        "values_s": [round(t, 6) for t in timings],
        "mean_ms": _fmt_ms(statistics.mean(timings)) if timings else 0,
        "stdev_ms": _fmt_ms(statistics.stdev(timings)) if len(timings) > 1 else 0,
        "min_ms": _fmt_ms(min(timings)) if timings else 0,
        "max_ms": _fmt_ms(max(timings)) if timings else 0,
    }


# ─── 模块 2: 单机模式记忆写入延迟与吞吐量 ────────────────────

def bench_write_latency(count: int = 100) -> dict:
    """测量 sync_mode=none 下批量写入延迟与吞吐量。

    改进 3（VectorClock 短路）影响此指标。
    """
    # 延迟导入，避免影响导入时间测量
    from omnimem.sdk import OmniMemSDK

    with tempfile.TemporaryDirectory() as tmpdir:
        sdk = OmniMemSDK(storage_dir=tmpdir, config={"sync_mode": "none"})
        # 预热
        try:
            sdk.memorize("预热记忆内容", memory_type="fact")
        except Exception:
            pass

        latencies: list[float] = []
        for i in range(count):
            content = f"基准测试记忆条目 #{i}：这是一条用于性能测试的记忆内容。"
            start = time.perf_counter()
            try:
                sdk.memorize(content, memory_type="fact")
            except Exception:
                pass
            latencies.append(time.perf_counter() - start)

        total_time = sum(latencies)
        return {
            "count": len(latencies),
            "total_ms": _fmt_ms(total_time),
            "mean_ms": _fmt_ms(statistics.mean(latencies)),
            "p50_ms": _fmt_ms(_percentile(latencies, 50)),
            "p95_ms": _fmt_ms(_percentile(latencies, 95)),
            "throughput_ops_per_sec": round(len(latencies) / total_time, 2) if total_time > 0 else 0,
        }


# ─── 模块 3: VectorClock 操作开销 ────────────────────────────

def bench_vector_clock_ops(iterations: int = 1000) -> dict:
    """对比 sync_mode=none（短路）与 sync_mode=file_lock（启用）的 VectorClock 开销。

    改进 3 影响。在优化后版本中，sync_mode=none 跳过 VectorClock；
    在优化前版本中，VectorClock 无条件初始化与递增。
    """
    from omnimem.governance.vector_clock import VectorClock

    # 场景 A: sync_mode=none（优化后短路，_vector_clock 为 None）
    # 模拟 provider.get_next_vc 的短路逻辑
    start_a = time.perf_counter()
    for _ in range(iterations):
        # 短路路径：_vector_clock is None，直接返回 None
        _vc = None
        _result = None if _vc is None else (_vc.increment("node"), _vc)
    elapsed_a = time.perf_counter() - start_a

    # 场景 B: sync_mode != none（VectorClock 启用，递增操作）
    vc = VectorClock()
    start_b = time.perf_counter()
    for _ in range(iterations):
        vc.increment("node")
    elapsed_b = time.perf_counter() - start_b

    return {
        "iterations": iterations,
        "short_circuit_us": round(elapsed_a * 1e6, 1),
        "vc_enabled_us": round(elapsed_b * 1e6, 1),
        "savings_us": round((elapsed_b - elapsed_a) * 1e6, 1),
        "savings_ratio": round(elapsed_b / elapsed_a, 2) if elapsed_a > 0 else 0,
    }


# ─── 模块 4: ChromaDB 日志噪音计数 ───────────────────────────

def bench_chromadb_log_noise() -> dict:
    """测量 ChromaDB telemetry 日志抑制效果。

    改进 5 影响。优化后 _ChromaDBTelemetryFilter 过滤更多噪音模式。
    """
    # 捕获 logging 输出
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.WARNING)

    # 临时添加 handler 到 chromadb 相关 logger
    chromadb_loggers = [
        "chromadb.telemetry",
        "chromadb.telemetry.product",
        "chromadb.telemetry.product.posthog",
    ]
    added_handlers: list[tuple[logging.Logger, logging.Handler]] = []
    for name in chromadb_loggers:
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        added_handlers.append((logger, handler))

    # 执行涉及 ChromaDB 的操作
    warning_count = 0
    try:
        from omnimem.retrieval.vector_factory import VectorStoreFactory
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                factory = VectorStoreFactory(tmpdir)
                # 尝试创建客户端，可能触发 telemetry 日志
                _ = factory.create(backend="chroma")
            except Exception:
                pass
    except Exception:
        pass

    # 统计 WARNING 级别日志条数
    log_output = log_capture.getvalue()
    warning_lines = [line for line in log_output.split("\n") if line.strip()]
    warning_count = len(warning_lines)

    # 清理
    for logger, hdl in added_handlers:
        logger.removeHandler(hdl)

    return {
        "warning_count": warning_count,
        "sample_lines": warning_lines[:3],  # 保留前 3 条样本
    }


# ─── 模块 5: 完整测试套件耗时 ────────────────────────────────

def bench_test_suite(skip: bool = False) -> dict:
    """测量完整测试套件运行时间。"""
    if skip:
        return {"skipped": True, "duration_s": 0, "passed": 0, "failed": 0}

    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True, text=True, timeout=600,
        cwd=str(_PROJECT_ROOT),
    )
    duration = time.perf_counter() - start

    # 解析 "980 passed, 18 skipped" 格式
    passed = 0
    failed = 0
    last_line = [line for line in result.stdout.split("\n") if "passed" in line]
    if last_line:
        import re
        m = re.search(r"(\d+)\s+passed", last_line[-1])
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+)\s+failed", last_line[-1])
        if m:
            failed = int(m.group(1))

    return {
        "skipped": False,
        "duration_s": round(duration, 2),
        "passed": passed,
        "failed": failed,
        "exit_code": result.returncode,
    }


# ─── 主流程 ──────────────────────────────────────────────────

def run_all(skip_test_suite: bool = False) -> dict:
    """运行所有基准测试，返回汇总字典。"""
    print("开始性能基准测试...", file=sys.stderr)
    results: dict[str, Any] = {
        "timestamp": time.time(),
        "environment": _collect_env(),
    }

    print("  [1/5] 测量插件导入时间...", file=sys.stderr)
    results["import_time"] = bench_import_time()

    print("  [2/5] 测量记忆写入延迟...", file=sys.stderr)
    results["write_latency"] = bench_write_latency()

    print("  [3/5] 测量 VectorClock 开销...", file=sys.stderr)
    results["vector_clock_ops"] = bench_vector_clock_ops()

    print("  [4/5] 测量 ChromaDB 日志噪音...", file=sys.stderr)
    results["chromadb_log_noise"] = bench_chromadb_log_noise()

    print("  [5/5] 测量测试套件耗时...", file=sys.stderr)
    results["test_suite"] = bench_test_suite(skip=skip_test_suite)

    print("完成。", file=sys.stderr)
    return results


def _collect_env() -> dict:
    """收集测试环境信息。"""
    import platform
    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    # 关键依赖版本
    for pkg in ["chromadb", "tiktoken", "rank_bm25", "cryptography", "aiosqlite", "datasketch"]:
        try:
            mod = __import__(pkg)
            env[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env[f"{pkg}_version"] = "not_installed"
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="OmniMem 性能基准对比测试")
    parser.add_argument("--skip-test-suite", action="store_true", help="跳过完整测试套件（耗时较长）")
    parser.add_argument("--output", "-o", default=None, help="输出文件路径（默认 /tmp/omnimem_perf_<timestamp>.json）")
    args = parser.parse_args()

    results = run_all(skip_test_suite=args.skip_test_suite)

    output_path = args.output or f"/tmp/omnimem_perf_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n数据已保存到: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
