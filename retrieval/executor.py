"""共享检索线程池 — 全进程单池 + 引用计数管理。

原实现每个 HybridOrchestrator 实例各建一个 cpu+4 线程池，多 Provider/SDK 实例并存时
线程数线性膨胀。改为全进程共享单池，实例持引用，最后一个 shutdown 时真正关闭。
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_shared_executor: ThreadPoolExecutor | None = None
_shared_executor_refs: int = 0
_shared_executor_lock = threading.Lock()


def acquire_shared_executor(max_workers: int) -> ThreadPoolExecutor:
    """获取共享线程池（首次调用创建，规格由首个调用方决定）。"""
    global _shared_executor, _shared_executor_refs
    with _shared_executor_lock:
        if _shared_executor is None:
            _shared_executor = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="omnimem_retrieval"
            )
            logger.info("Shared retrieval executor created (max_workers=%d)", max_workers)
        _shared_executor_refs += 1
        return _shared_executor


def release_shared_executor(wait: bool = True, timeout: float = 10.0) -> None:
    """释放共享线程池引用，归零时真正关闭。

    ★ P1-6：原先 ``shutdown(wait=True)`` 是**在持有 ``_shared_executor_lock`` 的情况下**
    做的，而 join 会等所有 worker 跑完当前任务 —— 只要有一个 worker 卡在模型加载或向量
    查询里，关闭就永久阻塞，且期间任何新的 ``acquire_shared_executor`` 也一起挂住。
    现在：引用计数在锁内改完就放锁，关闭在锁外做，并且等待本身有界。
    """
    global _shared_executor, _shared_executor_refs
    with _shared_executor_lock:
        if _shared_executor is None:
            return
        _shared_executor_refs -= 1
        if _shared_executor_refs > 0:
            return
        executor = _shared_executor
        _shared_executor = None
        _shared_executor_refs = 0

    if not wait:
        executor.shutdown(wait=False, cancel_futures=True)
        return

    box: dict[str, BaseException] = {}

    def _drain() -> None:
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException as exc:  # noqa: BLE001 - 只用于回传日志
            box["error"] = exc

    thread = threading.Thread(target=_drain, name="omnimem-exec-shutdown", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        logger.warning(
            "共享检索线程池在 %.1fs 内未能优雅关闭，放弃等待（残留 worker 会跑完当前任务后自行退出）",
            timeout,
        )
    elif "error" in box:
        logger.warning("共享检索线程池关闭异常: %s", box["error"])
