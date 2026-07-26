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


def release_shared_executor(wait: bool = True) -> None:
    """释放共享线程池引用，归零时真正关闭。"""
    global _shared_executor, _shared_executor_refs
    with _shared_executor_lock:
        if _shared_executor is None:
            return
        _shared_executor_refs -= 1
        if _shared_executor_refs <= 0:
            _shared_executor.shutdown(wait=wait)
            _shared_executor = None
            _shared_executor_refs = 0
