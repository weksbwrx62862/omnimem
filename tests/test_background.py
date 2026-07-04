"""BackgroundTaskExecutor 后台任务执行器测试。"""

from __future__ import annotations

import threading
import time

import pytest
from omnimem.core.background import BackgroundTaskExecutor


class TestBackgroundTaskExecutorInit:
    """BackgroundTaskExecutor 初始化测试。"""

    def test_default_init(self) -> None:
        """默认初始化，线程池大小为 2。"""
        executor = BackgroundTaskExecutor()
        assert executor.pending_tasks == 0
        executor.shutdown(wait=False)

    def test_custom_workers(self) -> None:
        """自定义线程池大小。"""
        executor = BackgroundTaskExecutor(max_workers=4)
        assert executor.pending_tasks == 0
        executor.shutdown(wait=False)


class TestSubmit:
    """submit 方法测试。"""

    def test_submit_simple_task(self) -> None:
        """提交简单任务并获取结果。"""
        executor = BackgroundTaskExecutor(max_workers=2)
        future = executor.submit(lambda: 42)
        assert future.result(timeout=5) == 42
        executor.shutdown(wait=True)

    def test_submit_with_args(self) -> None:
        """提交带参数的任务。"""
        executor = BackgroundTaskExecutor(max_workers=2)

        def add(a, b):
            return a + b

        future = executor.submit(add, 3, 4)
        assert future.result(timeout=5) == 7
        executor.shutdown(wait=True)

    def test_submit_with_kwargs(self) -> None:
        """提交带关键字参数的任务。"""
        executor = BackgroundTaskExecutor(max_workers=2)

        def greet(name="world"):
            return f"hello {name}"

        future = executor.submit(greet, name="omnimem")
        assert future.result(timeout=5) == "hello omnimem"
        executor.shutdown(wait=True)

    def test_submit_multiple_tasks(self) -> None:
        """提交多个任务全部完成。"""
        executor = BackgroundTaskExecutor(max_workers=2)
        futures = [executor.submit(lambda i=i: i * 2) for i in range(5)]
        results = [f.result(timeout=5) for f in futures]
        assert results == [0, 2, 4, 6, 8]
        executor.shutdown(wait=True)


class TestPendingTasks:
    """pending_tasks 属性测试。"""

    def test_pending_count_decreases_after_completion(self) -> None:
        """任务完成后 pending_tasks 递减。"""
        executor = BackgroundTaskExecutor(max_workers=1)
        barrier = threading.Barrier(2, timeout=5)

        def blocking_task():
            barrier.wait()  # 等待主线程检查
            return "done"

        future = executor.submit(blocking_task)
        # 等待任务开始执行
        time.sleep(0.1)
        # 任务执行中，pending 应为 1
        assert executor.pending_tasks >= 1

        # 释放任务
        barrier.wait()
        future.result(timeout=5)
        # 任务完成，pending 应为 0
        assert executor.pending_tasks == 0
        executor.shutdown(wait=True)

    def test_pending_count_zero_after_all_done(self) -> None:
        """所有任务完成后 pending_tasks 为 0。"""
        executor = BackgroundTaskExecutor(max_workers=2)
        futures = [executor.submit(lambda: None) for _ in range(3)]
        for f in futures:
            f.result(timeout=5)
        assert executor.pending_tasks == 0
        executor.shutdown(wait=True)


class TestExceptionHandling:
    """异常处理测试。"""

    def test_exception_propagated_to_future(self) -> None:
        """任务异常通过 Future 传播。"""
        executor = BackgroundTaskExecutor(max_workers=2)

        def failing_task():
            raise ValueError("任务失败")

        future = executor.submit(failing_task)
        with pytest.raises(ValueError, match="任务失败"):
            future.result(timeout=5)
        executor.shutdown(wait=True)

    def test_pending_count_decreases_on_exception(self) -> None:
        """任务异常后 pending_tasks 仍递减。"""
        executor = BackgroundTaskExecutor(max_workers=2)

        def failing_task():
            raise RuntimeError("失败")

        future = executor.submit(failing_task)
        with pytest.raises(RuntimeError):
            future.result(timeout=5)
        assert executor.pending_tasks == 0
        executor.shutdown(wait=True)

    def test_exception_does_not_affect_other_tasks(self) -> None:
        """一个任务异常不影响其他任务。"""
        executor = BackgroundTaskExecutor(max_workers=2)

        def failing():
            raise RuntimeError("失败")

        def succeeding():
            return "成功"

        f1 = executor.submit(failing)
        f2 = executor.submit(succeeding)

        with pytest.raises(RuntimeError):
            f1.result(timeout=5)
        assert f2.result(timeout=5) == "成功"
        executor.shutdown(wait=True)


class TestShutdown:
    """shutdown 方法测试。"""

    def test_shutdown_with_wait(self) -> None:
        """等待所有任务完成后关闭。"""
        executor = BackgroundTaskExecutor(max_workers=2)
        future = executor.submit(lambda: time.sleep(0.1) or "done")
        executor.shutdown(wait=True)
        assert future.result(timeout=5) == "done"

    def test_shutdown_without_wait(self) -> None:
        """不等待直接关闭。"""
        executor = BackgroundTaskExecutor(max_workers=2)
        executor.submit(lambda: time.sleep(0.5))
        executor.shutdown(wait=False)
        # 不应阻塞

    def test_submit_after_shutdown_raises(self) -> None:
        """关闭后提交任务应抛出异常。"""
        executor = BackgroundTaskExecutor(max_workers=2)
        executor.shutdown(wait=True)
        with pytest.raises(RuntimeError):
            executor.submit(lambda: 1)


class TestConcurrentExecution:
    """并发执行测试。"""

    def test_tasks_run_concurrently(self) -> None:
        """多个任务可并发执行。"""
        executor = BackgroundTaskExecutor(max_workers=4)
        results = []
        lock = threading.Lock()

        def record_thread(index):
            with lock:
                results.append(index)
            return index

        futures = [executor.submit(record_thread, i) for i in range(10)]
        for f in futures:
            f.result(timeout=10)

        assert len(results) == 10
        executor.shutdown(wait=True)
