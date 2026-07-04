"""熔断器三态转换单元测试。"""

from __future__ import annotations

import time

from omnimem.retrieval.circuit_breaker import CircuitBreaker


def test_initial_state_is_closed() -> None:
    """熔断器初始状态应为 CLOSED。"""
    breaker = CircuitBreaker(threshold=3, cooldown=60.0)
    assert breaker.state == CircuitBreaker.CLOSED
    assert breaker.threshold == 3
    assert breaker.cooldown == 60.0


def test_closed_to_open_after_threshold_failures() -> None:
    """连续失败达到阈值后应切换到 OPEN。"""
    breaker = CircuitBreaker(threshold=3, cooldown=60.0)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitBreaker.CLOSED
    breaker.record_failure()
    assert breaker.state == CircuitBreaker.OPEN


def test_open_uses_fallback_and_does_not_call_fn() -> None:
    """OPEN 状态下应直接返回 fallback，不执行原函数。"""
    breaker = CircuitBreaker(threshold=1, cooldown=60.0)
    breaker.record_failure()
    assert breaker.state == CircuitBreaker.OPEN

    called = False

    def _fn() -> str:
        nonlocal called
        called = True
        return "ok"

    result = breaker.call(_fn, fallback=lambda: "fallback")
    assert result == "fallback"
    assert not called


def test_open_to_half_open_after_cooldown_then_recover() -> None:
    """OPEN 状态冷却期满后尝试恢复，成功后应回到 CLOSED。"""
    breaker = CircuitBreaker(threshold=1, cooldown=0.1)
    breaker.record_failure()
    assert breaker.state == CircuitBreaker.OPEN
    time.sleep(0.15)
    result = breaker.call(lambda: "ok", fallback=lambda: "fallback")
    assert result == "ok"
    assert breaker.state == CircuitBreaker.CLOSED


def test_half_open_to_closed_on_success() -> None:
    """HALF_OPEN 成功后应恢复到 CLOSED。"""
    breaker = CircuitBreaker(threshold=3, cooldown=60.0)
    breaker._state = CircuitBreaker.HALF_OPEN
    breaker._failures = 1
    breaker.record_success()
    assert breaker.state == CircuitBreaker.CLOSED
    assert breaker._failures == 0


def test_half_open_to_open_on_failure() -> None:
    """HALF_OPEN 失败后应回到 OPEN。"""
    breaker = CircuitBreaker(threshold=1, cooldown=60.0)
    breaker._state = CircuitBreaker.HALF_OPEN
    breaker.record_failure()
    assert breaker.state == CircuitBreaker.OPEN


def test_reset_manual() -> None:
    """手动 reset 后熔断器应回到 CLOSED 并清零计数器。"""
    breaker = CircuitBreaker(threshold=1, cooldown=60.0)
    breaker.record_failure()
    assert breaker.state == CircuitBreaker.OPEN
    breaker.reset()
    assert breaker.state == CircuitBreaker.CLOSED


def test_success_clears_failure_count_in_closed() -> None:
    """CLOSED 状态下成功应清零失败计数器。"""
    breaker = CircuitBreaker(threshold=3, cooldown=60.0)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state == CircuitBreaker.CLOSED


def test_should_skip_reflects_open_state() -> None:
    """should_skip 在 OPEN 且未冷却时返回 True，冷却后返回 False。"""
    breaker = CircuitBreaker(threshold=1, cooldown=0.2)
    breaker.record_failure()
    assert breaker.should_skip() is True
    time.sleep(0.25)
    assert breaker.should_skip() is False
    assert breaker.state == CircuitBreaker.HALF_OPEN


def test_call_success_in_closed_keeps_closed() -> None:
    """CLOSED 状态下成功调用应保持 CLOSED。"""
    breaker = CircuitBreaker(threshold=3, cooldown=60.0)
    result = breaker.call(lambda: "ok", fallback=lambda: "fallback")
    assert result == "ok"
    assert breaker.state == CircuitBreaker.CLOSED


def test_call_failure_in_closed_increments_count() -> None:
    """CLOSED 状态下失败调用应累计失败次数并触发 OPEN。"""
    breaker = CircuitBreaker(threshold=2, cooldown=60.0)

    def _fail() -> str:
        raise RuntimeError("boom")

    breaker.call(_fail, fallback=lambda: "fallback")
    assert breaker.state == CircuitBreaker.CLOSED
    breaker.call(_fail, fallback=lambda: "fallback")
    assert breaker.state == CircuitBreaker.OPEN
