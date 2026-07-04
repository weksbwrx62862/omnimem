"""三态熔断器实现。

向量检索连续故障时自动降级到纯 BM25。
"""

from __future__ import annotations

import logging
import time

from omnimem.utils.metrics import get_alert_manager, set_circuit_breaker_state

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """三态熔断器 — 向量检索连续故障时自动降级到纯 BM25。

    状态机：
      CLOSED → (阈值次故障) → OPEN → (冷却后) → HALF_OPEN → (成功) → CLOSED
                                                    HALF_OPEN → (失败) → OPEN

    使用：
        breaker = CircuitBreaker(threshold=3, cooldown=60)
        result = breaker.call(lambda: risky_op(), fallback=lambda: safe_op())
    """

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, threshold: int = 3, cooldown: float = 60.0, on_recover=None):
        self._state = self.CLOSED
        self._failures = 0
        self._threshold = threshold
        self._cooldown = cooldown
        self._last_failure_time = 0.0
        self._on_recover = on_recover
        # 初始化熔断器状态指标
        set_circuit_breaker_state(self._state)

    @property
    def state(self) -> str:
        return self._state

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def cooldown(self) -> float:
        return self._cooldown

    def _transition_to(self, new_state: str, reason: str = "") -> None:
        """状态转换辅助方法 — 触发告警并更新指标。"""
        old_state = self._state
        if old_state == new_state:
            return
        self._state = new_state
        # 更新熔断器状态指标
        set_circuit_breaker_state(new_state)
        # 仅在关键转换时触发告警
        if old_state == self.CLOSED and new_state == self.OPEN:
            get_alert_manager().fire(
                name="circuit_breaker_open",
                severity="critical",
                message=f"检索熔断器 CLOSED→OPEN（连续失败 {self._failures} 次）",
                old_state=old_state,
                new_state=new_state,
                failures=self._failures,
                threshold=self._threshold,
                reason=reason,
            )
        elif old_state == self.OPEN and new_state == self.HALF_OPEN:
            get_alert_manager().fire(
                name="circuit_breaker_half_open",
                severity="info",
                message="检索熔断器 OPEN→HALF_OPEN（冷却期已过，尝试恢复）",
                old_state=old_state,
                new_state=new_state,
                reason=reason,
            )
        elif old_state == self.HALF_OPEN and new_state == self.CLOSED:
            get_alert_manager().fire(
                name="circuit_breaker_recovered",
                severity="info",
                message="检索熔断器 HALF_OPEN→CLOSED（已恢复正常）",
                old_state=old_state,
                new_state=new_state,
                reason=reason,
            )

    def call(self, fn, fallback):
        """执行 fn()，故障时返回 fallback()。"""
        now = time.time()
        if self._state == self.OPEN:
            if now - self._last_failure_time > self._cooldown:
                self._transition_to(self.HALF_OPEN, reason="cooldown elapsed")
                logger.info("CircuitBreaker: OPEN→HALF_OPEN (cooldown elapsed)")
            else:
                logger.warning(
                    "CircuitBreaker: OPEN, circuit open (%.1fs remaining)",
                    self._cooldown - (now - self._last_failure_time),
                )
                return fallback()
        try:
            result = fn()
            # 成功 — 恢复
            if self._state == self.HALF_OPEN:
                self._transition_to(self.CLOSED, reason="recovered after half-open success")
                self._failures = 0
                logger.info("CircuitBreaker: HALF_OPEN→CLOSED (recovered)")
                if self._on_recover:
                    self._on_recover()
            elif self._failures > 0:
                self._failures = 0
            return result
        except Exception:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self._threshold:
                self._transition_to(self.OPEN, reason=f"{self._failures} consecutive failures")
                logger.error("CircuitBreaker: CLOSED→OPEN (%d consecutive failures)", self._failures)
            return fallback()

    def reset(self) -> None:
        """手动重置熔断器。"""
        self._transition_to(self.CLOSED, reason="manual reset")
        self._failures = 0

    def record_failure(self) -> None:
        """记录一次故障，达到阈值时自动 OPEN。"""
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self._threshold:
            self._transition_to(self.OPEN, reason=f"{self._failures} consecutive failures")
            logger.error("CircuitBreaker: CLOSED→OPEN (%d consecutive failures)", self._failures)

    def record_success(self) -> None:
        """记录一次成功，HALF_OPEN→CLOSED 或清零计数器。"""
        if self._state == self.HALF_OPEN:
            self._transition_to(self.CLOSED, reason="recovered after half-open success")
            self._failures = 0
            logger.info("CircuitBreaker: HALF_OPEN→CLOSED (recovered)")
            if self._on_recover:
                self._on_recover()
        elif self._failures > 0:
            self._failures = 0

    def should_skip(self) -> bool:
        """OPEN 且未冷却时返回 True，调用方应跳过向量检索。"""
        if self._state != self.OPEN:
            return False
        if time.time() - self._last_failure_time > self._cooldown:
            self._transition_to(self.HALF_OPEN, reason="cooldown elapsed")
            logger.info("CircuitBreaker: OPEN→HALF_OPEN (cooldown elapsed)")
            return False
        return True
