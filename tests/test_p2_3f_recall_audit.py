"""P2-3f: recall 审计覆盖（tests/test_p2_3f_recall_audit.py）。

缺陷：外部套件 test_security_audit::test_recall_audited 真失败 —— 参数校验失败与
no_results 这两条提前返回的路径完全不写审计，等于「查过什么」无痕可查。

约束：
  - 每次 omni_recall 调用恰好留下一条 operation="recall" 的审计记录；
  - status="found" 由 RecallService 自己写，handler 不重复写；
  - 审计写入异常绝不阻断检索；
  - 查询词进审计前经过脱敏与截断。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from omnimem.handlers import recall as recall_module
from omnimem.handlers.deps import HandlerDependencies


class RecordingAuditLogger:
    """记录 log() 调用；raise_on_log=True 时模拟审计子系统故障。"""

    def __init__(self, *, raise_on_log: bool = False) -> None:
        self.rows: list[dict[str, Any]] = []
        self.raise_on_log = raise_on_log

    def log(
        self,
        operation: str,
        *,
        memory_id: str | None = None,
        details: dict[str, Any] | None = None,
        result: str = "success",
        instance_id: str | None = None,
    ) -> None:
        if self.raise_on_log:
            raise RuntimeError("audit db locked")
        self.rows.append(
            {
                "operation": operation,
                "memory_id": memory_id,
                "details": details,
                "result": result,
                "instance_id": instance_id,
            }
        )


class FakeProvider:
    def __init__(self, audit_logger: RecordingAuditLogger) -> None:
        self._audit_logger = audit_logger
        self._instance_id = "inst-p2-3f"


def _recall(provider: Any, args: dict[str, Any]) -> dict[str, Any]:
    return json.loads(recall_module.handle_recall(provider, args))


@pytest.fixture()
def audit_logger() -> RecordingAuditLogger:
    return RecordingAuditLogger()


@pytest.fixture()
def provider(audit_logger: RecordingAuditLogger) -> FakeProvider:
    return FakeProvider(audit_logger)


class TestValidationErrorIsAudited:
    def test_invalid_mode_writes_one_recall_row(self, provider, audit_logger):
        result = _recall(provider, {"query": "检索审计", "mode": "bm25"})
        assert result["status"] == "error"
        assert len(audit_logger.rows) == 1
        row = audit_logger.rows[0]
        assert row["operation"] == "recall"
        assert row["result"] == "error"
        assert row["instance_id"] == "inst-p2-3f"

    def test_details_carry_query_reason_hits_elapsed(self, provider, audit_logger):
        _recall(provider, {"query": "检索审计", "mode": "not-a-mode"})
        details = audit_logger.rows[0]["details"]
        assert details["query"] == "检索审计"
        assert details["mode"] == "not-a-mode"
        assert details["status"] == "error"
        assert "invalid mode" in details["reason"]
        assert details["hits"] == 0
        assert isinstance(details["elapsed_ms"], (int, float))
        assert details["elapsed_ms"] >= 0

    def test_empty_query_is_audited(self, provider, audit_logger):
        result = _recall(provider, {"query": "   "})
        assert result["status"] == "error"
        assert len(audit_logger.rows) == 1
        assert audit_logger.rows[0]["result"] == "error"


class TestNonFoundOutcomeIsAudited:
    def test_no_results_writes_recall_row(self, provider, audit_logger, monkeypatch):
        monkeypatch.setattr(
            recall_module.RecallService,
            "handle",
            lambda self, args: {"status": "no_results", "query": args["query"], "message": "No relevant memories found."},
        )
        result = _recall(provider, {"query": "不存在的东西"})
        assert result["status"] == "no_results"
        assert len(audit_logger.rows) == 1
        row = audit_logger.rows[0]
        assert row["result"] == "no_results"
        assert row["details"]["reason"] == "No relevant memories found."

    def test_found_is_not_double_audited(self, provider, audit_logger, monkeypatch):
        """found 由 RecallService 审计，handler 不重复写（一次调用一条记录）。"""
        monkeypatch.setattr(
            recall_module.RecallService,
            "handle",
            lambda self, args: {"status": "found", "query": args["query"], "memories": [{"memory_id": "a" * 12}]},
        )
        result = _recall(provider, {"query": "命中查询"})
        assert result["status"] == "found"
        assert audit_logger.rows == []


class TestAsyncPathMatchesSync:
    @pytest.mark.asyncio
    async def test_async_validation_error_audited(self, provider, audit_logger):
        result = json.loads(await recall_module.async_handle_recall(provider, {"query": "x", "mode": "bm25"}))
        assert result["status"] == "error"
        assert len(audit_logger.rows) == 1
        assert audit_logger.rows[0]["operation"] == "recall"

    @pytest.mark.asyncio
    async def test_async_no_results_audited(self, provider, audit_logger, monkeypatch):
        async def _async_handle(self, args):
            return {"status": "no_results", "query": args["query"], "message": "No relevant memories found."}

        monkeypatch.setattr(recall_module.RecallService, "async_handle", _async_handle)
        result = json.loads(await recall_module.async_handle_recall(provider, {"query": "不存在"}))
        assert result["status"] == "no_results"
        assert len(audit_logger.rows) == 1
        assert audit_logger.rows[0]["result"] == "no_results"


class TestAuditFailureDoesNotBreakRecall:
    def test_recall_still_returns_when_audit_raises(self, monkeypatch):
        logger = RecordingAuditLogger(raise_on_log=True)
        provider = FakeProvider(logger)
        result = _recall(provider, {"query": "检索审计", "mode": "bm25"})
        assert result["status"] == "error"
        assert "invalid mode" in result["reason"]

    def test_missing_audit_logger_is_tolerated(self):
        provider = FakeProvider(None)
        result = _recall(provider, {"query": "检索审计", "mode": "bm25"})
        assert result["status"] == "error"


class TestQueryTextIsSanitized:
    def test_api_key_in_query_is_masked(self, provider, audit_logger):
        secret_query = "如何配置 sk-abcdefghijklmnopqrstuvwxyz123456"
        _recall(provider, {"query": secret_query, "mode": "bm25"})
        recorded = audit_logger.rows[0]["details"]["query"]
        assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in recorded
        assert "sk-***" in recorded

    def test_overlong_query_is_truncated(self, provider, audit_logger):
        _recall(provider, {"query": "a" * 5_000, "mode": "bm25"})
        recorded = audit_logger.rows[0]["details"]["query"]
        assert len(recorded) <= 200

    def test_audit_logger_protocol_only_needs_log(self):
        """HandlerDependencies 的 audit_logger 走 log() 关键字协议，不依赖真实 AuditLogger。"""
        deps = HandlerDependencies(audit_logger=RecordingAuditLogger(), instance_id="i")
        recall_module._audit_recall(deps, {"query": "q"}, {"status": "no_results"}, 0.0)
        assert deps.audit_logger.rows[0]["instance_id"] == "i"
