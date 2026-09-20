"""改进项 #9 recall 分发接线单测（离线）。

验证 RecallService._run_agentic 复用 AgenticRetriever 且对底层异常非致命回落，
以及 handler 允许 agentic 模式通过校验。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from omnimem.handlers.recall import _validate_recall_args
from omnimem.services.recall_service import RecallService


class _FakeRetriever:
    def __init__(self, mapping: dict[str, list[dict[str, Any]]]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def search(self, query: str, top_k: int = 20, mode: str = "hybrid") -> list[dict[str, Any]]:
        self.calls.append(query)
        return list(self.mapping.get(query, []))


def _service(retriever: Any) -> RecallService:
    deps = SimpleNamespace(retriever=retriever, config=SimpleNamespace(get=lambda *_a, **_k: 3))
    return RecallService(deps=deps)


def test_run_agentic_returns_docs():
    fake = _FakeRetriever({"q": [{"memory_id": "m1", "content": "a", "score": 0.9},
                                 {"memory_id": "m2", "content": "b", "score": 0.8},
                                 {"memory_id": "m3", "content": "c", "score": 0.7}]})
    out = _service(fake)._run_agentic("q", {})
    assert isinstance(out, list)
    assert {d["memory_id"] for d in out} == {"m1", "m2", "m3"}
    assert "q" in fake.calls


def test_run_agentic_nonfatal_on_retriever_error():
    class Boom:
        def search(self, query, top_k=20, mode="hybrid"):
            raise RuntimeError("down")

    out = _service(Boom())._run_agentic("q", {})
    assert out == []  # 异常吞掉，回落空结果


def test_handler_allows_agentic_mode():
    assert _validate_recall_args({"query": "x", "mode": "agentic"}) is None
    assert _validate_recall_args({"query": "x", "mode": "bogus"}) is not None
