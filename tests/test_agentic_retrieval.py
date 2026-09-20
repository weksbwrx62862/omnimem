"""改进项 #9 Agentic 迭代检索单测（离线，注入 FakeRetriever，无需 LLM/检索后端）。

覆盖：空查询、单轮充分、多跳补检、max_steps 触顶、跨查询去重融合、
底层异常吞掉、启发式默认分解、to_recall_dict 结构。
"""
from __future__ import annotations

from typing import Any

from omnimem.retrieval.agentic import (
    AgenticConfig,
    AgenticResult,
    AgenticRetriever,
)
from omnimem.retrieval.rrf import RRFFusion


class FakeRetriever:
    """按查询精确匹配返回预置结果；记录调用次序。"""

    def __init__(self, mapping: dict[str, list[dict[str, Any]]], default=None) -> None:
        self.mapping = mapping
        self.default = list(default or [])
        self.calls: list[str] = []

    def search(self, query: str, top_k: int = 20, mode: str = "hybrid") -> list[dict[str, Any]]:
        self.calls.append(query)
        return list(self.mapping.get(query, self.default))


class BoomRetriever:
    def search(self, query: str, top_k: int = 20, mode: str = "hybrid"):
        raise RuntimeError("downstream down")


def _doc(mid: str, content: str = "", score: float = 0.5) -> dict[str, Any]:
    return {"memory_id": mid, "content": content or f"body {mid}", "score": score}


# ── 空查询 ────────────────────────────────────────────
def test_empty_query_is_noop():
    fake = FakeRetriever({})
    res = AgenticRetriever(fake).run("   ")
    assert isinstance(res, AgenticResult)
    assert res.results == []
    assert res.steps == []
    assert fake.calls == []


# ── 单轮充分：只跑种子查询 ────────────────────────────
def test_single_shot_sufficiency():
    seed = "who owns the cat"
    fake = FakeRetriever({seed: [_doc("m1"), _doc("m2"), _doc("m3")]})
    res = AgenticRetriever(fake, config=AgenticConfig(min_hits=3)).run(seed)
    assert len(res.steps) == 1
    assert fake.calls == [seed]  # 判据满足 → 不再补检
    assert res.steps[0].new_hits == 3
    assert {d["memory_id"] for d in res.results} == {"m1", "m2", "m3"}


# ── 多跳：补检查询浮出新文档，步数=2 ──────────────────
def test_multihop_followup():
    seed = "capital of the neighbor of france"
    followup = "germany capital berlin"
    fake = FakeRetriever({
        seed: [_doc("m1")],
        followup: [_doc("m2"), _doc("m3"), _doc("m4")],
    })
    rewrite = lambda q, gathered, step: [followup]  # noqa: E731
    res = AgenticRetriever(
        fake, rewrite_fn=rewrite, config=AgenticConfig(min_hits=3)
    ).run(seed)
    assert len(res.steps) == 2
    assert fake.calls == [seed, followup]
    assert res.steps[1].new_hits == 3
    assert {d["memory_id"] for d in res.results} == {"m1", "m2", "m3", "m4"}


# ── max_steps 触顶：判据始终不满足 ────────────────────
def test_max_steps_cap():
    fake = FakeRetriever({})  # 一切查询都无命中
    rewrite = lambda q, gathered, step: [f"sub-{step}"]  # noqa: E731
    res = AgenticRetriever(
        fake,
        rewrite_fn=rewrite,
        config=AgenticConfig(max_steps=3, min_hits=99),
    ).run("root query")
    assert len(res.steps) == 3
    assert res.results == []
    # 每轮都发起了检索
    assert len(fake.calls) == 3


# ── 跨查询去重 + RRF 融合 ─────────────────────────────
def test_dedup_and_merge_across_queries():
    seed = "alpha"
    followup = "alpha beta"
    fake = FakeRetriever({
        seed: [_doc("m1"), _doc("m2")],
        followup: [_doc("m1"), _doc("m2"), _doc("m3")],
    })
    rewrite = lambda q, gathered, step: [followup]  # noqa: E731
    res = AgenticRetriever(
        fake,
        rewrite_fn=rewrite,
        fusion=RRFFusion(k=60, min_rrf=0.0),
        config=AgenticConfig(min_hits=5),
    ).run(seed)
    ids = [d["memory_id"] for d in res.results]
    assert sorted(ids) == ["m1", "m2", "m3"]      # 每个 id 只出现一次
    # m1/m2 在两轮都命中 → 融合分应高于只命中一次的 m3，排在前面
    assert set(ids[:2]) == {"m1", "m2"}


# ── 底层检索异常被吞掉，不影响编排 ────────────────────
def test_search_exception_swallowed():
    res = AgenticRetriever(BoomRetriever()).run("anything")
    assert res.results == []
    assert len(res.steps) >= 1  # 仍记录了步，只是命中为 0


# ── 启发式默认分解：用命中高频词补查询 ────────────────
def test_default_rewrite_uses_bridge_terms():
    fake = FakeRetriever({})
    eng = AgenticRetriever(fake, config=AgenticConfig(queries_per_step=2))
    gathered = [_doc("m1", content="apple banana banana cherry")]
    out = eng._default_rewrite("apple", gathered, step=1)
    assert out  # 至少产出一条
    assert all("apple" in q for q in out)         # 保留原查询词
    assert any("banana" in q for q in out)         # 高频桥接词优先
    assert not any(q.strip() == "apple" for q in out)  # 不重复纯原查询


# ── 默认判据：命中数阈值 ──────────────────────────────
def test_default_judge_threshold():
    eng = AgenticRetriever(FakeRetriever({}), config=AgenticConfig(min_hits=2))
    assert eng._default_judge("q", [_doc("a"), _doc("b")]) is True
    assert eng._default_judge("q", [_doc("a")]) is False


# ── to_recall_dict 结构 ───────────────────────────────
def test_to_recall_dict_shape():
    seed = "shape"
    fake = FakeRetriever({seed: [_doc("m1"), _doc("m2"), _doc("m3")]})
    res = AgenticRetriever(fake, config=AgenticConfig(min_hits=3)).run(seed)
    payload = res.to_recall_dict()
    assert payload["status"] == "ok"
    assert payload["mode"] == "agentic"
    assert payload["count"] == len(res.results)
    assert payload["steps"] == 1
    assert "_agentic" not in payload
    explained = res.to_recall_dict(explain=True)
    assert "_agentic" in explained
    assert explained["_agentic"]["steps"][0]["queries"] == [seed]
