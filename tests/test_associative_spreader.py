"""联想扩散引擎 AssociativeSpreader 单测（改进项 #2 覆盖，离线，注入 Fake KG/Retriever）。

覆盖：空实体早退、KG 多跳（含跳数置信度衰减）、语义扩散降权、内容/ID 去重、
top_k 截断与按置信度排序、下游异常非致命、无通道与 __repr__。
"""
from __future__ import annotations

from typing import Any

from omnimem.associative.spreader import (
    _BASE_ASSOC_SCORE,
    _KG_CONFIDENCE_DECAY,
    AssociativeSpreader,
)


class FakeKG:
    def __init__(self, neighbors: dict[str, list[dict[str, Any]]], raises: bool = False) -> None:
        self._neighbors = neighbors
        self._raises = raises
        self.calls: list[str] = []

    def get_neighbors(self, entity: str, depth: int = 1) -> list[dict[str, Any]]:
        self.calls.append(entity)
        if self._raises:
            raise RuntimeError("kg down")
        return list(self._neighbors.get(entity, []))


class FakeRetriever:
    def __init__(self, hits: dict[str, list[dict[str, Any]]], raises: bool = False) -> None:
        self._hits = hits
        self._raises = raises

    def vector_search(self, entity: str, top_k: int = 3) -> list[dict[str, Any]]:
        if self._raises:
            raise RuntimeError("vec down")
        return [dict(h) for h in self._hits.get(entity, [])]


def _spreader(entities: list[str], *, kg=None, retriever=None, **kw) -> AssociativeSpreader:
    sp = AssociativeSpreader(knowledge_graph=kg, retriever=retriever, **kw)
    sp._extract_entities = lambda text: list(entities)  # 绕过需要分词的 KG 实体抽取
    return sp


# ── 早退 ──────────────────────────────────────────────
def test_empty_entities_returns_empty():
    sp = _spreader([])
    assert sp.spread("whatever") == []


def test_no_channels_returns_empty():
    sp = _spreader(["A"])  # kg/retriever 均 None
    assert sp.spread("q") == []


# ── KG 单跳 ───────────────────────────────────────────
def test_kg_one_hop_confidence_and_shape():
    kg = FakeKG({"A": [{"subject": "A", "predicate": "USES", "object": "B", "confidence": 0.9}]})
    sp = _spreader(["A"], kg=kg)
    res = sp.spread("q", top_k=5)
    assert len(res) >= 1
    r = next(x for x in res if x["content"] == "A USES B")
    assert r["_assoc_type"] == "kg_spread"
    assert r["_source"] == "association"
    assert r["_spread_depth"] == 0
    assert r["confidence"] == 0.9  # decay**0
    assert r["score"] == round(max(0.9 * 0.8, _BASE_ASSOC_SCORE), 4)


# ── KG 两跳：置信度按深度衰减 + 排序 ──────────────────
def test_kg_two_hop_decay_and_order():
    kg = FakeKG({
        "A": [{"subject": "A", "predicate": "USES", "object": "B", "confidence": 0.9}],
        "B": [{"subject": "B", "predicate": "NEEDS", "object": "C", "confidence": 0.9}],
    })
    sp = _spreader(["A"], kg=kg)
    res = sp.spread("q", top_k=5)
    by_content = {x["content"]: x for x in res}
    assert by_content["A USES B"]["_spread_depth"] == 0
    assert by_content["B NEEDS C"]["_spread_depth"] == 1
    assert by_content["B NEEDS C"]["confidence"] == round(0.9 * _KG_CONFIDENCE_DECAY, 4)
    confs = [x["confidence"] for x in res]
    assert confs == sorted(confs, reverse=True)  # 按置信度降序


# ── 分数下限：低置信三元组仍过过滤线 ──────────────────
def test_score_floored_at_base():
    kg = FakeKG({"A": [{"subject": "A", "predicate": "P", "object": "Z", "confidence": 0.1}]})
    sp = _spreader(["A"], kg=kg)
    r = sp.spread("q")[0]
    assert r["score"] == _BASE_ASSOC_SCORE  # max(0.1*0.8, 0.35)


# ── 语义扩散：降权 + 标注 ─────────────────────────────
def test_semantic_spread_scaled_and_tagged():
    rt = FakeRetriever({"A": [{"memory_id": "m9", "content": "near text", "score": 0.9}]})
    sp = _spreader(["A"], retriever=rt)
    res = sp.spread("q")
    assert len(res) == 1
    r = res[0]
    assert r["_source"] == "association"
    assert r["_assoc_type"] == "semantic_spread"
    assert r["type"] == "association"
    assert r["score"] == round(max(0.9 * 0.6, _BASE_ASSOC_SCORE), 4)
    assert r["confidence"] == round(0.9 * 0.6, 4)


def test_semantic_skips_existing_ids():
    rt = FakeRetriever({"A": [{"memory_id": "m9", "content": "x", "score": 0.9}]})
    sp = _spreader(["A"], retriever=rt)
    assert sp.spread("q", existing_ids={"m9"}) == []


# ── 去重：跨通道内容去重 ──────────────────────────────
def test_content_dedup_across_channels():
    kg = FakeKG({"A": [{"subject": "A", "predicate": "P", "object": "B"}]})
    rt = FakeRetriever({"A": [{"memory_id": "z", "content": "a p b", "score": 0.9}]})
    sp = _spreader(["A"], kg=kg, retriever=rt)
    res = sp.spread("q")
    # KG 产出 "A P B"（小写 "a p b"）与语义命中同内容 → 只保留一条
    assert sum(1 for x in res if x["content"].lower() == "a p b") == 1


# ── top_k 截断 ────────────────────────────────────────
def test_top_k_caps_results():
    kg = FakeKG({"A": [{"subject": "A", "predicate": f"P{i}", "object": f"B{i}", "confidence": 0.9} for i in range(10)]})
    sp = _spreader(["A"], kg=kg)
    assert len(sp.spread("q", top_k=2)) <= 2


# ── 下游异常非致命 ────────────────────────────────────
def test_kg_failure_is_nonfatal():
    rt = FakeRetriever({"A": [{"memory_id": "m1", "content": "sem", "score": 0.8}]})
    sp = _spreader(["A"], kg=FakeKG({}, raises=True), retriever=rt)
    res = sp.spread("q")
    assert [x["content"] for x in res] == ["sem"]  # KG 挂不影响语义通道


def test_semantic_failure_is_nonfatal():
    kg = FakeKG({"A": [{"subject": "A", "predicate": "P", "object": "B", "confidence": 0.9}]})
    sp = _spreader(["A"], kg=kg, retriever=FakeRetriever({}, raises=True))
    res = sp.spread("q")
    assert any(x["_assoc_type"] == "kg_spread" for x in res)


# ── max_depth 约束 ────────────────────────────────────
def test_max_depth_limits_expansion():
    kg = FakeKG({
        "A": [{"subject": "A", "predicate": "USES", "object": "B", "confidence": 0.9}],
        "B": [{"subject": "B", "predicate": "NEEDS", "object": "C", "confidence": 0.9}],
    })
    sp = _spreader(["A"], kg=kg, max_depth=0)
    contents = {x["content"] for x in sp.spread("q")}
    assert "A USES B" in contents      # 深度 0 的三元组仍在
    assert "B NEEDS C" not in contents  # 不再向 B 扩散下一跳


# ── __repr__ ──────────────────────────────────────────
def test_repr_contains_wiring():
    sp = _spreader(["A"], kg=FakeKG({}), retriever=FakeRetriever({}), max_depth=3)
    text = repr(sp)
    assert "AssociativeSpreader" in text and "max_depth=3" in text
