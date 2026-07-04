"""同义词扩展器单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omnimem.retrieval.synonym_expander import SynonymExpander


@pytest.fixture
def synonym_map() -> dict[str, list[str]]:
    """测试用同义词映射。"""
    return {
        "python": ["py", "python3"],
        "数据库": ["db", "database"],
    }


def test_synonym_expander_loads_from_map(synonym_map: dict[str, list[str]]) -> None:
    """Expander 应接受显式同义词映射。"""
    expander = SynonymExpander(synonym_map=synonym_map)
    assert expander._synonym_map == synonym_map


def test_synonym_expander_load_empty_when_file_missing() -> None:
    """配置文件缺失时应降级为空映射。"""
    with patch.object(SynonymExpander, "load_synonyms", return_value={}):
        expander = SynonymExpander()
    assert expander._synonym_map == {}


def test_synonym_expander_extends_query(synonym_map: dict[str, list[str]]) -> None:
    """Expander 应使用同义词扩展查询并合并结果。"""
    expander = SynonymExpander(synonym_map=synonym_map)

    bm25 = MagicMock()
    base_result = [{"memory_id": "m1", "content": "python 简介", "score": 1.0}]
    expanded_result = [{"memory_id": "m2", "content": "py 入门", "score": 0.9}]

    def _fake_search(_query: str, top_k: int) -> list[dict]:  # noqa: ARG001
        if _query == "python 教程":
            return base_result
        if _query in ("py 教程", "python3 教程"):
            return expanded_result
        return []

    bm25.search.side_effect = _fake_search
    results = expander.search(bm25, "python 教程", top_k=5)
    memory_ids = {r["memory_id"] for r in results}
    assert "m1" in memory_ids
    assert "m2" in memory_ids


def test_synonym_expander_deduplicates_by_memory_id(synonym_map: dict[str, list[str]]) -> None:
    """扩展查询产生重复 memory_id 时应去重。"""
    expander = SynonymExpander(synonym_map=synonym_map)

    bm25 = MagicMock()
    same_result = [{"memory_id": "m1", "content": "python", "score": 1.0}]

    def _fake_search(_query: str, top_k: int) -> list[dict]:  # noqa: ARG001
        return same_result

    bm25.search.side_effect = _fake_search
    results = expander.search(bm25, "python 教程", top_k=5)
    assert len(results) == 1


def test_synonym_expander_no_match_returns_base(synonym_map: dict[str, list[str]]) -> None:
    """查询未命中同义词映射时只返回基础结果。"""
    expander = SynonymExpander(synonym_map=synonym_map)

    bm25 = MagicMock()
    base_result = [{"memory_id": "m1", "content": "其他", "score": 1.0}]
    bm25.search.return_value = base_result

    results = expander.search(bm25, "rust 教程", top_k=5)
    assert results == base_result
    assert bm25.search.call_count == 1


def test_synonym_expander_load_synonyms_file(tmp_path: Path) -> None:
    """SynonymExpander 应能从 JSON 文件加载同义词映射。"""
    import json

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    synonyms_path = config_dir / "synonyms.json"
    data = {"ai": ["人工智能", "artificial intelligence"]}
    synonyms_path.write_text(json.dumps(data), encoding="utf-8")

    # 构造一个链式 parent 对象，使得 Path(__file__).parent.parent == tmp_path
    # 这样 / config / synonyms.json 就能找到临时文件
    class _FakePath:
        def __init__(self) -> None:
            self._parent = self

        @property
        def parent(self) -> _FakePath:
            return self._parent

    fake_root = _FakePath()
    fake_root._parent = _FakePath()
    fake_root._parent._parent = tmp_path  # type: ignore[assignment]

    with patch("omnimem.retrieval.synonym_expander.Path") as mock_path_cls:
        mock_path_cls.return_value = fake_root
        expander = SynonymExpander()

    assert expander._synonym_map == data
