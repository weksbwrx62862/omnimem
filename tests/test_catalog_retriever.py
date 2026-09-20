"""retrieval.catalog.CatalogRetriever 离线单元测试。

以最小替身挂载 index / wing_room / vector / bm25，覆盖：
  - _infer_hall / _infer_wing：关键字命中、大小写、未命中
  - _match_rooms：完全包含 / 部分 / 字符重叠 / 无匹配
  - _locate_directories：hint 优先、search_l0 异常降级、无 wing/hall → 空
  - search：无目录 → 空；有目录 → 走向量+BM25 过滤、去重、排序、top_k 截断
  - _search_in_directories：向量/BM25 异常降级；跨目录合并去重
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from omnimem.retrieval.catalog import CatalogRetriever


@pytest.fixture
def deps() -> dict[str, Any]:
    index = MagicMock()
    index.search_l0.return_value = []
    index.search_by_directory.return_value = []
    vector = MagicMock()
    vector.search.return_value = []
    bm25 = MagicMock()
    bm25.search.return_value = []
    wing_room = MagicMock()
    return {"index": index, "vector": vector, "bm25": bm25, "wing_room": wing_room}


@pytest.fixture
def cat(deps: dict[str, Any]) -> CatalogRetriever:
    return CatalogRetriever(
        index=deps["index"],
        wing_room=deps["wing_room"],
        vector_retriever=deps["vector"],
        bm25_retriever=deps["bm25"],
    )


# ── _infer_hall ──


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("用户偏好深色主题", "preferences"),
        ("my preferences", "preferences"),
        ("记住这个事实", "facts"),
        ("how to fix it", "corrections"),  # "fix" 命中 corrections
        ("教程步骤", "skills"),
        ("流程指南", "procedures"),
        ("tool call", "actions"),
        ("经验教训", "reasoning"),
        ("随便聊聊", ""),
    ],
)
def test_infer_hall(cat: CatalogRetriever, q: str, expected: str) -> None:
    assert cat._infer_hall(q) == expected


def test_infer_hall_case_insensitive(cat: CatalogRetriever) -> None:
    assert cat._infer_hall("MY PREFERENCES") == "preferences"


# ── _infer_wing ──


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("我的私人秘密", "personal"),
        ("团队协作", "team"),
        ("公开信息", "public"),
        ("public data", "public"),
        ("无关内容", ""),
    ],
)
def test_infer_wing(cat: CatalogRetriever, q: str, expected: str) -> None:
    assert cat._infer_wing(q) == expected


def test_infer_wing_case_insensitive(cat: CatalogRetriever) -> None:
    assert cat._infer_wing("PRIVATE things") == "personal"


# ── _match_rooms ──


def test_match_rooms_full_containment(cat: CatalogRetriever) -> None:
    # room_lower in q_lower → score=3 排最前
    out = cat._match_rooms("work meeting notes", ["work", "meeting", "zzz"])
    assert out[0] in ("work", "meeting")
    assert "zzz" not in out


def test_match_rooms_partial_token(cat: CatalogRetriever) -> None:
    # 分词后 token 长度 >=2 → 部分匹配 score=2
    out = cat._match_rooms("关于项目管理的讨论", ["项目", "无关"])
    assert "项目" in out


def test_match_rooms_char_overlap(cat: CatalogRetriever) -> None:
    # 字符重叠 score=1
    out = cat._match_rooms("abc def", ["zzz"])
    assert out == []


def test_match_rooms_sorted_by_score(cat: CatalogRetriever) -> None:
    out = cat._match_rooms("work meeting notes", ["notes", "work"])
    # "work" 完全包含；"notes" 完全包含
    assert set(out) == {"work", "notes"}


def test_match_rooms_empty_rooms(cat: CatalogRetriever) -> None:
    assert cat._match_rooms("q", []) == []


# ── _locate_directories ──


def test_locate_returns_empty_when_no_hint_no_signal(
    cat: CatalogRetriever,
    deps: dict[str, Any],
) -> None:
    dirs = cat._locate_directories("无关内容", "", "")
    assert dirs == []
    deps["index"].search_l0.assert_not_called()


def test_locate_hall_hint_only(cat: CatalogRetriever, deps: dict[str, Any]) -> None:
    dirs = cat._locate_directories("q", "", "preferences")
    assert dirs == [{"wing": "", "hall": "preferences", "room": ""}]
    deps["index"].search_l0.assert_called_once_with(wing="", hall="preferences")


def test_locate_wing_hint_only(cat: CatalogRetriever, deps: dict[str, Any]) -> None:
    dirs = cat._locate_directories("q", "personal", "")
    assert dirs == [{"wing": "personal", "hall": "", "room": ""}]


def test_locate_hints_take_priority_over_inferred(
    cat: CatalogRetriever,
    deps: dict[str, Any],
) -> None:
    # query 含 "偏好" 会推 preferences，但 hint_hall 覆盖
    dirs = cat._locate_directories("偏好设置", "", "facts")
    assert dirs[0]["hall"] == "facts"


def test_locate_matched_rooms_expanded_to_dirs(
    cat: CatalogRetriever,
    deps: dict[str, Any],
) -> None:
    deps["index"].search_l0.return_value = ["work", "meeting", "notes", "other"]
    dirs = cat._locate_directories("work meeting notes", "", "preferences")
    assert len(dirs) == 3  # 只保留前 3
    assert all(d["wing"] == "" and d["hall"] == "preferences" for d in dirs)


def test_locate_search_l0_exception_degrades_to_empty_room(
    cat: CatalogRetriever,
    deps: dict[str, Any],
) -> None:
    deps["index"].search_l0.side_effect = RuntimeError("boom")
    dirs = cat._locate_directories("q", "", "preferences")
    # rooms 空 → 走 else 分支
    assert dirs == [{"wing": "", "hall": "preferences", "room": ""}]


# ── search ──


def test_search_returns_empty_when_no_dirs_located(
    cat: CatalogRetriever,
    deps: dict[str, Any],
) -> None:
    out = cat.search("无关内容")
    assert out == []
    deps["vector"].search.assert_not_called()


def test_search_filters_to_directory_ids(
    cat: CatalogRetriever,
    deps: dict[str, Any],
) -> None:
    deps["index"].search_by_directory.return_value = [
        {"memory_id": "m1"},
        {"memory_id": "m2"},
    ]
    deps["vector"].search.return_value = [
        {"memory_id": "m1", "score": 0.9},
        {"memory_id": "m3", "score": 0.8},  # 目录外 → 过滤
    ]
    deps["bm25"].search.return_value = [
        {"memory_id": "m2", "score": 0.6},
        {"memory_id": "m1", "score": 0.5},  # 已由 vector 命中 → 去重
    ]
    out = cat.search("偏好设置", top_k=10, hint_hall="preferences")
    mids = [r["memory_id"] for r in out]
    assert mids == ["m1", "m2"]
    assert out[0]["_source"] == "catalog"
    assert out[1]["_source"] == "catalog_bm25"
    assert out[0]["catalog_dir"] == "/preferences/"


def test_search_sorts_by_score_and_caps_top_k(
    cat: CatalogRetriever,
    deps: dict[str, Any],
) -> None:
    deps["index"].search_by_directory.return_value = [
        {"memory_id": f"m{i}"} for i in range(10)
    ]
    deps["vector"].search.return_value = [
        {"memory_id": f"m{i}", "score": i * 0.1} for i in range(10)
    ]
    out = cat.search("q", top_k=3, hint_hall="preferences")
    assert len(out) == 3
    assert [r["memory_id"] for r in out] == ["m9", "m8", "m7"]


def test_search_by_directory_exception_skips_dir(
    cat: CatalogRetriever,
    deps: dict[str, Any],
) -> None:
    deps["index"].search_by_directory.side_effect = RuntimeError("boom")
    out = cat.search("偏好", hint_hall="preferences")
    assert out == []


def test_search_vector_exception_falls_back_to_bm25(
    cat: CatalogRetriever,
    deps: dict[str, Any],
) -> None:
    deps["index"].search_by_directory.return_value = [{"memory_id": "m1"}]
    deps["vector"].search.side_effect = RuntimeError("vec down")
    deps["bm25"].search.return_value = [{"memory_id": "m1", "score": 0.7}]
    out = cat.search("偏好", hint_hall="preferences")
    assert len(out) == 1
    assert out[0]["_source"] == "catalog_bm25"


def test_search_bm25_exception_falls_back_to_vector(
    cat: CatalogRetriever,
    deps: dict[str, Any],
) -> None:
    deps["index"].search_by_directory.return_value = [{"memory_id": "m1"}]
    deps["vector"].search.return_value = [{"memory_id": "m1", "score": 0.7}]
    deps["bm25"].search.side_effect = RuntimeError("bm25 down")
    out = cat.search("偏好", hint_hall="preferences")
    assert len(out) == 1
    assert out[0]["_source"] == "catalog"


def test_search_empty_dir_entries_skip(cat: CatalogRetriever, deps: dict[str, Any]) -> None:
    deps["index"].search_by_directory.return_value = []
    out = cat.search("偏好", hint_hall="preferences")
    assert out == []


def test_search_missing_memory_id_ignored(
    cat: CatalogRetriever,
    deps: dict[str, Any],
) -> None:
    deps["index"].search_by_directory.return_value = [{"memory_id": "m1"}]
    deps["vector"].search.return_value = [
        {"score": 0.9},  # 无 memory_id
        {"memory_id": "", "score": 0.8},  # 空 memory_id
        {"memory_id": "m1", "score": 0.7},
    ]
    out = cat.search("偏好", hint_hall="preferences")
    assert len(out) == 1
    assert out[0]["memory_id"] == "m1"


def test_search_multiple_dirs_dedup_across(
    cat: CatalogRetriever,
    deps: dict[str, Any],
) -> None:
    deps["index"].search_l0.return_value = ["roomA", "roomB"]
    deps["index"].search_by_directory.side_effect = [
        [{"memory_id": "m1"}, {"memory_id": "m2"}],
        [{"memory_id": "m1"}, {"memory_id": "m3"}],  # m1 重复 → 只入一次
    ]
    deps["vector"].search.return_value = [
        {"memory_id": "m1", "score": 0.9},
        {"memory_id": "m2", "score": 0.8},
        {"memory_id": "m3", "score": 0.7},
    ]
    out = cat.search("work notes", hint_hall="preferences")
    mids = [r["memory_id"] for r in out]
    assert mids == ["m1", "m2", "m3"]


def test_search_catalog_dir_label_contains_wing_hall_room(
    cat: CatalogRetriever,
    deps: dict[str, Any],
) -> None:
    deps["index"].search_l0.return_value = ["myroom"]
    deps["index"].search_by_directory.return_value = [{"memory_id": "m1"}]
    deps["vector"].search.return_value = [{"memory_id": "m1", "score": 0.5}]
    out = cat.search("myroom", hint_wing="personal", hint_hall="facts")
    assert out[0]["catalog_dir"] == "personal/facts/myroom"
