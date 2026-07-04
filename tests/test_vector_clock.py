"""Governance VectorClock 模块测试。

覆盖: VectorClock 初始化、递增、比较、合并、序列化、持久化
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from omnimem.governance.vector_clock import VectorClock


class TestVectorClockInit:
    def test_empty_init(self):
        vc = VectorClock()
        assert vc.to_dict() == {}

    def test_init_with_clock(self):
        vc = VectorClock({"node-a": 3, "node-b": 1})
        assert vc.to_dict() == {"node-a": 3, "node-b": 1}

    def test_init_copies_clock(self):
        original = {"a": 1}
        vc = VectorClock(original)
        original["a"] = 999
        assert vc.to_dict()["a"] == 1


class TestVectorClockIncrement:
    def test_increment_new_node(self):
        vc = VectorClock()
        vc.increment("node-1")
        assert vc.to_dict()["node-1"] == 1

    def test_increment_existing_node(self):
        vc = VectorClock({"node-1": 5})
        vc.increment("node-1")
        assert vc.to_dict()["node-1"] == 6

    def test_chainable(self):
        vc = VectorClock()
        result = vc.increment("a").increment("b")
        assert result is vc


class TestVectorClockCompare:
    def test_equal_clocks(self):
        vc1 = VectorClock({"a": 1, "b": 2})
        vc2 = VectorClock({"a": 1, "b": 2})
        assert vc1.compare(vc2) == 0

    def test_before(self):
        vc1 = VectorClock({"a": 1, "b": 1})
        vc2 = VectorClock({"a": 2, "b": 2})
        assert vc1.compare(vc2) == -1

    def test_after(self):
        vc1 = VectorClock({"a": 3, "b": 3})
        vc2 = VectorClock({"a": 1, "b": 1})
        assert vc1.compare(vc2) == 1

    def test_concurrent(self):
        vc1 = VectorClock({"a": 3, "b": 1})
        vc2 = VectorClock({"a": 1, "b": 3})
        assert vc1.compare(vc2) == 0  # 并发冲突

    def test_empty_vs_nonempty(self):
        vc1 = VectorClock()
        vc2 = VectorClock({"a": 1})
        assert vc1.compare(vc2) == -1


class TestVectorClockMerge:
    def test_merge_disjoint(self):
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"b": 2})
        merged = vc1.merge(vc2)
        assert merged.to_dict() == {"a": 1, "b": 2}

    def test_merge_overlapping(self):
        vc1 = VectorClock({"a": 3, "b": 1})
        vc2 = VectorClock({"a": 1, "b": 5})
        merged = vc1.merge(vc2)
        assert merged.to_dict() == {"a": 3, "b": 5}

    def test_merge_returns_new_instance(self):
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"b": 2})
        merged = vc1.merge(vc2)
        assert merged is not vc1
        assert merged is not vc2


class TestVectorClockSerialization:
    def test_to_json_from_json(self):
        vc = VectorClock({"a": 3, "b": 1})
        json_str = vc.to_json()
        restored = VectorClock.from_json(json_str)
        assert restored.to_dict() == vc.to_dict()

    def test_from_json_invalid(self):
        vc = VectorClock.from_json("not valid json{{{")
        assert vc.to_dict() == {}

    def test_from_dict(self):
        vc = VectorClock.from_dict({"x": 5, "y": 3.7})
        assert vc.to_dict()["x"] == 5
        assert vc.to_dict()["y"] == 3  # float → int


class TestVectorClockPersistence:
    def test_save_and_load(self):
        vc = VectorClock({"node-1": 10, "node-2": 20})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vc.json"
            assert vc.save(path) is True
            loaded = VectorClock.load(path)
            assert loaded.to_dict() == vc.to_dict()

    def test_load_nonexistent_file(self):
        result = VectorClock.load(Path("/nonexistent/path/vc.json"))
        # load returns None for missing file
        assert result is None or result.to_dict() == {}

    def test_save_creates_parent_dirs(self):
        vc = VectorClock({"a": 1})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "vc.json"
            assert vc.save(path) is True
            assert path.exists()
