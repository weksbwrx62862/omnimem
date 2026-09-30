"""L2 结构化记忆模块测试 — 包括 WingRoom / DrawerCloset / ThreeLevelIndex / WriteOp / Saga 补偿。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from omnimem.memory.drawer_closet import DrawerClosetStore, WriteOp
from omnimem.memory.index import ThreeLevelIndex
from omnimem.memory.wing_room import WingRoomManager


class TestWingRoomManager(unittest.TestCase):
    """WingRoomManager 测试。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.wrm = WingRoomManager(Path(self.tmpdir))

    def test_resolve_wing(self) -> None:
        self.assertEqual(self.wrm.resolve_wing("personal"), "personal")
        self.assertEqual(self.wrm.resolve_wing("project"), "projects")
        self.assertEqual(self.wrm.resolve_wing("team"), "team")
        self.assertEqual(self.wrm.resolve_wing("secret"), "personal")

    def test_resolve_hall(self) -> None:
        self.assertEqual(self.wrm.resolve_hall("fact"), "facts")
        self.assertEqual(self.wrm.resolve_hall("preference"), "preferences")
        self.assertEqual(self.wrm.resolve_hall("correction"), "corrections")

    def test_resolve_room_tech_keyword(self) -> None:
        room = self.wrm.resolve_room("使用python编写爬虫", "personal")
        self.assertEqual(room, "python")

    def test_resolve_room_chinese_topic(self) -> None:
        room = self.wrm.resolve_room("量子计算是未来技术", "personal")
        self.assertIn("量子", room)

    def test_resolve_room_fallback_hash(self) -> None:
        room = self.wrm.resolve_room("xxx", "personal", "fact")
        self.assertTrue(room.startswith("fact-"))

    def test_get_room_path(self) -> None:
        path = self.wrm.get_room_path("personal", "facts", "quantum")
        self.assertTrue(path.exists())
        self.assertIn("personal", str(path))
        self.assertIn("facts", str(path))
        self.assertIn("quantum", str(path))

    def test_list_wings(self) -> None:
        self.wrm.get_room_path("personal", "facts", "test")
        wings = self.wrm.list_wings()
        self.assertIn("personal", wings)

    def test_sanitize_name(self) -> None:
        self.assertEqual(WingRoomManager._sanitize_name("a/b\\c"), "a-b-c")
        self.assertEqual(WingRoomManager._sanitize_name(""), "unnamed")


class TestDrawerClosetStore(unittest.TestCase):
    """DrawerClosetStore 测试。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.store = DrawerClosetStore(Path(self.tmpdir))

    def test_add_and_get(self) -> None:
        mid = self.store.add(
            wing="personal", room="test", content="测试内容", memory_type="fact", confidence=3
        )
        self.assertTrue(mid)
        result = self.store.get(mid)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "测试内容")  # type: ignore[index]
        self.assertEqual(result["type"], "fact")  # type: ignore[index]

    def test_add_creates_disk_files(self) -> None:
        mid = self.store.add(wing="personal", room="test", content="磁盘测试")
        self.store.flush()
        palace = Path(self.tmpdir)
        drawer_files = list(palace.rglob(f"drawer/{mid}.md"))
        closet_files = list(palace.rglob(f"closet/{mid}.md"))
        self.assertTrue(len(drawer_files) > 0, "Drawer file should exist")
        self.assertTrue(len(closet_files) > 0, "Closet file should exist")

    def test_search_by_type(self) -> None:
        self.store.add(wing="personal", room="r1", content="事实1", memory_type="fact")
        self.store.add(wing="personal", room="r2", content="偏好1", memory_type="preference")
        facts = self.store.search(memory_type="fact")
        prefs = self.store.search(memory_type="preference")
        self.assertTrue(any(f["type"] == "fact" for f in facts))
        self.assertTrue(any(p["type"] == "preference" for p in prefs))

    def test_search_by_wing(self) -> None:
        self.store.add(wing="personal", room="r1", content="个人记忆")
        self.store.add(wing="shared", room="r2", content="共享记忆")
        personal = self.store.search(wing="personal")
        shared = self.store.search(wing="shared")
        self.assertTrue(all(m["wing"] == "personal" for m in personal))
        self.assertTrue(all(m["wing"] == "shared" for m in shared))

    def test_search_by_content(self) -> None:
        self.store.add(wing="personal", room="r1", content="Python是最好的语言")
        results = self.store.search_by_content("Python")
        self.assertTrue(len(results) > 0)
        self.assertIn("Python", results[0]["content"])

    def test_warm_up(self) -> None:
        mid = "warm-test-001"
        entries = [
            {
                "memory_id": mid,
                "content": "预热内容",
                "summary": "预热摘要",
                "type": "fact",
                "wing": "personal",
            }
        ]
        self.store.warm_up(entries)
        result = self.store.get(mid)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "预热内容")  # type: ignore[index]

    def test_update_privacy(self) -> None:
        mid = self.store.add(wing="personal", room="r1", content="隐私测试", privacy="personal")
        ok = self.store.update_privacy(mid, "secret", new_wing="personal")
        self.assertTrue(ok)
        result = self.store.get(mid)
        assert result is not None
        self.assertEqual(result["privacy"], "secret")

    def test_get_nonexistent(self) -> None:
        result = self.store.get("nonexistent-id")
        self.assertIsNone(result)

    def test_lru_eviction(self) -> None:
        store = DrawerClosetStore(Path(self.tmpdir) / "evict", max_index_size=3)
        ids = []
        for i in range(5):
            ids.append(store.add(wing="personal", room=f"r{i}", content=f"内容{i}"))
        self.assertLessEqual(len(store._closet_index), 3)

    def test_closet_summary_no_newline(self) -> None:
        mid = self.store.add(wing="personal", room="r1", content="第一行\n第二行\n第三行")
        result = self.store.get(mid)
        self.assertIn("第一行", result["summary"])  # type: ignore[index]
        self.assertNotIn("\n", result["summary"])  # type: ignore[index]


class TestDrawerClosetEdgeCases(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.store = DrawerClosetStore(Path(self.tmpdir))

    def test_add_empty_content(self) -> None:
        mid = self.store.add(wing="personal", room="r1", content="")
        self.assertTrue(mid)
        result = self.store.get(mid)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "")

    def test_add_very_long_content(self) -> None:
        long_content = "A" * 10000
        mid = self.store.add(wing="personal", room="r1", content=long_content)
        self.assertTrue(mid)
        result = self.store.get(mid)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], long_content)

    def test_add_special_unicode(self) -> None:
        content = "emoji: \U0001f600\U0001f680 zero-width:\u200b\u200c end"
        mid = self.store.add(wing="personal", room="r1", content=content)
        self.assertTrue(mid)
        result = self.store.get(mid)
        self.assertIsNotNone(result)
        self.assertIn("\U0001f600", result["content"])
        self.assertIn("\u200b", result["content"])

    def test_add_duplicate_id(self) -> None:
        mid = self.store.add(wing="personal", room="r1", content="原始内容", memory_id="dup-001")
        self.assertEqual(mid, "dup-001")
        mid2 = self.store.add(wing="personal", room="r1", content="覆盖内容", memory_id="dup-001")
        self.assertEqual(mid2, "dup-001")
        result = self.store.get("dup-001")
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "覆盖内容")

    def test_get_nonexistent(self) -> None:
        result = self.store.get("nonexistent-id-xyz")
        self.assertIsNone(result)

    def test_search_empty_store(self) -> None:
        results = self.store.search(memory_type="fact")
        self.assertEqual(len(results), 0)
        results = self.store.search_by_content("anything")
        self.assertEqual(len(results), 0)

    def test_lru_eviction(self) -> None:
        store = DrawerClosetStore(Path(self.tmpdir) / "lru-edge", max_index_size=3)
        ids = []
        for i in range(5):
            ids.append(store.add(wing="personal", room=f"r{i}", content=f"内容{i}"))
        self.assertLessEqual(len(store._closet_index), 3)
        store.get(ids[0])
        self.assertIn(ids[0], store._closet_index)

    def test_write_buffer_flush(self) -> None:
        store = DrawerClosetStore(Path(self.tmpdir) / "buffer", max_index_size=100)
        mids = []
        for i in range(3):
            mids.append(store.add(wing="personal", room=f"r{i}", content=f"缓冲{i}"))
        store.flush()
        for mid in mids:
            result = store.get(mid)
            self.assertIsNotNone(result)


class TestThreeLevelIndex(unittest.TestCase):
    """ThreeLevelIndex 测试。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.index = ThreeLevelIndex(Path(self.tmpdir))

    def tearDown(self) -> None:
        self.index.close()

    def test_add_and_get(self) -> None:
        self.index.add(
            memory_id="idx-001", wing="personal", hall="facts", room="test", content="索引测试"
        )
        self.index.flush()
        result = self.index.get("idx-001")
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "索引测试")  # type: ignore[index]

    def test_search_l0(self) -> None:
        self.index.add(memory_id="l0-1", wing="personal", hall="facts", room="room-a", content="c1")
        self.index.add(memory_id="l0-2", wing="personal", hall="facts", room="room-b", content="c2")
        self.index.flush()
        rooms = self.index.search_l0(wing="personal", hall="facts")
        self.assertIn("room-a", rooms)
        self.assertIn("room-b", rooms)

    def test_search_l1(self) -> None:
        self.index.add(
            memory_id="l1-1", wing="personal", hall="facts", room="r", content="c", type="fact"
        )
        self.index.flush()
        results = self.index.search_l1(wing="personal")
        self.assertTrue(len(results) > 0)

    def test_search_l2_keyword(self) -> None:
        self.index.add(
            memory_id="l2-1", wing="personal", hall="facts", room="r", content="量子计算"
        )
        self.index.flush()
        results = self.index.search_l2(keyword="量子")
        self.assertTrue(len(results) > 0)

    def test_remove(self) -> None:
        self.index.add(memory_id="rm-1", wing="personal", hall="facts", room="r", content="c")
        self.index.flush()
        ok = self.index.remove("rm-1")
        self.assertTrue(ok)
        self.index.flush()
        result = self.index.get("rm-1")
        self.assertIsNone(result)

    def test_update_privacy(self) -> None:
        self.index.add(
            memory_id="up-1",
            wing="personal",
            hall="facts",
            room="r",
            content="c",
            privacy="personal",
        )
        self.index.flush()
        ok = self.index.update_privacy("up-1", "secret")
        self.assertTrue(ok)
        self.index.flush()
        result = self.index.get("up-1")
        self.assertEqual(result["privacy"], "secret")  # type: ignore[index]

    def test_batch_commit(self) -> None:
        for i in range(10):
            self.index.add(
                memory_id=f"batch-{i}", wing="personal", hall="facts", room="r", content=f"c{i}"
            )
        self.index.flush()
        for i in range(10):
            result = self.index.get(f"batch-{i}")
            self.assertIsNotNone(result, f"batch-{i} should exist")


# ═══════════════════════════════════════════════════════════════
# WriteOp 缓冲测试
# ═══════════════════════════════════════════════════════════════

class TestWriteOpBuffer(unittest.TestCase):
    """验证写入缓冲从 partial 改为 WriteOp 后的行为。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.store = DrawerClosetStore(Path(self.tmpdir), write_buffer_threshold=20)

    def test_add_stores_writeop_in_buffer(self) -> None:
        mid = self.store.add(wing="personal", room="test", content="测试写入缓冲")
        self.assertEqual(len(self.store._write_buffer), 2)
        self.assertEqual(self.store._pending_disk_writes, 2)
        drawer_file = self.store._palace_dir / "personal" / "fact" / "test" / "drawer" / f"{mid}.md"
        self.assertFalse(drawer_file.exists())
        self.store.flush()
        self.assertEqual(len(self.store._write_buffer), 0)
        self.assertTrue(drawer_file.exists())
        self.assertIn("测试写入缓冲", drawer_file.read_text(encoding="utf-8"))

    def test_flush_write_buffer_writes_files(self) -> None:
        mid = self.store.add(wing="personal", room="test", content="flush 测试")
        self.store.flush()
        drawer_files = list(Path(self.tmpdir).rglob(f"drawer/{mid}.md"))
        closet_files = list(Path(self.tmpdir).rglob(f"closet/{mid}.md"))
        self.assertEqual(len(drawer_files), 1)
        self.assertEqual(len(closet_files), 1)
        drawer_text = drawer_files[0].read_text(encoding="utf-8")
        closet_text = closet_files[0].read_text(encoding="utf-8")
        self.assertIn("flush 测试", drawer_text)
        self.assertIn("flush 测试", closet_text)

    def test_auto_flush_on_threshold(self) -> None:
        store = DrawerClosetStore(Path(self.tmpdir), write_buffer_threshold=3)
        mids = []
        for i in range(2):
            mid = store.add(wing="personal", room=f"r{i}", content=f"内容{i}")
            mids.append(mid)
            self.assertEqual(len(store._write_buffer), (i + 1) * 2,
                             f"第{i+1}次 add 后 buffer 应保留")
        mid3 = store.add(wing="personal", room="r2", content="内容2")
        self.assertEqual(len(store._write_buffer), 0, "达到阈值后 buffer 应清空")
        drawer_file = Path(self.tmpdir) / "personal" / "fact" / "r2" / "drawer" / f"{mid3}.md"
        self.assertTrue(drawer_file.exists(), "达到阈值后文件应落盘")

    def test_read_after_flush(self) -> None:
        mid = self.store.add(wing="personal", room="test", content="持久化读取测试")
        self.store.flush()
        result = self.store.get(mid)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "持久化读取测试")

    def test_writeop_is_serializable_dataclass(self) -> None:
        self.store.add(wing="personal", room="test", content="序列化检查")
        for op in self.store._write_buffer:
            self.assertIsInstance(op, WriteOp)
            self.assertTrue(hasattr(op, "op_type"))
            self.assertTrue(hasattr(op, "path"))
            self.assertTrue(hasattr(op, "content"))
            self.assertTrue(hasattr(op, "memory_type"))
            self.assertTrue(hasattr(op, "confidence"))
            self.assertTrue(hasattr(op, "privacy"))
            self.assertTrue(hasattr(op, "stored_at"))


# ═══════════════════════════════════════════════════════════════
# Saga 补偿集成测试
# ═══════════════════════════════════════════════════════════════

class TestDrawerClosetSagaCompensation:
    """验证 DrawerClosetStore.add() 的 Saga 补偿语义。"""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.palace_dir = Path(self.tmpdir) / "palace"
        self.store = DrawerClosetStore(
            palace_dir=self.palace_dir,
            write_buffer_threshold=20,
        )

    def test_normal_write_success(self) -> None:
        memory_id = self.store.add(
            wing="test_wing", room="test_room", content="测试内容", memory_type="fact",
        )
        assert memory_id, "memory_id 不应为空"
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0
        assert memory_id in self.store._closet_index, "closet_index 应包含 memory_id"
        assert memory_id in self.store._id_to_path, "id_to_path 应包含 memory_id"

        meta = self.store._meta_store.get(memory_id)
        assert meta is not None, "MetaStore 应能查到记录"
        assert meta["memory_id"] == memory_id
        assert meta["wing"] == "test_wing"
        assert meta["room"] == "test_room"
        assert meta["type"] == "fact"

        self.store.flush()
        drawer_path = self.store._id_to_path[memory_id]
        assert drawer_path.exists(), f"drawer 文件应存在: {drawer_path}"
        drawer_text = drawer_path.read_text(encoding="utf-8")
        assert "测试内容" in drawer_text

        closet_path = drawer_path.parent.parent / "closet" / f"{memory_id}.md"
        assert closet_path.exists(), f"closet 文件应存在: {closet_path}"

    def test_meta_store_failure_compensation(self) -> None:
        with patch.object(
            self.store._meta_store, "add",
            side_effect=RuntimeError("mock meta store failure"),
        ):
            memory_id = self.store.add(
                wing="test_wing", room="test_room", content="补偿测试内容", memory_type="fact",
            )
        assert memory_id, "memory_id 不应为空"
        assert memory_id not in self.store._closet_index, "compensate 应从 closet_index 中移除 memory_id"
        meta = self.store._meta_store.get(memory_id)
        assert meta is None, "MetaStore 不应保留该记录"

    def test_compensation_when_buffer_not_flushed(self) -> None:
        self.store._WRITE_BUFFER_THRESHOLD = 1000
        with patch.object(
            self.store._meta_store, "add",
            side_effect=RuntimeError("mock meta store failure"),
        ):
            memory_id = self.store.add(
                wing="w1", room="r1", content="未 flush 补偿测试", memory_type="t1",
            )
        assert memory_id, "memory_id 不应为空"
        assert memory_id not in self.store._closet_index, "closet_index 应被清理"
        assert memory_id not in self.store._id_to_path, "id_to_path 应被清理"
        type_set = self.store._type_index.get("t1", set())
        assert memory_id not in type_set, "type_index 中 t1 集合应不含 memory_id"
        wing_set = self.store._wing_index.get("w1", set())
        assert memory_id not in wing_set, "wing_index 中 w1 集合应不含 memory_id"
        files = list(self.palace_dir.rglob(f"{memory_id}.md"))
        assert files == [], "未 flush 时不应有文件落盘"

    def test_compensation_after_flush(self) -> None:
        store = DrawerClosetStore(palace_dir=self.palace_dir, write_buffer_threshold=1)
        first_id = store.add(
            wing="w_flush", room="r_flush", content="第一条已 flush 内容", memory_type="fact",
        )
        first_drawer = store._id_to_path[first_id]
        assert first_drawer.exists(), "第一条 drawer 文件应已落盘"

        with patch.object(
            store._meta_store, "add",
            side_effect=RuntimeError("mock meta store failure after flush"),
        ):
            second_id = store.add(
                wing="w_flush", room="r_flush", content="第二条将触发补偿的内容", memory_type="fact",
            )

        second_drawer = self.palace_dir / "w_flush" / "fact" / "r_flush" / "drawer" / f"{second_id}.md"
        second_closet = self.palace_dir / "w_flush" / "fact" / "r_flush" / "closet" / f"{second_id}.md"
        assert not second_drawer.exists(), f"compensate 应删除已落盘的 drawer 文件: {second_drawer}"
        assert not second_closet.exists(), f"compensate 应删除已落盘的 closet 文件: {second_closet}"
        assert second_id not in store._closet_index, "第二条 closet_index 应被清理"
        assert second_id not in store._id_to_path, "第二条 id_to_path 应被清理"
        assert first_id in store._closet_index, "第一条索引不应被误清理"
        assert first_drawer.exists(), "第一条文件不应被误删"
        first_meta = store._meta_store.get(first_id)
        assert first_meta is not None, "第一条 MetaStore 记录应保留"

    def test_compensation_cleans_all_indexes(self) -> None:
        with patch.object(
            self.store._meta_store, "add",
            side_effect=RuntimeError("mock meta store failure"),
        ):
            memory_id = self.store.add(wing="w1", room="r1", content="c1", memory_type="t1")
        assert memory_id not in self.store._closet_index, "closet_index 应被清理"
        assert memory_id not in self.store._id_to_path, "id_to_path 应被清理"
        t1_set = self.store._type_index.get("t1", set())
        assert memory_id not in t1_set, "type_index['t1'] 不应含 memory_id"
        w1_set = self.store._wing_index.get("w1", set())
        assert memory_id not in w1_set, "wing_index['w1'] 不应含 memory_id"
        assert self.store._meta_store.get(memory_id) is None, "MetaStore 不应保留补偿后的记录"
