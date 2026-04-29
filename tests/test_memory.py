"""L2 结构化记忆模块测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omnimem.memory.drawer_closet import DrawerClosetStore
from omnimem.memory.index import ThreeLevelIndex
from omnimem.memory.wing_room import WingRoomManager


class TestWingRoomManager(unittest.TestCase):
    """WingRoomManager 测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.wrm = WingRoomManager(Path(self.tmpdir))

    def test_resolve_wing(self):
        self.assertEqual(self.wrm.resolve_wing("personal"), "personal")
        self.assertEqual(self.wrm.resolve_wing("project"), "projects")
        self.assertEqual(self.wrm.resolve_wing("team"), "shared")
        self.assertEqual(self.wrm.resolve_wing("secret"), "personal")

    def test_resolve_hall(self):
        self.assertEqual(self.wrm.resolve_hall("fact"), "facts")
        self.assertEqual(self.wrm.resolve_hall("preference"), "preferences")
        self.assertEqual(self.wrm.resolve_hall("correction"), "corrections")

    def test_resolve_room_tech_keyword(self):
        room = self.wrm.resolve_room("使用python编写爬虫", "personal")
        self.assertEqual(room, "python")

    def test_resolve_room_chinese_topic(self):
        room = self.wrm.resolve_room("量子计算是未来技术", "personal")
        self.assertIn("量子", room)

    def test_resolve_room_fallback_hash(self):
        room = self.wrm.resolve_room("xxx", "personal", "fact")
        self.assertTrue(room.startswith("fact-"))

    def test_get_room_path(self):
        path = self.wrm.get_room_path("personal", "facts", "quantum")
        self.assertTrue(path.exists())
        self.assertIn("personal", str(path))
        self.assertIn("facts", str(path))
        self.assertIn("quantum", str(path))

    def test_list_wings(self):
        self.wrm.get_room_path("personal", "facts", "test")
        wings = self.wrm.list_wings()
        self.assertIn("personal", wings)

    def test_sanitize_name(self):
        self.assertEqual(WingRoomManager._sanitize_name("a/b\\c"), "a-b-c")
        self.assertEqual(WingRoomManager._sanitize_name(""), "unnamed")


class TestDrawerClosetStore(unittest.TestCase):
    """DrawerClosetStore 测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = DrawerClosetStore(Path(self.tmpdir))

    def test_add_and_get(self):
        mid = self.store.add(
            wing="personal", room="test", content="测试内容", memory_type="fact", confidence=3
        )
        self.assertTrue(mid)
        result = self.store.get(mid)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "测试内容")
        self.assertEqual(result["type"], "fact")

    def test_add_creates_disk_files(self):
        mid = self.store.add(wing="personal", room="test", content="磁盘测试")
        self.store.flush()
        palace = Path(self.tmpdir)
        drawer_files = list(palace.rglob(f"drawer/{mid}.md"))
        closet_files = list(palace.rglob(f"closet/{mid}.md"))
        self.assertTrue(len(drawer_files) > 0, "Drawer file should exist")
        self.assertTrue(len(closet_files) > 0, "Closet file should exist")

    def test_search_by_type(self):
        self.store.add(wing="personal", room="r1", content="事实1", memory_type="fact")
        self.store.add(wing="personal", room="r2", content="偏好1", memory_type="preference")
        facts = self.store.search(memory_type="fact")
        prefs = self.store.search(memory_type="preference")
        self.assertTrue(any(f["type"] == "fact" for f in facts))
        self.assertTrue(any(p["type"] == "preference" for p in prefs))

    def test_search_by_wing(self):
        self.store.add(wing="personal", room="r1", content="个人记忆")
        self.store.add(wing="shared", room="r2", content="共享记忆")
        personal = self.store.search(wing="personal")
        shared = self.store.search(wing="shared")
        self.assertTrue(all(m["wing"] == "personal" for m in personal))
        self.assertTrue(all(m["wing"] == "shared" for m in shared))

    def test_search_by_content(self):
        self.store.add(wing="personal", room="r1", content="Python是最好的语言")
        results = self.store.search_by_content("Python")
        self.assertTrue(len(results) > 0)
        self.assertIn("Python", results[0]["content"])

    def test_warm_up(self):
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
        self.assertEqual(result["content"], "预热内容")

    def test_update_privacy(self):
        mid = self.store.add(wing="personal", room="r1", content="隐私测试", privacy="personal")
        ok = self.store.update_privacy(mid, "secret", new_wing="personal")
        self.assertTrue(ok)
        result = self.store.get(mid)
        self.assertEqual(result["privacy"], "secret")

    def test_get_nonexistent(self):
        result = self.store.get("nonexistent-id")
        self.assertIsNone(result)

    def test_lru_eviction(self):
        store = DrawerClosetStore(Path(self.tmpdir) / "evict", max_index_size=3)
        ids = []
        for i in range(5):
            ids.append(store.add(wing="personal", room=f"r{i}", content=f"内容{i}"))
        self.assertLessEqual(len(store._closet_index), 3)

    def test_closet_summary_no_newline(self):
        mid = self.store.add(wing="personal", room="r1", content="第一行\n第二行\n第三行")
        result = self.store.get(mid)
        self.assertIn("第一行", result["summary"])
        self.assertNotIn("\n", result["summary"])


class TestThreeLevelIndex(unittest.TestCase):
    """ThreeLevelIndex 测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index = ThreeLevelIndex(Path(self.tmpdir))

    def tearDown(self):
        self.index.close()

    def test_add_and_get(self):
        self.index.add(
            memory_id="idx-001", wing="personal", hall="facts", room="test", content="索引测试"
        )
        self.index.flush()
        result = self.index.get("idx-001")
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "索引测试")

    def test_search_l0(self):
        self.index.add(memory_id="l0-1", wing="personal", hall="facts", room="room-a", content="c1")
        self.index.add(memory_id="l0-2", wing="personal", hall="facts", room="room-b", content="c2")
        self.index.flush()
        rooms = self.index.search_l0(wing="personal", hall="facts")
        self.assertIn("room-a", rooms)
        self.assertIn("room-b", rooms)

    def test_search_l1(self):
        self.index.add(
            memory_id="l1-1", wing="personal", hall="facts", room="r", content="c", type="fact"
        )
        self.index.flush()
        results = self.index.search_l1(wing="personal")
        self.assertTrue(len(results) > 0)

    def test_search_l2_keyword(self):
        self.index.add(
            memory_id="l2-1", wing="personal", hall="facts", room="r", content="量子计算"
        )
        self.index.flush()
        results = self.index.search_l2(keyword="量子")
        self.assertTrue(len(results) > 0)

    def test_remove(self):
        self.index.add(memory_id="rm-1", wing="personal", hall="facts", room="r", content="c")
        self.index.flush()
        ok = self.index.remove("rm-1")
        self.assertTrue(ok)
        self.index.flush()
        result = self.index.get("rm-1")
        self.assertIsNone(result)

    def test_update_privacy(self):
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
        self.assertEqual(result["privacy"], "secret")

    def test_batch_commit(self):
        for i in range(10):
            self.index.add(
                memory_id=f"batch-{i}", wing="personal", hall="facts", room="r", content=f"c{i}"
            )
        self.index.flush()
        for i in range(10):
            result = self.index.get(f"batch-{i}")
            self.assertIsNotNone(result, f"batch-{i} should exist")
