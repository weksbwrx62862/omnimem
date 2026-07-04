"""DrawerClosetStore 写入缓冲安全化测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omnimem.memory.drawer_closet import DrawerClosetStore, WriteOp


class TestWriteOpBuffer(unittest.TestCase):
    """验证写入缓冲从 partial 改为 WriteOp 后的行为。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.store = DrawerClosetStore(Path(self.tmpdir), write_buffer_threshold=20)

    def test_add_stores_writeop_in_buffer(self) -> None:
        """add() 后将 WriteOp 存入缓冲区，不立即落盘。"""
        mid = self.store.add(wing="personal", room="test", content="测试写入缓冲")
        # 批量缓冲：每次 add 产生 drawer + closet 两个 WriteOp
        self.assertEqual(len(self.store._write_buffer), 2)
        self.assertEqual(self.store._pending_disk_writes, 2)
        # 验证文件尚未落盘
        drawer_file = self.store._palace_dir / "personal" / "fact" / "test" / "drawer" / f"{mid}.md"
        self.assertFalse(drawer_file.exists())
        # flush 后文件落盘
        self.store.flush()
        self.assertEqual(len(self.store._write_buffer), 0)
        self.assertTrue(drawer_file.exists())
        self.assertIn("测试写入缓冲", drawer_file.read_text(encoding="utf-8"))

    def test_flush_write_buffer_writes_files(self) -> None:
        """_flush_write_buffer() 正确写入 drawer 和 closet 文件。"""
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
        """达到 write_buffer_threshold * 2 个待写入操作时自动 flush。"""
        store = DrawerClosetStore(Path(self.tmpdir), write_buffer_threshold=3)
        mids = []
        for i in range(2):
            mid = store.add(wing="personal", room=f"r{i}", content=f"内容{i}")
            mids.append(mid)
            # 未达到阈值（3*2=6 个 WriteOp），缓冲区保留
            self.assertEqual(len(store._write_buffer), (i + 1) * 2,
                             f"第{i+1}次 add 后 buffer 应保留")
        # 第 3 次 add 后达到阈值，触发自动 flush
        mid3 = store.add(wing="personal", room="r2", content="内容2")
        self.assertEqual(len(store._write_buffer), 0, "达到阈值后 buffer 应清空")
        drawer_file = Path(self.tmpdir) / "personal" / "fact" / "r2" / "drawer" / f"{mid3}.md"
        self.assertTrue(drawer_file.exists(), "达到阈值后文件应落盘")

    def test_read_after_flush(self) -> None:
        """写入内容可被正确读取。"""
        mid = self.store.add(wing="personal", room="test", content="持久化读取测试")
        self.store.flush()
        result = self.store.get(mid)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "持久化读取测试")  # type: ignore[index-error]

    def test_writeop_is_serializable_dataclass(self) -> None:
        """WriteOp 为 dataclass，字段可访问且不含闭包函数。"""
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


if __name__ == "__main__":
    unittest.main()
