"""DrawerClosetStore Saga 补偿语义集成测试。

验证 P0-4:使用真实 DrawerClosetStore(非 mock)验证 Saga 失败时的补偿行为,
覆盖已 flush 和未 flush 两种缓冲状态。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from omnimem.memory.drawer_closet import DrawerClosetStore


class TestDrawerClosetSagaCompensation:
    """验证 DrawerClosetStore.add() 的 Saga 补偿语义。"""

    def setup_method(self) -> None:
        """每个测试用例使用独立的临时目录 + 真实 SQLite MetaStore。

        注意:MetaStore 由 DrawerClosetStore 内部创建(palace_dir/.meta 目录),
        无需外部注入。这里直接通过 DrawerClosetStore 构造真实环境。
        """
        self.tmpdir = tempfile.mkdtemp()
        self.palace_dir = Path(self.tmpdir) / "palace"
        # 默认 write_buffer_threshold=20,避免在测试中意外触发自动 flush
        self.store = DrawerClosetStore(
            palace_dir=self.palace_dir,
            write_buffer_threshold=20,
        )

    # ─── 正常路径 ─────────────────────────────────────────────

    def test_normal_write_success(self) -> None:
        """正常写入:Saga 全部步骤成功,数据落盘且索引就绪。"""
        memory_id = self.store.add(
            wing="test_wing",
            room="test_room",
            content="测试内容",
            memory_type="fact",
        )

        # 1. 返回有效的 memory_id
        assert memory_id, "memory_id 不应为空"
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0

        # 2. 内存索引已就绪
        assert memory_id in self.store._closet_index, "closet_index 应包含 memory_id"
        assert memory_id in self.store._id_to_path, "id_to_path 应包含 memory_id"

        # 3. MetaStore 已成功写入(真实 SQLite)
        meta = self.store._meta_store.get(memory_id)
        assert meta is not None, "MetaStore 应能查到记录"
        assert meta["memory_id"] == memory_id
        assert meta["wing"] == "test_wing"
        assert meta["room"] == "test_room"
        assert meta["type"] == "fact"

        # 4. 显式 flush 后文件落盘
        self.store.flush()

        # 5. drawer 文件存在
        drawer_path = self.store._id_to_path[memory_id]
        assert drawer_path.exists(), f"drawer 文件应存在: {drawer_path}"
        drawer_text = drawer_path.read_text(encoding="utf-8")
        assert "测试内容" in drawer_text

        # 6. closet 文件存在(同目录下 closet 子目录)
        closet_path = drawer_path.parent.parent / "closet" / f"{memory_id}.md"
        assert closet_path.exists(), f"closet 文件应存在: {closet_path}"

    # ─── Saga 失败补偿 ─────────────────────────────────────────

    def test_meta_store_failure_compensation(self) -> None:
        """MetaStore.add 抛异常时,Saga 内部捕获并触发补偿,索引被清理。"""
        # mock MetaStore.add 抛异常,触发 Saga 补偿
        with patch.object(
            self.store._meta_store,
            "add",
            side_effect=RuntimeError("mock meta store failure"),
        ):
            # Saga 内部捕获异常,不应抛出
            memory_id = self.store.add(
                wing="test_wing",
                room="test_room",
                content="补偿测试内容",
                memory_type="fact",
            )

        # 1. 不抛异常(已通过到达此处证明)
        assert memory_id, "memory_id 不应为空"

        # 2. 补偿已清理 closet_index
        assert memory_id not in self.store._closet_index, (
            "compensate 应从 closet_index 中移除 memory_id"
        )

        # 3. MetaStore 中查不到记录(从未成功写入或被 compensate delete 清理)
        meta = self.store._meta_store.get(memory_id)
        assert meta is None, "MetaStore 不应保留该记录"

        # 注意:此时 write_buffer 默认阈值较高,文件可能尚未 flush,
        # compensate 对未落盘文件安全失败(unlink missing_ok=True)。

    def test_compensation_when_buffer_not_flushed(self) -> None:
        """缓冲未 flush 时补偿:四个内存索引全部被清理。

        设置较高的 _WRITE_BUFFER_THRESHOLD,确保 add() 不会触发自动 flush,
        此时 drawer/closet 文件尚未落盘,compensate 仅清理内存索引。
        """
        # 设置较高的阈值,确保不会自动 flush
        self.store._WRITE_BUFFER_THRESHOLD = 1000

        with patch.object(
            self.store._meta_store,
            "add",
            side_effect=RuntimeError("mock meta store failure"),
        ):
            memory_id = self.store.add(
                wing="w1",
                room="r1",
                content="未 flush 补偿测试",
                memory_type="t1",
            )

        # 1. 不抛异常
        assert memory_id, "memory_id 不应为空"

        # 2. 四个内存索引全部被清理
        assert memory_id not in self.store._closet_index, "closet_index 应被清理"
        assert memory_id not in self.store._id_to_path, "id_to_path 应被清理"

        # 3. 二级倒排索引也被清理
        type_set = self.store._type_index.get("t1", set())
        assert memory_id not in type_set, "type_index 中 t1 集合应不含 memory_id"

        wing_set = self.store._wing_index.get("w1", set())
        assert memory_id not in wing_set, "wing_index 中 w1 集合应不含 memory_id"

        # 4. 文件应未落盘(因为没触发 flush)
        # 通过 rglob 检查 palace_dir 下不存在该 memory_id 的文件
        files = list(self.palace_dir.rglob(f"{memory_id}.md"))
        assert files == [], "未 flush 时不应有文件落盘"

    def test_compensation_after_flush(self) -> None:
        """flush 后补偿:文件已落盘,compensate 能删除文件。

        实现思路:
          1. 设置 write_buffer_threshold=1,使 _pending_disk_writes >= 2 时触发自动 flush
          2. 第一次 add:正常写入,触发 flush,文件落盘,MetaStore 写入成功
          3. mock MetaStore.add 抛异常,第二次 add:文件已 flush 落盘,
             Saga 调用 _write_meta 时抛异常,触发 compensate,删除已落盘文件
          4. 验证第二次的 drawer/closet 文件被删除
        """
        # 重新构造 store,threshold=1,这样 1 条 add 后(2 个 WriteOp)就触发自动 flush
        store = DrawerClosetStore(
            palace_dir=self.palace_dir,
            write_buffer_threshold=1,
        )

        # 第一次正常写入(触发自动 flush,文件落盘)
        first_id = store.add(
            wing="w_flush",
            room="r_flush",
            content="第一条已 flush 内容",
            memory_type="fact",
        )
        # 验证第一条已落盘
        first_drawer = store._id_to_path[first_id]
        assert first_drawer.exists(), "第一条 drawer 文件应已落盘"

        # mock MetaStore.add 抛异常,第二次 add 触发 Saga 补偿
        with patch.object(
            store._meta_store,
            "add",
            side_effect=RuntimeError("mock meta store failure after flush"),
        ):
            second_id = store.add(
                wing="w_flush",
                room="r_flush",
                content="第二条将触发补偿的内容",
                memory_type="fact",
            )

        # 第二次的文件应已落盘(因为 threshold=1 触发 flush),
        # 然后 compensate 应删除它们
        second_drawer = self.palace_dir / "w_flush" / "fact" / "r_flush" / "drawer" / f"{second_id}.md"
        second_closet = self.palace_dir / "w_flush" / "fact" / "r_flush" / "closet" / f"{second_id}.md"

        assert not second_drawer.exists(), (
            f"compensate 应删除已落盘的 drawer 文件: {second_drawer}"
        )
        assert not second_closet.exists(), (
            f"compensate 应删除已落盘的 closet 文件: {second_closet}"
        )

        # 第二条的内存索引也应被清理
        assert second_id not in store._closet_index, "第二条 closet_index 应被清理"
        assert second_id not in store._id_to_path, "第二条 id_to_path 应被清理"

        # 第一条应仍完好(补偿只针对失败的 memory_id)
        assert first_id in store._closet_index, "第一条索引不应被误清理"
        assert first_drawer.exists(), "第一条文件不应被误删"
        first_meta = store._meta_store.get(first_id)
        assert first_meta is not None, "第一条 MetaStore 记录应保留"

    def test_compensation_cleans_all_indexes(self) -> None:
        """补偿清理所有四个内存索引:closet_index、id_to_path、type_index、wing_index。"""
        with patch.object(
            self.store._meta_store,
            "add",
            side_effect=RuntimeError("mock meta store failure"),
        ):
            memory_id = self.store.add(
                wing="w1",
                room="r1",
                content="c1",
                memory_type="t1",
            )

        # 1. closet_index 被清理
        assert memory_id not in self.store._closet_index, (
            "closet_index 应被清理"
        )

        # 2. id_to_path 被清理
        assert memory_id not in self.store._id_to_path, (
            "id_to_path 应被清理"
        )

        # 3. type_index 对应 type 的 set 不含 memory_id
        t1_set = self.store._type_index.get("t1", set())
        assert memory_id not in t1_set, "type_index['t1'] 不应含 memory_id"

        # 4. wing_index 对应 wing 的 set 不含 memory_id
        w1_set = self.store._wing_index.get("w1", set())
        assert memory_id not in w1_set, "wing_index['w1'] 不应含 memory_id"

        # 5. MetaStore 中也不存在该记录
        assert self.store._meta_store.get(memory_id) is None, (
            "MetaStore 不应保留补偿后的记录"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
