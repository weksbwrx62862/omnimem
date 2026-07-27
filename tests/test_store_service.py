"""MemoryStoreService 存储服务层测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omnimem.core.store_service import MemoryStoreService


@pytest.fixture
def mock_deps():
    """提供 mock 依赖组件。"""
    store = MagicMock()
    store.add.return_value = "mem-001"

    perception = MagicMock()
    perception._extract_core_fact.return_value = "核心事实内容足够长"

    provenance = MagicMock()
    provenance.track.return_value = {"source": "test", "method": "test"}

    return store, perception, provenance


@pytest.fixture
def service(mock_deps):
    """提供 MemoryStoreService 实例。"""
    store, perception, provenance = mock_deps
    return MemoryStoreService(
        store=store,
        perception=perception,
        provenance=provenance,
        session_id="session-test",
        turn_count=0,
    )


class TestMemoryStoreServiceInit:
    """MemoryStoreService 初始化测试。"""

    def test_default_init(self, mock_deps) -> None:
        """默认初始化属性正确。"""
        store, perception, provenance = mock_deps
        svc = MemoryStoreService(store=store, perception=perception, provenance=provenance)
        assert svc._store is store
        assert svc._perception is perception
        assert svc._provenance is provenance
        assert svc._session_id == ""
        assert svc._turn_count == 0
        assert svc._last_save_turn == 0

    def test_init_with_params(self, mock_deps) -> None:
        """传入参数初始化。"""
        store, perception, provenance = mock_deps
        svc = MemoryStoreService(
            store=store, perception=perception, provenance=provenance,
            session_id="s-1", turn_count=5,
        )
        assert svc._session_id == "s-1"
        assert svc._turn_count == 5


class TestProperties:
    """属性读写测试。"""

    def test_turn_count_property(self, service) -> None:
        """turn_count 属性读写。"""
        assert service.turn_count == 0
        service.turn_count = 10
        assert service.turn_count == 10

    def test_last_save_turn_property(self, service) -> None:
        """last_save_turn 属性读写。"""
        assert service.last_save_turn == 0
        service.last_save_turn = 5
        assert service.last_save_turn == 5


class TestExtractCoreFact:
    """extract_core_fact 方法测试。"""

    def test_delegates_to_perception(self, service) -> None:
        """委托给 perception._extract_core_fact。"""
        result = service.extract_core_fact("用户喜欢Python")
        service._perception._extract_core_fact.assert_called_once_with("用户喜欢Python")
        assert result == "核心事实内容足够长"


class TestStoreCorrection:
    """store_correction 方法测试。"""

    def test_store_with_correction_target(self, service) -> None:
        """有 correction_target 时使用目标值。"""
        signals = MagicMock()
        signals.correction_target = "纠正后的内容"

        result = service.store_correction(signals, "原始内容")
        assert result == "mem-001"
        # 验证 store.add 被调用，且 content 包含"纠正:"
        call_args = service._store.add.call_args
        assert "纠正:" in call_args.kwargs.get("content", call_args[1].get("content", ""))

    def test_store_without_correction_target(self, service) -> None:
        """无 correction_target 时使用 extract_core_fact。"""
        signals = MagicMock()
        signals.correction_target = None

        result = service.store_correction(signals, "用户纠正了某个信息")
        assert result == "mem-001"
        service._perception._extract_core_fact.assert_called()

    def test_store_returns_none_on_failure(self, service) -> None:
        """存储失败时返回 None。"""
        service._store.add.return_value = None
        signals = MagicMock()
        signals.correction_target = "目标"
        result = service.store_correction(signals, "内容")
        assert result is None


class TestStoreReinforcement:
    """store_reinforcement 方法测试。"""

    def test_store_with_reinforcement_target(self, service) -> None:
        """有 reinforcement_target 时使用目标值。"""
        signals = MagicMock()
        signals.reinforcement_target = "强化内容"

        result = service.store_reinforcement(signals, "原始内容")
        assert result == "mem-001"
        call_args = service._store.add.call_args
        assert "确认:" in call_args.kwargs.get("content", call_args[1].get("content", ""))

    def test_store_without_reinforcement_target(self, service) -> None:
        """无 reinforcement_target 时使用 extract_core_fact。"""
        signals = MagicMock()
        signals.reinforcement_target = None

        result = service.store_reinforcement(signals, "用户确认了偏好")
        assert result == "mem-001"


class TestStoreFact:
    """store_fact 方法测试。"""

    def test_store_with_fact_content(self, service) -> None:
        """有 fact_content 时直接使用。"""
        signals = MagicMock()
        signals.fact_content = "事实内容"
        signals.has_preference = False

        result = service.store_fact(signals, "原始内容")
        assert result == "mem-001"

    def test_store_with_preference(self, service) -> None:
        """has_preference=True 时 memory_type 为 preference。"""
        signals = MagicMock()
        signals.fact_content = "偏好内容"
        signals.has_preference = True

        service.store_fact(signals, "原始内容")
        call_args = service._store.add.call_args
        assert call_args.kwargs.get("memory_type", call_args[1].get("memory_type", "")) == "preference"

    def test_store_without_fact_content(self, service) -> None:
        """无 fact_content 时使用 extract_core_fact。"""
        signals = MagicMock()
        signals.fact_content = None
        signals.has_preference = False

        service.store_fact(signals, "用户陈述了一个事实")
        service._perception._extract_core_fact.assert_called()


class TestAutoCheckpoint:
    """auto_checkpoint 方法测试。"""

    def test_checkpoint_not_due(self, service) -> None:
        """轮次间隔不足时不存档。"""
        service.turn_count = 5
        service.last_save_turn = 0
        # save_interval=15，5-0 < 15，不应存档
        result = service.auto_checkpoint("用户内容", save_interval=15)
        assert result is False

    def test_checkpoint_due(self, service) -> None:
        """轮次间隔足够时存档。"""
        service.turn_count = 14  # 下次 +1 = 15
        service.last_save_turn = 0
        result = service.auto_checkpoint("用户内容", save_interval=15)
        assert result is True
        service._store.add.assert_called_once()

    def test_checkpoint_increments_turn(self, service) -> None:
        """存档时 turn_count 递增。"""
        service.turn_count = 0
        service.auto_checkpoint("内容", save_interval=1)
        assert service.turn_count == 1

    def test_checkpoint_with_empty_content(self, service) -> None:
        """空内容时使用 turn 编号作为兜底。"""
        service.turn_count = 14
        service.last_save_turn = 0
        result = service.auto_checkpoint("", save_interval=15)
        assert result is True


class TestEmergencySave:
    """emergency_save 方法测试。"""

    def test_save_user_messages(self, service) -> None:
        """保存用户消息的核心事实。"""
        messages = [
            {"role": "user", "content": "这是一条很长很长的用户消息，包含重要的事实信息"},
            {"role": "assistant", "content": "好的，我记住了"},
        ]
        result = service.emergency_save(messages)
        assert "Emergency saved" in result
        service._store.add.assert_called_once()

    def test_skip_non_user_messages(self, service) -> None:
        """非用户消息应被跳过。"""
        messages = [
            {"role": "assistant", "content": "这是一条很长的助手消息"},
            {"role": "system", "content": "系统消息也很长"},
        ]
        result = service.emergency_save(messages)
        assert "0 core facts" in result

    def test_save_list_content(self, service) -> None:
        """content 为列表格式时正确处理。"""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "这是一条很长很长的消息内容"},
            ]},
        ]
        result = service.emergency_save(messages)
        assert "Emergency saved" in result

    def test_save_limits_to_10(self, service) -> None:
        """最多保存 10 条核心事实。"""
        messages = [
            {"role": "user", "content": f"用户消息内容编号{i}，这是一条很长的事实"}
            for i in range(20)
        ]
        result = service.emergency_save(messages)
        assert "10 core facts" in result


class TestExtractSessionMemories:
    """extract_session_memories 方法测试。"""

    def test_extract_from_user_messages(self, service) -> None:
        """从用户消息中提取隐式记忆。"""
        service._perception.extract_implicit_memories.return_value = ["隐式记忆1"]

        def strip_fn(_x):
            return _x
        def should_store_fn(_x):
            return True
        memorize_fn = MagicMock()

        messages = [
            {"role": "user", "content": "这是一条足够长的用户消息内容，用于测试隐式记忆提取功能是否正常工作，需要超过五十个字符才能通过过滤检查"},
        ]
        count = service.extract_session_memories(messages, strip_fn, should_store_fn, memorize_fn)
        assert count == 1
        memorize_fn.assert_called_once()

    def test_skip_short_messages(self, service) -> None:
        """过短的用户消息被跳过。"""
        def strip_fn(_x):
            return _x
        def should_store_fn(_x):
            return True
        memorize_fn = MagicMock()

        messages = [
            {"role": "user", "content": "短消息"},
        ]
        count = service.extract_session_memories(messages, strip_fn, should_store_fn, memorize_fn)
        assert count == 0

    def test_filter_by_should_store(self, service) -> None:
        """should_store 返回 False 时跳过存储。"""
        service._perception.extract_implicit_memories.return_value = ["记忆1"]

        def strip_fn(_x):
            return _x
        def should_store_fn(_x):
            return False  # 全部拒绝
        memorize_fn = MagicMock()

        messages = [
            {"role": "user", "content": "这是一条足够长的用户消息内容，用于测试隐式记忆提取功能是否正常工作，需要超过五十个字符才能通过过滤检查"},
        ]
        count = service.extract_session_memories(messages, strip_fn, should_store_fn, memorize_fn)
        assert count == 0


class TestStoreDelegation:
    """store_delegation 方法测试。"""

    def test_store_delegation_success(self, service) -> None:
        """成功存储委托记录。"""
        result = service.store_delegation("任务描述", "执行结果", "child-session-123")
        assert result == "mem-001"
        service._store.add.assert_called_once()

    def test_store_delegation_no_child_session(self, service) -> None:
        """无子会话 ID 时使用 unknown。"""
        service.store_delegation("任务", "结果", "")
        call_args = service._store.add.call_args
        assert call_args.kwargs.get("room", call_args[1].get("room", "")) == "unknown"

    def test_store_delegation_failure(self, service) -> None:
        """存储失败时返回 None。"""
        service._store.add.return_value = None
        result = service.store_delegation("任务", "结果")
        assert result is None
