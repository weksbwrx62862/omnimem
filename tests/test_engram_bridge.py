"""EngramBridge 共享记忆桥接层测试。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from omnimem.core.engram_bridge import (
    Engram,
    EngramBridge,
    MemoryFederation,
    SharedMemorySync,
    create_engram_bridge,
    create_memory_federation,
    create_shared_memory_sync,
)


def _make_engram(
    eid: str = "eng-001",
    content: str = "测试内容",
    confidence: float = 3.0,
    source: str = "instance-1",
) -> Engram:
    """创建测试用 Engram 实例。"""
    now = datetime.now()
    return Engram(
        id=eid,
        content=content,
        memory_type="fact",
        confidence=confidence,
        source_instance=source,
        created_at=now,
        updated_at=now,
        metadata={},
        tags=[],
        relationships=[],
    )


class TestEngramDataclass:
    """Engram 数据类测试。"""

    def test_to_dict(self) -> None:
        """to_dict 应包含所有字段。"""
        engram = _make_engram()
        d = engram.to_dict()
        assert d["id"] == "eng-001"
        assert d["content"] == "测试内容"
        assert "created_at" in d
        assert "updated_at" in d

    def test_from_dict(self) -> None:
        """from_dict 应正确还原 Engram。"""
        engram = _make_engram()
        d = engram.to_dict()
        restored = Engram.from_dict(d)
        assert restored.id == engram.id
        assert restored.content == engram.content
        assert restored.confidence == engram.confidence

    def test_from_dict_with_defaults(self) -> None:
        """from_dict 缺少可选字段时使用默认值。"""
        d = {
            "id": "eng-002",
            "content": "内容",
            "memory_type": "fact",
            "confidence": 3.0,
            "source_instance": "inst-1",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        engram = Engram.from_dict(d)
        assert engram.metadata == {}
        assert engram.tags == []
        assert engram.relationships == []


class TestEngramBridgeInit:
    """EngramBridge 初始化测试。"""

    def test_default_init(self) -> None:
        """默认初始化。"""
        bridge = EngramBridge(instance_id="test-inst")
        assert bridge.instance_id == "test-inst"
        assert bridge.plur_endpoint == "http://localhost:8080"
        assert bridge.sync_interval == 300
        assert bridge.auto_sync is True
        assert len(bridge._local_cache) == 0

    def test_custom_init(self) -> None:
        """自定义参数初始化。"""
        bridge = EngramBridge(
            instance_id="inst-2",
            plur_endpoint="http://plur:9090",
            sync_interval=600,
            auto_sync=False,
        )
        assert bridge.plur_endpoint == "http://plur:9090"
        assert bridge.sync_interval == 600
        assert bridge.auto_sync is False


class TestConvertToEngram:
    """convert_to_engram 方法测试。"""

    def test_convert_with_memory_id(self) -> None:
        """有 memory_id 时使用原始 ID。"""
        bridge = EngramBridge(instance_id="inst-1")
        omni = {"memory_id": "mem-001", "content": "测试内容", "confidence": 4}
        engram = bridge.convert_to_engram(omni)
        assert engram.id == "mem-001"
        assert engram.content == "测试内容"
        assert engram.confidence == 4.0

    def test_convert_without_memory_id(self) -> None:
        """无 memory_id 时自动生成 ID。"""
        bridge = EngramBridge(instance_id="inst-1")
        omni = {"content": "测试内容"}
        engram = bridge.convert_to_engram(omni)
        assert engram.id  # 非空
        assert len(engram.id) == 16  # SHA256 前16位

    def test_convert_preserves_metadata(self) -> None:
        """转换时保留 wing/room/privacy 等元数据。"""
        bridge = EngramBridge(instance_id="inst-1")
        omni = {"content": "内容", "wing": "personal", "room": "fact", "privacy": "personal"}
        engram = bridge.convert_to_engram(omni)
        assert engram.metadata["wing"] == "personal"
        assert engram.metadata["room"] == "fact"


class TestExtractTags:
    """_extract_tags 方法测试。"""

    def test_extract_correction_tag(self) -> None:
        """内容含"纠正"时提取 correction 标签。"""
        bridge = EngramBridge(instance_id="inst-1")
        tags = bridge._extract_tags("用户纠正了之前的说法")
        assert "correction" in tags

    def test_extract_preference_tag(self) -> None:
        """内容含"偏好"时提取 preference 标签。"""
        bridge = EngramBridge(instance_id="inst-1")
        tags = bridge._extract_tags("用户的偏好是暗色主题")
        assert "preference" in tags

    def test_no_tags(self) -> None:
        """内容无匹配关键词时返回空列表。"""
        bridge = EngramBridge(instance_id="inst-1")
        tags = bridge._extract_tags("普通内容没有关键词")
        assert tags == []


class TestSyncToPlur:
    """sync_to_plur 方法测试。"""

    @pytest.mark.asyncio
    async def test_sync_success(self) -> None:
        """成功同步记忆到 Plur。"""
        bridge = EngramBridge(instance_id="inst-1")
        memories = [
            {"memory_id": "mem-1", "content": "事实1"},
            {"memory_id": "mem-2", "content": "事实2"},
        ]
        result = await bridge.sync_to_plur(memories)
        assert result.success is True
        assert result.synced_count == 2
        assert result.failed_count == 0

    @pytest.mark.asyncio
    async def test_sync_partial_failure(self) -> None:
        """部分记忆同步失败。"""
        bridge = EngramBridge(instance_id="inst-1")
        # 模拟 _store_to_plur 第二次调用失败
        call_count = {"n": 0}
        original_store = bridge._store_to_plur

        async def mock_store(engram):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("存储失败")
            return await original_store(engram)

        bridge._store_to_plur = mock_store
        memories = [
            {"memory_id": "mem-1", "content": "事实1"},
            {"memory_id": "mem-2", "content": "事实2"},
        ]
        result = await bridge.sync_to_plur(memories)
        assert result.failed_count == 1

    @pytest.mark.asyncio
    async def test_sync_empty_list(self) -> None:
        """空记忆列表同步。"""
        bridge = EngramBridge(instance_id="inst-1")
        result = await bridge.sync_to_plur([])
        assert result.success is True
        assert result.synced_count == 0


class TestFetchFromPlur:
    """fetch_from_plur 方法测试。"""

    @pytest.mark.asyncio
    async def test_fetch_all(self) -> None:
        """无过滤条件时返回全部缓存。"""
        bridge = EngramBridge(instance_id="inst-1")
        bridge._remote_cache["eng-1"] = _make_engram("eng-1", "内容1")
        bridge._remote_cache["eng-2"] = _make_engram("eng-2", "内容2")

        result = await bridge.fetch_from_plur()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_fetch_with_query(self) -> None:
        """按关键词过滤。"""
        bridge = EngramBridge(instance_id="inst-1")
        bridge._remote_cache["eng-1"] = _make_engram("eng-1", "Python开发")
        bridge._remote_cache["eng-2"] = _make_engram("eng-2", "Java开发")

        result = await bridge.fetch_from_plur(query="Python")
        assert len(result) == 1
        assert result[0].content == "Python开发"

    @pytest.mark.asyncio
    async def test_fetch_with_limit(self) -> None:
        """限制返回数量。"""
        bridge = EngramBridge(instance_id="inst-1")
        for i in range(10):
            bridge._remote_cache[f"eng-{i}"] = _make_engram(f"eng-{i}", f"内容{i}")

        result = await bridge.fetch_from_plur(limit=3)
        assert len(result) == 3


class TestResolveConflict:
    """resolve_conflict 方法测试。"""

    @pytest.mark.asyncio
    async def test_local_strategy(self) -> None:
        """local 策略返回本地版本。"""
        bridge = EngramBridge(instance_id="inst-1")
        local = _make_engram("eng-1", "本地内容")
        remote = _make_engram("eng-1", "远程内容")
        result = await bridge.resolve_conflict(local, remote, strategy="local")
        assert result.content == "本地内容"

    @pytest.mark.asyncio
    async def test_remote_strategy(self) -> None:
        """remote 策略返回远程版本。"""
        bridge = EngramBridge(instance_id="inst-1")
        local = _make_engram("eng-1", "本地内容")
        remote = _make_engram("eng-1", "远程内容")
        result = await bridge.resolve_conflict(local, remote, strategy="remote")
        assert result.content == "远程内容"

    @pytest.mark.asyncio
    async def test_highest_confidence_strategy(self) -> None:
        """highest_confidence 策略返回置信度高的版本。"""
        bridge = EngramBridge(instance_id="inst-1")
        local = _make_engram("eng-1", "低置信度", confidence=2.0)
        remote = _make_engram("eng-1", "高置信度", confidence=5.0)
        result = await bridge.resolve_conflict(local, remote, strategy="highest_confidence")
        assert result.content == "高置信度"

    @pytest.mark.asyncio
    async def test_merge_strategy(self) -> None:
        """merge 策略合并标签和元数据。"""
        bridge = EngramBridge(instance_id="inst-1")
        local = _make_engram("eng-1", "本地内容", confidence=3.0)
        local.tags = ["tag1"]
        local.metadata = {"key1": "val1"}
        remote = _make_engram("eng-1", "远程内容", confidence=5.0)
        remote.updated_at = local.updated_at + timedelta(seconds=1)
        remote.tags = ["tag2"]
        remote.metadata = {"key2": "val2"}

        result = await bridge.resolve_conflict(local, remote, strategy="merge")
        assert result.confidence == 5.0  # 取最大
        assert "tag1" in result.tags
        assert "tag2" in result.tags

    @pytest.mark.asyncio
    async def test_unknown_strategy_raises(self) -> None:
        """未知策略抛出 ValueError。"""
        bridge = EngramBridge(instance_id="inst-1")
        local = _make_engram()
        remote = _make_engram()
        with pytest.raises(ValueError, match="Unknown conflict resolution strategy"):
            await bridge.resolve_conflict(local, remote, strategy="invalid")


class TestGetSyncStatus:
    """get_sync_status 方法测试。"""

    def test_initial_status(self) -> None:
        """初始状态。"""
        bridge = EngramBridge(instance_id="inst-1")
        status = bridge.get_sync_status()
        assert status["instance_id"] == "inst-1"
        assert status["last_sync"] is None
        assert status["local_cache_size"] == 0
        assert status["remote_cache_size"] == 0


class TestFactoryFunctions:
    """工厂函数测试。"""

    def test_create_engram_bridge(self) -> None:
        """create_engram_bridge 创建正确实例。"""
        bridge = create_engram_bridge(instance_id="test", plur_endpoint="http://test:8080")
        assert isinstance(bridge, EngramBridge)
        assert bridge.instance_id == "test"

    def test_create_shared_memory_sync(self) -> None:
        """create_shared_memory_sync 创建正确实例。"""
        bridge = create_engram_bridge(instance_id="test")
        sync = create_shared_memory_sync(bridge=bridge, auto_sync_interval=60)
        assert isinstance(sync, SharedMemorySync)
        assert sync.auto_sync_interval == 60

    def test_create_memory_federation(self) -> None:
        """create_memory_federation 创建正确实例。"""
        fed = create_memory_federation()
        assert isinstance(fed, MemoryFederation)


class TestMemoryFederation:
    """MemoryFederation 联邦查询测试。"""

    @pytest.mark.asyncio
    async def test_register_and_query(self) -> None:
        """注册实例后可联邦查询。"""
        fed = create_memory_federation()
        bridge = create_engram_bridge(instance_id="inst-1")
        bridge._remote_cache["eng-1"] = _make_engram("eng-1", "Python知识")
        fed.register_instance("inst-1", bridge)

        results = await fed.federated_query("Python")
        assert "inst-1" in results
        assert len(results["inst-1"]) == 1

    @pytest.mark.asyncio
    async def test_aggregate_memories(self) -> None:
        """聚合多实例记忆并去重。"""
        fed = create_memory_federation()
        bridge1 = create_engram_bridge(instance_id="inst-1")
        bridge1._remote_cache["eng-1"] = _make_engram("eng-1", "知识1", confidence=5.0)
        fed.register_instance("inst-1", bridge1)

        result = await fed.aggregate_memories("知识")
        assert len(result) >= 1

    def test_get_federation_status(self) -> None:
        """获取联邦状态。"""
        fed = create_memory_federation()
        bridge = create_engram_bridge(instance_id="inst-1")
        fed.register_instance("inst-1", bridge)

        status = fed.get_federation_status()
        assert "inst-1" in status["registered_instances"]
        assert status["instance_count"] == 1
