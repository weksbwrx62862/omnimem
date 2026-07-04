"""
EngramBridge 集成测试
"""

import os

# 添加路径
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engram_bridge import (
    Engram,
    create_engram_bridge,
    create_memory_federation,
    create_shared_memory_sync,
)
from plur_client import create_mock_server


@pytest.fixture
def mock_server():
    """创建模拟服务器"""
    return create_mock_server()


@pytest.fixture
def bridge():
    """创建测试用 EngramBridge"""
    return create_engram_bridge(
        instance_id="test-instance-1",
        plur_endpoint="http://localhost:8080"
    )


@pytest.fixture
def sample_omni_memory():
    """创建样本 OmniMem 记忆"""
    return {
        "memory_id": "test-memory-001",
        "content": "用户偏好简洁直接的中文回复",
        "type": "preference",
        "confidence": 5,
        "stored_at": datetime.now().isoformat(),
        "wing": "personal",
        "room": None,
        "privacy": "personal",
        "original_content": "用户偏好简洁直接的中文回复"
    }


class TestEngram:
    """测试 Engram 数据类"""

    def test_engram_creation(self):
        """测试 Engram 创建"""
        engram = Engram(
            id="test-001",
            content="测试内容",
            memory_type="fact",
            confidence=4,
            source_instance="instance-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"key": "value"},
            tags=["test"],
            relationships=[]
        )

        assert engram.id == "test-001"
        assert engram.content == "测试内容"
        assert engram.confidence == 4

    def test_engram_to_dict(self):
        """测试 Engram 转字典"""
        engram = Engram(
            id="test-001",
            content="测试内容",
            memory_type="fact",
            confidence=4,
            source_instance="instance-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"key": "value"},
            tags=["test"],
            relationships=[]
        )

        data = engram.to_dict()
        assert data["id"] == "test-001"
        assert data["content"] == "测试内容"
        assert "created_at" in data

    def test_engram_from_dict(self):
        """测试从字典创建 Engram"""
        data = {
            "id": "test-001",
            "content": "测试内容",
            "memory_type": "fact",
            "confidence": 4,
            "source_instance": "instance-1",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {"key": "value"},
            "tags": ["test"],
            "relationships": []
        }

        engram = Engram.from_dict(data)
        assert engram.id == "test-001"
        assert engram.content == "测试内容"


class TestEngramBridge:
    """测试 EngramBridge"""

    def test_bridge_initialization(self, bridge):
        """测试桥接初始化"""
        assert bridge.instance_id == "test-instance-1"
        assert bridge.plur_endpoint == "http://localhost:8080"
        assert len(bridge._local_cache) == 0
        assert len(bridge._remote_cache) == 0

    def test_convert_to_engram(self, bridge, sample_omni_memory):
        """测试记忆转换"""
        engram = bridge.convert_to_engram(sample_omni_memory)

        assert engram.content == "用户偏好简洁直接的中文回复"
        assert engram.memory_type == "preference"
        assert engram.confidence == 5
        assert engram.source_instance == "test-instance-1"
        assert "preference" in engram.tags

    def test_extract_tags(self, bridge):
        """测试标签提取"""
        content1 = "这是一个教训类的记忆"
        tags1 = bridge._extract_tags(content1)
        assert "lesson" in tags1

        content2 = "用户偏好设置"
        tags2 = bridge._extract_tags(content2)
        assert "preference" in tags2

    @pytest.mark.asyncio
    async def test_sync_to_plur(self, bridge, sample_omni_memory):
        """测试同步到 Plur"""
        memories = [sample_omni_memory]

        # 模拟 Plur 存储
        bridge._remote_cache = {}

        result = await bridge.sync_to_plur(memories)

        assert result.success is True
        assert result.synced_count == 1
        assert result.failed_count == 0

    @pytest.mark.asyncio
    async def test_fetch_from_plur(self, bridge):
        """测试从 Plur 获取"""
        # 预填充缓存
        engram = Engram(
            id="test-001",
            content="测试内容",
            memory_type="fact",
            confidence=4,
            source_instance="instance-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
            tags=["test"],
            relationships=[]
        )
        bridge._remote_cache["test-001"] = engram

        engrams = await bridge.fetch_from_plur(query="测试")

        assert len(engrams) == 1
        assert engrams[0].content == "测试内容"

    @pytest.mark.asyncio
    async def test_resolve_conflict(self, bridge):
        """测试冲突解决"""
        local = Engram(
            id="test-001",
            content="本地内容",
            memory_type="fact",
            confidence=3,
            source_instance="instance-1",
            created_at=datetime.now() - timedelta(hours=1),
            updated_at=datetime.now() - timedelta(hours=1),
            metadata={},
            tags=["local"],
            relationships=[]
        )

        remote = Engram(
            id="test-001",
            content="远程内容",
            memory_type="fact",
            confidence=4,
            source_instance="instance-2",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
            tags=["remote"],
            relationships=[]
        )

        # 测试合并策略
        merged = await bridge.resolve_conflict(local, remote, strategy="merge")

        assert merged.id == "test-001"
        assert merged.confidence == 4  # 取最高置信度
        assert "local" in merged.tags
        assert "remote" in merged.tags

    def test_get_sync_status(self, bridge):
        """测试获取同步状态"""
        status = bridge.get_sync_status()

        assert status["instance_id"] == "test-instance-1"
        assert status["last_sync"] is None
        assert status["local_cache_size"] == 0


class TestSharedMemorySync:
    """测试 SharedMemorySync"""

    @pytest.fixture
    def sync(self, bridge):
        """创建测试用同步管理器"""
        return create_shared_memory_sync(bridge=bridge)

    def test_sync_initialization(self, sync):
        """测试同步管理器初始化"""
        assert sync.auto_sync_interval == 300
        assert sync.conflict_strategy == "merge"
        assert sync._is_running is False

    @pytest.mark.asyncio
    async def test_start_stop(self, sync):
        """测试启动和停止"""
        await sync.start()
        assert sync._is_running is True
        assert sync._sync_task is not None

        await sync.stop()
        assert sync._is_running is False

    @pytest.mark.asyncio
    async def test_perform_sync(self, sync, sample_omni_memory):
        """测试执行同步"""
        # 预填充本地缓存
        engram = sync.bridge.convert_to_engram(sample_omni_memory)
        sync.bridge._local_cache[engram.id] = engram

        result = await sync.perform_sync()

        assert result.success is True
        assert result.synced_count >= 0


class TestMemoryFederation:
    """测试 MemoryFederation"""

    @pytest.fixture
    def federation(self):
        """创建测试用联邦"""
        return create_memory_federation()

    def test_register_instance(self, federation, bridge):
        """测试注册实例"""
        federation.register_instance("instance-1", bridge)

        assert "instance-1" in federation._bridges
        assert len(federation._bridges) == 1

    @pytest.mark.asyncio
    async def test_federated_query(self, federation, bridge):
        """测试联邦查询"""
        # 预填充缓存
        engram = Engram(
            id="test-001",
            content="测试内容",
            memory_type="fact",
            confidence=4,
            source_instance="instance-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
            tags=["test"],
            relationships=[]
        )
        bridge._remote_cache["test-001"] = engram

        federation.register_instance("instance-1", bridge)

        results = await federation.federated_query("测试")

        assert "instance-1" in results
        assert len(results["instance-1"]) == 1

    def test_get_federation_status(self, federation, bridge):
        """测试获取联邦状态"""
        federation.register_instance("instance-1", bridge)

        status = federation.get_federation_status()

        assert status["instance_count"] == 1
        assert "instance-1" in status["registered_instances"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
