"""govern 处理器测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from omnimem.handlers.govern import _scan_memory_conflicts


def _make_provider(memories: list[dict[str, Any]], config: dict[str, Any] | None = None) -> Any:
    """构造仅包含必要属性的模拟 Provider。"""
    return SimpleNamespace(
        _store=SimpleNamespace(search=lambda *_, **__: memories),
        _config=config or {},
    )


def _memory(mid: str, content: str, stored_at: str) -> dict[str, Any]:
    """构造一条可冲突扫描的记忆。"""
    return {
        "memory_id": mid,
        "content": content,
        "type": "fact",
        "stored_at": stored_at,
    }


def test_small_group_all_compared() -> None:
    """组大小低于默认阈值时，应完整比较并返回冲突对。"""
    memories = [
        _memory("m0", "项目编号 使用aws", "2024-01-01T00:00:00Z"),
        _memory("m1", "项目编号 使用腾讯云", "2024-01-02T00:00:00Z"),
        _memory("m2", "项目编号 方案A", "2024-01-03T00:00:00Z"),
    ]
    provider = _make_provider(memories)
    conflicts = _scan_memory_conflicts(provider)
    assert len(conflicts) == 1
    pair = conflicts[0]
    assert pair["memory_a"]["memory_id"] in ("m0", "m1")
    assert pair["memory_b"]["memory_id"] in ("m0", "m1")
    assert pair["conflict_type"] == "semantic_contradiction"


def test_large_group_takes_recent_n() -> None:
    """组大小超过阈值时，仅最近的 N 条参与比较，旧冲突对应被排除。"""
    # 同组内 m0/m1 存在云厂商互斥冲突，但其余记忆时间更近
    memories = [
        _memory("m0", "项目编号 使用aws", "2024-01-01T00:00:00Z"),
        _memory("m1", "项目编号 使用腾讯云", "2024-01-02T00:00:00Z"),
        _memory("m2", "项目编号 初版", "2024-01-05T00:00:00Z"),
        _memory("m3", "项目编号 修订版", "2024-01-06T00:00:00Z"),
        _memory("m4", "项目编号 终稿", "2024-01-07T00:00:00Z"),
    ]
    provider = _make_provider(memories, {"conflict_scan_max_group_size": 3})
    conflicts = _scan_memory_conflicts(provider)
    # 最近 3 条为 m2/m3/m4，它们之间无冲突
    assert conflicts == []


def test_custom_config_threshold() -> None:
    """自定义阈值应生效：每组最多取最近的 N 条。"""
    memories = [
        _memory("m0", "项目编号 使用aws", "2024-01-01T00:00:00Z"),
        _memory("m1", "项目编号 使用腾讯云", "2024-01-02T00:00:00Z"),
        _memory("m2", "项目编号 初版", "2024-01-05T00:00:00Z"),
        _memory("m3", "项目编号 修订版", "2024-01-06T00:00:00Z"),
    ]
    provider = _make_provider(memories, {"conflict_scan_max_group_size": 2})
    conflicts = _scan_memory_conflicts(provider)
    # 按关键词分组后，m0/m1 组大小为 2 仍完整比较并产生冲突；m2/m3 组无冲突
    assert len(conflicts) == 1
    pair = conflicts[0]
    assert pair["memory_a"]["memory_id"] in ("m0", "m1")
    assert pair["memory_b"]["memory_id"] in ("m0", "m1")


def test_conflict_result_structure() -> None:
    """返回的 conflict 列表应保持既定字段结构。"""
    memories = [
        _memory("m0", "项目编号 使用aws", "2024-01-01T00:00:00Z"),
        _memory("m1", "项目编号 使用腾讯云", "2024-01-02T00:00:00Z"),
    ]
    provider = _make_provider(memories)
    conflicts = _scan_memory_conflicts(provider)
    assert len(conflicts) == 1
    pair = conflicts[0]
    assert set(pair.keys()) == {
        "memory_a",
        "memory_b",
        "overlap",
        "negation_in",
        "conflict_type",
    }
    for key in ("memory_a", "memory_b"):
        assert set(pair[key].keys()) == {"memory_id", "content", "type"}


def test_fallback_when_no_config() -> None:
    """Provider 无配置时，应使用默认值 50 并完整比较。"""
    memories = [
        _memory("m0", "项目编号 使用aws", "2024-01-01T00:00:00Z"),
        _memory("m1", "项目编号 使用腾讯云", "2024-01-02T00:00:00Z"),
        _memory("m2", "项目编号 方案A", "2024-01-03T00:00:00Z"),
    ]
    provider = SimpleNamespace(
        _store=SimpleNamespace(search=lambda *_, **__: memories),
        # 不传入 _config，验证 extract_deps 回退到空字典
    )
    conflicts = _scan_memory_conflicts(provider)
    assert len(conflicts) == 1


def test_provider_config_attribute() -> None:
    """Provider 使用 config 而非 _config 属性时，应能读取阈值。"""
    memories = [
        _memory("m0", "项目编号 使用aws", "2024-01-01T00:00:00Z"),
        _memory("m1", "项目编号 使用腾讯云", "2024-01-02T00:00:00Z"),
        _memory("m2", "项目编号 初版", "2024-01-05T00:00:00Z"),
        _memory("m3", "项目编号 修订版", "2024-01-06T00:00:00Z"),
    ]
    provider = SimpleNamespace(
        _store=SimpleNamespace(search=lambda *_, **__: memories),
        config={"conflict_scan_max_group_size": 2},
    )
    conflicts = _scan_memory_conflicts(provider)
    # 与 test_custom_config_threshold 一致：按关键词分组后 m0/m1 产生冲突
    assert len(conflicts) == 1
    pair = conflicts[0]
    assert pair["memory_a"]["memory_id"] in ("m0", "m1")
    assert pair["memory_b"]["memory_id"] in ("m0", "m1")
