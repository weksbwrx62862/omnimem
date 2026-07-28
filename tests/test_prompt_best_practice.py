"""工具描述固化最佳实践的锁定测试 — 防未来重构丢失写入指引。"""

from __future__ import annotations


class TestMemorizeBestPracticeInSchema:
    """omni_memorize 工具描述(Hermes + MCP 两入口)必须含单句/唯一命名空间指引。"""

    def test_hermes_schema_has_best_practice(self) -> None:
        from omnimem.handlers.schemas import get_tool_schemas

        tool = next(t for t in get_tool_schemas() if t["name"] == "omni_memorize")
        desc = tool["description"]
        assert "BEST PRACTICE" in desc
        assert "single" in desc.lower() and "unique" in desc.lower()
        content_desc = tool["parameters"]["properties"]["content"]["description"]
        assert "SINGLE" in content_desc

    def test_mcp_schema_has_best_practice(self) -> None:
        from omnimem.mcp_server import OmniMemMCPServer

        srv = OmniMemMCPServer.__new__(OmniMemMCPServer)
        tools = OmniMemMCPServer.list_tools(srv)
        tool = next(t for t in tools if t["name"] == "omni_memorize")
        assert "BEST PRACTICE" in tool["description"]
        content_desc = tool["inputSchema"]["properties"]["content"]["description"]
        assert "SINGLE" in content_desc

    def test_two_entrypoints_consistent(self) -> None:
        # 两入口都应含 8 类 memory_type 与最佳实践, 防再次漂移
        from omnimem.handlers.schemas import get_tool_schemas
        from omnimem.mcp_server import OmniMemMCPServer

        h = next(t for t in get_tool_schemas() if t["name"] == "omni_memorize")
        srv = OmniMemMCPServer.__new__(OmniMemMCPServer)
        m = next(t for t in OmniMemMCPServer.list_tools(srv) if t["name"] == "omni_memorize")
        h_types = set(h["parameters"]["properties"]["memory_type"]["enum"])
        m_types = set(m["inputSchema"]["properties"]["memory_type"]["enum"])
        assert h_types == m_types
        assert ("BEST PRACTICE" in h["description"]) and ("BEST PRACTICE" in m["description"])
