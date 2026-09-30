"""压缩引擎测试 — 微压缩 + 行压缩 + 流水线 + LLM 摘要 + Mermaid 画布。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from omnimem.compression.line_compress import structured_line_compress
from omnimem.compression.llm_summary import (
    StructuredSummary,
    _extract_without_llm,
    _parse_llm_response,
    llm_summarize,
)
from omnimem.compression.micro import microcompact
from omnimem.compression.pipeline import CompressionPipeline


# ═══════════════════════════════════════════════════════════════
# 微压缩测试
# ═══════════════════════════════════════════════════════════════

class TestMicrocompact:
    def test_removes_duplicate_lines(self):
        lines = ["hello", "hello", "world"]
        result = microcompact(lines)
        assert len(result) == 2
        assert result[0] == "hello"
        assert result[1] == "world"

    def test_removes_noise(self):
        lines = ["---", "===", "valid content"]
        result = microcompact(lines)
        assert len(result) == 1
        assert "valid content" in result[0]

    def test_preserves_key_markers(self):
        lines = ["DECISION: use Python", "normal line"]
        result = microcompact(lines)
        assert any("DECISION" in line for line in result)

    def test_preserves_empty_line_structure(self):
        lines = ["line1", "", "", "line2"]
        result = microcompact(lines)
        assert "" in result

    def test_empty_input(self):
        result = microcompact([])
        assert result == []

    def test_short_lines_filtered(self):
        lines = ["ab", "this is long enough"]
        result = microcompact(lines)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════
# 结构化行压缩测试
# ═══════════════════════════════════════════════════════════════

class TestStructuredLineCompress:
    def test_merges_similar_lines(self):
        lines = ["用户喜欢Python", "用户偏好Python", "用户爱好编程"]
        result = structured_line_compress(lines)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_empty_lines_preserved(self):
        lines = ["line1", "", "line2"]
        result = structured_line_compress(lines)
        assert any("line1" in line for line in result)
        assert any("line2" in line for line in result)

    def test_empty_input(self):
        result = structured_line_compress([])
        assert result == []

    def test_long_line_truncated(self):
        long_line = "a" * 300
        lines = [long_line]
        result = structured_line_compress(lines)
        assert len(result[0]) <= 200

    def test_redundant_phrases_removed(self):
        lines = ["I think this is important", "basically it works"]
        result = structured_line_compress(lines)
        combined = " ".join(result)
        assert "I think " not in combined
        assert "basically " not in combined


# ═══════════════════════════════════════════════════════════════
# 压缩流水线测试
# ═══════════════════════════════════════════════════════════════

class TestCompressionPipeline:
    def test_pipeline_runs(self):
        pipeline = CompressionPipeline()
        result = pipeline.compress("测试文本 " * 100)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_pipeline_without_llm(self):
        pipeline = CompressionPipeline(llm_call_fn=None)
        result = pipeline.compress("some content to compress")
        assert isinstance(result, str)

    def test_pipeline_short_content(self):
        pipeline = CompressionPipeline()
        result = pipeline.compress("short")
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════
# LLM 摘要测试
# ═══════════════════════════════════════════════════════════════

class TestStructuredSummary:
    def test_default_values(self):
        s = StructuredSummary()
        assert s.goal == ""
        assert s.progress == ""
        assert s.decisions == ""
        assert s.key_info == ""
        assert s.open_issues == ""
        assert s.next_steps == ""

    def test_to_text_skips_empty_fields(self):
        s = StructuredSummary(goal="test goal", progress="done")
        text = s.to_text()
        assert "Goal: test goal" in text
        assert "Progress: done" in text
        assert "Decisions" not in text

    def test_to_text_all_fields(self):
        s = StructuredSummary(
            goal="g", progress="p", decisions="d",
            key_info="k", open_issues="o", next_steps="n",
        )
        text = s.to_text()
        assert text.count("\n") == 5

    def test_to_dict(self):
        s = StructuredSummary(goal="g", key_info="k")
        d = s.to_dict()
        assert d["goal"] == "g"
        assert d["key_info"] == "k"
        assert d["progress"] == ""


class TestParseLLMResponse:
    def test_valid_json(self):
        data = {"goal": "build app", "progress": "50%", "decisions": "use Python",
                "key_info": "fastapi", "open_issues": "testing", "next_steps": "deploy"}
        result = _parse_llm_response(json.dumps(data))
        assert result.goal == "build app"
        assert result.progress == "50%"

    def test_json_in_markdown_block(self):
        data = {"goal": "test", "progress": "done"}
        wrapped = f"```json\n{json.dumps(data)}\n```"
        result = _parse_llm_response(wrapped)
        assert result.goal == "test"

    def test_malformed_json_fallback(self):
        result = _parse_llm_response("this is not json at all")
        assert result.key_info == "this is not json at all"

    def test_partial_json_missing_keys(self):
        data = {"goal": "only goal"}
        result = _parse_llm_response(json.dumps(data))
        assert result.goal == "only goal"
        assert result.progress == ""


class TestExtractWithoutLLM:
    def test_extracts_first_line_as_goal(self):
        result = _extract_without_llm("Build a web app\nSome details")
        assert result.goal == "Build a web app"

    def test_extracts_decisions(self):
        msg = "决定使用Python开发\n选择FastAPI框架\n普通内容"
        result = _extract_without_llm(msg)
        assert "Python" in result.decisions

    def test_empty_input(self):
        result = _extract_without_llm("")
        assert result.goal == ""


class TestLLMSummarize:
    def test_without_llm_fn(self):
        result = llm_summarize("some messages", llm_call_fn=None)
        assert isinstance(result, StructuredSummary)

    def test_with_mock_llm(self):
        mock_fn = MagicMock(return_value=json.dumps(
            {"goal": "test", "progress": "done", "decisions": "",
             "key_info": "", "open_issues": "", "next_steps": ""}
        ))
        result = llm_summarize("messages", llm_call_fn=mock_fn)
        assert result.goal == "test"
        mock_fn.assert_called_once()

    def test_llm_failure_fallback(self):
        mock_fn = MagicMock(side_effect=RuntimeError("LLM down"))
        result = llm_summarize("fallback test", llm_call_fn=mock_fn)
        assert isinstance(result, StructuredSummary)
        assert result.progress == "See conversation history"


# ═══════════════════════════════════════════════════════════════
# Mermaid 画布测试
# ═══════════════════════════════════════════════════════════════

class TestMermaidCanvas:
    """测试 MermaidCanvas：日志卸载、Mermaid 生成、refs 清理、索引持久化。"""

    @pytest.fixture
    def canvas(self, omni_tmp_path):
        from omnimem.compression.mermaid_canvas import MermaidCanvas
        return MermaidCanvas(omni_tmp_path)

    def test_offload_and_compress(self, canvas):
        logs = [
            {"tool_name": "search_files", "status": "success"},
            {"tool_name": "read_file", "status": "success"},
            {"tool_name": "patch", "status": "error"},
        ]
        mermaid, mapping = canvas.offload_and_compress(logs, "s1")
        assert "graph TD" in mermaid
        assert "T1" in mermaid
        assert "T2" in mermaid
        assert "T3" in mermaid
        assert len(mapping) == 3

    def test_recover_by_node_id(self, canvas):
        logs = [{"tool_name": "test", "status": "ok", "data": "hello"}]
        _, mapping = canvas.offload_and_compress(logs, "s1")
        text = canvas.recover_by_node_id("T1")
        assert text is not None
        assert "test" in text

    def test_recover_missing_node(self, canvas):
        assert canvas.recover_by_node_id("nonexistent") is None

    def test_stale_refs_cleanup(self, omni_tmp_path):
        from omnimem.compression.mermaid_canvas import MermaidCanvas
        config = {"max_refs_age_days": 1}
        canvas = MermaidCanvas(omni_tmp_path, config=config)
        old_file = canvas._refs_dir / "old.md"
        old_file.write_text("old data")
        import os
        os.utime(old_file, (0, 0))
        new_file = canvas._refs_dir / "new.md"
        new_file.write_text("new data")
        canvas._cleanup_stale_refs()
        assert not old_file.exists()
        assert new_file.exists()

    def test_index_persistence(self, canvas):
        logs = [{"tool_name": "test", "status": "ok"}]
        canvas.offload_and_compress(logs, "s1")
        index_path = canvas._refs_dir / "_index.json"
        assert index_path.exists()
        index = json.loads(index_path.read_text())
        assert "T1" in index

    def test_is_tool_log(self, canvas):
        assert canvas.is_tool_log("tool_call: search_files") is True
        assert canvas.is_tool_log("command: ls -la") is True
        assert canvas.is_tool_log("工具调用: 读取文件") is True
        assert canvas.is_tool_log("这是一个普通对话") is False

    def test_is_tool_log_custom_pattern(self, omni_tmp_path):
        from omnimem.compression.mermaid_canvas import MermaidCanvas
        config = {"mermaid_tool_log_patterns": [r"MY_TOOL_TAG"]}
        canvas = MermaidCanvas(omni_tmp_path, config=config)
        assert canvas.is_tool_log("MY_TOOL_TAG: something") is True
        assert canvas.is_tool_log("tool_call: normal") is False

    def test_empty_logs(self, canvas):
        mermaid, mapping = canvas.offload_and_compress([], "s1")
        assert mermaid == ""
        assert mapping == {}


class TestCompressionPipelineMermaid:
    """测试 CompressionPipeline 的 Mermaid 分支。"""

    def test_tool_log_goes_mermaid(self, omni_tmp_path):
        from omnimem.compression.mermaid_canvas import MermaidCanvas
        from omnimem.compression.pipeline import CompressionPipeline

        canvas = MermaidCanvas(omni_tmp_path)
        pipeline = CompressionPipeline(llm_call_fn=None, mermaid_canvas=canvas, session_key="test")
        content = "tool_call: search_files\nresult: found 3 files"
        result = pipeline.compress(content)
        assert "graph TD" in result

    def test_normal_content_skips_mermaid(self, omni_tmp_path):
        from omnimem.compression.mermaid_canvas import MermaidCanvas
        from omnimem.compression.pipeline import CompressionPipeline

        canvas = MermaidCanvas(omni_tmp_path)
        pipeline = CompressionPipeline(llm_call_fn=None, mermaid_canvas=canvas, session_key="test")
        content = "用户喜欢简洁的回答"
        result = pipeline.compress(content)
        assert "graph TD" not in result

    def test_no_canvas_fallback(self):
        from omnimem.compression.pipeline import CompressionPipeline

        pipeline = CompressionPipeline(llm_call_fn=None)
        content = "tool_call: search_files\nresult: found 3 files"
        result = pipeline.compress(content)
        assert isinstance(result, str)
