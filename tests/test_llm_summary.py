"""LLM Summary 模块测试。

覆盖: StructuredSummary, llm_summarize, _parse_llm_response, _extract_without_llm
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from omnimem.compression.llm_summary import (
    StructuredSummary,
    _extract_without_llm,
    _parse_llm_response,
    llm_summarize,
)


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
        assert text.count("\n") == 5  # 6 fields → 5 newlines

    def test_to_dict(self):
        s = StructuredSummary(goal="g", key_info="k")
        d = s.to_dict()
        assert d["goal"] == "g"
        assert d["key_info"] == "k"
        assert d["progress"] == ""


class TestParseLLMResponse:
    def test_valid_json(self):
        data = {
            "goal": "build app",
            "progress": "50%",
            "decisions": "use Python",
            "key_info": "fastapi",
            "open_issues": "testing",
            "next_steps": "deploy",
        }
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
        mock_fn = MagicMock(return_value=json.dumps({
            "goal": "test", "progress": "done", "decisions": "",
            "key_info": "", "open_issues": "", "next_steps": "",
        }))
        result = llm_summarize("messages", llm_call_fn=mock_fn)
        assert result.goal == "test"
        mock_fn.assert_called_once()

    def test_llm_failure_fallback(self):
        mock_fn = MagicMock(side_effect=RuntimeError("LLM down"))
        result = llm_summarize("fallback test", llm_call_fn=mock_fn)
        assert isinstance(result, StructuredSummary)
        # Should fall back to _extract_without_llm
        assert result.progress == "See conversation history"
