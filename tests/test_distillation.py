"""DistillationEngine 蒸馏引擎测试。"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

from omnimem.core.distillation import DistillationEngine, _escape_fts5_query


class TestEscapeFts5Query:
    """FTS5 特殊字符转义测试。"""

    def test_normal_text_no_escape(self) -> None:
        """普通文本无需转义。"""
        assert _escape_fts5_query("hello") == "hello"

    def test_special_chars_escaped(self) -> None:
        """FTS5 特殊字符应被引号包裹。"""
        result = _escape_fts5_query("a+b")
        assert result == 'a"+"b'

    def test_multiple_special_chars(self) -> None:
        """多个特殊字符均被转义。"""
        result = _escape_fts5_query("a + b - c")
        assert '"+"' in result
        assert '"-"' in result


class TestDistillationEngineInit:
    """DistillationEngine 初始化测试。"""

    def test_default_init(self) -> None:
        """无参数初始化，属性应为默认值。"""
        engine = DistillationEngine()
        assert engine._llm_fn is None
        assert engine._store is None
        assert engine._memorize_fn is None
        assert engine._config is None
        assert engine._distill_count == 0

    def test_init_with_all_params(self) -> None:
        """传入全部参数初始化。"""
        llm_fn = MagicMock()
        store = MagicMock()
        memorize_fn = MagicMock()
        config = MagicMock()

        engine = DistillationEngine(
            llm_fn=llm_fn, store=store,
            memorize_fn=memorize_fn, config=config,
        )
        assert engine._llm_fn is llm_fn
        assert engine._store is store
        assert engine._memorize_fn is memorize_fn
        assert engine._config is config


class TestDistillRecentFacts:
    """distill_recent_facts 方法测试。"""

    def test_no_llm_fn_returns_unavailable(self) -> None:
        """无 LLM 函数时返回 llm_unavailable。"""
        engine = DistillationEngine()
        result = engine.distill_recent_facts()
        assert result["status"] == "llm_unavailable"
        assert result["input_count"] == 0

    def test_too_soon_skips(self) -> None:
        """两次蒸馏间隔太短时跳过。"""
        engine = DistillationEngine(llm_fn=MagicMock())
        engine._last_distill_time = time.time()  # 刚蒸馏过
        result = engine.distill_recent_facts()
        assert result["status"] == "skipped"
        assert result["reason"] == "too_soon"

    def test_already_distilled_this_turn(self) -> None:
        """同一轮次已蒸馏过时跳过。"""
        engine = DistillationEngine(llm_fn=MagicMock())
        engine._last_distill_time = 0  # 确保不触发 too_soon
        engine._last_distill_turn = 5
        result = engine.distill_recent_facts(turn_count=5)
        assert result["status"] == "skipped"
        assert result["reason"] == "already_distilled_this_turn"

    def test_no_facts_below_minimum(self) -> None:
        """原始事实数量不足时返回 no_facts。"""
        store = MagicMock()
        store._meta_store = None
        store.search_by_content.return_value = []
        store._closet_index = {}

        engine = DistillationEngine(llm_fn=MagicMock(), store=store)
        engine._last_distill_time = 0
        result = engine.distill_recent_facts(min_facts=3)
        assert result["status"] == "no_facts"

    def test_llm_call_exception_returns_unavailable(self) -> None:
        """LLM 调用异常时返回 llm_unavailable。"""
        llm_fn = MagicMock(side_effect=RuntimeError("LLM error"))
        store = MagicMock()
        store._meta_store = None
        store.search_by_content.return_value = [
            {"content": "事实1"}, {"content": "事实2"}, {"content": "事实3"},
        ]
        store._closet_index = {}

        engine = DistillationEngine(llm_fn=llm_fn, store=store)
        engine._last_distill_time = 0
        result = engine.distill_recent_facts(min_facts=3)
        assert result["status"] == "llm_unavailable"

    def test_llm_returns_empty(self) -> None:
        """LLM 返回空字符串时蒸馏结果为空。"""
        llm_fn = MagicMock(return_value="")
        store = MagicMock()
        store._meta_store = None
        store.search_by_content.return_value = [
            {"content": "事实1"}, {"content": "事实2"}, {"content": "事实3"},
        ]
        store._closet_index = {}

        engine = DistillationEngine(llm_fn=llm_fn, store=store)
        engine._last_distill_time = 0
        result = engine.distill_recent_facts(min_facts=3)
        assert result["status"] == "distilled"
        assert result["distilled_count"] == 0

    def test_successful_distillation(self) -> None:
        """正常蒸馏流程：LLM 返回有效事实。"""
        llm_fn = MagicMock(return_value="用户偏好暗色主题\n用户使用Python开发")
        memorize_fn = MagicMock(return_value="mem-001")
        store = MagicMock()
        store._meta_store = None
        store.search_by_content.return_value = [
            {"content": "原始事实1"}, {"content": "原始事实2"}, {"content": "原始事实3"},
        ]
        store._closet_index = {}

        engine = DistillationEngine(
            llm_fn=llm_fn, store=store, memorize_fn=memorize_fn,
        )
        engine._last_distill_time = 0
        result = engine.distill_recent_facts(min_facts=3)
        assert result["status"] == "distilled"
        assert result["input_count"] == 3
        assert len(result["facts"]) >= 1


class TestParseDistilledFacts:
    """_parse_distilled_facts 静态方法测试。"""

    def test_parse_line_by_line(self) -> None:
        """按行解析蒸馏输出。"""
        raw = "用户偏好暗色主题\n用户使用Python开发\n这是一个重要的事实"
        facts = DistillationEngine._parse_distilled_facts(raw)
        assert len(facts) == 3

    def test_parse_json_array(self) -> None:
        """解析 JSON 数组格式输出。"""
        raw = json.dumps(["事实一很长的内容", "事实二也很长的内容"])
        facts = DistillationEngine._parse_distilled_facts(raw)
        assert len(facts) == 2

    def test_parse_json_dict_items(self) -> None:
        """解析 JSON 对象数组（含 content 字段）。"""
        raw = json.dumps([
            {"content": "这是一个很长的事实内容"},
            {"fact": "另一个很长的事实内容"},
        ])
        facts = DistillationEngine._parse_distilled_facts(raw)
        assert len(facts) == 2

    def test_skip_short_lines(self) -> None:
        """过短的行（<=5字符）应被过滤。"""
        raw = "短\n这是一个很长的事实内容不应该被过滤"
        facts = DistillationEngine._parse_distilled_facts(raw)
        assert all(len(f) > 5 for f in facts)

    def test_strip_numbering_prefix(self) -> None:
        """去除编号前缀（如 1. 2) 等）。"""
        raw = "1. 用户偏好暗色主题\n2) 用户使用Python开发"
        facts = DistillationEngine._parse_distilled_facts(raw)
        for f in facts:
            assert not f.startswith("1.")
            assert not f.startswith("2)")

    def test_skip_comment_lines(self) -> None:
        """跳过注释行和代码块标记。"""
        raw = "# 注释\n```代码块\n这是一个正常的长事实内容"
        facts = DistillationEngine._parse_distilled_facts(raw)
        for f in facts:
            assert not f.startswith("#")
            assert not f.startswith("```")

    def test_empty_input(self) -> None:
        """空输入返回空列表。"""
        facts = DistillationEngine._parse_distilled_facts("")
        assert facts == []


class TestGetStats:
    """get_stats 方法测试。"""

    def test_initial_stats(self) -> None:
        """初始统计值应为零。"""
        engine = DistillationEngine()
        stats = engine.get_stats()
        assert stats["total_distillations"] == 0
        assert stats["last_distill_turn"] == 0
        assert stats["last_distill_time"] == 0.0

    def test_stats_after_distillation(self) -> None:
        """蒸馏后统计值应更新。"""
        llm_fn = MagicMock(return_value="用户偏好暗色主题")
        memorize_fn = MagicMock(return_value="mem-001")
        store = MagicMock()
        store._meta_store = None
        store.search_by_content.return_value = [
            {"content": "事实1"}, {"content": "事实2"}, {"content": "事实3"},
        ]
        store._closet_index = {}

        engine = DistillationEngine(
            llm_fn=llm_fn, store=store, memorize_fn=memorize_fn,
        )
        engine._last_distill_time = 0
        engine.distill_recent_facts(turn_count=10, min_facts=3)

        stats = engine.get_stats()
        assert stats["total_distillations"] == 1
        assert stats["last_distill_turn"] == 10


class TestGetDistillModel:
    """_get_distill_model 方法测试。"""

    def test_no_config_returns_none(self) -> None:
        """无配置时返回 None。"""
        engine = DistillationEngine()
        assert engine._get_distill_model() is None

    def test_config_with_model(self) -> None:
        """配置中指定模型时返回模型名。"""
        config = MagicMock()
        config.get.return_value = "gpt-4o-mini"
        engine = DistillationEngine(config=config)
        assert engine._get_distill_model() == "gpt-4o-mini"

    def test_config_with_empty_model(self) -> None:
        """配置中模型名为空时返回 None。"""
        config = MagicMock()
        config.get.return_value = ""
        engine = DistillationEngine(config=config)
        assert engine._get_distill_model() is None
