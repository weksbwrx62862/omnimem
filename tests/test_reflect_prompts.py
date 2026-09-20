"""deep/reflect/prompts.py 单元测试：LLM 输出解析与生成回退路径。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from omnimem.deep.reflect import prompts as prompts_mod
from omnimem.deep.reflect import synthesis as synthesis_mod
from omnimem.deep.reflect.disposition import Disposition


def _parse(raw: str):
    return prompts_mod._parse_llm_output(raw)


def _is_keyword_stuffing(text: str) -> bool:
    return synthesis_mod._is_keyword_stuffing(text)


class _Engine:
    """供 _generate_with_llm 使用的最小 stub。"""

    _parse_llm_output = staticmethod(prompts_mod._parse_llm_output)
    _is_keyword_stuffing = staticmethod(_is_keyword_stuffing)

    def __init__(self, llm_fn=None, llm_client=None):
        self._llm_fn = llm_fn
        self._llm_client = llm_client


def _generate(engine, query="测试主题", contents=None, disposition=None, max_tokens=800):
    return prompts_mod._generate_with_llm(
        engine,
        query,
        contents or ["内容一", "内容二"],
        disposition or Disposition(),
        max_tokens=max_tokens,
    )


# ─── _parse_llm_output ────────────────────────────────────────


def test_parse_json_primary_keys():
    raw = json.dumps(
        {"observation": "观察X", "mental_model": "模型Y", "confidence": 0.82},
        ensure_ascii=False,
    )
    obs, model, conf = _parse(raw)
    assert obs == "观察X"
    assert model == "模型Y"
    assert conf == pytest.approx(0.82)


def test_parse_json_alternate_keys():
    raw = json.dumps({"obs": "A", "model": "B", "conf": 0.4})
    obs, model, conf = _parse(raw)
    assert obs == "A"
    assert model == "B"
    assert conf == pytest.approx(0.4)


def test_parse_json_confidence_clamped():
    raw = json.dumps({"observation": "O", "mental_model": "M", "confidence": 5.0})
    _, _, conf = _parse(raw)
    assert conf == 1.0

    raw = json.dumps({"observation": "O", "mental_model": "M", "confidence": -3})
    _, _, conf = _parse(raw)
    assert conf == 0.0


def test_parse_json_confidence_bad_string_falls_back_to_default():
    raw = json.dumps({"observation": "O", "mental_model": "M", "confidence": "not-a-number"})
    obs, model, conf = _parse(raw)
    assert obs == "O"
    assert model == "M"
    assert conf == 0.5


def test_parse_json_missing_confidence_defaults_half():
    raw = json.dumps({"observation": "只有观察"})
    obs, model, conf = _parse(raw)
    assert obs == "只有观察"
    assert model == ""
    assert conf == 0.5


def test_parse_json_empty_dict_falls_through_to_markers():
    # JSON 中 observation/mental_model 都为空 → 不返回 JSON 路径，落入 marker 解析。
    raw = "{}"
    obs, model, conf = _parse(raw)
    # 无 marker 且 observation/model 都空 → fallback：整段作为 observation，默认 0.5。
    assert obs == "{}"
    assert model == ""
    assert conf == 0.5


def test_parse_invalid_json_falls_back_to_chinese_markers():
    raw = (
        "这不是合法 JSON {\n"
        "【观察】\n用户偏好深色主题\n"
        "【心智模型】\n深色主题减少夜间疲劳。\n"
        "【置信度】\n0.75"
    )
    obs, model, conf = _parse(raw)
    assert "深色主题" in obs
    assert "夜间疲劳" in model
    assert conf == pytest.approx(0.75)


def test_parse_english_colon_markers():
    raw = (
        "observation: 用户强调过两次\n"
        "mental_model: 反复提及意味着优先级高\n"
        "confidence: 0.9"
    )
    # 该实现只识别中文标记，英文 colon 不匹配时会落入 fallback。
    obs, model, conf = _parse(raw)
    # 兜底：整段作为 observation
    assert "用户强调过两次" in obs
    assert conf == 0.5
    # model 未被解析出
    assert model == "" or model == raw[:500].strip()


def test_parse_colon_chinese_markers():
    raw = "观察：简单复述\n心智模型：复述不产生新洞察\n置信度：0.6"
    obs, model, conf = _parse(raw)
    assert "简单复述" in obs
    assert "新洞察" in model
    assert conf == pytest.approx(0.6)


def test_parse_confidence_over_one_clamped():
    raw = "【观察】\nOK\n【心智模型】\nOK的因果\n【置信度】\n1.7"
    _, _, conf = _parse(raw)
    assert conf == 1.0


def test_parse_bad_confidence_string_falls_back():
    raw = "【观察】\nOK\n【心智模型】\nOK的因果\n【置信度】\nabc"
    obs, model, conf = _parse(raw)
    assert obs == "OK"
    assert model == "OK的因果"
    # conf_match 找不到数字 → confidence 保持 None → 默认 0.5
    assert conf == 0.5


def test_parse_no_markers_uses_full_text_and_truncates_500():
    raw = "x" * 800
    obs, model, conf = _parse(raw)
    assert obs == raw[:500]
    assert model == ""
    assert conf == 0.5


# ─── _generate_with_llm ───────────────────────────────────────


def test_generate_bails_without_llm():
    engine = _Engine(llm_fn=None, llm_client=None)
    assert _generate(engine) is None


def test_generate_uses_llm_fn_when_available():
    calls = {"n": 0}

    def _llm_fn(prompt, system, max_tokens):  # noqa: ARG001
        calls["n"] += 1
        return (
            "【观察】\n用户近期在多个会话中反复偏好简洁回复并明确拒绝冗余解释。\n"
            "【心智模型】\n因为工作繁忙所以用户偏好简洁回复。\n"
            "【置信度】\n0.78"
        )

    engine = _Engine(llm_fn=_llm_fn, llm_client=None)
    result = _generate(engine)
    assert result is not None
    obs, model, conf = result
    assert "简洁回复" in obs
    assert "工作繁忙" in model
    assert conf == pytest.approx(0.78)
    assert calls["n"] == 1


def test_generate_retries_when_observation_too_short():
    """observation <15 字符触发截断重试，最后一次仍返回给调用方。"""
    attempts = {"n": 0}

    def _llm_fn(prompt, system, max_tokens):  # noqa: ARG001
        attempts["n"] += 1
        return (
            "【观察】\n短。\n"
            "【心智模型】\n由于外部约束用户偏好简洁回复而不是详细解释。\n"
            "【置信度】\n0.6"
        )

    engine = _Engine(llm_fn=_llm_fn, llm_client=None)
    result = _generate(engine)
    # 前两次 obs 太短触发 continue，第三次（最后一次）不再重试，直接返回解析结果。
    assert result is not None
    assert attempts["n"] == 3
    obs, model, conf = result
    assert obs == "短。"
    assert conf == pytest.approx(0.6)


def test_generate_falls_back_to_client_when_fn_returns_empty():
    def _llm_fn(prompt, system, max_tokens):  # noqa: ARG001
        return ""

    class _Client:
        def __init__(self):
            self.invoked = 0

        def call_sync(self, prompt, system, max_tokens, temperature):  # noqa: ARG002
            self.invoked += 1
            return SimpleNamespace(
                content=(
                    "【观察】\n用户经常询问图表渲染问题。\n"
                    "【心智模型】\n因为项目重度使用可视化所以用户经常询问图表渲染。\n"
                    "【置信度】\n0.66"
                )
            )

    client = _Client()
    engine = _Engine(llm_fn=_llm_fn, llm_client=client)
    result = _generate(engine)
    assert result is not None
    assert client.invoked >= 1
    _, _, conf = result
    assert conf == pytest.approx(0.66)


def test_generate_retries_on_llm_fn_exception():
    attempts = {"n": 0}

    def _llm_fn(prompt, system, max_tokens):  # noqa: ARG001
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")
        return (
            "【观察】\n第三次终于成功产出内容。\n"
            "【心智模型】\n由于前两次偶发网络异常第三次终于成功产出内容。\n"
            "【置信度】\n0.55"
        )

    engine = _Engine(llm_fn=_llm_fn, llm_client=None)
    result = _generate(engine)
    assert result is not None
    assert attempts["n"] == 3


def test_generate_returns_none_after_three_empty_responses():
    def _llm_fn(prompt, system, max_tokens):  # noqa: ARG001
        return "   "

    engine = _Engine(llm_fn=_llm_fn, llm_client=None)
    assert _generate(engine) is None


def test_generate_returns_parsed_tuple_when_confidence_marker_never_appears():
    """截断路径：缺少【置信度】标记 → 前两次 continue，第三次落到 _parse_llm_output 兜底。"""
    attempts = {"n": 0}

    def _llm_fn(prompt, system, max_tokens):  # noqa: ARG001
        attempts["n"] += 1
        return "【观察】\n只有观察没有后续结构内容"

    engine = _Engine(llm_fn=_llm_fn, llm_client=None)
    result = _generate(engine)
    assert attempts["n"] == 3
    assert result is not None
    obs, model, conf = result
    # 中文 marker 匹配到 【观察】，因缺 model/conf marker → model 空、conf 默认 0.5
    assert obs == "只有观察没有后续结构内容"
    assert model == ""
    assert conf == 0.5


def test_generate_returns_stuffed_model_on_final_attempt():
    """关键词堆砌检查在前两次触发重试，第三次直接返回。"""
    attempts = {"n": 0}

    def _llm_fn(prompt, system, max_tokens):  # noqa: ARG001
        attempts["n"] += 1
        return (
            "【观察】\n观察内容足够长以通过短观察检查所以不算截断。\n"
            "【心智模型】\n简洁,高效,优雅\n"
            "【置信度】\n0.7"
        )

    engine = _Engine(llm_fn=_llm_fn, llm_client=None)
    result = _generate(engine)
    assert attempts["n"] == 3
    assert result is not None
    obs, model, conf = result
    assert obs == "观察内容足够长以通过短观察检查所以不算截断。"
    # 最后一次不再丢弃堆砌结果
    assert model == "简洁,高效,优雅"
    assert conf == pytest.approx(0.7)


def test_generate_recovers_when_keyword_stuffing_resolves_on_last_attempt():
    """前两次堆砌触发重试，第三次输出正常句子 → 一次成功返回。"""
    attempts = {"n": 0}

    def _llm_fn(prompt, system, max_tokens):  # noqa: ARG001
        attempts["n"] += 1
        if attempts["n"] < 3:
            model_line = "简洁,高效,优雅"
        else:
            model_line = "因为工程实践要求简洁所以用户偏好简洁高效优雅的表达风格。"
        return (
            "【观察】\n观察内容足够长以通过短观察检查所以不算截断。\n"
            f"【心智模型】\n{model_line}\n"
            "【置信度】\n0.7"
        )

    engine = _Engine(llm_fn=_llm_fn, llm_client=None)
    result = _generate(engine)
    assert result is not None
    assert attempts["n"] == 3
    _, model, _ = result
    assert "工程实践" in model


def test_generate_handles_generic_exception_and_retries():
    attempts = {"n": 0}

    def _llm_fn(prompt, system, max_tokens):  # noqa: ARG001
        attempts["n"] += 1
        raise ValueError("always fails")

    engine = _Engine(llm_fn=_llm_fn, llm_client=None)
    # 顶层 except 会记录 warning；三次重试均失败 → 返回 None
    result = _generate(engine)
    assert result is None
    # 每次 _llm_fn 内部 except 已吞掉异常，返回 raw=None → 走 empty 分支 continue
    assert attempts["n"] == 3
