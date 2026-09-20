"""deep/reflect/synthesis.py 单元测试：规则归纳、关键词与堆砌修复。"""

from __future__ import annotations

import pytest
from omnimem.deep.reflect import synthesis as synth_mod
from omnimem.deep.reflect.disposition import ReflectionContext


class _Engine:
    """把 synthesis 模块函数按 ReflectEngine mixin 的挂载方式绑定到 stub。"""

    _smart_extract_keywords = synth_mod._smart_extract_keywords
    _extract_content_phrases = synth_mod._extract_content_phrases
    _rule_based_observation = synth_mod._rule_based_observation
    _rule_based_synthesize = synth_mod._rule_based_synthesize
    _generate_model_from_observations = synth_mod._generate_model_from_observations
    _generate_observation_from_facts = synth_mod._generate_observation_from_facts
    _generate_model_from_facts = synth_mod._generate_model_from_facts
    _post_process_mental_model = synth_mod._post_process_mental_model
    _is_keyword_stuffing = staticmethod(synth_mod._is_keyword_stuffing)


@pytest.fixture
def eng():
    return _Engine()


# ─── _smart_extract_keywords ─────────────────────────────


def test_extract_keywords_empty_returns_list():
    assert _Engine._smart_extract_keywords(None, "") == []


def test_extract_keywords_filters_stopwords_and_bad_endings():
    text = "关于性能问题、系统需要优化、用户可以接受"
    kws = _Engine._smart_extract_keywords(None, text)
    # 整段作为中文片段保留（停用词只精确匹配，非前缀过滤）
    assert kws == ["关于性能问题", "系统需要优化", "用户可以接受"]
    # 纯停用词或以停用词结尾的短片段仍会被剔除
    text2 = "关于、进行、需要"
    assert _Engine._smart_extract_keywords(None, text2) == []


def test_extract_keywords_english_lowercased_and_stopwords_removed():
    text = "The System requires Optimization and Refactoring"
    kws = _Engine._smart_extract_keywords(None, text)
    assert "system" in kws
    assert "the" not in kws
    assert "and" not in kws


def test_extract_keywords_respects_max_and_dedups():
    text = "关键词一、关键词一、关键词二、关键词三、关键词四、关键词五、关键词六、关键词七"
    kws = _Engine._smart_extract_keywords(None, text, max_keywords=3)
    assert len(kws) <= 3
    assert len(kws) == len(set(kws))


# ─── _extract_content_phrases ────────────────────────────


def test_extract_phrases_filters_short_and_bad_prefixes():
    texts = [
        "短",  # <4
        "R1 首轮测试片段",  # R1 前缀
        "[标记] 测试片段",  # [ 前缀
        "关于系统性能优化的详细观察记录",  # 关于前缀
        "系统响应时间显著下降导致用户体验受损",  # 应保留
    ]
    phrases = _Engine._extract_content_phrases(None, texts, max_phrases=5)
    assert any("系统响应时间" in p for p in phrases)
    assert not any(p.startswith(("R1", "[", "关于")) for p in phrases)
    assert all(len(p) >= 4 for p in phrases)


def test_extract_phrases_respects_max_limit():
    texts = ["有效短语一测试内容", "有效短语二测试内容", "有效短语三测试内容"]
    phrases = _Engine._extract_content_phrases(None, texts, max_phrases=2)
    assert len(phrases) == 2


# ─── _is_keyword_stuffing ────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", True),
        ("短", True),  # <4
        ("用户在多次会话中强调要简洁的回复。", False),  # 完整单句
        ("简洁,高效,优雅", True),  # 短逗号列表
        ("1. 苹果, 2. 香蕉, 3. 橙子.", False),  # 编号列举 + 句末 → 排除
        ("1. 用户偏好, 2. 系统响应, 3. 性能优化", True),  # 编号但缺句末且无动词 → 特征3 命中
        ("a,b,c,d,e", True),  # 分隔符密度过高
    ],
)
def test_is_keyword_stuffing_cases(text, expected):
    assert synth_mod._is_keyword_stuffing(text) is expected


# ─── _rule_based_observation ─────────────────────────────


def test_rule_observation_empty_returns_blank(eng):
    assert eng._rule_based_observation("q", ReflectionContext()) == ""


def test_rule_observation_uses_mental_models_branch(eng):
    ctx = ReflectionContext(
        mental_models=[{"content": "已有心智模型"}],
        observations=[{"content": "观察A"}],
    )
    text = eng._rule_based_observation("q", ctx)
    assert "已有心智模型支撑" in text
    assert "观察A" in text


def test_rule_observation_observations_branch(eng):
    ctx = ReflectionContext(
        observations=[{"content": "系统响应时间显著下降导致用户体验受损"}],
    )
    text = eng._rule_based_observation("q", ctx)
    assert "基于 1 条观察的归纳" in text


def test_rule_observation_facts_branch(eng):
    ctx = ReflectionContext(
        facts=[{"content": "用户在会话中反复询问图表渲染问题"}],
    )
    text = eng._rule_based_observation("q", ctx)
    assert "基于 1 条记忆的归纳" in text


def test_rule_observation_appends_expanded_context(eng):
    ctx = ReflectionContext(
        facts=[{"content": "主要记忆内容较长以通过短片段筛选"}],
        expanded=[{"content": "扩展开来的关联记忆文本内容"}],
    )
    text = eng._rule_based_observation("q", ctx)
    assert "关联上下文" in text
    assert "扩展开来的关联记忆文本内容" in text


# ─── _rule_based_synthesize ──────────────────────────────


def test_rule_synthesize_empty_returns_low_confidence(eng):
    obs, model, conf = eng._rule_based_synthesize("q", ReflectionContext(), 0.0)
    assert obs == ""
    assert model == ""
    assert conf == 0.2


def test_rule_synthesize_observations_confidence_formula(eng):
    ctx = ReflectionContext(
        observations=[{"content": "有效片段测试内容一"} for _ in range(5)],
    )
    _, _, conf = eng._rule_based_synthesize("q", ctx, 0.0)
    # min(0.45, 0.25 + n_sources*0.02); n=5 → 0.25 + 0.10 = 0.35
    assert conf == pytest.approx(0.35)


def test_rule_synthesize_facts_bonus_when_phrases_present(eng):
    ctx = ReflectionContext(
        facts=[{"content": "系统响应时间显著下降导致用户体验受损"} for _ in range(3)],
    )
    _, model, conf = eng._rule_based_synthesize("q", ctx, 0.0)
    # 3 facts → base 0.30 + 3*0.02 = 0.36; +0.05 phrase bonus → 0.41 (capped at 0.45)
    assert conf == pytest.approx(0.41)
    assert model  # 非空


# ─── _generate_model_from_observations / facts ───────────


def test_generate_model_from_observations_empty_returns_blank(eng):
    assert eng._generate_model_from_observations([], "q") == ""


def test_generate_model_from_observations_two_phrases(eng):
    obs = [
        "系统响应时间显著下降导致用户体验受损",
        "缓存命中率降低到不足百分之三十",
    ]
    model = eng._generate_model_from_observations(obs, "性能")
    assert "在「性能」方面" in model
    assert "同时" in model
    assert "基于2条观察" in model


def test_generate_model_from_observations_no_signal(eng):
    # 极短、无中文有效片段 → 走"信息有限"兜底
    obs = ["a,b,c,d,e"]
    model = eng._generate_model_from_observations(obs, "主题")
    # 因未产生 phrases/keywords，最可能落到"关键词兜底"或"信息有限"两种文本
    assert "主题" in model


def test_generate_observation_from_facts_needs_at_least_two(eng):
    assert eng._generate_observation_from_facts(["单条记忆"], "q") == ""


def test_generate_observation_from_facts_groups_keywords(eng):
    facts = [
        "系统性能,响应慢于预期",
        "系统性能,吞吐显著下降",
        "用户体验,视觉反馈延迟",
    ]
    text = eng._generate_observation_from_facts(facts, "q")
    assert text  # 有内容
    assert "系统性能" in text
    assert "一致趋势" in text  # ≥2 分组走该模板
    assert "记录显示" in text  # 单条分组走该模板


def test_generate_model_from_facts_empty_returns_blank(eng):
    assert eng._generate_model_from_facts([], "q") == ""


def test_generate_model_from_facts_with_phrases(eng):
    facts = [
        "系统响应时间显著下降导致用户体验受损",
        "缓存命中率降低到不足百分之三十",
        "并发访问压力增加触发限流保护",
    ]
    model = eng._generate_model_from_facts(facts, "性能")
    assert "关于「性能」" in model
    assert "此外" in model or "尚需更多验证" in model


# ─── _post_process_mental_model ──────────────────────────


def test_post_process_passthrough_empty(eng):
    assert eng._post_process_mental_model("", 0.3) == ""


def test_post_process_passthrough_non_stuffed(eng):
    text = "用户在多次会话中反复强调要简洁的回复。"
    assert eng._post_process_mental_model(text, 0.9) == text


def test_post_process_low_confidence_reconstructs_with_two_meaningful(eng):
    stuffed = "用户偏好,系统响应,简洁回复"
    result = eng._post_process_mental_model(stuffed, 0.3)
    assert result.startswith("当前记忆显示在")
    assert "用户偏好" in result and "系统响应" in result


def test_post_process_low_confidence_single_meaningful(eng):
    stuffed = "甲乙丙,短,小"
    result = eng._post_process_mental_model(stuffed, 0.3)
    assert "甲乙丙" in result
    assert "用户关注的主要领域" in result


def test_post_process_low_confidence_no_meaningful_falls_back(eng):
    stuffed = "a,b,c,d,e"
    result = eng._post_process_mental_model(stuffed, 0.3)
    assert "初步阶段" in result


def test_post_process_high_confidence_marks_warning(eng):
    stuffed = "简洁,高效,优雅"
    result = eng._post_process_mental_model(stuffed, 0.8)
    assert result.startswith("[⚠ 质量警告]")
    assert stuffed in result
