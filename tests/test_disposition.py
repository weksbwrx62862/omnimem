"""deep/reflect/disposition.py 性格修饰与语气单测（离线）。"""
from __future__ import annotations

from omnimem.deep.reflect.disposition import Disposition, _apply_disposition


# ── clamp / to_dict ────────────────────────────────────
def test_clamp_bounds():
    d = Disposition(skepticism=0, literalness=9, empathy=-3).clamp()
    assert d.skepticism == 1
    assert d.literalness == 5
    assert d.empathy == 1


def test_to_dict_shape():
    assert Disposition(skepticism=2, literalness=3, empathy=4).to_dict() == {
        "skepticism": 2, "literalness": 3, "empathy": 4,
    }


# ── 怀疑度前缀 ────────────────────────────────────────
def test_high_skepticism_prefixes_both_fields():
    obs, model = _apply_disposition("外部事实", "内部模型", Disposition(skepticism=5, empathy=1, literalness=3))
    assert obs.startswith("在缺乏更多证据的情况下，暂且认为：")
    assert model.startswith("在缺乏更多证据的情况下，暂且认为：")


def test_low_skepticism_no_prefix():
    obs, model = _apply_disposition("obs", "model", Disposition(skepticism=1, empathy=1, literalness=3))
    assert obs == "obs"
    assert model == "model"


# ── 共情后缀（仅涉人内容）──────────────────────────────
def test_empathy_suffix_only_when_person_context():
    obs_person, _ = _apply_disposition("用户的体验", "model", Disposition(skepticism=1, empathy=4, literalness=3))
    assert obs_person.endswith("（需关注相关人的需求和感受）")
    obs_tech, _ = _apply_disposition("数据库索引", "model", Disposition(skepticism=1, empathy=4, literalness=3))
    assert "（需关注" not in obs_tech


# ── 字面度修饰 ────────────────────────────────────────
def test_high_literalness_appends_verifiability():
    _, model = _apply_disposition("obs", "核心结论", Disposition(skepticism=1, empathy=1, literalness=5))
    assert model.endswith("。（以上结论基于可验证的事实依据）")


def test_high_literalness_skips_when_period_terminated():
    _, model = _apply_disposition("obs", "结论。", Disposition(skepticism=1, empathy=1, literalness=5))
    assert model == "结论。"


def test_low_literalness_softens_speculation():
    _, model = _apply_disposition("obs", "发现核心规律", Disposition(skepticism=1, empathy=1, literalness=1))
    # 链式 replace: 核心规律->可能的规律, 再 规律->推测 => 发现可能的推测
    assert model == "发现可能的推测"
    assert "推测" in model and "可能的" in model
