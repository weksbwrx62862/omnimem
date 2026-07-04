"""反思数据模型与 Disposition 性格修饰。

职责：
  - 定义 Disposition / ReflectResult / ReflectionContext 数据类
  - 根据性格参数调整反思输出语气
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Disposition:
    """反思性格参数，控制 ReflectEngine 输出的语气和侧重。

    三个维度:
      skepticism (1-5): 怀疑度，越高越审慎，输出更保守
      literalness (1-5): 字面度，越高越精确，强调可验证性
      empathy (1-5): 共情度，越高越关注人的感受和影响
    """

    skepticism: int = 3  # 怀疑度 1-5
    literalness: int = 2  # 字面度 1-5
    empathy: int = 4  # 共情度 1-5

    def clamp(self) -> Disposition:
        """确保参数在合法范围内。"""
        return Disposition(
            skepticism=max(1, min(5, self.skepticism)),
            literalness=max(1, min(5, self.literalness)),
            empathy=max(1, min(5, self.empathy)),
        )

    def to_dict(self) -> dict[str, int]:
        """将性格参数序列化为字典。"""
        return {
            "skepticism": self.skepticism,
            "literalness": self.literalness,
            "empathy": self.empathy,
        }


@dataclass
class ReflectResult:
    """Reflect 结果。"""

    observation: str = ""
    mental_model: str = ""
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    disposition_used: dict[str, int] | None = None
    reflection_depth: int = 0  # 反思循环深度
    query: str = ""  # 反思查询关键词


@dataclass
class ReflectionContext:
    """反思循环中累积的上下文。"""

    query: str = ""
    mental_models: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    expanded: list[dict[str, Any]] = field(default_factory=list)


def _apply_disposition(observation: str, model: str, disposition: Disposition) -> tuple[str, str]:
    """根据 Disposition 参数调整反思输出的语气和侧重。

    Returns:
        (adjusted_observation, adjusted_model)
    """
    d = disposition.clamp()

    # ─── 怀疑度修饰 ───
    skepticism_prefixes = {
        1: "",
        2: "初步看来，",
        3: "据现有信息，",
        4: "需要谨慎对待以下判断：",
        5: "在缺乏更多证据的情况下，暂且认为：",
    }
    s_prefix = skepticism_prefixes.get(d.skepticism, "")

    # ─── 共情度修饰 ───
    # ★ 仅在内容涉及人/感受时添加共情后缀，技术/事实类内容不加
    _person_keywords = {
        "用户",
        "人",
        "感受",
        "情感",
        "心情",
        "体验",
        "偏好",
        "性格",
        "user",
        "people",
        "feeling",
        "emotion",
        "experience",
        "person",
    }
    has_person_context = any(kw in observation or kw in model for kw in _person_keywords)
    empathy_suffixes = {
        1: "",
        2: "",
        3: "（考虑当事人感受）" if has_person_context else "",
        4: "（需关注相关人的需求和感受）" if has_person_context else "",
        5: "（优先考虑对人的影响和情感因素）" if has_person_context else "",
    }
    e_suffix = empathy_suffixes.get(d.empathy, "")

    # ─── 字面度修饰 ───
    if d.literalness >= 4:
        # 高字面度：强调可验证性
        if model and not model.endswith("。"):
            model += "。（以上结论基于可验证的事实依据）"
    elif d.literalness <= 2:
        # 低字面度：允许推测
        if model and "可能" not in model and "或许" not in model:
            model = model.replace("核心规律", "可能的规律").replace("规律", "推测")

    adjusted_obs = f"{s_prefix}{observation}{e_suffix}" if s_prefix or e_suffix else observation
    adjusted_model = f"{s_prefix}{model}" if s_prefix else model

    return adjusted_obs, adjusted_model
