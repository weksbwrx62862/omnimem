"""查询质量评估与结果裁剪。

包含垃圾查询检测和 Token 预算裁剪逻辑。
"""

from __future__ import annotations

import re
from typing import Any

# 预编译垃圾查询检测常用正则，避免每次调用重复编译
_RE_HAS_CHINESE = re.compile(r"[\u4e00-\u9fff]")
_RE_CHINESE_REPEAT = re.compile(r"([\u4e00-\u9fff])\1{2,}")
_RE_CHINESE_SEGMENT = re.compile(r"[\u4e00-\u9fff]+")
_RE_ENGLISH_WORDS = re.compile(r"[a-zA-Z]{3,}")
_RE_ENGLISH_SEQ = re.compile(r"[a-zA-Z]{5,}")
_RE_ONLY_DIGITS_SPACES = re.compile(r"^[\d\s]+$")
_RE_ALPHANUMERIC = re.compile(r"^[a-zA-Z0-9]+$")

# 类级别垃圾查询白名单：避免每次 is_garbage_query 调用时重建集合
_GARBAGE_COMMON_WORDS = frozenset(
    {
        "what",
        "how",
        "why",
        "when",
        "where",
        "this",
        "that",
        "with",
        "from",
        "have",
        "will",
        "would",
        "could",
        "should",
        "about",
        "just",
        "like",
        "only",
        "some",
        "them",
        "than",
        "into",
        "over",
        "also",
        "back",
        "after",
        "used",
        "first",
        "well",
        "way",
        "even",
        "want",
        "because",
        "any",
        "these",
        "most",
        "make",
        "know",
        "time",
        "year",
        "good",
        "work",
        "qual",
        "user",
        "http",
        "html",
        "json",
        "api",
        "url",
        "app",
        "log",
    }
)


def is_garbage_query(query: str) -> bool:
    """检测查询是否为无意义/垃圾输入（QUAL-1修复）。

    以下情况判定为垃圾查询：
    1. 纯随机字符串（连续5+非词典字符且无中文/常见英文单词）
    2. 极短查询（<2字符）且无中文
    3. 纯数字/纯符号串
    4. 包含少量常见词但主体为随机字符（如 zzzzzxyz123test）
    5. 中文查询中连续3+相同中文字符（如"测试测试测试"）
    6. 查询长度<=3且只包含1个垃圾词时判定为垃圾

    Returns:
        True 表示应限制返回结果数量
    """
    q = query.strip()
    if not q or len(q) < 2:
        return True

    # 中文查询不再直接放行，增加重复字符检测
    if _RE_HAS_CHINESE.search(q):
        # 检测连续3+相同中文字符（如"测测测"、"啊啊啊啊"）
        if _RE_CHINESE_REPEAT.search(q):
            return True
        # 检测重复词组（如"测试测试测试"）
        cn_chars = _RE_CHINESE_SEGMENT.findall(q)
        for segment in cn_chars:
            if len(segment) >= 2:
                for unit_len in range(1, len(segment) // 2 + 1):
                    unit = segment[:unit_len]
                    if unit * 3 == segment[: unit_len * 3]:
                        return True
        return False

    words = _RE_ENGLISH_WORDS.findall(q.lower())
    word_set = set(words)
    matched_common = word_set & _GARBAGE_COMMON_WORDS

    # 增加匹配阈值 — 查询长度<=3且只包含1个垃圾词时判定为垃圾
    if len(matched_common) >= 2:
        common_char_len = sum(len(w) for w in words if w in matched_common)
        if common_char_len / len(q) > 0.4 and len(word_set) <= len(matched_common) + 2:
            return False

    if matched_common and len(q) > 8:
        non_word_chars = _RE_ENGLISH_WORDS.sub("", q)
        noise_ratio = len(non_word_chars) / len(q)
        if noise_ratio > 0.5:
            return True

    random_chars = re.sub(r"[a-zA-Z0-9\s]", "", q)
    if len(random_chars) > len(q) * 0.6:
        return True

    if _RE_ONLY_DIGITS_SPACES.match(q):
        return True

    alpha_seq = _RE_ENGLISH_SEQ.findall(q)
    for seq in alpha_seq:
        seq_lower = seq.lower()
        if seq_lower not in _GARBAGE_COMMON_WORDS:
            vowel_count = sum(1 for c in seq_lower if c in "aeiou")
            unique_chars = len(set(seq_lower))
            if vowel_count == 0 or unique_chars <= 2:
                return True

    # 查询长度<=3且只包含1个垃圾词时判定为垃圾
    if len(matched_common) == 1 and len(q) <= 3 and len(word_set) <= 1:
        return True

    return not matched_common and _RE_ALPHANUMERIC.match(q) is not None and len(q) > 8


def trim_to_budget(results: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    """裁剪结果到 Token 预算内。"""
    budget = max_tokens
    chars_per_token = 4
    trimmed = []
    used = 0
    for r in results:
        content = r.get("content", "")
        est_tokens = max(1, len(content) // chars_per_token)
        if used + est_tokens <= budget:
            trimmed.append(r)
            used += est_tokens
        else:
            continue
    return trimmed
