"""三元组提取与关系推理。

职责：
  - 从文本中提取 (subject, predicate, object) 三元组
  - 正则/LLM 两层回退抽取
  - 基于已有三元组推理隐含关系
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 模块级 LLM 客户端引用（由外部注入，避免绕过 LLMBackend 直连）
_llm_client: Any = None


def set_llm_client(client: Any) -> None:
    """注入 LLM 客户端，供 _extract_triples_llm 使用。"""
    global _llm_client
    _llm_client = client


# ─── 关系模式：从文本中提取 (主语, 关系, 宾语) 三元组 ─────────────────
_RELATION_PATTERNS = [
    # ── 中文关系 ──────────────────────────────────────────────

    # "A 使用/采用/基于 B"
    (
        r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,12})\s*(?:使用|采用|选用|基于|依赖|运行在)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "uses",
    ),
    # "A 属于/归入 B"
    (
        r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,12})\s*(?:属于|归入|隶属于)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "belongs_to",
    ),
    # "A 导致/引起/造成/触发/容易出现 B"
    (
        r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,12})\s*(?:导致|引起|造成|触发|容易出现|容易发生|容易产生)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "causes",
    ),
    # "A 替代/取代/替换/升级为 B"（支持"替代了"、"替换成"等后缀）
    (
        r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,12})\s*(?:替代|取代|替换|升级为)(?:了|成|掉)?\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "replaces",
    ),
    # "A 用/通过 B 替代/替换 C" → (A, replaces, C)
    (
        r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,12})\s*(?:用|通过)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)\s*(?:替代|取代|替换)(?:了|成|掉)?\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "replaces",
    ),
    # "A 优于/胜过/好于 B"
    (
        r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,12})\s*(?:优于|胜过|好于)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "better_than",
    ),
    # "A 包含/包括/由...组成 B"
    (
        r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,12})\s*(?:包含|包括|由.*组成)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "contains",
    ),
    # "A 在/于 B 中/里/上"
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,12})\s*(?:在|于)\s*([\u4e00-\u9fff]{2,8})\s*(?:中|里|上)", "located_in"),
    # "A 需要/必须/应该 B"（如 "需要 KL 散度"）
    (
        r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,12})\s*(?:需要|必须|应该)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "requires",
    ),
    # "A 连接/关联/对应/映射到 B"
    (
        r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,12})\s*(?:连接|关联|对应|映射到)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "connects_to",
    ),

    # ── 英文关系 ──────────────────────────────────────────────

    # "A uses/depends on/relies on B"
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:uses?|depends?\s+on|relies?\s+on)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "uses",
    ),
    # "A causes/leads to/triggers B"
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:causes?|leads?\s+to|triggers?)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "causes",
    ),
    # "A replaces/supersedes B"
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:replaces?|supersedes?)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "replaces",
    ),
    # "A contains/includes B"
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:contains?|includes?)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "contains",
    ),
    # "A in B" (e.g. "PPO in RLHF")
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+in\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "part_of",
    ),
    # "A for B" (e.g. "RLHF for training")
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+for\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "used_for",
    ),
]

# 否定关系模式
_NEGATION_PATTERNS = [
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:不|并非|没有|无法|不能)\s*(?:使用|采用|依赖|支持)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "not_uses",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:不同于|区别于|不是)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "differs_from",
    ),
]


def extract_triples(text: str, use_llm: bool = True) -> list[tuple[str, str, str]]:
    """从文本中提取 (主语, 关系, 宾语) 三元组。

    优先使用 TripleExtractor（正则→LLM 两层回退），
    复用 LLM 客户端避免每次重连。
    Returns: List of (subject, predicate, object) tuples
    """
    try:
        from omnimem.governance.triple_extractor import get_triple_extractor

        extractor = get_triple_extractor()
        return extractor.extract(text, use_llm=use_llm)
    except ImportError:
        pass

    # 回退：原有正则 + LLM 逻辑
    triples: list[tuple[str, str, str]] = []

    # 1. 正则提取（快速）
    for pattern, predicate in _RELATION_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                subj, obj = match[0], match[1]
                if subj and obj and subj != obj:
                    triples.append((subj, predicate, obj))

    # 否定关系
    for pattern, predicate in _NEGATION_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                subj, obj = match[0], match[1]
                if subj and obj and subj != obj:
                    triples.append((subj, predicate + "_not", obj))

    # 2. LLM 回退：正则结果不足 2 条且文本较长时使用
    if use_llm and len(triples) < 2 and len(text) > 30:
        try:
            llm_triples = _extract_triples_llm(text)
            # 去重合并
            existing = {(s.lower(), p.lower(), o.lower()) for s, p, o in triples}
            for s, p, o in llm_triples:
                key = (s.lower(), p.lower(), o.lower())
                if key not in existing:
                    triples.append((s, p, o))
                    existing.add(key)
        except Exception as e:
            logger.debug("LLM triple extraction failed, using regex only: %s", e)

    return triples


def _extract_triples_llm(text: str) -> list[tuple[str, str, str]]:
    """使用 LLM 从文本中抽取知识三元组（轻量级，仅在正则不足时调用）。

    通过模块级注入的 _llm_client 调用，不再自建 OpenAI 连接、
    自读 config.yaml 或硬编码模型名。
    """
    if _llm_client is None:
        logger.debug("LLM client not injected, skipping KG triple extraction")
        return []

    prompt = f"""从以下文本中抽取知识三元组 (subject, predicate, object)。
要求：
- 抽取 1-3 个最重要的三元组
- subject/object 是有意义的实体（技术名、工具、概念、人名、地名）
- predicate 是中文或英文关系词（使用、属于、推荐、避免、解决、配置、导致等）
- 跳过信息量不足的文本

文本：
{text[:500]}

返回 JSON 数组：[{{"s": "...", "p": "...", "o": "..."}}]
只返回 JSON。"""

    try:
        # 使用注入的 LLM 客户端（AsyncLLMClient 或 LLMBackend）
        resp = _llm_client.call_sync(prompt, max_tokens=500, temperature=0.3)
        raw = resp.content if hasattr(resp, 'content') else str(resp)
    except Exception as e:
        logger.debug("LLM triple extraction failed: %s", e)
        return []

    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return []
    try:
        items = json.loads(m.group())
    except json.JSONDecodeError:
        return []
    return [
        (item["s"].strip(), item["p"].strip(), item["o"].strip())
        for item in items
        if item.get("s") and item.get("p") and item.get("o")
    ]


def infer_relations(existing_triples: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """基于已有三元组推理隐含关系。

    推理规则:
      - 传递性: A uses B, B uses C → A uses C (transitive)
      - 互逆: A belongs_to B → B contains A
      - 替代链: A replaces B, B replaces C → A replaces C
    """
    inferred: list[tuple[str, str, str]] = []

    # 建立主语→(关系→宾语)的索引
    subj_map: dict[str, dict[str, list[str]]] = {}
    for t in existing_triples:
        s, p, o = t.get("subject", ""), t.get("predicate", ""), t.get("object", "")
        if not s or not p or not o:
            continue
        subj_map.setdefault(s, {}).setdefault(p, []).append(o)

    # 传递性推理: uses, causes, replaces
    transitive_preds = {"uses", "causes", "replaces"}
    for subj, pred_map in subj_map.items():
        for pred in transitive_preds:
            if pred in pred_map:
                for obj in pred_map[pred]:
                    # obj 的关系传递到 subj
                    if obj in subj_map and pred in subj_map[obj]:
                        for trans_obj in subj_map[obj][pred]:
                            if trans_obj != subj:  # 避免循环
                                inferred.append((subj, pred, trans_obj))

    # 互逆推理: belongs_to ↔ contains
    for subj, pred_map in subj_map.items():
        if "belongs_to" in pred_map:
            for obj in pred_map["belongs_to"]:
                inferred.append((obj, "contains", subj))

    return inferred
