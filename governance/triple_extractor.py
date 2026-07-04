"""TripleExtractor — mem0 风格的结构化三元组提取器。

相比 knowledge_graph.py 中原有的 extract_triples():
1. 复用 LLM 客户端（避免每次读取 config.yaml + 新建客户端）
2. 批量提取：缓存待处理文本，攒够批次后一次 LLM 调用处理多条
3. 与 EntityExtractor 联动，先识实体再抽取关系
4. 扩展中文关系模式，覆盖更多日常对话场景
5. 实体归一化 + 去重

与现有系统集成方式：
  - memorize handler 调用 TripleExtractor.extract() 替代 extract_triples()
  - 提取结果传给 knowledge_graph.add_triple_with_negation_check()
  - 作为替代/补充，不改变现有存储路径
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 噪声词黑名单：常见助词、代词、疑问词、副词碎片，不应作为 KG 实体
_NOISE_WORDS: set[str] = {
    "目前", "什么", "怎么", "怎样", "什么样", "为什么",
    "这个", "那个", "这", "那", "哪些", "哪个", "这里", "那里", "已经", "正在", "就是", "还有", "可以", "需要", "应该",
    "可能", "因为", "所以", "但是", "然而", "虽然", "如果", "一般", "比较",
    "非常", "很多", "一些", "部分", "各种", "不同", "大多", "通常", "经常",
    "之一", "其中", "属于", "进行", "没有", "不会", "不能", "不行",
    "自己", "别人", "大家", "有人", "所有", "任何", "其他", "另外",
    "情况", "问题", "事情", "东西", "地方", "时候", "里面",
    "也是", "也行", "也对",
    # 英文噪声
    "turn", "this", "that", "there", "here", "it", "he", "she",
    "then", "now", "what", "when", "where", "how", "why", "who",
    "已安装", "了", "着", "过", "的", "地", "得",
    # 数字/量词碎片
    "一个", "一种", "一套", "这台", "那台", "这个那个",
    "的一个", "这篇", "那篇", "这条", "那条",
    # 测试/排查类噪声
    "测试一下", "测试了", "验证一下", "排查一下",
    # 数字开头噪声
    "12月", "123", "345", "678",
}

def _is_valid_entity(token: str) -> bool:
    """检查 token 是否是有效的 KG 实体。

    过滤掉：
    - 单字或双字碎片
    - 纯数字/标点
    - 噪声词黑名单中的词（含前缀匹配）
    - 包含大量空格/换行的文本碎片
    """
    token = token.strip()
    if not token or len(token) < 2:
        return False
    # 短 token 特殊处理：2 字中文可以通过，2 字英文需要 3 字以上
    import re
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", token))
    has_alpha = bool(re.search(r"[A-Za-z]{2,}", token))
    if len(token) == 2 and not has_chinese:
        return False
    if token in _NOISE_WORDS:
        return False
    # 噪声前缀匹配：如果 token 开头匹配噪声词，则过滤
    for nw in _NOISE_WORDS:
        if len(nw) >= 2 and token.startswith(nw):
            return False
    # 不能全是空白字符
    if not token.replace("/", "").replace("-", "").replace("_", "").replace(".", "").strip():
        return False
    # 包含空格或换行的碎片
    if " " in token or "\n" in token:
        return False
    # 至少有一个中文字符或连续的英文字母序列
    if not has_chinese and not has_alpha:
        return False
    return True

# 扩展的中文关系模式（覆盖更多日常场景）
_EXTENDED_ZH_PATTERNS = [
    # 使用/采用
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:使用|采用|选用|基于|依赖|用到)\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})", "uses"),
    # 属于/归入/是/为
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:属于|归入|隶属于|是|为|作为)\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})", "belongs_to"),
    # 导致/引起/造成/容易出现
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:导致|引起|造成|触发|引发|容易出现|容易发生|容易产生)\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})", "causes"),
    # 替代/取代/升级为/改为（支持"替代了"、"替换成"等含后缀形式）
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:替代|取代|替换|更换|升级为|改为|换成)(?:了|成|掉)?\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})", "replaces"),
    # 用/通过 X 替代/替换 Y → (group1, replaces, group3)
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:用|通过)\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:替代|取代|替换|更换)(?:了|成|掉)?\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})", "replaces"),
    # 关联/连接/映射
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:连接|关联|对应|映射到|接口为)\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})", "connects_to"),
    # 包含/包括/由...组成
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:包含|包括|含有|由.*组成|内含)\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})", "contains"),
    # 位置/在/于
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:在|于)\s*([\u4e00-\u9fff]{2,10})\s*(?:中|里|上|内)", "located_in"),
    # 需要/必须/应该
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:需要|必须|应该)\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})", "requires"),
    # 推荐/建议/偏好
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:推荐|建议|偏好|喜欢|习惯用)\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})", "recommends"),
    # 配置/设置
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:配置|设置|安装|部署在)\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})", "configured_with"),
    # 解决/修复/处理
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:解决|修复|处理|应对)\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})", "resolves"),
    # 对比/优于
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:优于|胜过|好于|比.*好)\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})", "better_than"),
    # 与...集成/对接
    (r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:与|和)\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:集成|对接|打通)", "integrates_with"),
    # 通过...实现
    (r"通过\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})\s*(?:实现|达成|完成)\s*([\u4e00-\u9fffA-Za-z0-9_.-]{2,20})", "implements_via"),
]

# 英文关系模式补充
_EXTENDED_EN_PATTERNS = [
    (r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:integrates?\s+with|works?\s+with)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)", "integrates_with"),
    (r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:is\s+configured?\s+with|runs?\s+on)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)", "configured_with"),
    (r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:resolves?|fixes?|solves?)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)", "resolves"),
    (r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:recommends?|suggests?|prefers?)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)", "recommends"),
    # A in B (e.g. "PPO in RLHF")
    (r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+in\s+(\b[A-Za-z][A-Za-z0-9_.-]*)", "part_of"),
    # A for B (e.g. "RLHF for training")
    (r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+for\s+(\b[A-Za-z][A-Za-z0-9_.-]*)", "used_for"),
]

# 中英混合模式
_MIXED_PATTERNS = [
    # 中文名词+的+英文 — {2,4} 限制主语长度，避免捕获动词/连词碎片
    (r"([\u4e00-\u9fff]{2,4})\s*的\s*([A-Za-z][A-Za-z0-9_.-]*)", "has_property"),
    (r"([A-Za-z][A-Za-z0-9_.-]*)\s*的\s*([\u4e00-\u9fff]{2,4})", "has_property"),
    # 中文动词+英文技术名词
    (r"([\u4e00-\u9fff]{2,4})\s*(?:使用|配置|调用|启动|运行)\s*([A-Za-z][A-Za-z0-9_.-]{2,})", "uses"),
]

# 语义归一化映射：将同义关系映射到标准谓词
_PREDICATE_NORMALIZE: dict[str, str] = {
    "使用": "uses",
    "选用": "uses",
    "采用": "uses",
    "基于": "uses",
    "依赖": "uses",
    "用到": "uses",
    "调用": "uses",
    "属于": "belongs_to",
    "归入": "belongs_to",
    "是": "is_a",
    "为": "is_a",
    "导致": "causes",
    "引起": "causes",
    "造成": "causes",
    "触发": "causes",
    "引发": "causes",
    "容易出现": "causes",
    "容易发生": "causes",
    "容易产生": "causes",
    "替代": "replaces",
    "取代": "replaces",
    "替换": "replaces",
    "升级为": "replaces",
    "关联": "connects_to",
    "连接": "connects_to",
    "包含": "contains",
    "包括": "contains",
    "推荐": "recommends",
    "建议": "recommends",
    "偏好": "recommends",
    "解决": "resolves",
    "修复": "resolves",
    "处理": "resolves",
    "配置": "configured_with",
    "设置": "configured_with",
    "部署": "configured_with",
}


class TripleExtractor:
    """mem0 风格的结构化三元组提取器。

    特性：
    - 正则优先(快速) → LLM回退(深度)
    - 客户端复用：LLM client 只创建一次
    - 批量模式: 攒多条后一次 LLM 调用批量提取
    - 实体归一化
    """

    _BATCH_SIZE = 5  # 攒够 5 条文本后批量调用 LLM
    _BATCH_TIMEOUT = 30  # 即使不够 5 条，30 秒后也提交

    def __init__(self) -> None:
        self._llm_client: Any = None
        self._llm_model: str = "deepseek-v4-flash"
        # 批量处理队列
        self._batch_queue: deque[tuple[str, Any]] = deque()
        self._batch_lock = threading.Lock()
        self._batch_results: dict[str, list[tuple[str, str, str]]] = {}
        self._batch_last_flush = datetime.now(timezone.utc)
        # 实体提取器
        self._entity_extractor: Any = None

    def _get_llm_client(self) -> Any:
        """获取或创建复用的 LLM 客户端（避免每次读取 config）。"""
        if self._llm_client is not None:
            return self._llm_client

        try:
            import yaml
            from openai import OpenAI

            config_path = Path.home() / ".hermes" / "config.yaml"
            if not config_path.exists():
                return None
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            providers = cfg.get("providers", {})
            ds = providers.get("openai", {})
            if not ds.get("api_key"):
                return None

            self._llm_client = OpenAI(
                api_key=ds["api_key"],
                base_url=ds.get("base_url", ""),
                timeout=30,
                max_retries=2,
            )
            return self._llm_client
        except Exception as e:
            logger.debug("TripleExtractor: LLM client init failed: %s", e)
            return None

    def _get_entity_extractor(self) -> Any:
        """获取 EntityExtractor 实例。"""
        if self._entity_extractor is not None:
            return self._entity_extractor
        try:
            from omnimem.retrieval.entity_extractor import EntityExtractor

            self._entity_extractor = EntityExtractor()
        except Exception:
            self._entity_extractor = None
        return self._entity_extractor

    def extract(self, text: str, use_llm: bool = True) -> list[tuple[str, str, str]]:
        """从文本提取三元组。

        Args:
            text: 输入文本
            use_llm: 是否启用 LLM 回退

        Returns:
            List of (subject, predicate, object) tuples
        """
        if not text or len(text) < 3:
            return []

        triples: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        # 1. 正则提取（快速，覆盖已知模式）
        for pattern_set in [_EXTENDED_ZH_PATTERNS, _EXTENDED_EN_PATTERNS, _MIXED_PATTERNS]:
            for pattern, predicate in pattern_set:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        # 处理 2 组模式：(subject, object)
                        if len(match) == 2:
                            subj, obj = match[0].strip(), match[1].strip()
                        # 处理 3 组模式（如"用 X 替代 Y"→ (actor, tool, target)）
                        # 工具组(group2)替代目标组(group3)
                        elif len(match) >= 3:
                            subj, obj = match[1].strip(), match[2].strip()
                        else:
                            continue
                    else:
                        continue
                    subj = self._normalize_entity(subj)
                    obj = self._normalize_entity(obj)
                    if not subj or not obj or subj == obj:
                        continue
                    if not _is_valid_entity(subj) or not _is_valid_entity(obj):
                        continue
                    key = (subj, predicate, obj)
                    if key not in seen:
                        seen.add(key)
                        triples.append((subj, predicate, obj))

        # 2. 如果正则结果不够，使用 LLM 回退
        if use_llm and len(triples) < 2 and len(text) > 30:
            llm_triples = self._extract_via_llm(text)
            for s, p, o in llm_triples:
                if not _is_valid_entity(s) or not _is_valid_entity(o):
                    continue
                key = (s, p, o)
                if key not in seen:
                    seen.add(key)
                    triples.append((s, p, o))

        return triples

    def extract_with_entities(self, text: str) -> tuple[list[str], list[tuple[str, str, str]]]:
        """同时提取实体和三元组。

        Returns:
            (entities, triples)
        """
        triples = self.extract(text)

        # 从三元组中收集实体
        entities: set[str] = set()
        for s, _, o in triples:
            entities.add(s)
            entities.add(o)

        # 补充 EntityExtractor 的结果
        extractor = self._get_entity_extractor()
        if extractor:
            extra = extractor.extract(text, max_entities=8)
            for e in extra:
                entities.add(self._normalize_entity(e))

        return list(entities), triples

    def _extract_via_llm(self, text: str) -> list[tuple[str, str, str]]:
        """通过 LLM 提取三元组。"""
        client = self._get_llm_client()
        if not client:
            return []

        prompt = f"""从以下文本中抽取知识三元组 (subject, predicate, object)。

规则：
- 抽取最重要的 1-3 个三元组
- subject/object 是有意义的名词短语（技术名、工具、概念、人名、地名、项目名）
- predicate 是英文关系词：uses, belongs_to, causes, replaces, connects_to, contains, \
located_in, recommends, configured_with, resolves, implements_via, has_property, is_a
- 如果没有明显的关系，返回空数组

文本：
{text[:600]}

返回 JSON 数组：[{{"s":"...","p":"...","o":"..."}}]
只返回 JSON。"""

        try:
            resp = client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content or ""

            import json

            m = re.search(r"\[[\s\S]*\]", raw)
            if not m:
                return []
            items = json.loads(m.group())
            return [
                (self._normalize_entity(item["s"]), item["p"].strip(), self._normalize_entity(item["o"]))
                for item in items
                if item.get("s") and item.get("p") and item.get("o")
            ]
        except Exception as e:
            logger.debug("TripleExtractor LLM failed: %s", e)
            return []

    def _normalize_entity(self, entity: str) -> str:
        """实体归一化：去空格、统一大小写（英文）、去除标点。"""
        entity = entity.strip()
        entity = re.sub(r'\s+', ' ', entity)
        entity = re.sub(r'[，。！？、；：""''（）【】《》…—·]', '', entity)
        # 英文小写归一化
        if re.match(r'^[A-Za-z]', entity):
            entity = entity.lower()
        return entity

    # ─── 批量模式（减少 LLM 调用次数） ────────────────────────

    def batch_extract(self, texts: list[str]) -> list[list[tuple[str, str, str]]]:
        """批量提取三元组（一次 LLM 调用处理多条文本）。

        先用正则快速提取，结果不足的才进入批量 LLM。
        """
        results: list[list[tuple[str, str, str]]] = []

        # 第一遍：正则提取
        llm_candidates: list[int] = []
        for i, text in enumerate(texts):
            triples = self.extract(text, use_llm=False)
            results.append(triples)
            if len(triples) < 2 and len(text) > 30:
                llm_candidates.append(i)

        # 第二遍：LLM 批量处理正则不足的
        if llm_candidates:
            llm_texts = [texts[i] for i in llm_candidates]
            llm_results = self._extract_batch_via_llm(llm_texts)
            for idx, llm_triples in zip(llm_candidates, llm_results):
                seen = {(s, p, o) for s, p, o in results[idx]}
                for s, p, o in llm_triples:
                    key = (s, p, o)
                    if key not in seen:
                        results[idx].append((s, p, o))

        return results

    def _extract_batch_via_llm(self, texts: list[str]) -> list[list[tuple[str, str, str]]]:
        """一次 LLM 调用处理多条文本。"""
        client = self._get_llm_client()
        if not client:
            return [[] for _ in texts]

        parts = "\n\n---\n\n".join(f"[{i}]\n{t[:400]}" for i, t in enumerate(texts))

        prompt = f"""从以下 {len(texts)} 段文本中分别抽取知识三元组。

规则同单条提取，但需要为每个文本序号返回对应的三元组数组。

文本：
{parts}

返回 JSON 数组：[[{{\"s\":\"...\",\"p\":\"...\",\"o\":\"...\"}},...]]
数组长度必须等于 {len(texts)}，空文本返回 []。
只返回 JSON。"""

        try:
            resp = client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            raw = resp.choices[0].message.content or ""
            import json

            m = re.search(r"\[[\s\S]*\]", raw)
            if not m:
                return [[] for _ in texts]
            items = json.loads(m.group())
            results: list[list[tuple[str, str, str]]] = []
            for group in items:
                triples = []
                if isinstance(group, list):
                    for item in group:
                        if isinstance(item, dict) and all(k in item for k in ("s", "p", "o")):
                            triples.append((self._normalize_entity(item["s"]), item["p"].strip(), self._normalize_entity(item["o"])))
                results.append(triples)

            # 补齐不足的组
            while len(results) < len(texts):
                results.append([])
            return results
        except Exception as e:
            logger.debug("TripleExtractor batch LLM failed: %s", e)
            return [[] for _ in texts]


# ─── 全局单例 ────────────────────────────────────────────

_triple_extractor: TripleExtractor | None = None
_triple_lock = threading.Lock()


def get_triple_extractor() -> TripleExtractor:
    """获取全局 TripleExtractor 单例（复用 LLM 客户端）。"""
    global _triple_extractor
    if _triple_extractor is None:
        with _triple_lock:
            if _triple_extractor is None:
                _triple_extractor = TripleExtractor()
    return _triple_extractor
