"""实体抽取与归一化。

职责：
  - 从文本中提取候选实体（jieba + 规则正则 + 人名检测）
  - 实体查询归一化
  - POLE+O 实体分类
  - LLM 深度实体/三元组抽取回退
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _normalize_kg_entity(entity: str) -> str:
    """知识图谱实体查询归一化：与 TripleExtractor 保持一致。

    去除首尾空格、内部连续空格，英文开头则小写。
    """
    entity = entity.strip()
    entity = re.sub(r"\s+", " ", entity)
    entity = re.sub(r"[，。！？、；：\"\"''（）【】《》…—·]", "", entity)
    if re.match(r"^[A-Za-z]", entity):
        entity = entity.lower()
    return entity


# ─── 实体抽取模式 ────────────────────────────────────────────

# 中文实体模式：人名/地名/机构/技术术语
_ZH_ENTITY_PATTERNS = [
    r"[\u4e00-\u9fff]{2,4}(?=公司|团队|项目|系统|框架|平台|模块|服务|接口|数据库)",  # 组织/系统名
    r"(?<=用户|客户|同事|老板|领导|朋友)[\u4e00-\u9fff]{2,3}",  # 人名
]

# 英文实体模式
_EN_ENTITY_PATTERNS = [
    r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b",  # CamelCase
    r"\b[A-Z]{2,}\b",  # 缩写 API, SQL, etc.
    r"\b[a-z]+(?:-[a-z]+)+\b",  # kebab-case
    # 技术名词：不用 \b（中英混合时无效），但用前后断言防止子串匹配
    r"(?<![A-Za-z])(Python|Java|Go|Rust|TypeScript|React|Vue|Docker|K8s|Redis|MySQL|PostgreSQL|MongoDB|Neo4j|ChromaDB|SQLite)(?![A-Za-z])",
]

# 通用实体模式：从关系三元组中提取的实体
_GENERIC_ENTITY_PATTERNS = [
    # 中文关键词前面的名词（如"前端使用React"中的"前端"）
    r"(?<=[，。、\s])[\u4e00-\u9fff]{2,6}(?=使用|采用|选用|基于|依赖|运行)",
    # 中文关键词后面的英文技术名词
    r"(?:使用|采用|选用|基于|依赖)\s*([A-Z][A-Za-z0-9_.-]*)",
]


# ─── 噪声实体过滤集（借鉴 EntityExtractor._GENERIC_NOUNS） ────
# 用于过滤 jieba 和正则提取中产生的无意义 token
_NOISE_ENTITIES: set[str] = {
    # 通用概念词（不含有效信息）
    "问题", "方法", "方案", "结果", "数据", "功能", "配置", "部署",
    "测试", "需求", "设计", "架构", "实现", "开发", "优化", "模块",
    "系统", "服务", "接口", "组件", "引擎", "管道", "通道", "版本",
    "发布", "编辑", "文档", "搜索", "索引", "缓存", "进程", "线程",
    # 虚词/连词/碎片
    "这个", "那个", "这些", "那些", "这里", "那里", "什么", "怎么",
    "需要正确", "支持需要", "进行", "使用", "通过", "包括", "相关", "关于",
    "方面", "内容", "信息", "操作", "方式", "时间", "地方",
    "东西", "事情", "部分", "类型", "状态",
    # 常见人名检测误报（姓氏+通用词开头）
    "关于遗", "对于", "由于", "在于", "基于", "鉴于",
    "包名冲", "这条记忆",
    # 英文虚词
    "the", "this", "that", "with", "from", "into", "about",
    "than", "also", "very", "just",
}

# ─── jieba 技术词典（惰性注册一次） ────────────────────────────
# 覆盖 KG/OmniMem 领域的专有词，确保 jieba 不切碎
_TECH_DICT: list[str] = [
    "三元组支持", "知识图谱", "联想扩散", "语义扩散",
    "PrimingCache", "AssociativeSpreader",
    "RRFFusion", "CrossEncoder", "HybridRetriever",
    "感知引擎", "证据组", "多跳查询", "遗忘曲线",
    "启动效应", "启动缓存", "上下文压缩", "内存管理",
    "数据目录", "命名空间", "包名冲突",
    "RLHF", "GRPO", "PPO", "KL散度", "reward_hacking",
    "Qoder", "Hermes", "OmniMem", "WeRSS",
]

_tech_dict_registered: bool = False


def _ensure_tech_dict() -> None:
    """向 jieba 注册技术词典，仅首次调用生效。"""
    global _tech_dict_registered
    if _tech_dict_registered:
        return
    try:
        import jieba

        for term in _TECH_DICT:
            jieba.suggest_freq(term, tune=True)
        _tech_dict_registered = True
        logger.debug("jieba tech dict registered: %d terms", len(_TECH_DICT))
    except ImportError:
        pass


# jieba 词性标注中可参与 N-gram 合并的内容词
# 只保留名词类（n/vn/nz/eng/ns/nt/nr），排除动词（v）避免动名碎片
_MERGEABLE_POS: set[str] = {"n", "vn", "nz", "eng", "ns", "nt", "nr"}

# ─── 裸中文人名检测（预编译常量，避免每次调用重建）──────────
_CN_SURNAMES = (
    "王李张刘陈杨黄赵周吴徐孙马胡朱郭何罗高林郑梁谢唐许冯宋韩邓彭曹曾田萧"
    "潘袁蔡蒋余于杜叶程苏魏吕丁任卢姚钟姜崔谭陆汪范金石廖贾夏韦付方白邹孟"
    "熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤温康施文"
    "牛樊葛邢安齐易乔伍庞颜倪庄聂章鲁岳翟殷詹申欧耿关兰焦俞左柳甘祝包宁尚"
)
_CN_MULTI_SURNAMES = "司马|欧阳|上官|皇甫|诸葛|令狐|司徒"
_NAME_BREAK_CHARS = set(
    "的在了和是就也都很到说要去了不有这那个什么怎么"
    "可以需要通过使用进行包括负责参与处理完成检查确认"
    "执行部署配置优化升级返回发送接收"
)
# 预编译：单姓+1~2个中文字符（简单字符类，快速匹配）
_SINGLE_NAME_RE = re.compile(rf"[{_CN_SURNAMES}][\u4e00-\u9fff]{{1,2}}")
_MULTI_NAME_RE = re.compile(rf"({_CN_MULTI_SURNAMES})[\u4e00-\u9fff]{{0,2}}")

# 概念词（不应被识别为人名）
_CONCEPT_WORDS: set[str] = {
    "问题", "方法", "方案", "结果", "数据", "功能", "配置", "部署",
    "测试", "需求", "设计", "架构", "实现", "开发", "优化", "模块",
    "系统", "服务", "接口", "组件", "引擎", "管道", "通道", "版本",
}


def extract_entities(text: str) -> list[str]:
    """从文本中提取实体。

    ★ P1方案四：优先使用 jieba 分词 + 词性标注（若可用），
    与规则正则互补，提升通用命名实体覆盖率。
    增强：技术词典注册 + N-gram 合并 + 噪声过滤。
    """
    entities: set[str] = set()

    # 注册技术词典，确保 jieba 不切碎技术专有词
    _ensure_tech_dict()

    # 优先路径：jieba NER + 全词表收集（供后续 N-gram 合并）
    _all_jieba_words: list[tuple[str, str]] = []
    try:
        import jieba.posseg as pseg

        for word, flag in pseg.lcut(text):
            if len(word) < 2:
                continue
            _all_jieba_words.append((word, flag))
            if flag in ("nr", "ns", "nt", "nz") and word not in _NOISE_ENTITIES:
                entities.add(word)
    except ImportError:
        pass

    # N-gram 合并：相邻内容词组成复合实体
    for i in range(len(_all_jieba_words) - 1):
        w1, p1 = _all_jieba_words[i]
        w2, p2 = _all_jieba_words[i + 1]
        if p1 in _MERGEABLE_POS and p2 in _MERGEABLE_POS:
            candidate = w1 + w2
            if 2 <= len(candidate) <= 12 and candidate not in _NOISE_ENTITIES:
                entities.add(candidate)

    # 中文实体（规则正则，与 jieba 互补）
    for pattern in _ZH_ENTITY_PATTERNS:
        matches = re.findall(pattern, text)
        entities.update(matches)

    # 英文实体
    for pattern in _EN_ENTITY_PATTERNS:
        matches = re.findall(pattern, text)
        entities.update(matches)

    # 通用实体模式
    for pattern in _GENERIC_ENTITY_PATTERNS:
        matches = re.findall(pattern, text)
        entities.update(matches)

    # 从三元组中提取的实体（主语和宾语也是实体）
    # 限制长度 ≤8 避免正则贪婪捕获的长碎片
    # use_llm=False 避免每次调用都发起 LLM API 请求
    # 延迟导入 extraction.extract_triples 避免循环依赖
    from omnimem.deep.kg.extraction import extract_triples

    triples = extract_triples(text, use_llm=False)
    for subj, _, obj in triples:
        if len(subj) <= 8 and subj not in _NOISE_ENTITIES:
            entities.add(subj)
        if len(obj) <= 8 and obj not in _NOISE_ENTITIES:
            entities.add(obj)

    # ★ 裸中文人名检测（预编译正则，替换原滑动窗口循环）
    seen_names: set[str] = set()
    for match in _SINGLE_NAME_RE.finditer(text):
        cand = match.group(0)
        if cand in entities or cand in seen_names:
            continue
        if cand in _CONCEPT_WORDS:
            continue
        if cand[-1] in _NAME_BREAK_CHARS:
            continue
        entities.add(cand)
        seen_names.add(cand)
    # 复姓（司马/欧阳等）+0~2中文字
    for match in _MULTI_NAME_RE.finditer(text):
        cand = match.group(0)
        if len(cand) < 3:
            continue
        if cand in entities or cand in seen_names:
            continue
        if cand in _CONCEPT_WORDS:
            continue
        entities.add(cand)
        seen_names.add(cand)

    # 去除太短的实体和噪声词
    return [e for e in entities if len(e) >= 2 and e not in _NOISE_ENTITIES]


# ─── POLE+O 实体分类 (参考 neo4j-labs/agent-memory) ────────

_POLEO_TYPES: dict[str, str] = {
    "person": "Person",
    "org": "Organization",
    "location": "Location",
    "event": "Event",
    "object": "Object",
}


def _classify_entity_poleo(name: str) -> str:
    """快速规则分类实体到 POLE+O 类型。LLM 提取后可覆盖此分类。

    优先级: Location/Organization已知 > Person > 后缀匹配Organization > Location > Event > Object
    """
    # ── 已知地名（优先级最高，避免姓氏误判）──
    known_locations = {"北京", "上海", "深圳", "杭州", "广州", "成都", "浙江", "金华",
                       "中国", "美国", "日本", "旧金山", "硅谷"}
    if name in known_locations:
        return "location"

    # ── 已知组织名（优先级高于人名模式）──
    known_orgs = {"OpenAI", "Google", "Meta", "Microsoft", "Apple", "Amazon",
                  "Anthropic", "Nous Research", "Hermes", "GitHub"}
    if name in known_orgs:
        return "org"

    # ── Person: 中文人名 + 角色称谓 ──
    _cn_surnames = (
        "王李张刘陈杨黄赵周吴徐孙马胡朱郭何罗高林郑梁谢唐许冯宋韩邓彭曹曾田萧"
        "潘袁蔡蒋余于杜叶程苏魏吕丁任卢姚钟姜崔谭陆汪范金石廖贾夏韦付方白邹孟"
        "熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤温康施文"
        "牛樊葛邢安齐易乔伍庞颜倪庄聂章鲁岳翟殷詹申欧耿关兰焦俞左柳甘祝包宁尚"
        "司马欧阳上官皇甫诸葛令狐司徒"
    )
    if len(name) >= 2 and len(name) <= 4 and re.match(r"^[\u4e00-\u9fff]+$", name):
        # 排除明确非人名的概念词
        concept_words = {
            "问题", "方法", "方案", "结果", "数据", "功能", "配置", "部署",
            "测试", "需求", "设计", "架构", "实现", "开发", "优化", "模块",
            "系统", "服务", "接口", "组件", "引擎", "管道", "通道",
        }
        # 姓氏匹配 或 结尾为常见人名后缀
        if name[0] in _cn_surnames and name not in concept_words:
            return "person"
        # 称谓后缀: 总/工/老师/经理/教授/博士/老板/先生/女士
        if re.search(r"(总|工|老师|经理|教授|博士|老板|先生|女士|同学)$", name):
            return "person"

    # 英文人名: First Last (CamelCase with space concept)
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", name):
        return "person"

    # ── Organization ──
    org_suffixes = (
        "公司|团队|部门|组|实验室|小组|工作室|组织|中心|研究院|研究所|"
        "学院|大学|集团|银行|基金|协会|联盟|社区|平台"
    )
    if re.search(f"({org_suffixes})$", name):
        return "org"

    # ── Location ──
    location_suffixes = "市|省|区|县|镇|乡|村|街道|路|大厦|广场|园区|国家|大陆|岛|湾|湖|山|海"
    if re.search(f"({location_suffixes})$", name):
        return "location"

    # ── Event ──
    event_keywords = (
        "会议|峰会|大会|发布会|上线|部署|测试|回归|修复|审查|合并|"
        "发布|版本|里程碑|迭代|冲刺|评审|复盘|事故|故障|演练"
    )
    if re.search(f"({event_keywords})", name):
        return "event"

    # ── Object (默认) ──
    return "object"


def extract_entities_llm(
    texts: list[str],
    llm_call: Callable[[str], str] | None = None,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """使用 LLM 深度提取实体和三元组（POLE+O 分类版）。

    设计: 批量处理，后台异步触发，不影响主路径的正则快速提取。
    无 LLM 客户端时回退到规则分类。

    Args:
        texts: 待提取的记忆文本列表
        llm_call: LLM 调用函数（接收prompt返回text），可选

    Returns:
        (entities, triples) — entities 带 POLE+O 分类前缀
    """
    if not texts or not llm_call:
        all_entities: set[str] = set()
        all_triples: list[tuple[str, str, str]] = []
        for text in texts:
            all_entities.update(extract_entities(text))
            # 延迟导入避免循环依赖
            from omnimem.deep.kg.extraction import extract_triples

            all_triples.extend(extract_triples(text))
        classified = [f"{_classify_entity_poleo(e)}:{e}" for e in all_entities]
        return classified, all_triples

    prompt = (
        "Extract entities and relationships from the following texts. "
        "Classify each entity into: Person, Organization, Location, Event, Object. "
        "Return JSON: "
        '{"entities": [{"name": "...", "type": "..."}], '
        '"triples": [["subject", "predicate", "object"]]}'
        "\n\nTexts:\n"
    )
    combined = "\n---\n".join(texts[:5])
    prompt += combined[:3000]

    try:
        response = llm_call(prompt)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in response")
        data = json.loads(json_match.group())
        entities = data.get("entities", [])
        triples_data = data.get("triples", [])

        classified_entities = [
            f"{e.get('type', 'Object')}:{e.get('name', '')}" for e in entities
        ]
        llm_triples = [
            (t[0], t[1], t[2]) for t in triples_data if len(t) >= 3
        ]
        return classified_entities, llm_triples
    except Exception as e:
        logger.warning("LLM entity extraction failed, fallback regex: %s", e)
        all_entities = set()
        all_triples: list[tuple[str, str, str]] = []
        for text in texts:
            all_entities.update(extract_entities(text))
            from omnimem.deep.kg.extraction import extract_triples

            all_triples.extend(extract_triples(text))
        return [f"Object:{e}" for e in all_entities], all_triples
