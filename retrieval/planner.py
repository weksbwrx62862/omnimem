"""查询规划器（HMS Planner 的规则版实现）。

HMS 的核心启示：Agent 记忆问题不在"记不住"而在"找不到"——
检索应是根据查询意图驱动的证据构建，而非所有通道等权一刀切。

本模块实现纯规则版 Planner：解析查询意图（时间/实体/计数/偏好/寒暄），
输出各检索通道的差异化权重乘数，替代"6 通道全跑 + 等权融合"。

设计约束:
  - 纯规则 + 正则，零 LLM 调用、零延迟（LLM 版可平滑升级，接口不变）
  - 输出为**局部权重**（channel_weights 乘数），绝不写共享 _source_weights，
    避免并发检索互相污染（对应 C7 修复的教训）
  - 未命中任何意图 → is_default=True（全 1.0），调用方可直接跳过，零开销
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── 意图信号正则（模块级预编译，避免每次调用重复编译） ──

# 计数类：how many/how much/几个/多少（LongMemEval 已知瓶颈，BM25 加权最有效）
_RE_COUNT = re.compile(
    r"\b(how many|how much|count|total|number of)\b|"
    r"几个|多少(?:次|条|笔|项|个)?|几次|几条|几项|几笔|"
    r"数量|总数|统计|频率",
    re.IGNORECASE,
)

# 时间类：绝对/相对时间窗（HMS Temporal 算子）
_RE_TEMPORAL = re.compile(
    r"今天|昨天|前天|明天|上周|本周|这周|下周|上个月|这个月|这月|下个月|"
    r"上月|本月|下月|去年|今年|明年|最近|之前|以前|当时|那阵子|那段时间|"
    r"上回|前些天|前几天|上星期|上礼拜|"
    r"星期[一二三四五六日天]|周[一二三四五六日天]|"
    r"\d{4}年|\d{1,2}月|\d{1,2}日|\d{1,2}号|"
    r"\b(yesterday|today|ago|recent|previous|earlier|"
    r"last|this|next)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

# 实体类：引号专名 / 中文称谓 / 组织后缀 / CamelCase（HMS Entity-bridge 算子）
_RE_ENTITY_QUOTED = re.compile(r'["“「『]([^"”」』]{2,20})["”」』]')
_RE_ENTITY_ZH_SUFFIX = re.compile(
    r"[\u4e00-\u9fff]{1,8}(?:总|经理|老师|博士|医生|先生|女士|主任|部长|所长|院长|主席|工程师|同学)"
)
# 2-4 字中文人名（常见姓氏开头 + 名），要求后接动作/语境词。
# 中文 2 字词大量以姓氏开头（分析/方案/安排/处理/讨论/经济…），
# 因此只匹配"姓氏+名+语境词"的三元组（如"王明说""小李去了"），
# 且整体不得是常见动词/名词组合。单独 2 字（如"王明"）不判定。
_RE_ENTITY_ZH_NAME = re.compile(
    r"(?:[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    r"谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛"
    r"雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪"
    r"祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危"
    r"江童颜郭梅盛林刁钟徐邱骆高夏蔡田胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁"
    r"宣贲邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊於惠甄麹家封芮羿储靳汲"
    r"邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘斜厉戎祖"
    r"武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴郁胥"
    r"能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀郏浦尚农温"
    r"别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广"
    r"禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢"
    r"关蒯相查后荆红游竺权逯盖益桓公])"
    r"(?P<name>[\u4e00-\u9fff]{1,2})"
    r"(?P<ctx>[的说了问讲去过吗呢阿呀说|]|[和跟与同]|\\s|$)"
)
# 常见"姓氏开头"的动词/名词组合，命中则不是人名（白名单排除）
_RE_NAME_FALSE_POSITIVE = re.compile(
    r"^(分析|方案|安排|处理|讨论|经济|解释|介绍|建议|教育|建设|设计|计划|"
    r"解决|决定|表达|表明|表示|要求|需要|研究|开发|管理|经营|监督|动员|"
    r"劳动|告诉|请求|支持|控制|操作|执行|运行|跟踪|检查|测试|比较|统计)"
)
_RE_ENTITY_ORG = re.compile(
    r"[\u4e00-\u9fff]{2,12}(?:公司|集团|银行|医院|大学|学院|科技|证券|基金|研究院|"
    r"事务所|实验室|工作室|团队|机构|中心)"
)
_RE_ENTITY_CAMEL = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:[A-Z][a-zA-Z]{2,})*\b")

# 偏好类：查询本身表达偏好意图（配合现有 preference_rewrite 使用）
_RE_PREFERENCE = re.compile(
    r"推荐|建议|喜欢|偏好|倾向|适合|"
    r"\b(prefer|recommend|suggest|best|ideal|favourite|favorite|suitable)\b",
    re.IGNORECASE,
)

# 寒暄类：无信息量查询（问候/致谢），应快速低噪声召回
_RE_CONVERSATIONAL = re.compile(
    r"^(你好|您好|hello|hi|hey|在吗|谢谢|感谢|thanks|再见|拜拜|"
    r"早上好|中午好|晚上好|辛苦了|好的|嗯|哦|ok|好)\b",
    re.IGNORECASE,
)

# 全部已知通道名（只对存在的通道生效，未知通道由调用方忽略）
_KNOWN_CHANNELS: tuple[str, ...] = (
    "vector", "bm25", "catalog", "graph", "temporal", "entity",
)

# 通道裁剪阈值：权重低于该值的通道直接跳过（不发起检索）
CHANNEL_SKIP_THRESHOLD: float = 0.3


@dataclass
class QueryPlan:
    """查询规划结果。

    Attributes:
        intent: 命中的意图（general/temporal/entity/count/preference/conversational/mixed）
        channel_weights: 通道名 → 权重乘数（仅含调用方关心的已知通道，值域 [0.0, 3.0]）
        top_k_scale: 召回规模缩放（<1.0 表示削减候选数）
        rationale: 命中理由（可追踪/调试）
        is_default: True 表示无任何加权（全 1.0），调用方可直接跳过
    """

    intent: str = "general"
    channel_weights: dict[str, float] = field(default_factory=dict)
    top_k_scale: float = 1.0
    rationale: str = ""
    is_default: bool = True

    def weight_for(self, channel: str) -> float:
        """查询某通道的权重乘数（未配置则 1.0）。"""
        return self.channel_weights.get(channel, 1.0)


class QueryPlanner:
    """规则版查询规划器：query → QueryPlan。"""

    def plan(self, query: str) -> QueryPlan:
        """解析查询意图，生成通道权重计划。

        可多意图组合（如"上周处理了几个问题"= temporal + count 同时加权）。
        """
        if not query or not query.strip():
            return QueryPlan()

        hits: list[str] = []
        q = query.strip()

        if _RE_COUNT.search(q):
            hits.append("count")
        if _RE_TEMPORAL.search(q):
            hits.append("temporal")
        if _RE_PREFERENCE.search(q):
            hits.append("preference")
        if self._is_entity_query(q):
            hits.append("entity")

        if not hits:
            # 无信息量寒暄 → 削减召回规模，快速返回
            if _RE_CONVERSATIONAL.search(q) and len(q) <= 16:
                weights = dict.fromkeys(_KNOWN_CHANNELS, 0.5)
                return QueryPlan(
                    intent="conversational",
                    channel_weights=weights,
                    top_k_scale=0.6,
                    rationale="寒暄/无信息量查询，削减通道权重与召回规模",
                    is_default=False,
                )
            return QueryPlan()  # general：全 1.0，调用方零开销跳过

        weights = self._compose_weights(hits)
        intent = "mixed" if len(hits) > 1 else hits[0]
        rationale = "意图命中: " + "+".join(hits)
        return QueryPlan(
            intent=intent,
            channel_weights=weights,
            top_k_scale=1.0,
            rationale=rationale,
            is_default=False,
        )

    # ── 内部 ──

    @staticmethod
    def _is_entity_query(query: str) -> bool:
        """启发式实体检测：引号专名 / 中文称谓 / 组织后缀 / CamelCase。"""
        if _RE_ENTITY_QUOTED.search(query):
            return True
        if _RE_ENTITY_ZH_SUFFIX.search(query):
            return True
        if _RE_ENTITY_ZH_NAME.search(query) and not _RE_NAME_FALSE_POSITIVE.search(query):
            return True
        if _RE_ENTITY_ORG.search(query):
            return True
        if _RE_ENTITY_CAMEL.search(query):
            return True
        return False

    @staticmethod
    def _compose_weights(hits: list[str]) -> dict[str, float]:
        """按命中的意图集合合成通道权重（乘数叠加）。"""
        weights: dict[str, float] = dict.fromkeys(_KNOWN_CHANNELS, 1.0)

        for intent in hits:
            if intent == "count":
                # COUNT 类查询：BM25 对精确数字敏感，大幅加权词法通道
                weights["bm25"] *= 2.5
            elif intent == "temporal":
                # 时序类查询：temporal 通道主导，graph 辅助（时间线常关联实体）
                weights["temporal"] *= 3.0
                weights["graph"] *= 1.3
            elif intent == "entity":
                # 实体类查询：graph 多跳 + entity 通道主导（HMS 中 Graph 算子贡献最大）
                weights["graph"] *= 2.5
                weights["entity"] *= 2.0
                weights["vector"] *= 1.2
            elif intent == "preference":
                # 偏好类：语义匹配更有效，小幅加权 vector
                weights["vector"] *= 1.2

        return weights
