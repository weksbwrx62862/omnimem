#!/usr/bin/env python
"""SF（Selective Forgetting，选择性遗忘）专项自测 runner。

参照 MemoryAgentBench FactConsolidation 设计思想，自建矛盾事实序列，
验证 OmniMem 冲突仲裁在 single-hop 与 multi-hop 上的真实水位：

  - 矛盾事实对：旧事实（序号小）→ 干扰事实若干 → 新矛盾事实（序号大），
    全部包装成「Fact #N: ...」自然语句按序号递增注入（模拟增量注入）
  - single-hop：直接问被更新的事实，期望取最新值
  - multi-hop：被替换事实参与推理链（A→B→答案），期望答案随新链改变
  - 判分：检索层 SubEM（子串精确匹配）——新事实（含答案链）出现且
    旧矛盾事实不出现 / 排名靠后 → 记为新事实生效
  - 二段归因：融合层（retriever.search 出口）与阅读层（recall memories 出口，
    含 ContextManager.refine 语义去重）分开记录，区分「检索丢失」与「阅读丢失」
  - 行业水位：MemoryAgentBench 实证所有方法 SF multi-hop 最高仅 28%，
    结果仅记录、不设阈值

用法:
    cd ~/.hermes/plugins/omnimem
    .venv/bin/python benchmarks/diagnosis/sf_selftest.py            # 全量 50 题
    .venv/bin/python benchmarks/diagnosis/sf_selftest.py --limit 1  # 冒烟测试
    .venv/bin/python benchmarks/diagnosis/sf_selftest.py --cases mh --keep-storage

输出:
    benchmarks/results/diagnosis/sf_selftest_details.jsonl  逐题明细
    benchmarks/results/diagnosis/sf_selftest_scores.json    分项分数与归因汇总
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# 项目根目录（脚本位于 benchmarks/diagnosis/ 下，根为上上级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ======================================================================
# 性能设施：模型加载缓存（仅测试进程内生效，不修改生产代码/行为）
# 每题需新建 SDK（独立存储目录），若无缓存则每题重复加载
# SentenceTransformer（~13s）/ CrossEncoder（~20s），50 题不可行；
# 缓存后模型权重跨题复用，嵌入/排序结果与逐题加载完全一致。
# ======================================================================
_ST_MODEL_CACHE: dict[Any, Any] = {}


def _patch_model_loading() -> None:
    """对 sentence_transformers 的模型构造加进程级缓存（纯性能优化）。"""
    try:
        import sentence_transformers
    except ImportError:
        return

    _orig_st = sentence_transformers.SentenceTransformer
    _orig_ce = sentence_transformers.CrossEncoder

    def _cached_st(path, *a, **kw):
        key = str(path)
        if key not in _ST_MODEL_CACHE:
            _ST_MODEL_CACHE[key] = _orig_st(path, *a, **kw)
        return _ST_MODEL_CACHE[key]

    def _cached_ce(path, *a, **kw):
        key = str(path)
        if key not in _ST_MODEL_CACHE:
            _ST_MODEL_CACHE[key] = _orig_ce(path, *a, **kw)
        return _ST_MODEL_CACHE[key]

    sentence_transformers.SentenceTransformer = _cached_st
    sentence_transformers.CrossEncoder = _cached_ce


_patch_model_loading()

from omnimem.sdk import OmniMemSDK  # noqa: E402

logger = logging.getLogger("sf_selftest")

# ======================================================================
# 一、自测数据构造（全部内嵌生成，无外部数据文件）
# ======================================================================

# 无关领域干扰事实池（天气/美食/地理/历史/科学/体育/艺术等，
# 不含任何题目实体名，且不含否定词以免自身误触发冲突仲裁）
_DISTRACTOR_POOL: list[str] = [
    "昨天东京下了整整一天的雨",
    "撒哈拉沙漠是世界最大的热带沙漠",
    "北极的冬季会出现极夜现象",
    "昆明的气候四季如春",
    "台风多发生在夏秋季节的西北太平洋",
    "四川火锅以麻辣鲜香著称",
    "意大利的那不勒斯是披萨的发源地",
    "日本寿司讲究鱼肉的新鲜度",
    "法国的鹅肝是传统美食之一",
    "兰州拉面讲究一清二白三红四绿",
    "珠穆朗玛峰是世界上海拔最高的山峰",
    "尼罗河是世界上最长的河流",
    "意大利的首都是罗马",
    "挪威的峡湾景观闻名世界",
    "敦煌莫高窟保存着大量佛教壁画",
    "万里长城全长超过两万公里",
    "丝绸之路连接了中国与欧洲",
    "巴黎的埃菲尔铁塔建于1889年",
    "兵马俑是秦始皇陵的陪葬坑",
    "古埃及人建造了巨大的金字塔",
    "水在标准大气压下100摄氏度沸腾",
    "光速约为每秒30万公里",
    "鲸鱼属于哺乳动物",
    "蜂蜜是少数不会变质的食物之一",
    "量子计算机使用量子比特进行运算",
    "大熊猫的主要食物是竹子",
    "蜂鸟是唯一能够倒飞的鸟类",
    "人体骨骼共有206块",
    "巴西足球队获得过五次世界杯冠军",
    "乒乓球是中国的国球",
    "马拉松全程约为42公里",
    "奥运会每四年举办一次",
    "蒙娜丽莎是达芬奇的代表作",
    "贝多芬创作了九部交响曲",
    "维生素C有助于增强免疫力",
    "摩卡咖啡的名字源于也门的港口城市",
    "深圳夏天的平均气温约为28度",
    "挪威的森林覆盖率很高",
    "咖啡因有提神醒脑的作用",
    "一年有二十四个节气",
]

# single-hop 题目定义：(实体, 主题模板, 旧值, 新值, 问题, 答案)
# 主题覆盖：人物关系 / 居住地 / 职业 / 爱好 / 物品 / 宠物 / 学历 / 语言
_SINGLE_HOP_CASES: list[tuple[str, str, str, str, str, str]] = [
    # ── 人物关系 ──
    ("Alice", "{e}的丈夫是{v}", "Bob", "Charlie", "Alice 的丈夫是谁？", "Charlie"),
    ("李雷", "{e}的妻子是{v}", "韩梅梅", "王芳", "李雷的妻子是谁？", "王芳"),
    ("Emma", "{e}的哥哥是{v}", "Frank", "David", "Emma 的哥哥是谁？", "David"),
    ("陈静", "{e}的姐姐是{v}", "孙悦", "马丽", "陈静的姐姐是谁？", "马丽"),
    ("Kevin", "{e}的室友是{v}", "Martin", "Oscar", "Kevin 的室友是谁？", "Oscar"),
    ("刘洋", "{e}的导师是{v}", "胡军", "郭涛", "刘洋的导师是谁？", "郭涛"),
    ("Grace", "{e}的儿子是{v}", "Ivan", "Peter", "Grace 的儿子是谁？", "Peter"),
    ("周杰", "{e}的女儿是{v}", "吴敏", "朱婷", "周杰的女儿是谁？", "朱婷"),
    # ── 居住地 ──
    ("张伟", "{e}居住在{v}", "北京", "成都", "张伟居住在哪座城市？", "成都"),
    ("Helen", "{e}居住在{v}", "伦敦", "曼彻斯特", "Helen 居住在哪座城市？", "曼彻斯特"),
    ("杨帆", "{e}居住在{v}", "上海", "杭州", "杨帆居住在哪座城市？", "杭州"),
    # ── 职业 ──
    ("赵磊", "{e}的职业是{v}", "医生", "律师", "赵磊的职业是什么？", "律师"),
    ("Nancy", "{e}的职业是{v}", "教师", "设计师", "Nancy 的职业是什么？", "设计师"),
    ("徐强", "{e}的职业是{v}", "厨师", "工程师", "徐强的职业是什么？", "工程师"),
    # ── 爱好 ──
    ("Julia", "{e}的爱好是{v}", "绘画", "攀岩", "Julia 的爱好是什么？", "攀岩"),
    ("林晓", "{e}的爱好是{v}", "摄影", "烘焙", "林晓的爱好是什么？", "烘焙"),
    ("Sam", "{e}的爱好是{v}", "钓鱼", "滑雪", "Sam 的爱好是什么？", "滑雪"),
    # ── 物品归属 ──
    ("Victor", "{e}使用的手机品牌是{v}", "苹果", "华为", "Victor 使用的手机是什么品牌？", "华为"),
    ("高远", "{e}开的汽车品牌是{v}", "丰田", "比亚迪", "高远开的汽车是什么品牌？", "比亚迪"),
    ("Tina", "{e}用的笔记本电脑品牌是{v}", "联想", "戴尔", "Tina 用的笔记本电脑是什么品牌？", "戴尔"),
    # ── 宠物 ──
    ("Rachel", "{e}养的宠物是一只{v}", "猫", "狗", "Rachel 养的宠物是什么？", "狗"),
    ("郑浩", "{e}养的宠物是一只{v}", "鹦鹉", "仓鼠", "郑浩养的宠物是什么？", "仓鼠"),
    # ── 学历 / 语言 ──
    ("Wendy", "{e}毕业于{v}", "复旦大学", "浙江大学", "Wendy 毕业于哪所大学？", "浙江大学"),
    ("胡军", "{e}毕业于{v}", "清华大学", "北京大学", "胡军毕业于哪所大学？", "北京大学"),
    ("Ivan", "{e}会说的外语是{v}", "法语", "日语", "Ivan 会说的外语是什么？", "日语"),
]

# multi-hop 2 跳题目定义：
# (实体E, 关系rel, 旧值V_old, 新值V_new, 属性attr, 桥接模板, 旧桥接值A_old, 新桥接值A_new, 问题, 答案)
# 推理链：E --rel--> V --attr--> A，rel 被更新后答案应走 V_new 的桥接
_MULTI_HOP_2_CASES: list[tuple[str, str, str, str, str, str, str, str, str, str]] = [
    ("Alice", "丈夫", "Bob", "Charlie", "职业", "{v}是一名{a}", "医生", "工程师",
     "Alice 的丈夫的职业是什么？", "工程师"),
    ("李雷", "妻子", "韩梅梅", "王芳", "职业", "{v}是一名{a}", "记者", "会计",
     "李雷的妻子的职业是什么？", "会计"),
    ("Emma", "哥哥", "Frank", "David", "居住城市", "{v}居住在{a}", "东京", "大阪",
     "Emma 的哥哥居住在哪座城市？", "大阪"),
    ("陈静", "姐姐", "孙悦", "马丽", "爱好", "{v}的爱好是{a}", "摄影", "烘焙",
     "陈静的姐姐的爱好是什么？", "烘焙"),
    ("Kevin", "室友", "Martin", "Oscar", "职业", "{v}是一名{a}", "厨师", "律师",
     "Kevin 的室友的职业是什么？", "律师"),
    ("刘洋", "导师", "胡军", "郭涛", "毕业大学", "{v}毕业于{a}", "南京大学", "武汉大学",
     "刘洋的导师毕业于哪所大学？", "武汉大学"),
    ("Grace", "儿子", "Ivan", "Peter", "职业", "{v}是一名{a}", "教师", "医生",
     "Grace 的儿子的职业是什么？", "医生"),
    ("周杰", "女儿", "吴敏", "朱婷", "工作部门", "{v}在{a}工作", "市场部", "研发部",
     "周杰的女儿在哪个部门工作？", "研发部"),
    ("张伟", "妻子", "黄蕾", "徐静", "职业", "{v}是一名{a}", "护士", "药剂师",
     "张伟的妻子的职业是什么？", "药剂师"),
    ("David", "妻子", "Lily", "Nancy", "居住城市", "{v}居住在{a}", "伦敦", "曼彻斯特",
     "David 的妻子居住在哪座城市？", "曼彻斯特"),
    ("王芳", "丈夫", "张伟", "高远", "会说的外语", "{v}会说{a}", "法语", "西班牙语",
     "王芳的丈夫会说什么外语？", "西班牙语"),
    ("Julia", "丈夫", "Martin", "Sam", "最喜欢的运动", "{v}最喜欢的运动是{a}", "网球", "游泳",
     "Julia 的丈夫最喜欢的运动是什么？", "游泳"),
    ("林晓", "哥哥", "郑浩", "黄磊", "职业", "{v}是一名{a}", "警察", "消防员",
     "林晓的哥哥的职业是什么？", "消防员"),
    ("Rachel", "姐姐", "Tina", "Wendy", "居住城市", "{v}居住在{a}", "悉尼", "墨尔本",
     "Rachel 的姐姐居住在哪座城市？", "墨尔本"),
    ("赵磊", "女儿", "Lily", "Rachel", "毕业大学", "{v}毕业于{a}", "中山大学", "四川大学",
     "赵磊的女儿毕业于哪所大学？", "四川大学"),
    ("Oscar", "妻子", "Julia", "Helen", "爱好", "{v}的爱好是{a}", "园艺", "陶艺",
     "Oscar 的妻子的爱好是什么？", "陶艺"),
    ("孙悦", "丈夫", "徐强", "杨帆", "职业", "{v}是一名{a}", "司机", "快递员",
     "孙悦的丈夫的职业是什么？", "快递员"),
    ("马丽", "儿子", "Victor", "Kevin", "居住城市", "{v}居住在{a}", "西安", "南京",
     "马丽的儿子居住在哪座城市？", "南京"),
    ("吴敏", "室友", "Rachel", "Grace", "职业", "{v}是一名{a}", "翻译", "导游",
     "吴敏的室友的职业是什么？", "导游"),
    ("朱婷", "姐姐", "Nancy", "Tina", "毕业大学", "{v}毕业于{a}", "同济大学", "南开大学",
     "朱婷的姐姐毕业于哪所大学？", "南开大学"),
]

# multi-hop 3 跳题目定义（E1 --rel--> E2 --rel2--> E3 --attr--> A，rel 被更新）：
# (E1, rel, E2_old, E2_new, rel2, E3_old, E3_new, 桥2模板, 桥3模板, A_old, A_new, 问题, 答案)
_MULTI_HOP_3_CASES: list[
    tuple[str, str, str, str, str, str, str, str, str, str, str, str, str]
] = [
    ("李雷", "妻子", "韩梅梅", "王芳", "哥哥", "韩强", "王刚",
     "{e2}的哥哥是{e3}", "{e3}在{a}工作", "银行", "医院",
     "李雷的妻子的哥哥在哪里工作？", "医院"),
    ("Alice", "丈夫", "Bob", "Charlie", "姐姐", "Bella", "Catherine",
     "{e2}的姐姐是{e3}", "{e3}居住在{a}", "多伦多", "温哥华",
     "Alice 的丈夫的姐姐居住在哪座城市？", "温哥华"),
    ("张伟", "女儿", "黄蕾", "徐静", "导师", "孙教授", "钱教授",
     "{e2}的导师是{e3}", "{e3}研究{a}", "古代历史", "人工智能",
     "张伟的女儿的导师研究什么领域？", "人工智能"),
    ("Emma", "父亲", "Frank", "David", "兄弟", "Felix", "Daniel",
     "{e2}的兄弟是{e3}", "{e3}开一家{a}", "书店", "餐厅",
     "Emma 的父亲的兄弟开什么店？", "餐厅"),
    ("刘洋", "母亲", "陈静", "杨帆", "妹妹", "陈晨", "杨柳",
     "{e2}的妹妹是{e3}", "{e3}养了{a}", "三只猫", "两只狗",
     "刘洋的母亲的妹妹养了什么宠物？", "两只狗"),
]


def _fmt(template: str, **kw: str) -> str:
    """模板格式化（关键字与模板占位符对齐）。"""
    return template.format(**kw)


def _build_case(
    qid: str,
    hop: int,
    question: str,
    answer: str,
    core_facts: list[tuple[str, str]],  # (文本, 角色)，最后一条必须是新矛盾事实
    n_distract: int,
    distract_offset: int,
) -> dict[str, Any]:
    """组装一道自测题：核心事实 + 轮转抽取的干扰事实，统一按 Fact #N 递增编号。

    注入顺序 = 序号顺序：旧事实（含旧桥接）→ 干扰 → 新矛盾事实（含新桥接），
    保证新事实写入时旧事实已在库中，可触发冲突检测/仲裁。
    """
    old_side = core_facts[:-1]
    new_side = [core_facts[-1]]

    facts: list[dict[str, str]] = []
    idx = 0
    for text, role in old_side:
        idx += 1
        facts.append({"seq": idx, "text": f"Fact #{idx}: {text}", "role": role})
    for i in range(n_distract):
        idx += 1
        d = _DISTRACTOR_POOL[(distract_offset + i) % len(_DISTRACTOR_POOL)]
        facts.append({"seq": idx, "text": f"Fact #{idx}: {d}", "role": "distractor"})
    for text, role in new_side:
        idx += 1
        facts.append({"seq": idx, "text": f"Fact #{idx}: {text}", "role": role})

    return {
        "qid": qid,
        "hop": hop,
        "question": question,
        "answer": answer,
        "facts": facts,
        "total_facts": idx,
    }


def build_dataset() -> list[dict[str, Any]]:
    """构造全部自测题（single-hop 25 + multi-hop 25，其中 3 跳 5 题）。"""
    cases: list[dict[str, Any]] = []

    # ── single-hop：旧事实 #1 → 干扰 → 新事实（末尾）──
    for i, (entity, tpl, old_v, new_v, question, answer) in enumerate(_SINGLE_HOP_CASES):
        core = [
            (_fmt(tpl, e=entity, v=old_v), "old_fact"),
            (_fmt(tpl, e=entity, v=new_v), "new_fact"),
        ]
        n_distract = 5 + (i % 5)  # 干扰数量 5-9 轮转，保证题间多样
        cases.append(
            _build_case(f"sh_{i + 1:02d}", 1, question, answer, core, n_distract, i * 3)
        )

    # ── multi-hop 2 跳：旧关系 + 双侧桥接（旧链在前）→ 干扰 → 新关系（末尾）──
    for i, (e, rel, v_old, v_new, _attr, br_tpl, a_old, a_new, question, answer) in enumerate(
        _MULTI_HOP_2_CASES
    ):
        link_tpl = "{e}的" + rel + "是{v}"
        core = [
            (_fmt(link_tpl, e=e, v=v_old), "old_link"),
            (_fmt(br_tpl, v=v_old, a=a_old), "bridge_old"),
            (_fmt(br_tpl, v=v_new, a=a_new), "bridge_new"),
            (_fmt(link_tpl, e=e, v=v_new), "new_link"),
        ]
        n_distract = 5 + (i % 5)
        cases.append(
            _build_case(f"mh_{i + 1:02d}", 2, question, answer, core, n_distract, i * 3 + 1)
        )

    # ── multi-hop 3 跳：旧链桥接 → 新链桥接 → 干扰 → 新关系（末尾）──
    for i, (e1, rel, e2o, e2n, _rel2, e3o, e3n, br2, br3, a_old, a_new, question, answer) in enumerate(
        _MULTI_HOP_3_CASES
    ):
        link_tpl = "{e}的" + rel + "是{v}"
        core = [
            (_fmt(link_tpl, e=e1, v=e2o), "old_link"),
            (_fmt(br2, e2=e2o, e3=e3o), "bridge_old"),   # E2_old 的 rel2 是 E3_old
            (_fmt(br3, e3=e3o, a=a_old), "bridge_old2"),  # E3_old 的属性（旧值）
            (_fmt(br2, e2=e2n, e3=e3n), "bridge2_new"),   # E2_new 的 rel2 是 E3_new
            (_fmt(br3, e3=e3n, a=a_new), "bridge3_new"),  # E3_new 的属性（新值）
            (_fmt(link_tpl, e=e1, v=e2n), "new_link"),
        ]
        n_distract = 5 + (i % 4)
        cases.append(
            _build_case(f"mh3_{i + 1:02d}", 3, question, answer, core, n_distract, i * 3 + 2)
        )

    return cases


def _attach_judge_keys(cases: list[dict[str, Any]]) -> None:
    """为每题预计算各角色的判分关键词（结构化重建，避免解析自然语言）。

    single-hop / link 类：实体 + 值；桥接类：值实体 + 属性值。
    """
    for case in cases:
        qid = case["qid"]
        if qid.startswith("sh_"):
            i = int(qid.split("_")[1]) - 1
            entity, _tpl, old_v, new_v, _q, _a = _SINGLE_HOP_CASES[i]
            case["_judge_keys"] = {
                "old_fact": [entity, old_v],
                "new_fact": [entity, new_v],
            }
        elif qid.startswith("mh_"):
            i = int(qid.split("_")[1]) - 1
            e, _rel, v_old, v_new, _attr, _br, a_old, a_new, _q, _a = _MULTI_HOP_2_CASES[i]
            case["_judge_keys"] = {
                "old_link": [e, v_old],
                "new_link": [e, v_new],
                "bridge_old": [v_old, a_old],
                "bridge_new": [v_new, a_new],
            }
        else:  # mh3_
            i = int(qid.split("_")[1]) - 1
            e1, _rel, e2o, e2n, _rel2, e3o, e3n, _br2, _br3, a_old, a_new, _q, _a = _MULTI_HOP_3_CASES[i]
            case["_judge_keys"] = {
                "old_link": [e1, e2o],
                "new_link": [e1, e2n],
                "bridge_old": [e2o, e3o],
                "bridge2_new": [e2n, e3n],
                "bridge3_new": [e3n, a_new],
            }


# ======================================================================
# 二、判分（检索层 SubEM，二段：融合层 / 阅读层）
# ======================================================================


def _norm(text: str) -> str:
    """归一化：去空白 + 小写，用于子串匹配。"""
    return "".join(text.split()).lower()


def _find_rank(contexts: list[str], *keywords: str) -> int | None:
    """返回首个同时包含全部关键词的 context 下标（0-based 排名），未命中返回 None。"""
    kws = [_norm(k) for k in keywords if k]
    for i, ctx in enumerate(contexts):
        c = _norm(ctx)
        if all(k in c for k in kws):
            return i
    return None


def _extract_contexts(memories: list[dict[str, Any]]) -> list[str]:
    """从 recall 返回的 memories 提取文本（content + summary 拼接，防精炼改写漏判）。"""
    ctxs: list[str] = []
    for m in memories:
        text = (m.get("content") or "") + " " + (m.get("summary") or "")
        if text.strip():
            ctxs.append(text)
    return ctxs


def _rank_all(contexts: list[str], judge_keys: dict[str, list[str]]) -> dict[str, int | None]:
    """对每个角色关键词组计算在上下文中的排名。"""
    ranks: dict[str, int | None] = {}
    for role, kws in judge_keys.items():
        ranks[role] = _find_rank(contexts, *kws) if kws else None
    return ranks


def _judge_with_ranks(ranks: dict[str, int | None], answer: str, contexts: list[str]) -> dict[str, Any]:
    """基于角色排名做 SubEM 判分（single-hop / multi-hop 通用）。

    single-hop 判分：新事实（实体+新值）出现 且 旧事实不出现/排名靠后
    multi-hop 判分：新链 link + 桥接（含答案）均出现，且旧 link 不出现/排名靠后
    """
    new_link_rank = ranks.get("new_link", ranks.get("new_fact"))
    old_link_rank = ranks.get("old_link", ranks.get("old_fact"))
    bridge_rank = ranks.get("bridge_new")
    if bridge_rank is None:
        # 3 跳：两级新桥接都需命中
        b2, b3 = ranks.get("bridge2_new"), ranks.get("bridge3_new")
        bridge_rank = max([r for r in (b2, b3) if r is not None], default=None)
        bridge_all_hit = b2 is not None and b3 is not None
    else:
        bridge_all_hit = bridge_rank is not None

    new_in = new_link_rank is not None
    old_in = old_link_rank is not None
    strict_pass = bool(
        new_in and bridge_all_hit and (not old_in or new_link_rank < old_link_rank)
    )
    # 答案子串宽松命中（近似端到端 SubEM 上限）
    answer_hit = _find_rank(contexts, answer) is not None

    return {
        "ranks": ranks,
        "new_link_rank": new_link_rank,
        "old_link_rank": old_link_rank,
        "bridge_rank": bridge_rank,
        "answer_hit": answer_hit,
        "new_in_context": new_in,
        "old_in_context": old_in,
        "bridge_in_context": bridge_all_hit,
        "strict_pass": strict_pass,
    }


# ======================================================================
# 三、runner 主流程
# ======================================================================

# SDK 配置：沿用 run_longmemeval.py 的已调优配置，保持与既有评测可比
_SDK_CONFIG: dict[str, Any] = {
    "rrf_k": 10,
    "enable_reranker": True,
    "vector_weight": 2.0,
    "bm25_weight": 2.0,
    "recall_timeout_ms": 30000,
}


def run_case(
    case: dict[str, Any],
    top_k: int = 10,
    keep_storage_dir: Path | None = None,
) -> dict[str, Any]:
    """运行单题：独立临时存储 → 按序注入 → 融合层检索 + recall 检索 → 判分。

    检索做两次同参数调用：
      a. retriever.search（融合层出口，不含 ContextManager.refine 语义去重）
      b. sdk.recall（阅读层出口，生成器实际可见的最终上下文）
    两者对比可区分「检索丢失」与「阅读层 refine 去重误杀」。
    """
    qid = case["qid"]
    result: dict[str, Any] = {
        "qid": qid,
        "hop": case["hop"],
        "question": case["question"],
        "answer": case["answer"],
        "total_facts": case["total_facts"],
    }

    # 每题独立临时存储目录，避免题间污染
    if keep_storage_dir is not None:
        storage_dir = keep_storage_dir / qid
        storage_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir: str | None = None
    else:
        tmp_dir = tempfile.mkdtemp(prefix=f"omnimem_sf_{qid}_")
        storage_dir = Path(tmp_dir)

    try:
        sdk = OmniMemSDK(storage_dir=storage_dir, config=dict(_SDK_CONFIG))
        try:
            # ── 步骤 1: 按序号逐条注入（模拟增量注入，触发冲突检测/仲裁）──
            t0 = time.perf_counter()
            memorize_log: list[dict[str, Any]] = []
            for f in case["facts"]:
                r = sdk.memorize(content=f["text"], memory_type="fact")
                memorize_log.append(
                    {
                        "seq": f["seq"],
                        "role": f["role"],
                        "status": r.get("status", ""),
                        "conflict": r.get("conflict_warning"),
                    }
                )
            result["ingest_time_s"] = round(time.perf_counter() - t0, 2)
            result["memorize_log"] = memorize_log

            # ── 步骤 2a: 融合层检索（retriever.search 出口，不含 refine 去重）──
            raw_results = sdk._retriever.search(case["question"], mode="rag", top_k=top_k)
            fusion_ctxs = [
                (r.get("content") or "") + " " + (r.get("summary") or "")
                for r in raw_results
            ]
            fusion_ranks = _rank_all(fusion_ctxs, case["_judge_keys"])
            result["fusion_ranks"] = fusion_ranks
            result["fusion_contexts"] = [c[:160] for c in fusion_ctxs]

            # ── 步骤 2b: 阅读层检索（recall 出口 = 生成器可见上下文）──
            t1 = time.perf_counter()
            recall = sdk.recall(query=case["question"], mode="rag", top_k=top_k)
            result["recall_time_s"] = round(time.perf_counter() - t1, 2)
            memories = recall.get("memories", []) if recall.get("status") == "found" else []
            contexts = _extract_contexts(memories)
            result["recall_status"] = recall.get("status", "")
            result["retrieved_count"] = len(contexts)
            result["retrieved_contexts"] = [c[:160] for c in contexts]

            # ── 步骤 3: 阅读层判分（SubEM 主指标）──
            judge = _judge_with_ranks(_rank_all(contexts, case["_judge_keys"]), case["answer"], contexts)
            result.update(judge)

            # ── 步骤 4: 失败归因（二段：写入 / 融合 / 阅读）──
            new_log = next(
                (m for m in memorize_log if m["role"] in ("new_fact", "new_link")), {}
            )
            status = new_log.get("status", "")
            fusion_new_rank = fusion_ranks.get("new_link", fusion_ranks.get("new_fact"))
            if judge["strict_pass"]:
                mode = "pass"
            elif status == "duplicate_skipped":
                mode = "dedup_killed"  # 写入层：语义去重误杀，新矛盾事实未入库
            elif status not in ("stored", "ok"):
                mode = "new_not_stored"
            elif fusion_new_rank is None:
                mode = "fusion_miss"  # 融合层：新事实写入成功但检索未回
            elif not judge["new_in_context"]:
                mode = "refine_killed"  # 阅读层：融合层已检回但被 refine 语义去重丢弃
            elif not judge["bridge_in_context"]:
                mode = "bridge_not_retrieved"  # 多跳桥接（含答案）未在阅读层出现
            elif judge["old_in_context"] and judge["old_link_rank"] < judge["new_link_rank"]:
                mode = "old_dominates"  # 新旧都回但旧事实排名在前
            else:
                mode = "other"
            result["failure_mode"] = mode
            result["new_fact_write_status"] = status
            result["new_fact_conflict"] = new_log.get("conflict")
        finally:
            sdk.close()
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["failure_mode"] = "error"
        logger.error("题目 %s 运行异常: %s", qid, e, exc_info=True)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


def warmup() -> None:
    """预热：跑一个丢弃题，完成 embedding / reranker 模型首次加载。

    避免第一道正式题因模型冷启动（~30s）触发 recall 超时，保证计时公平。
    """
    tmp = tempfile.mkdtemp(prefix="omnimem_sf_warmup_")
    try:
        sdk = OmniMemSDK(storage_dir=tmp, config=dict(_SDK_CONFIG))
        try:
            sdk.memorize(content="Fact #1: 预热用的无关事实，东京今天多云", memory_type="fact")
            sdk.recall(query="预热查询：今天天气如何？", mode="rag", top_k=5)
        finally:
            sdk.close()
    except Exception as e:
        logger.warning("warmup 失败（继续正式评测）: %s", e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _summarize(results: list[dict[str, Any]], elapsed_s: float, argv_note: str) -> dict[str, Any]:
    """汇总分项分数与归因统计。"""

    def _agg(sub: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sub)
        strict = sum(1 for r in sub if r.get("strict_pass"))
        ans = sum(1 for r in sub if r.get("answer_hit"))
        modes: dict[str, int] = {}
        for r in sub:
            m = r.get("failure_mode", "error")
            modes[m] = modes.get(m, 0) + 1
        return {
            "total": n,
            "strict_pass": strict,
            "subem": round(strict / n, 4) if n else None,
            "answer_substring_hit": round(ans / n, 4) if n else None,
            "failure_modes": modes,
        }

    sh = [r for r in results if r["hop"] == 1]
    mh = [r for r in results if r["hop"] in (2, 3)]
    mh2 = [r for r in results if r["hop"] == 2]
    mh3 = [r for r in results if r["hop"] == 3]

    # 冲突仲裁触发统计（针对新矛盾事实那条写入）
    conflict_types: dict[str, int] = {}
    triggered = not_triggered = dedup_killed = 0
    for r in results:
        status = r.get("new_fact_write_status", "")
        conflict = r.get("new_fact_conflict")
        if status == "duplicate_skipped":
            dedup_killed += 1
        elif conflict:
            triggered += 1
            ctype = conflict.get("conflict_type", "unknown")
            conflict_types[ctype] = conflict_types.get(ctype, 0) + 1
        else:
            not_triggered += 1

    # 融合层对照统计：新事实在融合层被检回但阅读层消失（refine 去重误杀）
    refine_killed = sum(1 for r in results if r.get("failure_mode") == "refine_killed")
    fusion_new_hit = sum(
        1 for r in results
        if (r.get("fusion_ranks", {}).get("new_link", r.get("fusion_ranks", {}).get("new_fact")) is not None)
    )

    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(elapsed_s, 1),
            "argv": argv_note,
            "judge": "检索层 SubEM（阅读层出口：新事实+桥接出现且旧事实不出现/靠后）",
            "industry_reference": "MemoryAgentBench: 所有方法 SF multi-hop 最高 28%",
        },
        "single_hop": _agg(sh),
        "multi_hop": _agg(mh),
        "multi_hop_2hop": _agg(mh2),
        "multi_hop_3hop": _agg(mh3),
        "conflict_arbitration": {
            "new_fact_write_dedup_killed": dedup_killed,
            "conflict_triggered": triggered,
            "conflict_not_triggered": not_triggered,
            "conflict_types": conflict_types,
        },
        "retrieval_layers": {
            "fusion_new_fact_hit": fusion_new_hit,
            "refine_dedup_killed": refine_killed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OmniMem SF（选择性遗忘）专项自测")
    parser.add_argument("--limit", type=int, default=0, help="每类最多跑 N 题（0=全部），冒烟测试用")
    parser.add_argument("--cases", choices=["all", "sh", "mh"], default="all", help="题目范围")
    parser.add_argument("--top-k", type=int, default=10, help="recall top-k（默认 10）")
    parser.add_argument(
        "--keep-storage", action="store_true",
        help="保留每题存储目录到 results/diagnosis/sf_selftest_storage/（默认用完即删）",
    )
    parser.add_argument("--no-warmup", action="store_true", help="跳过模型预热（调试用）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cases = build_dataset()
    _attach_judge_keys(cases)

    if args.cases == "sh":
        cases = [c for c in cases if c["hop"] == 1]
    elif args.cases == "mh":
        cases = [c for c in cases if c["hop"] in (2, 3)]
    if args.limit > 0:
        sh = [c for c in cases if c["hop"] == 1][: args.limit]
        mh = [c for c in cases if c["hop"] in (2, 3)][: args.limit]
        cases = sh + mh

    out_dir = PROJECT_ROOT / "benchmarks" / "results" / "diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "sf_selftest_details.jsonl"
    scores_path = out_dir / "sf_selftest_scores.json"
    storage_root = out_dir / "sf_selftest_storage" if args.keep_storage else None

    if not args.no_warmup:
        print("预热模型（embedding + reranker 首次加载）...", flush=True)
        t0 = time.perf_counter()
        warmup()
        print(f"预热完成 ({time.perf_counter() - t0:.1f}s)", flush=True)

    logger.info("SF 自测开始：%d 题 (top_k=%d)", len(cases), args.top_k)
    results: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    # 逐题运行（先 single-hop 后 multi-hop）
    ordered = [c for c in cases if c["hop"] == 1] + [c for c in cases if c["hop"] in (2, 3)]
    for i, case in enumerate(ordered):
        t0 = time.perf_counter()
        r = run_case(case, top_k=args.top_k, keep_storage_dir=storage_root)
        r["case_time_s"] = round(time.perf_counter() - t0, 2)
        results.append(r)
        mark = "✓" if r.get("strict_pass") else "✗"
        print(
            f"[{i + 1}/{len(ordered)}] {r['qid']} ({r['case_time_s']}s) "
            f"{mark} mode={r.get('failure_mode')} "
            f"fusion_new={r.get('fusion_ranks', {}).get('new_link', r.get('fusion_ranks', {}).get('new_fact'))} "
            f"final_new={r.get('new_link_rank')} final_old={r.get('old_link_rank')}",
            flush=True,
        )

    elapsed = time.perf_counter() - t_start

    # 逐题明细 jsonl
    with open(details_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总 json
    argv_note = (
        f"cases={args.cases} limit={args.limit} top_k={args.top_k} "
        f"keep_storage={args.keep_storage}"
    )
    summary = _summarize(results, elapsed, argv_note)
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print("\n" + "=" * 60)
    print("SF 自测汇总（检索层 SubEM，阅读层出口）")
    print("=" * 60)
    print(f"single-hop : {summary['single_hop']['strict_pass']}/{summary['single_hop']['total']}"
          f" = {summary['single_hop']['subem']}")
    print(f"multi-hop  : {summary['multi_hop']['strict_pass']}/{summary['multi_hop']['total']}"
          f" = {summary['multi_hop']['subem']}")
    if summary["multi_hop_2hop"]["total"]:
        print(f"  2-hop    : {summary['multi_hop_2hop']['strict_pass']}/{summary['multi_hop_2hop']['total']}"
              f" = {summary['multi_hop_2hop']['subem']}")
    if summary["multi_hop_3hop"]["total"]:
        print(f"  3-hop    : {summary['multi_hop_3hop']['strict_pass']}/{summary['multi_hop_3hop']['total']}"
              f" = {summary['multi_hop_3hop']['subem']}")
    print(f"冲突仲裁: 触发={summary['conflict_arbitration']['conflict_triggered']} "
          f"未触发={summary['conflict_arbitration']['conflict_not_triggered']} "
          f"去重误杀={summary['conflict_arbitration']['new_fact_write_dedup_killed']} "
          f"类型={summary['conflict_arbitration']['conflict_types']}")
    print(f"检索分层: 融合层新事实命中={summary['retrieval_layers']['fusion_new_fact_hit']} "
          f"阅读层refine去重误杀={summary['retrieval_layers']['refine_dedup_killed']}")
    print(f"明细: {details_path}")
    print(f"汇总: {scores_path}")
    print(f"总耗时: {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
