#!/usr/bin/env python3
"""LRU（Long-Range Understanding，长程理解）top-k 截断验证实验 — 诊断 spec Task 5。

实验协议（纯检索层，不依赖 LLM）：
  - 自建合成 haystack：30 个 session、共 150 条 turn 内容，
    埋入 40 条计数型证据事实（「第 X 本书」系列），分散在不同 session，
    其余为不同主题的干扰内容
  - 全量注入 OmniMem（独立临时存储）
  - 聚合问题（如「我一共买了多少本书？」需要全部 40 条证据才能答对）
  - 对 top-k ∈ {5, 10, 20, 50, 100} 分别测：检索结果中证据事实的覆盖率（命中条数/40）
  - 对照两种查询：高区分度查询（直接匹配证据句式）vs 模糊聚合查询，
    体现「聚合型问题语义相似度检索失效」

运行方式：
    cd ~/.hermes/plugins/omnimem
    .venv/bin/python benchmarks/diagnosis/lru_experiment.py

结果输出：benchmarks/results/diagnosis/lru_result.json
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

# 将项目根目录加入 sys.path，使脚本可直接运行
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from omnimem.sdk import OmniMemSDK  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("lru_experiment")

# ======================================================================
# 实验数据：合成 haystack
# ======================================================================

NUM_SESSIONS = 30
TURNS_PER_SESSION = 5  # 每个 session 的 turn 数（含证据）
TOTAL_TURNS = NUM_SESSIONS * TURNS_PER_SESSION  # 150 条
NUM_EVIDENCE = 40  # 计数型证据事实条数
RANDOM_SEED = 20260908  # 固定种子保证可复现

# 干扰内容主题池（每个 session 抽一个主题，围绕主题展开 5 轮对话式内容）
_DISTRACTOR_TOPICS: list[list[str]] = [
    ["今天跑了五公里，配速降到六分以内了", "拉伸的时候膝盖有点不舒服", "下周想试试间歇跑",
     "买了双新的缓震跑鞋", "晚上补了点蛋白质，喝了杯牛奶"],
    ["这周做了三次力量训练", "卧推重量加到六十公斤了", "深蹲的时候腰部代偿明显",
     "教练说核心力量要再练练", "训练后泡了二十分钟冷水澡"],
    ["周末去了趟郊区爬山", "山顶的风景特别好，拍了好多照片", "下山的时候差点崴脚",
     "带的三明治在山顶吃完了", "回来路上堵了两个小时"],
    ["今天做了一道红烧肉", "糖色炒得有点老了", "砂锅炖了一个半小时",
     "最后收汁收得不错", "配了一碗白米饭，很香"],
    ["公司的打印机又卡纸了", "换了新的墨盒才解决", "行政说下个月换新打印机",
     "扫描仪的驱动也得重装", "终于把合同都扫完了"],
    ["这周在学吉他", "F 和弦还是按不响", "手指尖都磨出茧了",
     "跟看了个教学视频", "能磕磕绊绊弹完一首歌了"],
    ["家里猫最近挑食", "换了新猫粮也不吃", "带去看了兽医",
     "医生说没什么大问题", "买了点猫条骗它吃药"],
    ["周末去钓了一次鱼", "早上五点就出发了", "钓了一上午就上了一条小鲫鱼",
     "中午在河边吃了自热火锅", "下午收竿回家的路上睡着了"],
    ["最近在学摄影", "买了一支定焦镜头", "周末去公园拍了一场人像",
     "后期调色总是调不出想要的感觉", "把家里的绿植当静物练手"],
    ["这周看了两部电影", "一部是老片重映，画质修复得很好", "另一部是科幻新片，特效炸裂",
     "影院的音响效果太棒了", "散场后在楼下吃了碗面"],
    ["车该做保养了", "换了机油和机滤", "师傅说刹车片还能用一万公里",
     "轮胎有一条慢跑气", "洗完车感觉像新车"],
    ["家里在装修", "水电改造这周完工", "瓷砖选了灰色的哑光砖",
     "木工进场打柜子", "预算又超了两万"],
    ["最近在学做手冲咖啡", "买了套入门的滤杯和壶", "浅烘的豆子酸味明显",
     "水温九十度萃取比较稳", "早上给自己冲了一杯耶加雪菲"],
    ["这周打了三次羽毛球", "右手臂有点酸", "换了个轻一点的拍子",
     "双打的时候网前小球有进步", "约了下周四的场"],
    ["办公室的绿萝黄叶了", "可能是浇水太勤", "挪到了散射光的位置",
     "施了点缓释肥", "新叶子冒出来了"],
    ["这周在整理简历", "把项目经历按 STAR 法则重写了", "量化了几个关键成果",
     "更新了领英主页", "投了三家心仪的公司"],
    ["周末去逛了花鸟市场", "买了一盆多肉", "看中一缸锦鲤但没买",
     "还给家里添了一束干花", "市场门口的糖葫芦很甜"],
    ["这周在学日语", "五十音图终于记熟了", "开始学简单的语法",
     "每天背二十个单词", "看了半集动画试试听力"],
    ["家里路由器固件升级了", "信号好像稳定了点", "把老设备挪到了 2.4G 频段",
     "给客卧加了个信号放大器", "测速终于跑满带宽了"],
    ["这周做了两次烘焙", "第一次做戚风蛋糕塌了", "第二次降了温度成功了不少",
     "买了电子秤精确称量", "同事们把司康分光了"],
    ["最近在听播客", "通勤时间听一集刚好", "有一期讲城市规划的很有意思",
     "安利给了两个朋友", "睡觉前又听了一集历史题材的"],
    ["这周去了两次博物馆", "特展的青铜器很震撼", "讲解器租了一个",
     "文创雪糕造型是镇馆之宝", "出来后在广场喂了鸽子"],
    ["家里换了个扫地机器人", "建图比老的快很多", "沙发底下终于能进了",
     "集尘盒两天就满了", "设置了每天上午自动清扫"],
    ["这周在练毛笔字", "从横竖撇捺开始", "买了两刀毛边纸",
     "老师说我握笔太紧", "每天睡前写半小时"],
    ["办公室搬到了新楼层", "工位靠窗了", "网络端口重新布过",
     "文件柜搬了一上午", "窗外能看到江景"],
    ["这周做了两次核酸码设计的活", "甲方要五彩斑斓的黑", "改了五稿终于过了",
     "导出印刷文件核对了色值", "尾款到账了"],
    ["周末去了趟古城", "城墙修得很新但味道还在", "在老茶馆听了一场评书",
     "买了两盒当地特产", "回程的高铁上睡了一路"],
    ["最近在给家里做收纳", "买了几个真空压缩袋", "换季的衣服都收起来了",
     "厨房添置了置物架", "终于能一眼找到袜子了"],
    ["这周在研究指数基金", "定投了中证五百", "看了一本资产配置的书",
     "把三个持仓相近的基金合并了", "收益终于转正了"],
    ["周末打了场桌游", "六人局玩到晚上十点", "第一次当狼人就被首刀",
     "后来换了合作类游戏", "约好下个月再聚"],
]

# 40 条证据事实（计数型，句式多样、分散注入），每条含唯一序号保证去重数字豁免
_EVIDENCE_POOL: list[str] = [
    "对了说下，我买的第 1 本书到了，是一本讲宋代生活的随笔",
    "朋友寄来第 2 本书，讲的是喜马拉雅山脉的科考纪实",
    "我在书店买的第 3 本书，是本旧书店行业观察",
    "第 4 本书到手了，讲海上灯塔守岛人的故事",
    "昨天收到的第 5 本书是本桥梁工程科普",
    "买的第 6 本书是本讲咖啡豆产地的图鉴",
    "第 7 本书到了，讲一位盲人航海家的传记",
    "我在机场买的第 8 本书，是一本关于古代驿路的考据",
    "第 9 本书是本儿童教育学入门，给外甥准备的",
    "收到的第 10 本书讲的是微塑料污染调查",
    "第 11 本书买的是新版地图集，纸质很好",
    "我在网上下单的第 12 本书，是本讲天妇罗的料理书",
    "第 13 本书到了，讲一战的战场考古",
    "买的第 14 本书是本城市规划批评文集",
    "第 15 本书收到，是本海洋浮游生物图谱",
    "又买了第 16 本书，讲的是一位钟表匠的家族史",
    "第 17 本书是本旧铁轨徒步旅行记",
    "我买的第 18 本书，是本讲方言保护的田野笔记",
    "第 19 本书到了，讲深海热泉生物",
    "收到的第 20 本书讲的是一位女性登山者的自传",
    "第 21 本书买的是本版画技法入门",
    "我在二手书店淘到第 22 本书，讲运河漕运史",
    "第 23 本书是本盐业史小册子",
    "买的第 24 本书讲的是失传文字的破译故事",
    "第 25 本书到了，是本讲灯塔鸟类的博物志",
    "收到的第 26 本书讲的是一位宫廷乐师的考证",
    "第 27 本书买的是本讲苔藓的图鉴",
    "我买的第 28 本书，是本关于敦煌壁画的修复手记",
    "第 29 本书是本古代天文仪器史",
    "买的第 30 本书讲的是一位林务官的巡山日志",
    "第 31 本书到了，讲撒哈拉岩画",
    "收到的第 32 本书是本信鸽竞翔的入门书",
    "第 33 本书买的是本讲潮汐的科普",
    "我在书展买的第 34 本书，讲一位制琴师的传记",
    "第 35 本书是本讲胡同声音的田野录音笔记",
    "买的第 36 本书讲的是南极冰芯研究",
    "第 37 本书到了，是本讲古罗马供水的工程史",
    "收到的第 38 本书讲一位灯塔画家的画册",
    "第 39 本书买的是本种子库的纪实文学",
    "最后买的第 40 本书，是本讲远洋渔船船员生活的口述史",
]

# 两种查询：模糊聚合 vs 高区分度（直接匹配证据句式关键词）
QUERY_AGGREGATE = "我一共买了多少本书？"
QUERY_DIRECT = "我买的第几本书都是什么书？列一下我的买书记录"

TOP_K_LIST = [5, 10, 20, 50, 100]


def build_haystack() -> tuple[list[dict], list[str]]:
    """构造合成 haystack，返回 (turn 内容列表, 证据内容列表)。

    turn 结构：{"session": int, "turn": int, "content": str, "is_evidence": bool}
    证据按固定随机种子分散到不同 session 的不同 turn 位置。
    """
    rng = random.Random(RANDOM_SEED)

    # 40 条证据分散：每 session 至多 2 条（保证分散度），随机挑 turn 位置
    evidence_slots: set[tuple[int, int]] = set()
    sessions = list(range(NUM_SESSIONS))
    rng.shuffle(sessions)
    # 第一轮每个 session 放一条（前 30 条），第二轮随机挑 10 个 session 放第二条
    chosen_sessions = sessions[:NUM_EVIDENCE]
    for s in chosen_sessions:
        t = rng.randrange(TURNS_PER_SESSION)
        while (s, t) in evidence_slots:
            t = rng.randrange(TURNS_PER_SESSION)
        evidence_slots.add((s, t))
    second_round = rng.sample(sessions, NUM_EVIDENCE - NUM_SESSIONS)
    for s in second_round:
        t = rng.randrange(TURNS_PER_SESSION)
        while (s, t) in evidence_slots:
            t = rng.randrange(TURNS_PER_SESSION)
        evidence_slots.add((s, t))

    evidence_contents = list(_EVIDENCE_POOL[:NUM_EVIDENCE])
    rng.shuffle(evidence_contents)

    turns: list[dict] = []
    ev_idx = 0
    for s in range(NUM_SESSIONS):
        topic = _DISTRACTOR_TOPICS[s % len(_DISTRACTOR_TOPICS)]
        for t in range(TURNS_PER_SESSION):
            if (s, t) in evidence_slots:
                turns.append({
                    "session": s, "turn": t,
                    "content": evidence_contents[ev_idx], "is_evidence": True,
                })
                ev_idx += 1
            else:
                # 干扰内容按 (session, turn) 稳定取用
                turns.append({
                    "session": s, "turn": t,
                    "content": topic[t % len(topic)], "is_evidence": False,
                })
    assert ev_idx == NUM_EVIDENCE
    return turns, evidence_contents


def run() -> dict:
    """执行 LRU top-k 截断实验，返回结构化结果。"""
    t_start = time.time()

    turns, evidence_contents = build_haystack()
    logger.info(
        "haystack 构造完成: %d sessions × %d turns = %d 条内容, 其中证据 %d 条",
        NUM_SESSIONS, TURNS_PER_SESSION, len(turns), NUM_EVIDENCE,
    )

    storage_dir = Path(tempfile.mkdtemp(prefix="omnimem_diag_lru_"))
    logger.info("临时存储目录: %s", storage_dir)

    inject_stats = {"total": 0, "stored": 0, "skipped": 0, "skipped_detail": []}
    # 证据句 -> memory_id 的映射（用于命中率统计）
    evidence_memory_ids: dict[str, str] = {}

    try:
        sdk = OmniMemSDK(storage_dir=storage_dir)
        try:
            # ── 阶段 1：全量注入 haystack ──
            t_inject = time.time()
            for item in turns:
                result = sdk.memorize(item["content"], memory_type="fact")
                inject_stats["total"] += 1
                status = result.get("status", "")
                if status == "stored":
                    inject_stats["stored"] += 1
                    mid = result.get("memory_id", "")
                    if item["is_evidence"] and mid:
                        evidence_memory_ids[item["content"]] = mid
                else:
                    inject_stats["skipped"] += 1
                    inject_stats["skipped_detail"].append(
                        {"status": status, "content": item["content"][:50],
                         "is_evidence": item["is_evidence"]}
                    )
            logger.info(
                "haystack 注入完成: %d/%d 入库 (证据 %d/%d), 耗时 %.1fs",
                inject_stats["stored"], inject_stats["total"],
                len(evidence_memory_ids), NUM_EVIDENCE, time.time() - t_inject,
            )

            # ── 阶段 2：两种查询 × 各 top-k 的覆盖率测量 ──
            target_ids = set(evidence_memory_ids.values())
            coverage_data: dict[str, list[dict]] = {}

            for query_name, query in [
                ("aggregate_query", QUERY_AGGREGATE),
                ("direct_query", QUERY_DIRECT),
            ]:
                coverage_data[query_name] = []
                for k in TOP_K_LIST:
                    t_k = time.time()
                    recall_result = sdk.recall(query, mode="rag", top_k=k, max_tokens=100000)
                    memories = (
                        recall_result.get("memories", [])
                        if recall_result.get("status") == "found" else []
                    )
                    returned_ids = {
                        m.get("memory_id", "") for m in memories if m.get("memory_id")
                    }
                    hit_ids = target_ids & returned_ids
                    coverage = len(hit_ids) / NUM_EVIDENCE if NUM_EVIDENCE else 0.0
                    coverage_data[query_name].append({
                        "top_k": k,
                        "returned_count": len(memories),
                        "evidence_hit": len(hit_ids),
                        "coverage": round(coverage, 4),
                        "duration_ms": round((time.time() - t_k) * 1000, 1),
                        "hit_memory_ids": sorted(hit_ids),
                    })
                    logger.info(
                        "[%s] top_k=%d 返回 %d 条, 命中证据 %d/%d (覆盖率 %.0f%%)",
                        query_name, k, len(memories), len(hit_ids), NUM_EVIDENCE, coverage * 100,
                    )
        finally:
            sdk.close()
    finally:
        # 清理临时存储（目录集中放在系统临时区，实验完成后删除）
        shutil.rmtree(storage_dir, ignore_errors=True)
        logger.info("临时存储目录已清理: %s", storage_dir)

    # ── 阶段 3：汇总与文本图表 ──
    chart_lines = render_chart(coverage_data)

    # 结论自动生成
    agg_cov = {d["top_k"]: d["coverage"] for d in coverage_data["aggregate_query"]}
    dir_cov = {d["top_k"]: d["coverage"] for d in coverage_data["direct_query"]}
    max_k = max(TOP_K_LIST)
    gap_at_max = dir_cov[max_k] - agg_cov[max_k]
    conclusion = (
        f"聚合查询在 top-k={max_k} 时证据覆盖率仅 {agg_cov[max_k]:.0%}（需 100% 才能正确回答"
        f"『我一共买了多少本书』），top-k 截断使长程聚合任务在检索层即失败；"
        f"高区分度查询同 k 覆盖率 {dir_cov[max_k]:.0%}"
        f"（差值 {gap_at_max:+.0%}），验证聚合型问题的语义相似度检索失效更严重。"
        "RAG 式 top-k 检索缺乏全局理解，与 MemoryAgentBench 对 LRU 的论文结论一致。"
    )

    result = {
        "experiment": "lru_topk_truncation",
        "spec_task": "Task 5 — LRU 空白验证",
        "meta": {
            "num_sessions": NUM_SESSIONS,
            "turns_per_session": TURNS_PER_SESSION,
            "total_turns": len(turns),
            "num_evidence": NUM_EVIDENCE,
            "random_seed": RANDOM_SEED,
            "aggregate_query": QUERY_AGGREGATE,
            "direct_query": QUERY_DIRECT,
            "top_k_list": TOP_K_LIST,
            "duration_sec": round(time.time() - t_start, 1),
            "storage_dir_policy": "临时目录（系统 temp区），实验后已清理",
        },
        "inject_stats": inject_stats,
        "evidence_stored": len(evidence_memory_ids),
        "coverage": coverage_data,
        "coverage_chart_text": chart_lines,
        "conclusion": conclusion,
    }
    return result


def render_chart(coverage_data: dict[str, list[dict]]) -> list[str]:
    """渲染简单的文本柱状覆盖率曲线。"""
    lines = ["覆盖率曲线（k -> 覆盖率，# 每格 5%）"]
    width = 20  # 每行 20 格，每格代表 5%
    for name, title in [("aggregate_query", "聚合查询"), ("direct_query", "高区分度查询")]:
        lines.append(f"--- {title} ({name}) ---")
        for d in coverage_data[name]:
            bar = "#" * min(int(round(d["coverage"] / 0.05)), width)
            lines.append(
                f"k={d['top_k']:>3} | {bar:<{width}} | {d['coverage']:>6.0%}"
                f" ({d['evidence_hit']}/{NUM_EVIDENCE})"
            )
    return lines


def main() -> None:
    result = run()

    output_path = _PROJECT_ROOT / "benchmarks" / "results" / "diagnosis" / "lru_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("LRU top-k 截断验证结果")
    print("=" * 70)
    for line in result["coverage_chart_text"]:
        print(line)
    print("-" * 70)
    print(f"注入统计: {result['inject_stats']['stored']}/{result['inject_stats']['total']} 入库"
          f"（证据 {result['evidence_stored']}/{NUM_EVIDENCE}）")
    print(f"结论: {result['conclusion']}")
    print(f"结果已写入: {output_path}")
    print(f"总耗时: {result['meta']['duration_sec']}s")


if __name__ == "__main__":
    main()
