#!/usr/bin/env python3
"""TTL（Test-Time Learning，测试时学习）空白验证实验 — 诊断 spec Task 4。

实验协议（参照 MemoryAgentBench MCC 增量分类任务，小规模简化版）：
  - 构造 4 个 LLM 训练数据中几乎不可能存在的自造工单类别
    （青鸟单/朱雀单/白泽单/玄豹单），类别名与内容主题的映射规则
    不写入 prompt，只能通过带标签样例归纳
  - 路径 A（记忆增强）：训练样例逐条 memorize 进 OmniMem（独立临时存储），
    每条测试样例先 recall 取 top-5 相似带标签样例作为上下文，再交 LLM 分类
  - 路径 B（直接问基线）：给出同样的类别说明，零样本直接分类，无任何样例
  - 对比两路径准确率：若 A 不优于 B，则「记忆提供 TTL 增量」为空白坐实；
    若 A 优于 B，记录增益幅度（说明记忆检索提供了有效的上下文信号）

运行方式：
    cd ~/.hermes/plugins/omnimem
    .venv/bin/python benchmarks/diagnosis/ttl_experiment.py

结果输出：benchmarks/results/diagnosis/ttl_result.json
"""

from __future__ import annotations

import json
import logging
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
logger = logging.getLogger("ttl_experiment")

# ======================================================================
# 实验数据：自造类别 + 带标签样例
# ======================================================================

# 四个类别均为编造的内部黑话，类别名与主题的映射规则只存在于样例中
CLASS_NAMES = ["青鸟单", "朱雀单", "白泽单", "玄豹单"]

# 每类 17 条训练样例（memorize 进记忆）+ 5 条测试样例（不进记忆）
TRAIN_SAMPLES: dict[str, list[str]] = {
    "青鸟单": [
        "昨晚的离线作业跑失败了，帮忙看下日志",
        "Flink 作业反压有点严重，延迟涨上去了",
        "数仓的 ods 层分区没跑出来，今早报表是空的",
        "调度平台上这个 Spark 任务一直在排队",
        "想给实时链路加一个 exactly-once 语义",
        "Kafka 消费组积压了两千万条消息",
        "这个特征工程的脚本内存吃太多，节点 OOM 了",
        "Hive 表小文件太多，查询巨慢",
        "帮我把这个 SQL 在集群上跑一遍验证口径",
        "训练任务到第三个 epoch 就挂了，显存不够",
        "ClickHouse 物化视图没同步，数据差了半天",
        "数据倾斜了，有个 reduce 跑了三个小时",
        "想扩容计算节点，队列等待时间太长",
        "这个指标加工逻辑要改，口径换了",
        "流水线构建产物上传失败，重试还是不行",
        "埋点日志有丢失，昨天少了两小时数据",
        "离线特征和在线特征对不齐，训练推理结果不一致",
    ],
    "朱雀单": [
        "首页轮播图在低端机上掉帧",
        "这个按钮的圆角和设计稿对不上",
        "移动端弹窗层级有 bug，被软键盘挡住了",
        "暗色模式下这个图标显示不对",
        "列表页滚动卡顿，帧率只有二十",
        "帮忙调一下渐变色值，视觉说要更饱和一点",
        "表格在 Safari 里错位了",
        "这个交互动效想做得更有弹性",
        "输入框的占位文案要改",
        "H5 页面在刘海屏上有白边",
        "导航栏吸顶的时候会抖一下",
        "表单校验的错误提示样式要统一",
        "图片懒加载后占位符高度塌了",
        "骨架屏闪烁太明显了",
        "这个字体在安卓机器上发虚",
        "页面首屏渲染要压到两秒以内",
        "下拉刷新的圈圈转个不停，页面卡死了",
    ],
    "白泽单": [
        "这份上线方案需要谁来审批",
        "帮我发起一个差旅报销流程",
        "入职文档的第三章信息过时了，更新一下",
        "供应商准入要走什么流程",
        "会议纪要我已经归档到知识库了",
        "这份设计文档需要补一个评审记录",
        "合同用印申请被驳回了，理由是什么",
        "新人培训的内部链接失效了",
        "项目立项需要准备哪些材料",
        "想查一下去年的复盘报告归档在哪",
        "变更单流转到谁那里了",
        "知识库检索太慢，找不到想要的文档",
        "请假流程走到二级主管就卡住了",
        "这份规范文档想发起一次评审会",
        "对外发布的白皮书需要法务过一遍",
        "帮我把这套操作手册整理成英文版",
        "归档的权限申请单还没批下来",
    ],
    "玄豹单": [
        "三号机柜的空调报警了，温度飙到三十五度",
        "想申请两根四万兆光模块",
        "机房断电演练计划安排在本周六",
        "核心交换机有个端口频繁掉线",
        "帮忙加一根从 A 栋到 B 栋的跳线",
        "新到货的服务器需要上架",
        "网络延迟从两毫秒涨到四十毫秒了",
        "想采购四块企业级固态硬盘",
        "UPS 电池需要更换了",
        "KVM 切换器接触不良",
        "防火墙策略要开放一个新端口",
        "这个配线架的标签乱了，要重新整理",
        "冷通道的盲板缺了几块",
        "万兆网卡驱动装不上",
        "核心网络设备要做双活改造",
        "想查一下机柜还剩多少 U 位",
        "办公区交换机要换成支持供电的那种",
    ],
}

TEST_SAMPLES: dict[str, list[str]] = {
    "青鸟单": [
        "今早跑批的作业超时了，能不能帮忙查下原因",
        "Doris 的导入任务卡住了，表里没数据",
        "想调整一下这个 Spark 任务的并行度",
        "集群上的任务昨晚又被调度器杀掉了",
        "消费积压越来越大，实时大盘延迟了",
    ],
    "朱雀单": [
        "商品详情页的大图加载太慢了",
        "这个选项卡切换的时候有白屏",
        "想把间距从八改成十二，视觉验收不过",
        "小程序里这张卡片的阴影渲染异常",
        "深色主题下文字对比度不够，看不清",
    ],
    "白泽单": [
        "离职交接清单需要哪些人签字",
        "这个采购申请单流转卡了三天了",
        "想把这次故障复盘沉淀成一篇文档",
        "合同续签要走什么审批链路",
        "新版发布流程的说明在哪能找到",
    ],
    "玄豹单": [
        "B2 机房的门禁读卡器坏了",
        "想再买两台绘图仪放实验室",
        "到货的内存条和主板不兼容",
        "链路聚合口偶尔丢包",
        "弱电井里的光纤被老鼠咬断了",
    ],
}

# 类别说明（刻意不泄露任何映射规则，两条路径完全一致）
CLASS_STATEMENT = (
    "你是一家公司的工单系统助手。该公司工单分为四个内部类别，"
    "类别名是公司自造的内部黑话，含义未在本文档中说明：青鸟单、朱雀单、白泽单、玄豹单。"
)

TOP_K = 5  # 路径 A 每次检索的相似样例数（即 few-shot 上下文规模）

# ======================================================================
# LLM 客户端（凭证读取参照 benchmarks/run_longmemeval.py 的既有模式）
# ======================================================================


def load_llm_config() -> tuple[str, str]:
    """读取 LLM 凭证，返回 (api_key, base_url)。

    优先环境变量 OPENAI_API_KEY / OPENAI_BASE_URL，
    未设置则回退到 hermes config.yaml 的 deepseek 配置。
    """
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if api_key.strip():
        return api_key, base_url or "https://api.deepseek.com"

    import yaml

    config_path = Path.home() / ".hermes" / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ds_cfg = cfg.get("providers", {}).get("deepseek", {})
    ds_key = ds_cfg.get("api_key", "")
    if not ds_key:
        raise RuntimeError("未找到可用的 LLM 凭证（环境变量或 hermes config.yaml 均为空）")
    return ds_key, ds_cfg.get("base_url", "https://api.deepseek.com")


def llm_classify(api_key: str, base_url: str, prompt: str, max_retries: int = 3) -> str:
    """调用 DeepSeek 分类，返回原始回复文本（失败返回空串）。"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=0,
                max_tokens=32,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 — 网络重试需捕获宽泛异常
            if attempt < max_retries - 1:
                wait = 2**attempt
                logger.warning("LLM 调用重试 (%d/%d): %s", attempt + 1, max_retries, e)
                time.sleep(wait)
            else:
                logger.error("LLM 调用最终失败: %s", e)
                return ""
    return ""


def parse_label(reply: str) -> str:
    """从回复中解析类别名，未识别返回空串。"""
    for name in CLASS_NAMES:
        if name in reply:
            return name
    return ""


# ======================================================================
# prompt 构造
# ======================================================================

_INSTRUCTION = (
    CLASS_STATEMENT
    + "\n请判断下面这条工单属于哪个类别。"
    "只输出类别名称（青鸟单/朱雀单/白泽单/玄豹单），不要输出其他内容。\n\n工单内容："
)


def build_prompt_baseline(text: str) -> str:
    """路径 B：零样本直接分类（与路径 A 共用完全相同的类别说明）。"""
    return _INSTRUCTION + text


def build_prompt_with_memory(text: str, retrieved_samples: list[str]) -> str:
    """路径 A：以检索到的带标签历史样例作为上下文进行分类。"""
    sample_lines = "\n".join(f"- {s}" for s in retrieved_samples)
    return (
        CLASS_STATEMENT
        + "\n以下是系统从记忆库中检索到的、与该工单相似的历史工单样例（含类别标注），"
        "请参考这些样例归纳判断规则：\n"
        f"{sample_lines}\n\n"
        "请判断下面这条工单属于哪个类别。"
        "只输出类别名称（青鸟单/朱雀单/白泽单/玄豹单），不要输出其他内容。\n\n工单内容："
        + text
    )


# ======================================================================
# 主流程
# ======================================================================


def run() -> dict:
    """执行 TTL 对比实验，返回结构化结果。"""
    t_start = time.time()

    api_key, base_url = load_llm_config()
    logger.info("LLM 凭证加载成功: %s", base_url)

    # 展平测试集（类别 -> 列表），保持固定顺序
    test_cases = [(label, text) for label, texts in TEST_SAMPLES.items() for text in texts]

    # 临时存储目录（实验后清理；失败时保留以便排查）
    storage_dir = Path(tempfile.mkdtemp(prefix="omnimem_diag_ttl_"))
    logger.info("临时存储目录: %s", storage_dir)

    records: list[dict] = []  # 每条测试的明细记录
    inject_stats = {"total": 0, "stored": 0, "skipped": 0, "skipped_detail": []}

    try:
        sdk = OmniMemSDK(storage_dir=storage_dir)
        try:
            # ── 阶段 1：逐条注入带标签训练样例（MCC 风格增量注入）──
            t_inject = time.time()
            for label, texts in TRAIN_SAMPLES.items():
                for text in texts:
                    content = f"[{label}] {text}"
                    result = sdk.memorize(content, memory_type="fact")
                    inject_stats["total"] += 1
                    status = result.get("status", "")
                    if status == "stored":
                        inject_stats["stored"] += 1
                    else:
                        inject_stats["skipped"] += 1
                        inject_stats["skipped_detail"].append({"status": status, "content": content})
            logger.info(
                "样例注入完成: %d/%d 入库, 耗时 %.1fs",
                inject_stats["stored"], inject_stats["total"], time.time() - t_inject,
            )

            # ── 阶段 2：逐条测试两路径 ──
            for idx, (gold, text) in enumerate(test_cases, 1):
                t_case = time.time()

                # 路径 A：先检索相似带标签样例
                recall_result = sdk.recall(text, mode="rag", top_k=TOP_K, max_tokens=4000)
                memories = recall_result.get("memories", []) if recall_result.get("status") == "found" else []
                retrieved_samples: list[str] = []
                retrieved_labels: list[str] = []
                for m in memories:
                    content = m.get("original_content") or m.get("content", "")
                    retrieved_samples.append(content)
                    # 从内容前缀解析样例标签，用于检索质量统计
                    matched = [n for n in CLASS_NAMES if f"[{n}]" in content]
                    retrieved_labels.append(matched[0] if matched else "未知")

                prompt_a = build_prompt_with_memory(text, retrieved_samples)
                reply_a = llm_classify(api_key, base_url, prompt_a)
                pred_a = parse_label(reply_a)

                # 路径 B：零样本直接分类
                prompt_b = build_prompt_baseline(text)
                reply_b = llm_classify(api_key, base_url, prompt_b)
                pred_b = parse_label(reply_b)

                records.append({
                    "gold": gold,
                    "text": text,
                    "pred_a": pred_a,
                    "pred_b": pred_b,
                    "reply_a_raw": reply_a,
                    "reply_b_raw": reply_b,
                    "a_correct": pred_a == gold,
                    "b_correct": pred_b == gold,
                    "retrieved_count": len(retrieved_samples),
                    "retrieved_labels": retrieved_labels,
                    "retrieved_same_class": retrieved_labels.count(gold),
                })
                logger.info(
                    "[%d/%d] gold=%s A=%s(%s) B=%s(%s) 检索同类 %d/%d 耗时 %.1fs",
                    idx, len(test_cases), gold,
                    pred_a, "对" if pred_a == gold else "错",
                    pred_b, "对" if pred_b == gold else "错",
                    retrieved_labels.count(gold), len(retrieved_samples),
                    time.time() - t_case,
                )
        finally:
            sdk.close()
    finally:
        # 清理临时存储（目录集中放在系统临时区，实验完成后删除）
        shutil.rmtree(storage_dir, ignore_errors=True)
        logger.info("临时存储目录已清理: %s", storage_dir)

    # ── 阶段 3：统计 ──
    total = len(records)
    acc_a = sum(r["a_correct"] for r in records) / total if total else 0.0
    acc_b = sum(r["b_correct"] for r in records) / total if total else 0.0

    per_class = {}
    for name in CLASS_NAMES:
        cls_records = [r for r in records if r["gold"] == name]
        if cls_records:
            per_class[name] = {
                "path_a_accuracy": round(sum(r["a_correct"] for r in cls_records) / len(cls_records), 4),
                "path_b_accuracy": round(sum(r["b_correct"] for r in cls_records) / len(cls_records), 4),
                "count": len(cls_records),
            }

    # 检索质量：top-k 检回样例中同类样例占比（检索信号有效性）
    retrieval_quality = {
        "top_k": TOP_K,
        "avg_retrieved": round(sum(r["retrieved_count"] for r in records) / total, 2) if total else 0,
        "avg_same_class_in_topk": round(
            sum(r["retrieved_same_class"] for r in records) / total, 2
        ) if total else 0,
        "zero_same_class_cases": sum(1 for r in records if r["retrieved_same_class"] == 0),
    }

    # 混淆抽样：路径 A 错例明细（用于人工检查）
    a_errors = [
        {"gold": r["gold"], "pred": r["pred_a"], "text": r["text"],
         "retrieved_labels": r["retrieved_labels"]}
        for r in records if not r["a_correct"]
    ]

    # 结论自动生成
    diff = acc_a - acc_b
    if diff <= 0.01:
        conclusion = (
            f"TTL 空白坐实：记忆增强路径准确率 {acc_a:.0%} 不优于直接问 LLM 的 {acc_b:.0%}"
            "（差值 {:.1%}）。OmniMem 记忆机制未能提供测试时学习增量能力。".format(diff)
        )
    else:
        conclusion = (
            f"记忆增强路径准确率 {acc_a:.0%} 高于直接问 LLM 的 {acc_b:.0%}（提升 {diff:.0%}）。"
            "说明记忆检索提供了有效的上下文信号，TTL 能力部分来自记忆而非模型参数，"
            "但增益仅限检索式 few-shot，非参数化学习。"
        )

    result = {
        "experiment": "ttl_incremental_classification",
        "spec_task": "Task 4 — TTL 空白验证",
        "meta": {
            "num_classes": len(CLASS_NAMES),
            "train_samples_per_class": {k: len(v) for k, v in TRAIN_SAMPLES.items()},
            "test_total": total,
            "top_k": TOP_K,
            "llm_model": "deepseek-chat",
            "duration_sec": round(time.time() - t_start, 1),
            "storage_dir_policy": "临时目录（系统 temp区），实验后已清理",
        },
        "inject_stats": inject_stats,
        "path_a_memory_enhanced": {
            "accuracy": round(acc_a, 4),
            "correct": sum(r["a_correct"] for r in records),
            "total": total,
        },
        "path_b_zero_shot_baseline": {
            "accuracy": round(acc_b, 4),
            "correct": sum(r["b_correct"] for r in records),
            "total": total,
        },
        "delta_a_minus_b": round(diff, 4),
        "per_class": per_class,
        "retrieval_quality": retrieval_quality,
        "path_a_errors": a_errors,
        "records": records,
        "conclusion": conclusion,
    }
    return result


def main() -> None:
    result = run()

    output_path = _PROJECT_ROOT / "benchmarks" / "results" / "diagnosis" / "ttl_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("TTL 增量分类验证结果")
    print("=" * 70)
    print(f"路径 A（记忆增强）准确率: {result['path_a_memory_enhanced']['accuracy']:.1%}")
    print(f"路径 B（直接问 LLM）准确率: {result['path_b_zero_shot_baseline']['accuracy']:.1%}")
    print(f"差值 (A - B): {result['delta_a_minus_b']:+.1%}")
    print(f"检索质量: top-{TOP_K} 平均检回 {result['retrieval_quality']['avg_retrieved']} 条, "
          f"其中同类样例 {result['retrieval_quality']['avg_same_class_in_topk']} 条, "
          f"零同类样例的测试 {result['retrieval_quality']['zero_same_class_cases']} 条")
    print(f"注入统计: {result['inject_stats']['stored']}/{result['inject_stats']['total']} 入库")
    print("-" * 70)
    print(f"结论: {result['conclusion']}")
    print(f"结果已写入: {output_path}")
    print(f"总耗时: {result['meta']['duration_sec']}s")


if __name__ == "__main__":
    main()
