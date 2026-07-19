# OmniMem 与开源项目 LongMemEval 评测对比报告

> 生成时间：2026-07-14 23:36:30
> OmniMem 结果文件：`benchmarks/results/longmemeval_v9b/scores.json`

## ⚠️ 重要警示

> **分数不可直接比较。** 以下各项目的分数使用了不同的 generator LLM、judge LLM 和 retrieval budget（top-k），评测条件存在显著差异，跨项目排名仅供参考，不能作为唯一的性能评判依据。

主要差异包括：

- **Generator LLM 不同**：Mem0 Platform/OSS 使用 GPT-5，HeLa-Mem 系列复现使用 gpt-4o-mini，论文 baseline 使用 GPT-4o。更强的 generator LLM 会显著拉高分数。
- **Judge LLM 不同**：不同项目使用不同的 judge 模型（GPT-5 / GPT-4o / gpt-4o-mini），评判尺度可能不一致。
- **Retrieval Budget 不同**：Mem0 Platform/OSS 使用 top_k=200，而多数论文复现项目未公开 retrieval budget。
- **Oracle 设置**：Offline Reading 为 oracle 上限（直接提供标准答案所在上下文），非真实检索场景。
- **评测范围不同**：OmniMem 为本地评测，题数与配置可能与其他项目不同。

## 一、总分对比

按总分降序排列：

| 排名 | 项目 | 总分 | Generator LLM | Judge LLM | Top-K | Oracle | 数据来源 |
|------|------|------|---------------|-----------|-------|--------|----------|
| 1 | OmniMem (本地评测) | 100.00 | N/A (本地评测) | N/A (本地评测) | N/A | 否 | OmniMem 本地评测 (longmemeval_v9b/scores.json) |
| 2 | Mem0 Platform (v3) | 94.40 | GPT-5 (内部) | GPT-5 | 200 | 否 | mem0ai/memory-benchmarks (2026/04) |
| 3 | Offline Reading (上限) | 91.84 | GPT-4o | GPT-4o | N/A | 是 ⚠️ | LongMemEval 论文 Figure 3a |
| 4 | Mem0 OSS (GPT-5 抽取) | 91.00 | GPT-5 | GPT-5 | 200 | 否 | mem0ai/memory-benchmarks (2026/04) |
| 5 | HeLa-Mem | 65.40 | gpt-4o-mini | gpt-4o-mini | N/A | 否 | ACL 2026 论文 |
| 6 | A-MEM | 62.60 | gpt-4o-mini | gpt-4o-mini | N/A | 否 | HeLa-Mem 论文复现 |
| 7 | NaiveRAG | 61.00 | gpt-4o-mini | gpt-4o-mini | N/A | 否 | HeLa-Mem 论文复现 |
| 8 | ChatGPT (论文 baseline) | 57.73 | GPT-4o | GPT-4o | N/A | 否 | LongMemEval 论文 Figure 3a |
| 9 | FullText | 56.80 | gpt-4o-mini | gpt-4o-mini | N/A | 否 | HeLa-Mem 论文复现 |
| 10 | Mem0 (HeLa-Mem 复现) | 53.61 | gpt-4o-mini | gpt-4o-mini | N/A | 否 | HeLa-Mem 论文复现 |
| 11 | MemoryOS | 44.80 | gpt-4o-mini | gpt-4o-mini | N/A | 否 | HeLa-Mem 论文复现 |
| 12 | LangMem | 37.20 | gpt-4o-mini | gpt-4o-mini | N/A | 否 | HeLa-Mem 论文复现 |
| 13 | Coze (论文 baseline) | 32.99 | GPT-4o | GPT-4o | N/A | 否 | LongMemEval 论文 Figure 3a |

## 二、分维度对比

仅展示提供分维度数据的项目：

| 维度 | Mem0 Platform (v3) | Mem0 OSS (GPT-5 抽取) | OmniMem (本地评测) |
|------|------|------|------|
| 单会话-用户 | 98.60 | 95.70 | 100.00 |
| 单会话-助手 | 98.20 | 92.90 | 0.00 |
| 单会话-偏好 | 96.70 | 93.30 | 100.00 |
| 知识更新 | 93.60 | 91.00 | 100.00 |
| 时序推理 | 97.00 | 94.70 | 0.00 |
| 多会话 | 88.00 | 83.50 | 0.00 |

## 三、评测条件标注

| 项目 | Generator LLM | Judge LLM | Top-K | Oracle | 备注 |
|------|---------------|-----------|-------|--------|------|
| Mem0 Platform (v3) | GPT-5 (内部) | GPT-5 | 200 | 否 | mem0ai/memory-benchmarks (2026/04) |
| Mem0 OSS (GPT-5 抽取) | GPT-5 | GPT-5 | 200 | 否 | mem0ai/memory-benchmarks (2026/04) |
| HeLa-Mem | gpt-4o-mini | gpt-4o-mini | N/A | 否 | ACL 2026 论文 |
| A-MEM | gpt-4o-mini | gpt-4o-mini | N/A | 否 | HeLa-Mem 论文复现 |
| NaiveRAG | gpt-4o-mini | gpt-4o-mini | N/A | 否 | HeLa-Mem 论文复现 |
| FullText | gpt-4o-mini | gpt-4o-mini | N/A | 否 | HeLa-Mem 论文复现 |
| Mem0 (HeLa-Mem 复现) | gpt-4o-mini | gpt-4o-mini | N/A | 否 | HeLa-Mem 论文复现 |
| MemoryOS | gpt-4o-mini | gpt-4o-mini | N/A | 否 | HeLa-Mem 论文复现 |
| LangMem | gpt-4o-mini | gpt-4o-mini | N/A | 否 | HeLa-Mem 论文复现 |
| ChatGPT (论文 baseline) | GPT-4o | GPT-4o | N/A | 否 | LongMemEval 论文 Figure 3a |
| Coze (论文 baseline) | GPT-4o | GPT-4o | N/A | 否 | LongMemEval 论文 Figure 3a |
| Offline Reading (上限) | GPT-4o | GPT-4o | N/A | 是 ⚠️ | LongMemEval 论文 Figure 3a |
| OmniMem (本地评测) | N/A (本地评测) | N/A (本地评测) | N/A | 否 | 模式=full | 题数=6 | max_sessions=15 | user_only=False |

### OmniMem 详细指标

**检索质量：**

| 指标 | 值 |
|------|------|
| Session Hit Rate | 100.00% |
| Turn Hit Rate | 100.00% |
| Avg Coverage | 33.33% |

**性能指标：**

| 指标 | 值 |
|------|------|
| 平均写入耗时 | 597.93 s |
| 平均检索耗时 | 5.25 s |
| 平均写入条数 | 155.50 |
| 总写入耗时 | 3587.58 s |
| 总检索耗时 | 31.53 s |

## 四、数据来源说明

- **Mem0 Platform (v3) / Mem0 OSS**：来自 mem0ai/memory-benchmarks 仓库（2026/04），使用 GPT-5 作为 generator 与 judge，top_k=200。
- **HeLa-Mem / A-MEM / NaiveRAG / FullText / Mem0(复现) / MemoryOS / LangMem**：来自 HeLa-Mem 论文（ACL 2026）及其复现实验，统一使用 gpt-4o-mini 作为 generator 与 judge。
- **ChatGPT / Coze / Offline Reading**：来自 LongMemEval 论文 Figure 3a，使用 GPT-4o 作为 generator 与 judge。Offline Reading 为 oracle 上限。
- **OmniMem**：本地 LongMemEval 评测结果，generator/judge 配置见评测脚本。

---

*本报告由 `benchmarks/comparison_report.py` 自动生成。*
