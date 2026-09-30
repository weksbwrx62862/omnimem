# OmniMem 薄弱环节诊断报告

> 诊断 spec：`.trae/specs/diagnose-memory-weaknesses/spec.md`
> 参照系：文章《Agent Memory 评测：∞Bench / LoCoMo / LongMemEval / MemoryAgentBench / AMA-Bench 拆解》
> 生成日期：2026-09-08
> ⚠️ 局限声明：LongMemEval 60 题全量基线评测（每题型 10 题）因单题 ingest ~335s、总时长预估 6h+，**已按用户要求暂停**（评测进程已停止，首题未完成，无该部分数据）。本报告的分题型结论基于**历史四轮（v8/v8e/v9/v9b，各 6 题、24 题次）**与失败归因，样本量为每题型 1~4 题次，方向性结论可信、量化结论需全量评测补足。其余各部分（差距矩阵、SF 50 题自测、TTL/LRU 实验）均为完整执行。

---

## 0. 执行摘要（薄弱点按优先级排序）

| 优先级 | 薄弱点 | 证据强度 | 实测影响 |
|---|---|---|---|
| **P0** | 阅读层语义去重会把**矛盾新事实当重复误杀**（SF 核心短板） | 实测 50 题 + 代码根因 | 新事实写入成功、融合层 50/50 命中，但阅读层 refine 去重丢弃 37/50 → SF single-hop 0%、multi-hop 16% |
| **P1** | 查询侧无时间感知（CP3 缺失）+ temporal 重排方向性错误 | 代码分析 + 历轮实测 | temporal-reasoning 历轮全 0；rerank 按 created_at 近因加分对「久远发生」问题方向相反 |
| **P1** | 跨 session 聚合检索薄弱（LRU / MR 能力） | 实测 + 代码分析 | multi-session 历轮全 0；top-k 截断下聚合证据覆盖率不足（详见 §5） |
| **P1** | assistant 侧发言检索弱（CP2 单一键的伴生问题） | 历轮实测 | single-session-assistant 历轮全 0、turn_hit 为 0 |
| **P2** | TTL 无参数化技能习得通道 | 实测 | 记忆增强 75% vs 直接问 25%——RAG 记忆已能提供 few-shot 式增益，但无参数化学习 |
| **P2** | ingest 性能瓶颈 | 实测 | ms15 单题 ingest 335s，评测/扩展样本成本放大 |
| **P2** | multi-hop 冲突传播无机制、时点查询未接入检索 | 代码分析 | UPDATE 仅标记直接冲突单条；TemporalKG query_at_time 无检索入口 |

---

## 1. 静态差距矩阵

见 `benchmarks/results/diagnosis/gap_matrix.md`（本报告姊妹文档）。四能力 × 四控制点 × 协议光谱全部对照，每项附代码路径证据。核心结论：

- **四能力**：AR 中强、SF 中（single-hop 强/multi-hop 缺）、LRU 弱、TTL 空白
- **四控制点**：CP1（原子事实粒度）与 CP4（评测侧 CoN）已落地；CP2（K=V+fact 键增强）与 CP3（query 侧时间感知）缺失
- **协议光谱**：当前 LongMemEval 型（逐 turn 批量摄入→独立作答），无法测 TTL 与 SF multi-hop 的真实水平

---

## 2. 分题型基线与失败归因（历史四轮 + attribution，LME 60 题暂停）

### 2.1 历轮分题型 QA（v8/v8e/v9/v9b，各 6 题同题集，可跨轮对比）

| 题型 | v8 | v8e | v9 | v9b | 稳定结论 |
|---|---|---|---|---|---|
| single-session-user | ✓ | ✓ | ✓ | ✓ | **稳定正确**（4/4） |
| single-session-preference | ✗ | ✓ | ✗ | ✓ | 不稳定（对错交替） |
| knowledge-update | ✗ | ✓ | ✓ | ✓ | v8e 起稳定正确 |
| multi-session | ✗ | ✗ | ✗ | ✗ | **稳定全错**（4/4） |
| temporal-reasoning | ✗ | ✗ | ✗ | ✗ | **稳定全错**（4/4） |
| single-session-assistant | ✗ | ✗ | ✗ | ✗ | **稳定全错**（4/4） |

max_sessions 5→15 未改变失败题型 → 失败与 haystack 规模无关，是结构性问题。

### 2.2 失败三分归因（attribution.py，基于历史 v9/v9b）

| 题型 | 主导损失段 | 判据 |
|---|---|---|
| multi-session | **检索丢失（session 级）** | 答案 session 全轮未进 top-k；写入比例 0.96 |
| temporal-reasoning | **检索丢失（turn 级）** | session 命中但含答案 turn 未检回；模型答"无日期信息" |
| single-session-assistant | **检索丢失（turn 级）** | session 命中但 assistant 答案 turn 未检回 |
| knowledge-update（v9b） | —（已答对） | 检索修复后转对 |

写入比例全部 ≥0.95 → **无构造丢失迹象**；错题主导损失在检索段。同时存在阅读丢失实例（v9 ssp 错题 turn_hit=Y、coverage=1.0 仍答错），印证三分（而非二分）归因的必要性，与 LongMemEval 论文「检索对但生成错占 40~50%」一致。

---

## 3. SF 专项自测（全量 50 题，完整执行）—— 本报告最重要发现

### 3.1 结果

| 分项 | 题数 | strict_pass | SubEM | answer_substring_hit |
|---|---|---|---|---|
| single-hop | 25 | 0 | **0.0** | 0.32 |
| multi-hop（2+3跳） | 25 | 4 | **0.16** | 0.76 |
| └ 2-hop | 20 | 4 | 0.20 | 0.80 |
| └ 3-hop | 5 | 0 | 0.0 | 0.60 |

对照行业水位：MemoryAgentBench 实证所有方法 SF multi-hop 最高 28%。OmniMem multi-hop 16% 低于但同量级。

### 3.2 分层定位：写入、检索、阅读三段全打通（重大发现）

| 环节 | 结果 |
|---|---|
| 冲突仲裁（写入侧） | 触发 39/50（78%），类型全为 semantic_contradiction，无去重误杀 → **仲裁检测有效** |
| 融合层检索（retriever.search 出口） | 新事实命中 **50/50（100%）** → **检索完全无问题** |
| 阅读层（ContextManager.refine 语义去重） | **去重误杀 37/50** → **真正的瓶颈** |

### 3.3 根因（代码级）

`context/manager.py` 的 `_is_duplicate` 用两级去重判定重复：
- 快速路径：指纹 Jaccard > 0.7
- 慢路径：Jaccard ∈ (0.4, 0.7] 时 embedding 相似度 > 0.92 判重

对「矛盾事实」（同一命题换取值，如 `Alice的丈夫是Bob` vs `Alice的丈夫是Charlie`），新旧事实指纹/语义高度相似 → 新事实被当作与已注入内容重复而丢弃。

**典型失败样例（sh_01）**：
- 问题：Alice 的丈夫是谁？ 答案：Charlie
- 融合层 ranks：`{old_fact: rank 0, new_fact: rank 6}` —— 新旧都检回了
- 阅读层出口（最终 retrieved）：只有 `Fact #1: Alice的丈夫是Bob` + 干扰项 —— **新事实被去重删除**
- 结果：答旧值 Bob → 0 分

**本质缺陷**：语义去重只防「记忆膨胀」，不理解「矛盾更新应保留新事实、抑制旧事实」。去重逻辑与 `is_superseded`/冲突仲裁两套语义互不相认——写入侧辛辛苦苦仲裁出的新事实，读取侧被当垃圾滤掉。这也解释了为何 knowledge-update 在 LongMemEval 上能对（单条直查、证据少），而 SF 序列注入场景（新旧同库）全面暴露。

---

## 4. TTL 验证（完整执行）

增量分类实验（4 个自造类别 × 17 训练样例注入记忆，20 测试题）：

| 路径 | 准确率 |
|---|---|
| 路径 A：记忆增强（recall top-k → LLM 分类） | **75%**（15/20） |
| 路径 B：零样本直接问 LLM | **25%**（5/20） |
| Δ | **+50%** |

结论：**记忆检索提供了有效的上下文信号**——OmniMem 能充当「外部技能存储」，TTL 能力部分来自记忆而非模型参数。但增益限于检索式 few-shot，**无参数化学习通道**（L4 LoRA 仅模拟占位）。「TTL 完全空白」的判断被修正为「检索式 few-shot 有效、参数化习得空白」。

## 5. LRU 验证（完整执行）

合成 30 session / 150 turn，埋 40 条计数证据，聚合查询「我一共买了多少本书？」：

| top_k | 证据覆盖率 |
|---|---|
| 5 | 12.5% |
| 10 | 22.5% |
| 20 | 45% |
| 50 | 82.5% |
| 100 | **97.5%**（仍无法 100% 正确计数） |

结论：top-k 截断使聚合任务在检索层即失败——即使取 100/150 turn 仍缺 1 条证据导致计数错误。与 MemoryAgentBench「RAG 系只能检索部分信息、缺全局理解」结论一致。高区分度查询对照（同 k 覆盖率低 10%）验证聚合型问题的语义相似度检索失效更严重。**无全库统计/聚合通道**。

---

## 6. 其他已确认薄弱点（静态证据）

1. **temporal rerank 语义方向错误**：`retrieval/registry.py` apply_temporal_rerank 基于 created_at 近因加分，未用已标注的 `_occurrence_time`；对「很久以前发生」类问题方向相反（旧答案被降权）——temporal-reasoning 全 0 的机制级解释
2. **multi-hop 冲突传播无机制**：UPDATE 仅标记直接冲突单条（memory_write_service.py），推理链下游不更新；contradiction 仅读取侧打包、不回写
3. **时点查询未接入检索**：TemporalKG query_at_time 仅内部定义，_GraphRetriever 只调 query_current
4. **图谱实体提取粗糙**：2-4 字中文全量正则匹配大量非实体
5. **ingest 性能瓶颈**：LLM 决策链（冲突检测+事实抽取+仲裁）逐条同步，ms15 单题 335s

---

## 7. 后续动作建议（供改进 spec 立项）

按「影响面 × 修复成本 × 论文已验证收益」：

1. **[P0] 去重矛盾感知**：`_is_duplicate` 增加冲突预检——若候选与已注入记忆构成语义矛盾（复用 contradiction 检测），跳过去重并交给 superseded/冲突处理。预期直接修复 SF single-hop（0%→高），multi-hop 同步受益
2. **[P1] 查询侧时间感知（CP3）**：query 时间范围抽取 → 按 `_occurrence_time` 过滤/加权；修复 temporal rerank 方向。论文参考：temporal 召回 +6.8~11.3%
3. **[P1] 键设计 role 感知/事实增强（CP2）**：K=V+fact 增强（Recall +9.4%）、assistant 侧检索弱修复
4. **[P1] 聚合通道**：为全局聚合查询（MR/LRU）提供统计/枚举通道或 LLM 规划多轮检索
5. **[P2] 评测补全**：恢复 60 题分层基线（建议降 max_sessions 控制成本，每题型 ≥10）验证历史方向性结论

## 8. 产物清单

| 产物 | 路径 |
|---|---|
| 差距矩阵 | `benchmarks/results/diagnosis/gap_matrix.md` |
| 历史四轮分题型分析+归因 | `benchmarks/results/diagnosis/historical_analysis.md` |
| SF 全量明细 | `benchmarks/results/diagnosis/sf_selftest_details.jsonl` |
| SF 分数 | `benchmarks/results/diagnosis/sf_selftest_scores.json` |
| TTL 结果 | `benchmarks/results/diagnosis/ttl_result.json` |
| LRU 结果 | `benchmarks/results/diagnosis/lru_result.json` |
| 三分归因工具 | `benchmarks/diagnosis/attribution.py`（用法见文件头） |
| 评测样本工具 | `benchmarks/diagnosis/make_stratified_sample.py` |
| 各实验脚本 | `benchmarks/diagnosis/{sf_selftest,ttl_experiment,lru_experiment}.py` |
| 分层样本 | `benchmarks/results/diagnosis/lme_sample_60.json` |
