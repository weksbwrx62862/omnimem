# OmniMem 静态差距矩阵（基于记忆评测基准框架）

> 参照系：《Agent Memory 评测：∞Bench / LoCoMo / LongMemEval / MemoryAgentBench / AMA-Bench 拆解》
> 四能力（AR/TTL/LRU/SF）× 检索四控制点（CP1-CP4）× 二段归因 × 协议光谱。
> 证据类型：静态代码分析 + 历轮 LongMemEval 评测数据（v8/v9b，各 6 题小样本）。
> 生成日期：2026-09-08。本文档为诊断 spec（Task 1）产出，不含改进方案。

## 1. 四能力对照（AR/TTL/LRU/SF）

| 能力 | OmniMem 对应机制 | 判断 | 证据（代码路径+简述） | 基准中的已知失败模式 |
|---|---|---|---|---|
| **AR** 精准检索 | HybridRetriever 多通道混合检索：vector+bm25+graph+temporal+catalog，RRF 融合，Planner 意图加权，同义扩展，重排 | **中强** | `retrieval/engine.py`（HybridRetriever.search）；`retrieval/fusion.py`（RRF 融合+superseded 过滤）；`retrieval/planner.py`（QueryPlanner 意图→通道权重）；`retrieval/synonym_expander.py`（同义扩展）；`retrieval/reranker.py`。v9b session_hit_rate=1.0、turn_hit_rate=1.0（`benchmarks/results/longmemeval_v9b/scores.json`） | LongMemEval：检索对了但生成错了占全部错误 40%~50%（阅读段损失）；assistant 侧发言检索是常见陷阱。OmniMem v9b single-session-assistant turn_hit_rate=0.0，与陷阱吻合 |
| **TTL** 测试时学习 | 仅实验性 L4 内化层：KVCache（高频记忆缓存预填充）+ LoRA（peft 缺失时纯模拟训练） | **空白**（实验占位，非 TTL） | `internalize/kv_cache.py`（KVCacheManager：access_count>10 缓存，是缓存非技能习得）；`internalize/lora_train.py`（"模拟训练模式——记录训练请求但不执行"）；`internalize/plugin.py`（@experimental_class 装饰）；`sdk.py` 中 `_lora_trainer`/`_kv_cache` 均为 `getattr(..., None)` 未入主链路。全局搜索无 in-context learning / 增量分类机制 | MemoryAgentBench：TTL 用增量分类任务（MCC 协议）测部署期不经训练习得新行为；RAG 系与 Long-Context 在 TTL 上差距最大。OmniMem 无任何 ICL 式技能积累通道，TTL 得分预期为"直接问 LLM"基线水平 |
| **LRU** 长程理解 | top-k 检索 + Gated 证据组织器（去重/时间排序）+ L3 Consolidation 升华 + ReflectEngine 反思 | **弱** | `services/recall_service.py`（top_k 默认 40，无全库扫描/全局统计）；`retrieval/organizer.py`（EvidenceOrganizer：仅聚合类查询触发，作用于已检回的 top-k 内）；`deep/consolidation.py`（ConsolidationEngine 四阶段升华，异步离线）；`deep/reflect/pipeline.py`（ReflectEngine 五步反思循环）。v9b overall_avg_coverage=0.3333、multi-session coverage=0.0 | MemoryAgentBench：RAG 系"只能检索部分信息、缺全局理解"，全局聚合（计数/跨 session 汇总）被 top-k 截断。OmniMem 无全库统计通道，multi-session 题型历轮全 0（v8/v9b）与此吻合 |
| **SF** 选择性遗忘 | 写入侧：ConflictResolver 仲裁 + LLM ADD/UPDATE/DELETE 决策 + TemporalKG（valid_at/superseded_by）；检索侧：is_superseded 过滤 + ContradictionDetector 冲突组打包；遗忘：FSRS v4 | **中**（single-hop 强，multi-hop 缺） | 写入：`governance/conflict.py`（ConflictResolver，latest/add_only 策略）；`services/memory_write_service.py` L292-318（M7-10 LLM 决策 UPDATE/DELETE，旧记忆标记 is_superseded）；`governance/temporal_kg.py`（TemporalTriple：valid_at/invalid_at/superseded_by，query_at_time 时点查询）。检索：`retrieval/fusion.py` L72-73、`retrieval/hybrid_orchestrator.py` L567-568（过滤 is_superseded）；`retrieval/contradiction.py`（冲突组+时间来源标注）。遗忘：`governance/forgetting.py` + `governance/fsrs_engine.py`（FSRS v4 四阶段归档）。v9b knowledge-update=1.0 | MemoryAgentBench：SF multi-hop 所有方法最高仅 28%（行业共同短板）。OmniMem UPDATE 只标记直接冲突的单条记忆（memory_write_service.py），被替换事实的下游推理链（关联记忆）无同步传播机制；KG 侧 superseded_by 也仅三元组级，且 `_GraphRetriever` 只调 query_current（`retrieval/registry.py`），历史时点查询未接入检索链路 |

## 2. 四控制点对照（CP1-CP4）

| 控制点 | 现状 | 论文参考收益 | 差距结论 | 证据代码路径 |
|---|---|---|---|---|
| **CP1** 值粒度 | 已落地 fact 粒度：写入主链路做原子事实抽取（LLM 失败回退规则），过程性/因果类型保留整句不拆分；长内容另有分段存储 | LongMemEval 比较显示粒度影响显著（fact 与 round 各有优劣） | **已落地（较强）** | `services/memory_write_service.py` L180-200（extract_facts 多条独立写入，metadata.is_atomic_fact 标记）；`perception/fact_extractor.py`（AtomicFactExtractor：短文本直存、LLM 抽取、正则回退）；评测侧 `benchmarks/retrieval_only_eval.py` --fact-extract 参数 |
| **CP2** 键设计 | K=V 单一键：向量索引的 embedding 输入即记忆原文，无 keyphrase 键、无 K=V+fact 拼接增强 | K=V+fact：Recall@k +9.4%、准确率 +5.4%（论文验证） | **缺失** | `retrieval/vector.py`（add/add_batch/_build_batch_entries：documents=content 原文，K=V）；全局搜索 keyphrase/K=V 无命中。注：`retrieval/synonym_expander.py` 是查询侧同义词扩展，非索引键设计；longmemeval_adapter.py 的偏好前缀注入 `[prefer like enjoy]` 是评测侧补丁（L561），非通用键增强 |
| **CP3** 查询时间感知 | 半成品：ingest 侧有 occurrence_time 标注，query 侧无时间范围抽取与索引过滤 | time-aware query expansion：temporal 召回 +6.8%~11.3%（论文验证） | **缺失（query 侧）** | 标注侧：`retrieval/temporal_separation.py`（parse_occurrence_time/annotate_results，融合后标注见 `retrieval/hybrid_orchestrator.py` L354）。query 侧仅有：(a) Planner temporal 意图→通道加权（`retrieval/planner.py` L199-201，temporal×3.0）；(b) 同义扩展+偏好改写（`retrieval/hybrid_orchestrator.py` L325-336，无时间语义）；(c) temporal rerank=近因衰减加分（`retrieval/registry.py` apply_temporal_rerank：基于 created_at 的 exp 衰减，方向是"越新越好"）。无任何"query 抽时间范围→按 _occurrence_time 过滤索引"机制 |
| **CP4** 阅读策略 | 评测侧已落地（CoN 两段式 + LLM Judge）；生产链路未接入（recall 直接拼接 context 返回） | CoN+JSON：oracle 检索下 +10 绝对百分点（论文验证）；LongMemEval 实证 15%~19% 实例为"检索对了但生成错了" | **评测已落地、生产未接入** | 评测侧：`benchmarks/retrieval_only_eval.py`（--chain-of-note、--judge-model、_generate_answer_with_con 两段式：逐条做笔记过滤 N/A→基于笔记作答，--con-max-notes 控成本）。生产侧：`services/recall_service.py`（handle：检索→Token 裁剪→返回拼接 context，无笔记/结构化阅读步骤） |

## 3. 协议光谱定位

**光谱**：一次性喂全文（∞Bench）→ 静态给定历史（LoCoMo）→ 摄入历史后独立作答（LongMemEval）→ 逐块增量注入+记忆指令、写后即冻（MemoryAgentBench）。

**OmniMem 当前位置**：LongMemEval 型协议。
- `benchmarks/run_longmemeval.py`（run_single_question）：每题执行 ingest → retrieval → generation → judge 四段，ingest 由 `benchmarks/longmemeval_adapter.py` 的 OmniMemMemoryProvider.ingest_sessions 逐 turn 批量调用 sdk.memorize()（无记忆指令、无冻结语义），摄入完成后独立作答。
- 补充设施：LoCoMo 适配器（`benchmarks/locomo_adapter.py`）与 STATE-Bench（`benchmarks/STATE-Bench/`，多域任务型评测）协议形态同类。

**与逐块增量注入协议的差距**：
1. **无记忆指令**：MemoryAgentBench 在逐块注入时即考验 Agent 决定"记什么"，OmniMem 评测协议由 adapter 无差别写入全部 turn。
2. **无写后即冻**：OmniMem 写入链路含 LLM ADD/UPDATE/DELETE 决策与冲突仲裁（`services/memory_write_service.py`），后续写入可改写早期记忆状态。该能力在逐块协议下才能与"查询期遗忘"区分开。
3. **无法测 TTL**：协议无部署期增量任务（增量分类/技能序列），TTL 空白被协议固化，现有评测永远测不到。
4. **SF 归因混淆**：knowledge-update 得分无法区分"写入期整合正确"与"查询期选择性遗忘正确"。

**影响**：当前协议可支撑 AR/SF(single-hop) 的部分结论，但 TTL、LRU（全局聚合）、SF multi-hop 三个维度的真实水平在现有协议下不可测，需补充逐块增量注入式评测（spec Task 4 的验证协议）。

## 4. 其他发现（探索中发现的额外薄弱点/强项）

### 强项（诊断中确认不是问题的部分）
1. **多通道检索框架完整且工程化**：插件化通道注册（`retrieval/registry.py`）、意图加权（`retrieval/planner.py`）、熔断降级（`retrieval/circuit_breaker.py`）、缓存（`retrieval/cache.py`）、自愈（`retrieval/self_healing.py`）、证据组织（`retrieval/organizer.py`）——超出一般 RAG 系统配置。
2. **时序知识图谱**：`governance/temporal_kg.py` 的 valid_at/invalid_at/superseded_by 三时序字段 + query_at_time 历史快照 + detect_contradiction + get_timeline，对标 Zep/Graphiti，是 SF 的潜在基础设施（但未接入检索主链路，见薄弱点 4）。
3. **写入治理链路成熟**：安全扫描、防递归注入、审计日志、RBAC（`services/memory_write_service.py` handle 全流程）。
4. **评测设施多元**：LongMemEval / LoCoMo / STATE-Bench / BM25-vs-FTS5（`benchmarks/fts5_recall_bench.py`）/ 检索质量专项（`benchmarks/retrieval_only_eval.py`）多套并存。

### 额外薄弱点
1. **temporal rerank 与 occurrence_time 语义脱节**：`retrieval/registry.py` apply_temporal_rerank 基于 created_at（提及时间）做近因加分，未使用已标注的 _occurrence_time；且对"很久以前发生"类 temporal-reasoning 问题，近因加分方向相反（旧答案被降权）。ingest 侧标注（temporal_separation.py）与 query 侧排序（registry.py）两套时间语义互不相认。
2. **assistant 侧发言检索弱**：v9b single-session-assistant turn_hit_rate=0.0。`benchmarks/longmemeval_adapter.py` L550 的 _is_substantive 过滤只影响写入侧；检索键无 role 感知（与 CP2 单一键设计问题叠加）。
3. **图谱通道实体提取粗糙**：`retrieval/registry.py` _GraphRetriever._extract_entities 用 2-4 字中文词组全量正则匹配（会匹配大量非实体词），且仅在时序查询时生效、固定 score=0.5，多跳遍历能力有限。
4. **TemporalKG 时点查询未接入检索**：query_at_time 仅在 `governance/temporal_kg.py` 内部定义，`retrieval/registry.py` _GraphRetriever.search 只调 query_current——历史时间点问答（"X 在 7 月时是什么"）检索链路无入口。
5. **ingest 性能瓶颈**：v9b avg_ingest_time_s=597.93（15 sessions）；写入链路的 LLM 决策（ADD/UPDATE/DELETE + 原子事实抽取 + 冲突检测）逐条同步调用，全量评测的时长成本会放大样本扩展难度（`benchmarks/results/longmemeval_v9b/scores.json` performance 段）。
6. **multi-hop 传播无机制**：`services/memory_write_service.py` 的 UPDATE 仅标记直接冲突记忆（update_field(target_id, is_superseded=True)），无沿推理链/关联图传播的钩子；contradiction.py 只在读取侧打包冲突证据组，不回写。
7. **样本量局限**：历轮 LongMemEval 评测（v8/v8e/v9/v9b）各 6 题，分题型结论为方向性证据（temporal-reasoning 与 multi-session 历轮全 0 模式稳定），量化结论需全量评测（spec Task 2）支撑。
