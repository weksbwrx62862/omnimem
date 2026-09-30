# LongMemEval 历史四轮（v8 / v8e / v9 / v9b）小样本分析

> 数据来源：`benchmarks/results/longmemeval_{v8,v8e,v9,v9b}/details.jsonl`（各 6 题，为同一组题，
> question_id 一致，可跨轮对比）。本分析是「60 题分题型基线」前的历史证据沉淀，
> 样本量极小（每题型 1 题），结论仅作方向性参考，需由 60 题基线验证。

## 1. 四轮配置与总览

| 轮次 | max_sessions | 正确/总数 | 正确题型 | 错误题型 |
|---|---|---|---|---|
| v8 | 5 | 1/6 (16.7%) | single-session-user | 其余 5 类 |
| v8e | 5 | 3/6 (50%) | ssu / ssp / ku | ms / tr / ssa |
| v9 | 5 | 2/6 (33.3%) | ssu / ku | ms / ssp / tr / ssa |
| v9b | 15 | 3/6 (50%) | ssu / ssp / ku | ms / tr / ssa |

- v8 → v8e 之间检索链路有修复（v8 全部 session_hit=False，v8e 起 5/6 题命中 session）。
- v9 与 v9b 失败模式完全一致：max_sessions 5→15 未改变失败题型，说明失败与
  haystack 规模关系不大，而是结构性问题。

## 2. 逐题跨轮对照（QA 正确性）

| question_id | 题型 | v8 | v8e | v9 | v9b |
|---|---|---|---|---|---|
| e47becba | single-session-user | ✓ | ✓ | ✓ | ✓ |
| 0a995998 | multi-session | ✗ | ✗ | ✗ | ✗ |
| 8a2466db | single-session-preference | ✗ | ✓ | ✗ | ✓ |
| gpt4_59149c77 | temporal-reasoning | ✗ | ✗ | ✗ | ✗ |
| 6a1eabeb | knowledge-update | ✗ | ✓ | ✓ | ✓ |
| 7161e7e2 | single-session-assistant | ✗ | ✗ | ✗ | ✗ |

**稳定失败**：temporal-reasoning（4/4 轮错）、multi-session（4/4 轮错）、
single-session-assistant（4/4 轮错）。
**稳定正确**：single-session-user（4/4 轮对）。
**不稳定**：single-session-preference（对错交替）、knowledge-update（v8 错、v8e 起对）。

## 3. 错题检索质量明细

（sess=session_hit，turn=turn_hit，cov=has_answer_coverage；Y/N/数值）

| question_id | 题型 | v8 | v8e | v9 | v9b |
|---|---|---|---|---|---|
| e47becba | ssu | N/N/0 | Y/Y/0.5 | Y/Y/0.5 | Y/Y/0.5 |
| 0a995998 | ms | N/N/0 | N/N/0 | N/N/0 | N/N/0 |
| 8a2466db | ssp | N/N/0 | Y/Y/1.0 | Y/Y/1.0 | Y/Y/1.0 |
| gpt4_59149c77 | tr | N/N/0 | Y/N/0 | Y/N/0 | Y/N/0 |
| 6a1eabeb | ku | N/N/0 | Y/Y/0.5 | Y/Y/0.5 | Y/Y/0.5 |
| 7161e7e2 | ssa | Y/N/0 | Y/N/0 | Y/N/0 | Y/N/0 |

写入完整性（ingest_count / actual_turns，v9b，max_sessions=15）：
0.96 / 0.96 / 0.97 / 0.99 / 0.95 / 0.98 —— 全部 ≥ 0.95，无写入量不足迹象。

## 4. 失败三分归因（用 benchmarks/diagnosis/attribution.py 对 v9/v9b 实测）

| 题型 | 归因 | 子类 | 依据 |
|---|---|---|---|
| multi-session | **检索丢失** | session 级 | 答案 session 全轮未进 top-k（sess_hit=False），写入比例 0.96 |
| temporal-reasoning | **检索丢失** | turn 级 | session 命中但含答案 turn 未检回（Y/N/0），模型答"无日期信息" |
| single-session-assistant | **检索丢失** | turn 级 | session 命中但 assistant 答案 turn 未检回（Y/N/0） |
| ssp(v9 错题) | **阅读丢失** | — | turn_hit=Y 且 coverage=1.0（证据全在上下文）仍答错 |
| ku(v8 错题) | 检索丢失为主 | — | v8 无 session 命中；v8e 起检索修复后转对 |

## 5. 初步观察（供 60 题基线验证的假设）

1. **错题主导损失段在检索段，而非构造段或阅读段**：三个稳定失败题型全部归因
   检索丢失（1 个 session 级 + 2 个 turn 级），写入比例均 ≥0.95。
2. **turn 级检索失败占 2/3**：session 命中但答案 turn 未检回，指向检索粒度/键
   设计问题（CP1 值粒度、CP2 键设计），与文章框架中「键设计只保留 user 侧发言
   是常见陷阱」的判断吻合（single-session-assistant 即 assistant turn 检索弱）。
3. **temporal-reasoning 的 session 命中但 turn 未命中**：与 CP3 缺失（query 侧
   无时间范围过滤）吻合——日期类证据在 query 改写后与时间条件不匹配，检不回。
4. **multi-session 的 session 级丢失**：跨 session 聚合题的证据分散在多个 session，
   单一 session 都未命中，指向多路召回/聚合能力（MR）薄弱。
5. **阅读丢失确实存在**：v9 的 single-session-preference 错题 turn_hit=Y、
   coverage=1.0 仍答错，与 LongMemEval 论文「15%~19% 实例检索对了但生成错」
   一致，验证了三分归因（而非二分）的必要性。
6. **样本局限**：每题型仅 1 题，无法区分「题型系统性薄弱」与「该题恰好难」；
  60 题基线（每题型 10 题）将解决此问题。

## 6. 性能基线（单题耗时）

| 轮次 | max_sessions | 平均 ingest | 平均检索 |
|---|---|---|---|
| v8 | 5 | 154.1s | 5.1s |
| v8e | 5 | 140.4s | 4.8s |
| v9 | 5 | 138.8s | 4.1s |
| v9b | 15 | 597.9s | 5.3s |
| 当前代码（冒烟） | 15 | 335.1s | 30.0s |

- v9b 单题 ingest ~600s 疑似叠加了更重的事实抽取配置；当前代码 ms15 实测 335s。
- 当前代码 ms15 单题核心耗时（ingest+检索+生成+判分）381.5s，60 题约 6.4h，
  超出 2.5h 预算，故正式基线降档 max_sessions（见冒烟校准结论）。

## 7. 抽样说明（60 题基线）

- 分层抽样：每题型 10 题（6 题型共 60 题），seed=42 可复现，
  脚本 `benchmarks/diagnosis/make_stratified_sample.py`，
  样本 `benchmarks/results/diagnosis/lme_sample_60.json`。
- **排除 30 道 abstention 题**（question_id 含 `_abs`，分布于 ms/ssu/tr/ku）：
  其答案 turn 缺失导致检索质量评估被跳过（skipped），无法参与三分归因。
  此为基线局限：abstention 能力（拒答）未被本轮基线覆盖。
- 评测输入文件 `lme_eval_input_61q.json` = 60 正式题 + 1 道牺牲题（knowledge-update，
  置于文件末尾永不被评测）：用于规避 `run_longmemeval.py` 的启动 bug——
  当 `--limit` 缺省或 ≥ 数据条数时 `n_types` 未定义直接 UnboundLocalError 崩溃，
  故用 `--limit 60` 配 61 题文件让每题型恰好取满 10 题。
