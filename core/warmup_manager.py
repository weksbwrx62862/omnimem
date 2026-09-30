"""后台预热管理器 — 负责 L3/L4 初始化、数据预热、检索引擎预热和启动审计。"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)

# ★ P1-8：预热要把**全部**活跃索引条目喂给 BM25 的差集更新，切片 = 批量删除。
#   与 governance/reconciler.py 的全表扫描口径一致，够覆盖单机记忆量。
_FULL_SCAN_LIMIT = 10_000_000


class WarmupManager:
    """管理 OmniMem 后台异步预热流程。

    职责:
      1. L3/L4 并行初始化（reflect + lora）
      2. 数据预热（索引 + BM25 重建 + index.db 同步）
      3. 检索引擎预热（SentenceTransformer + ChromaDB + 向量健康检查）
      4. 启动审计（一致性检查 + 自动修复）
    """

    _BG_INIT_BUDGET_SEC = 120.0

    def __init__(
        self,
        init_reflect_fn: Any,
        init_lora_fn: Any,
        index: Any,
        store: Any,
        retriever: Any,
        retrieval: Any,
        auditor: Any,
    ) -> None:
        self._init_reflect_fn = init_reflect_fn
        self._init_lora_fn = init_lora_fn
        self._index = index
        self._store = store
        self._retriever = retriever
        self._retrieval = retrieval
        self._auditor = auditor

    def run(self) -> None:
        """执行完整的后台预热流程（非阻塞调用，但自身是同步的）。"""
        logger.info("OmniMem background warmup: starting...")
        t0 = time.time()

        # L3/L4 并行初始化（★ P1-6: 有界。不能用 with ThreadPoolExecutor —— 它退出时
        # 隐式 shutdown(wait=True)，会把 as_completed 的超时又抵消掉）
        executor = ThreadPoolExecutor(max_workers=1)
        futures = {
            executor.submit(self._init_reflect_fn): "reflect",
            executor.submit(self._init_lora_fn): "lora",
        }
        done: set[str] = set()
        try:
            try:
                for future in as_completed(futures, timeout=self._BG_INIT_BUDGET_SEC):
                    name = futures[future]
                    done.add(name)
                    try:
                        future.result()
                    except Exception as e:
                        logger.warning("BG init %s failed: %s", name, e)
            except TimeoutError:
                logger.warning(
                    "BG init 超过 %.0fs 预算，跳过: %s",
                    self._BG_INIT_BUDGET_SEC,
                    ", ".join(sorted(set(futures.values()) - done)) or "未知",
                )
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

        # 数据预热（索引 + BM25）
        self._warmup_data()

        # 检索引擎预热（SentenceTransformer + ChromaDB）
        self._warmup_retrieval()

        # 启动审计
        self._startup_audit()

        elapsed = time.time() - t0
        logger.info("OmniMem background warmup: complete in %.1fs", elapsed)

    def _warmup_data(self) -> None:
        """数据预热：索引条目 → store 预热 + BM25 重建 + index.db 同步。"""
        try:
            # ★ P1-8：这里必须是**全量**活跃条目，不能是 LIMIT 切片。
            #   rebuild_bm25_from_entries → BM25Retriever.update_from_entries 的语义是
            #   「传进来的就是全世界的全部」：`ids_to_delete = current_ids - new_entries`。
            #   原先写死 limit=2000，副本实测 index 有 3 135 行 ⇒ 每次开机按 stored_at DESC
            #   只喂最新 2 000 条，剩下 1 135 条被当成"已删除"从 BM25 语料里剔掉并持久化
            #   （日志：BM25 rebuild (incremental): added=0, updated=0, deleted=1210）。
            #   结果老记忆的关键词通道每次启动都被削一次，而这正是 P1-1 那类"召不回"。
            indexed_entries = self._index.search_l1(limit=_FULL_SCAN_LIMIT)
            if indexed_entries:
                self._store.warm_up(indexed_entries[:500])
                rebuilt = self._retriever.rebuild_bm25_from_entries(indexed_entries)
                if rebuilt > 0:
                    logger.info(
                        "OmniMem: warmed up %d entries, rebuilt BM25 with %d entries",
                        min(len(indexed_entries), 500),
                        rebuilt,
                    )
            # index.db 全量同步 — 通过 MetaStore 高层接口
            try:
                stale, missing = self._store.meta_store.sync_from_index(self._index.db_path)
                if stale or missing:
                    logger.info("OmniMem: index.db synced — cleaned %d stale, added %d missing", stale, missing)
            except Exception as _e:
                logger.warning("OmniMem index.db sync skipped (non-fatal): %s", _e)
        except Exception as e:
            logger.warning("OmniMem warm-up/BM25 rebuild failed (non-fatal): %s", e)

    def _warmup_retrieval(self) -> None:
        """检索引擎预热 + 向量回填。"""
        try:
            self._retrieval.warmup()
            logger.info("OmniMem: retrieval engine warmup complete")
            try:
                self._backfill_vectors()
            except Exception as e:
                logger.warning("OmniMem: vector backfill pass failed: %s", e)
        except Exception as e:
            logger.warning("OmniMem retrieval warmup failed (non-fatal): %s", e)

    def _vector_covered_ids(self) -> set[str] | None:
        """已向量化记忆 id 集合；None = 拿不到（后端不支持枚举或检索器是旧对象）。"""
        fn = getattr(self._retriever, "vector_covered_memory_ids", None)
        if not callable(fn):
            return None
        try:
            return fn()
        except Exception:
            logger.warning("OmniMem: 向量覆盖率集合口径失败，退回标量口径", exc_info=True)
            return None

    def _backfill_vectors(self) -> None:
        """补回降级期间漏建向量的记忆：先排空死信队列，再按真实缺口决定是否全量重建。

        ★ P1-2 原实现只在 ``vec_count == 0`` 时重建。生产环境 vec_count=770 永远不为 0，
        于是 6 001 条 ``vector_index_pending.jsonl`` 无人消费、2 596 条记忆只有抽屉没有索引，
        检索覆盖率停在 17.2% —— 门控条件本身就永远不会触发。

        ★ P1-11 P1-2 的门控还是错的：它拿 ``vector_count()``（**向量行**数）和索引条目数比，
        而一条长记忆会被切成多个 chunk 行（见 ``retrieval/vector.py`` 的 ``_prepare_batch``），
        行数只会 ≥ 记忆数。线上实测向量 3 217 行 / 活跃记忆 2 742 条，判据「覆盖率 100%」
        是通过而非验证——真实缺口再多也报不出来。反例：80 条记忆切成 3 200 行，
        对着 2 742 条活跃记忆，旧判据照样放行。
        现在按**集合差**判定：``uncovered = 活跃 id − 已向量化 id``，只有真实缺口超容忍度才重建。
        后端无法枚举 id 时（如 Qdrant 的派生 UUID point id）退回标量口径，但明确告警它不可信。

        无论走哪条分支，最后都做一次孤儿向量清扫：重建是 upsert，永不回收
        「记忆已经不存在」的残留行（副本实测 144 条；另有 440 条 index 查不到但抽屉仍在，
        那些必须保留 —— 详见 ``_sweep_orphan_vectors``）。
        """
        health = self._retriever._check_vector_health()
        vec_count = health.get("vector_count", -1)
        if vec_count < 0:
            logger.warning("OmniMem: 向量通道不可用（breaker=%s），跳过回填",
                           health.get("breaker_state"))
            return

        pending_before = self._retriever.count_vector_pending()
        if pending_before:
            drained = self._retriever.drain_vector_pending()
            logger.warning(
                "OmniMem: 待回填向量队列 %d 条 → 已重放 %d 条（延后 %d，失败 %d）",
                pending_before, drained.get("replayed", 0),
                drained.get("deferred", 0), drained.get("failed", 0),
            )

        # ★ P1-8：这里同样是全量口径，limit=5000 会把老记忆挤出重建集
        indexed_entries = self._index.search_l1(limit=_FULL_SCAN_LIMIT) if self._index else []
        if not indexed_entries:
            self._sweep_orphan_vectors()
            return
        active_ids = {
            str(entry.get("memory_id") or entry.get("id") or "") for entry in indexed_entries
        } - {""}
        if not active_ids:
            self._sweep_orphan_vectors()
            return

        # 容忍度沿用 governance/auditor.quick_health_check 的口径（5%，下限 5 条）
        tolerance = max(len(active_ids) // 20, 5)
        covered = self._vector_covered_ids()

        if covered is None:
            vec_count = self._retriever.vector_count()
            logger.warning(
                "OmniMem: 向量后端不支持枚举 id，覆盖率退回标量口径"
                "（向量行=%d，活跃记忆=%d）——chunk 会让行数虚高，此判据只能发现灾难性缺失",
                vec_count, len(active_ids),
            )
            if vec_count >= len(active_ids) - tolerance:
                logger.warning(
                    "OmniMem: 向量覆盖核对**未验证**（标量口径放行：向量行=%d，活跃记忆=%d，容忍=%d）",
                    vec_count, len(active_ids), tolerance,
                )
                self._sweep_orphan_vectors()
                return
        else:
            uncovered = active_ids - covered
            if uncovered:
                logger.warning(
                    "OmniMem: 向量覆盖缺口 %d/%d 条活跃记忆（容忍 %d；向量行数 %d 仅供参考）",
                    len(uncovered), len(active_ids), tolerance, vec_count,
                )
            else:
                # ★ P1-11：判据通过也要留痕 —— 原实现静默 return，事后无法区分
                #   "核对过、确实全覆盖" 和 "核对根本没跑"（chunk 掩盖下的放行就是这样丢线索的）
                logger.info(
                    "OmniMem: 向量覆盖核对通过（活跃=%d 条全部已向量化，向量行=%d，容忍=%d）",
                    len(active_ids), vec_count, tolerance,
                )
            if len(uncovered) <= tolerance:
                self._sweep_orphan_vectors()
                return

        logger.warning(
            "OmniMem: 向量覆盖不足（活跃=%d 条，缺口超过容忍 %d，向量行=%d）"
            "且死信队列已空/不足，触发全量重建",
            len(active_ids), tolerance, vec_count,
        )
        result = self._retriever.rebuild_all_from_entries(indexed_entries)
        logger.info("OmniMem: vector rebuild complete: %s", result)
        from omnimem.retrieval.vector_store import _emit
        _emit("[OmniMem] 向量索引已自动重建")
        self._sweep_orphan_vectors()

    def _sweep_orphan_vectors(self) -> None:
        """删除「向量库里有，但记忆确实已经不存在」的向量行（含其 chunk 行）。

        ★ P1-11：重建走 upsert，永不回收这类残留，缺口只会在向量侧单向堆积。

        「不存在」的判据是 **抽屉 ∪ index 全表**，不是 index 单独一方 —— 副本实测
        （2026-09-30）向量库里 584 个「index 查不到」的记忆 id 中，**440 个磁盘抽屉还在**
        （正是 P1-1 那类索引缺行，以及已归档但按设计保留的行）。只按 index 判孤儿就会
        把这 440 条还能召回的记忆从向量通道抹掉，等于亲手造一个新 P1-1。抽屉扫不到
        （返回 None）时一律不动，与 P1-7/P1-8/P1-9 的 ``disk_memory_ids`` 约定一致。
        护栏：一次清扫量超过向量库的 10% 说明基线可疑（接错目录、id 口径不匹配都会
        表现为"全是孤儿"），拒绝执行并告警。
        """
        from omnimem.governance.reconciler import disk_memory_ids

        if not all(
            callable(getattr(self._retriever, name, None))
            for name in ("vector_orphan_ids", "vector_stored_ids", "delete_vectors_by_ids")
        ):
            logger.debug("OmniMem: 检索器缺少向量 id 级接口，跳过孤儿向量清扫")
            return

        palace_dir = None
        try:
            palace_dir = getattr(self._store.meta_store, "palace_dir", None)
        except Exception:
            logger.debug("OmniMem: 取不到 palace 目录，孤儿向量清扫无抽屉基线", exc_info=True)
        drawers = disk_memory_ids(palace_dir)
        if drawers is None:
            logger.warning('OmniMem: 无磁盘抽屉基线，拒绝清扫孤儿向量（镜像不能当"这条记忆不存在"）')
            return

        baseline: set[str] = set(drawers)
        scan = getattr(self._index, "search_all_for_retrieval", None)
        index_only_count = 0
        if callable(scan):
            try:
                rows = scan(limit=_FULL_SCAN_LIMIT)
            except Exception:
                logger.warning("OmniMem: 索引全表扫描失败，孤儿向量清扫只用抽屉作基线", exc_info=True)
                rows = []
            index_ids = {str(row.get("memory_id") or row.get("id") or "") for row in rows} - {""}
            index_only_count = len(index_ids - drawers)
            baseline |= index_ids
        else:
            logger.warning("OmniMem: 索引不支持全表 id 扫描，孤儿向量清扫只用抽屉作基线")
        baseline.discard("")
        if not baseline:
            return

        try:
            orphans = self._retriever.vector_orphan_ids(baseline)
            stored = self._retriever.vector_stored_ids()
        except Exception:
            logger.warning("OmniMem: 孤儿向量统计失败", exc_info=True)
            return
        if not orphans:
            return
        stored_total = len(stored) if stored is not None else len(orphans)
        ceiling = max(stored_total // 10, 5)
        if len(orphans) > ceiling:
            logger.warning(
                "OmniMem: 检出 %d 个孤儿向量 > 单次清扫上限 %d（向量库共 %d 行），"
                "基线可能不可信，拒绝清扫",
                len(orphans), ceiling, stored_total,
            )
            return
        deleted = self._retriever.delete_vectors_by_ids(orphans)
        logger.warning(
            "OmniMem: 已清扫 %d 个孤儿向量（抽屉与 index 都查不到该记忆；"
            "抽屉=%d，仅 index 有=%d，基线=%d）",
            deleted, len(drawers), index_only_count, len(baseline),
        )

    def _startup_audit(self) -> None:
        """启动时运行审计+修复，并**始终**上报待回填队列长度。"""
        try:
            pending = 0
            try:
                pending = self._retriever.count_vector_pending()
            except Exception:
                logger.debug("OmniMem: 待回填向量队列统计失败", exc_info=True)
            if pending:
                logger.warning(
                    "OmniMem startup: %d 条记忆仍待回填向量索引（vector_index_pending.jsonl）",
                    pending,
                )
            health = self._auditor.quick_health_check()
            if not health.get("healthy", True):
                audit = self._auditor.run_full_audit(limit=2000)
                if audit.get("total_issues", 0) > 0:
                    fixed = self._auditor.repair(audit)
                    logger.info(
                        "OmniMem startup audit: %d inconsistencies found, %d repaired",
                        audit["total_issues"], fixed,
                    )
        except Exception as e:
            logger.warning("OmniMem startup audit skipped (non-fatal): %s", e)
