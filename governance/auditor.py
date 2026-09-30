"""GovernanceAuditor — 后台治理巡检器。

P0方案六：长期运行的一致性保障机制。
定期检查并修复以下不一致：
  1. 幽灵索引：index 中有、MetaStore 无、**且磁盘抽屉也没了**的条目
  2. 漏索引：MetaStore 中有但 index/retriever 中缺失的条目
  3. 归档残留：已归档记忆在检索索引中的残留

设计原则（★ P1-9 更正）：
  - ~~以 MetaStore 为唯一事实来源~~ —— **错**。MetaStore 是并行双写的一方，降级/写失败
    时它自己缺行（副本实测 877 行 vs 磁盘 3 135 个抽屉）。以它为准的"幽灵清理"会删掉
    合法索引行（§5.9/§5.11）。删除类判据一律以磁盘抽屉为准，无法核验磁盘时不删。
  - "漏索引"方向仍可参考 MetaStore（补写无害），但补写前会查 store 里条目是否还存在。
  - 只读审计优先，修复操作需显式调用 repair() 或 _repair_from_metastore()
  - 利用现有 store/index/retriever 接口，不引入新存储
  - 轻量级实现，避免全量扫描导致的长耗时阻塞
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 健康检查统计 index 行数用：不能是普通分页尺度，否则大库上会误报覆盖不足
_HEALTH_SCAN_LIMIT = 10_000_000


class GovernanceAuditor:
    """治理巡检器：检测并修复 Store/Index/Retriever 之间的不一致。"""

    def __init__(
        self,
        store: Any,
        index: Any,
        retriever: Any,
        forgetting: Any,
    ):
        """初始化巡检器。

        Args:
            store: DrawerClosetStore 实例
            index: ThreeLevelIndex 实例
            retriever: HybridRetriever 实例
            forgetting: ForgettingCurve 实例
        """
        self._store = store
        self._index = index
        self._retriever = retriever
        self._forgetting = forgetting

    def _disk_memory_ids(self) -> set[str] | None:
        """磁盘抽屉 ID；拿不到可靠依据时返回 None —— 此时**一律不产生删除建议**。"""
        from omnimem.governance.reconciler import disk_memory_ids

        palace_dir = getattr(self._store.meta_store, "palace_dir", None)
        return disk_memory_ids(palace_dir)

    def run_full_audit(self, limit: int = 2000) -> dict[str, Any]:
        """执行全量一致性审计。

        ★ P1-9：ID 基准是 MetaStore，但**幽灵判定**必须再用磁盘抽屉否证一次：
        MetaStore 缺行不等于记忆不存在（并行双写会缺），只看 MetaStore 就会把
        合法的 index 行报成幽灵、随后被 repair() 删掉。

        Args:
            limit: 最多审计的条目数（防止大数据量时阻塞）

        Returns:
            审计结果字典，包含各类不一致条目列表
        """
        ghost_in_index: list[str] = []
        missing_in_index: list[str] = []
        ghost_in_retriever: list[str] = []
        missing_in_retriever: list[str] = []

        # ★ 1. 以 MetaStore 为基准获取有效记忆 ID
        meta_entries = self._store.meta_store.get_all(limit=limit)
        meta_ids: set[str] = {
            e.get("memory_id", "") for e in meta_entries if e.get("memory_id", "")
        }

        # 2. 获取 index 中的条目
        index_entries = self._index.search_all_for_retrieval(limit=limit)
        index_ids: set[str] = {
            e.get("memory_id", "") for e in index_entries if e.get("memory_id", "")
        }

        # ★ P1-9：删除判据是磁盘抽屉，不是 MetaStore（详见模块 docstring 的更正）
        disk_ids = self._disk_memory_ids()
        if disk_ids is None:
            logger.warning(
                "Audit: 无法核验磁盘抽屉，跳过幽灵索引检测（不做任何删除建议）"
            )
        else:
            # 幽灵索引：index 有、MetaStore 无、磁盘也没抽屉 —— 三者都缺才是幽灵
            for mid in index_ids:
                if mid not in meta_ids and mid not in disk_ids:
                    ghost_in_index.append(mid)

        # 检测漏索引：MetaStore 有但 index 无
        for mid in meta_ids:
            if mid not in index_ids:
                missing_in_index.append(mid)

        # 3. 检测已归档但在 index 中残留的条目
        try:
            archived = self._forgetting.get_archived_ids(limit=limit)
            for mid in archived:
                if mid in index_ids:
                    ghost_in_index.append(mid)
                if self._retriever.bm25_document_count > 0:
                    entry = self._store.get(mid)
                    if entry is None and mid in index_ids:
                        ghost_in_retriever.append(mid)
        except Exception as e:
            logger.warning("Audit archived check failed: %s", e)

        # 4. ChromaDB 条目数一致性检查（以 MetaStore 为基准）
        chroma_count: int = 0
        chroma_degraded: bool = False
        chroma_uncovered: int | None = None
        try:
            chroma_count = self._retriever.vector_count()
            meta_count = len(meta_ids)
            threshold = max(meta_count // 20, 5)
            # ★ P1-11：``vector_count()`` 是**向量行**数，长记忆会被切成多行，行数 ≥ 记忆数
            #   是正常状态。原先用 ``abs(meta - chroma) > threshold`` 双向比，100% 覆盖也会
            #   误报"偏差超阈值"（线上实测 3 217 行 / 2 742 条记忆），反过来 chunk 恰好吃掉
            #   缺口时又会漏报。能枚举 id 时按记忆集合比，否则只在"行数明显少于记忆数"时报。
            covered_fn = getattr(self._retriever, "vector_covered_memory_ids", None)
            covered = covered_fn() if callable(covered_fn) else None
            if covered is not None:
                chroma_uncovered = len(meta_ids - covered)
                if chroma_uncovered > threshold:
                    chroma_degraded = True
                    logger.warning(
                        "Audit: 向量覆盖缺口超阈值 — meta=%d, 已向量化=%d, 缺口=%d, 阈值=%d"
                        "（向量行数=%d 仅供参考）",
                        meta_count,
                        len(covered),
                        chroma_uncovered,
                        threshold,
                        chroma_count,
                    )
            elif chroma_count < meta_count - threshold:
                chroma_degraded = True
                logger.warning(
                    "Audit: ChromaDB 行数低于记忆数超阈值 — meta=%d, chroma=%d, 阈值=%d"
                    "（后端不支持枚举 id，只能判灾难性缺失）",
                    meta_count,
                    chroma_count,
                    threshold,
                )
        except Exception as e:
            logger.warning("Audit ChromaDB count check failed: %s", e)

        total_issues = (
            len(ghost_in_index)
            + len(missing_in_index)
            + len(ghost_in_retriever)
            + len(missing_in_retriever)
        )

        return {
            "ghost_in_index": list(set(ghost_in_index)),
            "missing_in_index": list(set(missing_in_index)),
            "ghost_in_retriever": list(set(ghost_in_retriever)),
            "missing_in_retriever": list(set(missing_in_retriever)),
            "total_issues": total_issues,
            "scanned_meta": len(meta_ids),
            "scanned_index": len(index_ids),
            "chroma_count": chroma_count,
            "chroma_degraded": chroma_degraded,
            "chroma_uncovered": chroma_uncovered,
        }

    def repair(self, audit_result: dict[str, Any]) -> int:
        """根据审计结果自动修复不一致。

        修复策略：
          - 幽灵索引：从 index 中删除
          - 漏索引：从 store 读取后重新写入 index 和 retriever
          - 归档残留：从 index/retriever 中删除

        Args:
            audit_result: run_full_audit() 的返回值

        Returns:
            成功修复的条目数
        """
        fixed = 0

        # 修复幽灵索引
        for mid in audit_result.get("ghost_in_index", []):
            try:
                if self._index.delete(mid):
                    logger.warning("Auditor: removed ghost index %s", mid)
                    fixed += 1
            except Exception as e:
                logger.warning("Auditor: failed to remove ghost index %s: %s", mid, e)

        # 修复漏索引
        for mid in audit_result.get("missing_in_index", []):
            try:
                entry = self._store.get(mid)
                if not entry:
                    continue
                self._index.add(
                    memory_id=mid,
                    wing=entry.get("wing", ""),
                    hall=entry.get("hall", entry.get("type", "fact")),
                    room=entry.get("room", ""),
                    content=entry.get("content", ""),
                    summary=entry.get("summary", ""),
                    type=entry.get("type", "fact"),
                    confidence=entry.get("confidence", 3),
                    privacy=entry.get("privacy", "personal"),
                    scope=entry.get("privacy", "personal"),
                    stored_at=entry.get("stored_at", ""),
                    provenance="",
                )
                self._retriever.add(
                    entry.get("content", ""),
                    memory_id=mid,
                    metadata={
                        "memory_id": mid,
                        "type": entry.get("type", "fact"),
                        "confidence": entry.get("confidence", 3),
                        "scope": entry.get("privacy", "personal"),
                        "privacy": entry.get("privacy", "personal"),
                        "wing": entry.get("wing", ""),
                        "room": entry.get("room", ""),
                        "stored_at": entry.get("stored_at", ""),
                    },
                )
                logger.warning("Auditor: re-indexed missing entry %s", mid)
                fixed += 1
            except Exception as e:
                logger.warning("Auditor: failed to re-index %s: %s", mid, e)

        return fixed

    def quick_health_check(self) -> dict[str, Any]:
        """快速健康检查：以**磁盘抽屉**为基准核对 index 覆盖率。

        ★ P1-9：原先的判据是 ``abs(meta_count - index_count) <= meta_count // 20``。
        MetaStore 只是并行双写的一方，它自己缺行就会让这套数据**永远**判为不健康
        （副本实测 meta=877 / index=3 037 → 阈值 43），于是每次开机都要跑一遍全量
        audit + repair —— 而 repair 又拿 MetaStore 当删除依据（§5.11）。基准换成抽屉后，
        这个门控才重新有意义：它报的是"磁盘有、索引没有"，也就是真的召不回的那批。

        Returns:
            健康状态摘要
        """
        meta_count = self._store.meta_store.count()
        index_count = len(self._index.search_all_for_retrieval(limit=_HEALTH_SCAN_LIMIT))
        retriever_count = self._retriever.bm25_document_count

        result: dict[str, Any] = {
            "meta_count": meta_count,
            "index_count": index_count,
            "retriever_bm25_count": retriever_count,
        }
        disk_ids = self._disk_memory_ids()
        if disk_ids is None:
            # 没有可靠基准就不报警、也不让上层去跑那轮"以镜像为准"的删除
            logger.warning("Audit: 无磁盘抽屉基准（未绑定 palace / 扫描为空），健康检查跳过判活")
            result["baseline"] = "none"
            result["healthy"] = True
            return result
        disk_total = len(disk_ids)
        result["baseline"] = "disk"
        result["disk_total"] = disk_total
        result["healthy"] = abs(disk_total - index_count) <= max(disk_total // 20, 5)
        return result

    def _repair_from_metastore(self, limit: int = 2000) -> int:
        """补齐 Index/Retriever 中缺失的条目，并清理真幽灵。

        ★ P1-9：标题里的"以 MetaStore 为唯一事实来源"已不成立 —— MetaStore 只用于
        **补写方向**（它有的、index 缺的，补上无害）。**删除方向**一律以磁盘抽屉为准，
        无法核验磁盘时一条都不删。

        Args:
            limit: 最多处理的条目数

        Returns:
            成功修复的条目数
        """
        fixed = 0

        # ★ 以 MetaStore 为基准获取所有记忆 ID
        meta_entries = self._store.meta_store.get_all(limit=limit)
        meta_ids: set[str] = {
            e.get("memory_id", "") for e in meta_entries if e.get("memory_id", "")
        }

        # 获取 Index 中已有的 ID
        index_entries = self._index.search_all_for_retrieval(limit=limit)
        index_ids: set[str] = {
            e.get("memory_id", "") for e in index_entries if e.get("memory_id", "")
        }

        # 遍历 MetaStore，检查 Index 和 Retriever 中的缺失
        for mid in meta_ids:
            entry = self._store.get(mid)
            if not entry:
                continue

            # 检查 Index 缺失
            if mid not in index_ids:
                try:
                    self._index.add(
                        memory_id=mid,
                        wing=entry.get("wing", ""),
                        hall=entry.get("hall", entry.get("type", "fact")),
                        room=entry.get("room", ""),
                        content=entry.get("content", ""),
                        summary=entry.get("summary", ""),
                        type=entry.get("type", "fact"),
                        confidence=entry.get("confidence", 3),
                        privacy=entry.get("privacy", "personal"),
                        scope=entry.get("privacy", "personal"),
                        stored_at=entry.get("stored_at", ""),
                        provenance="",
                    )
                    logger.info("Auditor._repair_from_metastore: 补充 Index 条目 %s", mid)
                    fixed += 1
                except Exception as e:
                    logger.warning("Auditor._repair_from_metastore: 补充 Index 失败 %s: %s", mid, e)

            # 检查 Retriever 缺失（通过 add 补充）
            try:
                self._retriever.add(
                    entry.get("content", ""),
                    memory_id=mid,
                    metadata={
                        "memory_id": mid,
                        "type": entry.get("type", "fact"),
                        "confidence": entry.get("confidence", 3),
                        "scope": entry.get("privacy", "personal"),
                        "privacy": entry.get("privacy", "personal"),
                        "wing": entry.get("wing", ""),
                        "room": entry.get("room", ""),
                        "stored_at": entry.get("stored_at", ""),
                    },
                )
            except Exception as e:
                logger.warning("Auditor._repair_from_metastore: 补充 Retriever 失败 %s: %s", mid, e)

        # 清理 Index 中真正消失的幽灵条目（★ P1-9：MetaStore 缺行 ≠ 记忆不存在）
        disk_ids = self._disk_memory_ids()
        if disk_ids is None:
            logger.warning("Auditor._repair_from_metastore: 无法核验磁盘，跳过幽灵清理")
            ghost_ids: set[str] = set()
        else:
            ghost_ids = (index_ids - meta_ids) - disk_ids
        for mid in ghost_ids:
            try:
                if self._index.delete(mid):
                    logger.info("Auditor._repair_from_metastore: 清理幽灵 Index 条目 %s", mid)
                    fixed += 1
            except Exception as e:
                logger.warning("Auditor._repair_from_metastore: 清理幽灵 Index 失败 %s: %s", mid, e)

        if fixed > 0:
            logger.info("Auditor._repair_from_metastore: 共修复 %d 条不一致", fixed)
        return fixed
