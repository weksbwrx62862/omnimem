"""索引维护（★ M6-9：从 hybrid_orchestrator.py 拆分）。"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class IndexAdminMixin:
    """索引重建与 sync_turn 条目维护。"""

    def enrich_for_rebuild(self, content: str, mem_type: str, room: str = "") -> str:
        """重建索引时为各类型附加可搜索描述。"""
        type_prefixes = {
            "secret": "[加密信息/密钥/凭证]",
            "skill": "[技能/步骤/教程]",
            "procedural": "[流程/操作/指南]",
            "reasoning": "[教训/经验/踩坑]",
            "action": "[Agent行为/工具调用]",
        }
        prefix = type_prefixes.get(mem_type, "")
        if prefix:
            return f"{prefix} {room} {content}"
        return content

    def rebuild_bm25_from_entries(self, entries: list[dict[str, Any]]) -> int:
        """从索引条目重建 BM25 检索通道（跨会话持久化恢复）。

        使用 BM25Retriever 的增量更新能力，避免全量重建。
        """
        bm25 = self._facade._bm25
        enriched_entries = []
        for entry in entries:
            memory_id = entry.get("memory_id", "")
            content = entry.get("content", "") or entry.get("summary", "")
            if not memory_id or not content:
                continue
            mem_type = entry.get("type", "fact")
            room = entry.get("room", "")
            enriched = self.enrich_for_rebuild(content, mem_type, room)
            enriched_entries.append({**entry, "content": enriched})
        result = bm25.update_from_entries(enriched_entries)
        total = result.get("added", 0) + result.get("updated", 0)
        logger.info(
            "BM25 rebuild (incremental): added=%d, updated=%d, deleted=%d",
            result.get("added", 0),
            result.get("updated", 0),
            result.get("deleted", 0),
        )
        return total

    def rebuild_all_from_entries(self, entries: list[dict[str, Any]]) -> dict[str, int]:
        """全量重建向量+BM25检索索引（解决历史向量退化问题）。

        向量索引使用分批并行 embedding 计算 + 批量写入；
        BM25 使用增量更新，避免全量重建。
        """
        facade = self._facade
        self.clear_all_cache()

        # 读取重建参数配置（默认 batch_size=32, max_workers=4）
        config = getattr(facade, "_config", None)
        batch_size = 32
        max_workers = 4
        if config is not None:
            try:
                batch_size = int(config.get("rebuild_batch_size", 32))
                max_workers = int(config.get("rebuild_max_workers", 4))
            except Exception:
                batch_size, max_workers = 32, 4
        batch_size = max(1, batch_size)
        max_workers = max(1, max_workers)

        # 1. ★ 不再预先清空向量索引：原实现在算 embedding 之前就把 entries 里的每个 id
        #   逐个 delete（facade._vector 并没有 reset()，走的是 else 分支），随后
        #   rebuild_vectors_parallel 一旦失败（模型加载锁超时→批次静默返回短列表→数量校验
        #   判失败）就一条都不写：**一次失败的重建会把可用索引清空**（副本实测
        #   vectors 975→193，检索 0 命中）。写入路径本身就是 upsert（按 id 覆盖），重建
        #   不需要先删——留着旧数据最多是"未更新"，删了再写失败就是"全丢"。
        # 2. 并行重建向量索引
        vec_count = 0
        try:
            vec_count = facade._vector.rebuild_vectors_parallel(
                entries, batch_size=batch_size, max_workers=max_workers
            )
        except Exception as e:
            logger.warning("rebuild_vectors_parallel failed: %s", e)
        if not vec_count:
            logger.error(
                "OmniMem: 向量重建未写入任何条目（entries=%d），已放弃并保持现有索引原样",
                len(entries),
            )

        # 3. BM25 增量更新
        bm25_count = 0
        try:
            bm25_entries = []
            for entry in entries:
                mid = entry.get("memory_id", "")
                content = entry.get("content", "")
                if not mid or not content:
                    continue
                mem_type = entry.get("type", "fact")
                room = entry.get("room", "")
                enriched = self.enrich_for_rebuild(content, mem_type, room)
                bm25_entries.append({**entry, "content": enriched})
            result = facade._bm25.update_from_entries(bm25_entries)
            bm25_count = result.get("added", 0) + result.get("updated", 0)
        except Exception as e:
            logger.warning("BM25 incremental rebuild failed: %s", e)

        facade._vector.flush()
        logger.info(
            "HybridRetriever rebuild: vector=%d, bm25=%d from %d entries",
            vec_count, bm25_count, len(entries),
        )
        return {"vector": vec_count, "bm25": bm25_count}

    def cleanup_sync_turn_entries(self) -> None:
        """清理旧的 sync_turn 条目，防止索引膨胀。"""
        facade = self._facade
        max_entries = facade._max_sync_turn_entries
        while len(facade._sync_turn_ids) > max_entries:
            old_id = facade._sync_turn_ids.popleft()
            try:
                facade._bm25.delete(old_id)
            except Exception as e:
                logger.warning("BM25 delete sync_turn %s failed: %s", old_id, e)
            try:
                facade._vector.delete(old_id)
            except Exception as e:
                logger.warning("Vector delete sync_turn %s failed: %s", old_id, e)
        # ★ 修复 S1（2026-09-28）：队列状态落盘，使 FIFO 淘汰能力跨 gateway 重启保持。
        # 不落盘则重启后 _sync_turn_ids 归零，上面的 while 永不触发、索引只增不减。
        if hasattr(facade, "persist_sync_turn_ids"):
            facade.persist_sync_turn_ids()

    def index_update(self, user_content: str, _assistant_content: str) -> None:
        """后台异步索引更新（从 sync_turn 调用）。"""
        import re
        import uuid

        clean_user = re.sub(
            r"### Relevant Memories(?:\s*\(prefetched\))?\s*\n.*"
            r"(?=\n(?!- )|\Z)",
            "",
            user_content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        clean_user = re.sub(r"^- \[cached\].*$", "", clean_user, flags=re.MULTILINE).strip()

        content = clean_user[:200].strip() if clean_user else ""
        if not content or len(content) < 5:
            return

        idx_id = f"sync-{uuid.uuid4().hex[:8]}"
        facade = self._facade
        self.clear_all_cache()
        facade._vector.add(content, memory_id=idx_id, metadata={"source": "sync_turn"})
        facade._bm25.add(content, memory_id=idx_id, metadata={"source": "sync_turn"})
        facade._sync_turn_ids.append(idx_id)
        self.cleanup_sync_turn_entries()
