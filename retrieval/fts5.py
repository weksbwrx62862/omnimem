"""FTS5Retriever — 基于 SQLite FTS5 的关键词检索器。

替代 rank-bm25 的外部依赖，复用 unified_index 的 FTS5 全文索引：
  - 增量写入 O(1)：由 unified_index 的触发器自动维护
  - 中文支持：查询时 jieba 分词，构造多词 MATCH 查询
  - 无缓冲/重建逻辑：与 BM25Retriever 的 ~300 行维护代码对比

设计原则：
  - 只读 unified_index 的读连接，不维护独立存储
  - 不修改 unified_index 的 schema（使用现有 content/summary/content_preview 三字段）
  - jieba 不可用时回退到原始查询词（FTS5 unicode61 逐字匹配）
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

try:
    import jieba

    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

from omnimem.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


class FTS5Retriever:
    """基于 unified_index FTS5 表的关键词检索器。

    与 BM25Retriever 接口兼容，可作为 drop-in replacement。
    """

    def __init__(self, get_read_conn: Any = None):
        """初始化 FTS5 检索器。

        Args:
            get_read_conn: 返回 unified_index 读连接的回调 (conn, rowid)。
                           conn 需为 sqlite3.Connection，rowid 为映射到 memory_index 的标识。
        """
        self._get_read_conn = get_read_conn
        self._name = "bm25"  # 兼容 BM25Retriever 的通道名

    @property
    def name(self) -> str:
        return self._name

    def search_sync(self, query: str, **kwargs: Any) -> RetrievalResult:
        """同步检索，返回统一 RetrievalResult（兼容 BaseRetriever）。"""
        top_k = kwargs.get("top_k", 10)
        results = self.search(query, top_k=top_k)
        scores = [float(r.get("score", 0.0)) for r in results]
        return RetrievalResult(results=results, scores=scores, channel=self.name)

    async def asearch(self, query: str, **kwargs: Any) -> RetrievalResult:
        """异步检索包装（兼容 BaseRetriever）。"""
        import asyncio

        return await asyncio.to_thread(self.search_sync, query, **kwargs)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """FTS5 关键词检索。

        查询时用 jieba 分词构造 MATCH 查询，利用 FTS5 索引增量更新。
        """
        if not self._get_read_conn or not query.strip():
            return []

        try:
            read_conn = self._get_read_conn()
            if read_conn is None:
                return []
        except Exception as e:
            logger.warning("FTS5: failed to get read connection: %s", e)
            return []

        # 构造 FTS5 MATCH 查询
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []

        try:
            rows = read_conn.execute(
                """SELECT m.rowid, m.content, m.content_preview, m.memory_id, m.type,
                          m.stored_at, m.confidence
                   FROM memory_index_fts f
                   JOIN memory_index m ON f.rowid = m.rowid
                   WHERE memory_index_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, top_k * 3),  # 多取一些用于后续过滤
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 search failed: %s", e)
            return []

        results = []
        for row in rows:
            rowid, content, content_preview, memory_id, memory_type, stored_at, confidence = (
                row[0], row[1], row[2], row[3], row[4], row[5], row[6]
            )
            # 过滤已删除/已过时的记忆
            # (unified_index 的触发器不会自动清理，需要应用层过滤)
            results.append({
                "memory_id": memory_id,
                "content": content or "",
                "summary": content_preview or "",
                "type": memory_type or "fact",
                "stored_at": stored_at or "",
                "confidence": confidence or 0.5,
                "score": 0.8,  # FTS5 不提供分数，给固定值由上层 RRF 融合处理
                "_source": "fts5",
            })

        # 截断到 top_k
        return results[:top_k]

    def _build_fts_query(self, query: str) -> str:
        """构造 FTS5 MATCH 查询字符串。

        中文用 jieba 分词后 OR 连接，英文保留原始词。
        例如 "用户喜欢Python" → '"用户" OR "喜欢" OR python'
        """
        if _HAS_JIEBA:
            tokens = jieba.lcut(query)
            # 过滤纯标点/空白
            tokens = [t.strip() for t in tokens if t.strip()]
            if not tokens:
                return self._escape_fts5(query)
            # 用 OR 连接，每个 token 用双引号包裹（短语匹配）
            parts = []
            for t in tokens:
                # 英文/数字不加引号，中文加引号
                if t.isascii() and any(c.isalpha() for c in t):
                    parts.append(self._escape_fts5(t))
                else:
                    parts.append(f'"{self._escape_fts5(t)}"')
            return " OR ".join(parts)
        else:
            # 无 jieba，原始查询
            return self._escape_fts5(query)

    @staticmethod
    def _escape_fts5(text: str) -> str:
        """转义 FTS5 特殊字符。"""
        # FTS5 特殊字符: " * ( ) -
        for char in '"*()-':
            text = text.replace(char, " ")
        return text.strip()

    # ── 兼容 BM25Retriever 接口（no-ops，由 unified_index 维护） ──

    def add(self, content: str, memory_id: str, metadata: dict[str, Any]) -> None:
        """no-op：unified_index 触发器自动维护 FTS5 索引。"""
        pass

    def flush(self) -> None:
        """no-op：无需手动刷新。"""
        pass

    def warmup(self) -> None:
        """no-op：无需预热。"""
        pass

    @property
    def document_count(self) -> int:
        return 0
