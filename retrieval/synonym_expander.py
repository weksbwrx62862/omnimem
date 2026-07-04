"""同义词扩展器。

为 BM25 关键词检索提供同义词扩展能力，弥补词袋模型的语义鸿沟。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from omnimem.retrieval.bm25 import BM25Retriever

logger = logging.getLogger(__name__)


class SynonymExpander:
    """同义词扩展器：加载同义词映射并扩展 BM25 查询。"""

    def __init__(self, synonym_map: dict[str, list[str]] | None = None) -> None:
        self._synonym_map = synonym_map or self.load_synonyms()

    @staticmethod
    def load_synonyms() -> dict[str, list[str]]:
        """从配置文件加载同义词映射。"""
        try:
            synonyms_path = Path(__file__).parent.parent / "config" / "synonyms.json"
            if synonyms_path.exists():
                with open(synonyms_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    result: dict[str, list[str]] = {}
                    for k, v in data.items():
                        if isinstance(v, list):
                            result[k] = v
                        elif isinstance(v, str):
                            result[k] = [v]
                    if result:
                        logger.info("Loaded %d synonym entries from %s", len(result), synonyms_path)
                        return result
            logger.warning(
                "synonyms.json not found or empty at %s, synonym expansion disabled",
                synonyms_path,
            )
        except Exception as e:
            logger.warning("Failed to load synonyms.json: %s, synonym expansion disabled", e)
        return {}

    def search(self, bm25: BM25Retriever, query: str, top_k: int) -> list[dict[str, Any]]:
        """BM25 检索通道（含同义词扩展）。

        同义词扩展 BM25 查询：弥补词袋模型的语义鸿沟（QUAL-3修复）
        注意：单字会被 _tokenize 丢弃，所以用2+字词
        扩展策略：上位词↔下位词 双向扩展 + 品种级细粒度覆盖
        """
        bm25_results = bm25.search(query, top_k=top_k)
        for key, synonyms in self._synonym_map.items():
            if key in query:
                for syn in synonyms:
                    expanded = query.replace(key, syn)
                    expanded_results = bm25.search(expanded, top_k=top_k)
                    existing_ids = {r.get("memory_id", "") for r in bm25_results}
                    for r in expanded_results:
                        if r.get("memory_id", "") not in existing_ids:
                            bm25_results.append(r)
                            existing_ids.add(r.get("memory_id", ""))
        return bm25_results
