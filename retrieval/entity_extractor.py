"""EntityExtractor - lightweight entity extraction for retrieval enhancement.

Inspired by mem0 three-signal fusion design, uses jieba + regex instead of spaCy.
Extracts 4 entity types for retrieval boost:
  1. PROPER - proper nouns (capitalized English, 2-4 char Chinese nouns)
  2. QUOTED - quoted text
  3. COMPOUND - compound nouns from jieba segmentation
  4. NOUN - single noun fallback

Entity boost: when query entities match memory entities, score += ENTITY_BOOST_WEIGHT.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

try:
    import jieba.posseg as pseg
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

ENTITY_BOOST_WEIGHT = 0.3

_GENERIC_NOUNS = {
    "problem", "method", "use", "implement", "include", "through",
    "situation", "content", "function", "system", "data", "operation",
    "time", "place", "thing", "part", "type", "kind", "example",
    "issue", "result", "status",
    "\u95ee\u9898", "\u65b9\u6cd5", "\u4f7f\u7528", "\u8fdb\u884c",
    "\u5b9e\u73b0", "\u76f8\u5173", "\u5305\u62ec", "\u5173\u4e8e",
    "\u901a\u8fc7", "\u60c5\u51b5", "\u65b9\u9762", "\u5185\u5bb9",
    "\u529f\u80fd", "\u7cfb\u7edf", "\u6570\u636e", "\u4fe1\u606f",
    "\u64cd\u4f5c", "\u65b9\u5f0f", "\u65f6\u95f4", "\u5730\u65b9",
    "\u4e1c\u897f", "\u4e8b\u60c5", "\u90e8\u5206", "\u7c7b\u578b",
    "\u7ed3\u679c", "\u72b6\u6001",
}

_VALUABLE_POS = {"nr", "ns", "nt", "nz", "eng", "vn", "n", "v"}

_PUNCT_NUM = re.compile(r"^[\d\s\W]+$")


class EntityExtractor:
    """Lightweight entity extractor using jieba + regex."""

    def extract(self, text: str, max_entities: int = 8) -> list[str]:
        """Extract entities from text.

        Args:
            text: input text
            max_entities: max entities to return

        Returns:
            Deduplicated entity list (order-preserving)
        """
        if not text or not text.strip():
            return []

        entities: list[str] = []
        seen: set[str] = set()

        # 1. Quoted text (highest priority)
        for ent in self._extract_quoted(text):
            ent = self._normalize(ent)
            if ent and ent not in seen and len(ent) >= 2:
                seen.add(ent)
                entities.append(ent)

        # 2. Proper nouns
        for ent in self._extract_proper(text):
            ent = self._normalize(ent)
            if ent and ent not in seen and not self._is_generic(ent):
                seen.add(ent)
                entities.append(ent)

        # 3. Jieba POS tagging (nouns, verbs, English words)
        if _HAS_JIEBA:
            for ent in self._extract_jieba_entities(text):
                ent = self._normalize(ent)
                if ent and ent not in seen and not self._is_generic(ent):
                    seen.add(ent)
                    entities.append(ent)

        return entities[:max_entities]

    def extract_from_metadata(self, metadata: dict[str, Any]) -> list[str]:
        """Extract stored entities from memory metadata."""
        return metadata.get("entities", [])

    def _extract_quoted(self, text: str) -> list[str]:
        """Extract quoted text."""
        patterns = [
            re.compile(r"\u300c([^\u300d]+)\u300d"),
            re.compile(r"\u300e([^\u300f]+)\u300f"),
            re.compile(r"\u201c([^\u201d]+)\u201d"),
            re.compile(r"'([^']+)'"),
        ]
        results = []
        for p in patterns:
            results.extend(p.findall(text))
        return results

    def _extract_proper(self, text: str) -> list[str]:
        """Extract proper nouns: capitalized English, Chinese named entities."""
        results = []
        # English proper nouns (2+ consecutive capitalized words, skip 1-char)
        en_proper = re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", text)
        results.extend(en_proper[:5])

        # Chinese named entities via regex (2-4 char sequences that are NOT
        # followed by common verb/adjective particles)
        # This avoids extracting "茅台是贵" when "茅台" is the entity
        zh_patterns = [
            # 2-char noun before common particles (是/的/了/在/和/与)
            re.compile(r"([\u4e00-\u9fff]{2,4})(?=[\u662f\u7684\u4e86\u5728\u548c\u4e0e\u6216\u800c\u4f46\u5374]|\s|[,.])"),
            # Standalone 2-4 char Chinese (fallback)
            re.compile(r"([\u4e00-\u9fff]{2,4})"),
        ]
        seen_zh: set[str] = set()
        for p in zh_patterns:
            for m in p.finditer(text):
                ent = m.group(1) if m.lastindex else m.group(0)
                if ent not in seen_zh and len(ent) >= 2:
                    seen_zh.add(ent)
                    results.append(ent)
                    if len(results) >= 10:
                        break
            if len(results) >= 10:
                break

        return results

    def _extract_jieba_entities(self, text: str) -> list[str]:
        """Extract entities using jieba POS tagging."""
        results = []
        try:
            words = pseg.cut(text)
            for word, pos in words:
                word = word.strip()
                if len(word) < 2:
                    continue
                if _PUNCT_NUM.match(word):
                    continue
                if pos in _VALUABLE_POS:
                    results.append(word)
        except Exception:
            pass
        return results

    def _normalize(self, text: str) -> str:
        """Normalize: strip whitespace and punctuation."""
        text = text.strip()
        text = re.sub(r"^[\s\W]+|[\s\W]+$", "", text)
        return text

    def _is_generic(self, word: str) -> bool:
        """Check if word is a generic noise noun."""
        return word.lower() in _GENERIC_NOUNS

    def compute_entity_overlap(
        self,
        query_entities: list[str],
        doc_entities: list[str],
    ) -> float:
        """Compute entity overlap score between query and document.

        Returns:
            Entity boost score (0.0 ~ ENTITY_BOOST_WEIGHT)
        """
        if not query_entities or not doc_entities:
            return 0.0

        q_set = {e.lower() for e in query_entities}
        d_set = {e.lower() for e in doc_entities}

        if not q_set or not d_set:
            return 0.0

        overlap = q_set & d_set
        if not overlap:
            return 0.0

        overlap_ratio = len(overlap) / len(q_set)
        return round(overlap_ratio * ENTITY_BOOST_WEIGHT, 6)
