"""HMS Entity-bridge 算子 — 规范实体消歧（Canonical Entity Resolution）。

对应 HMS 论文的 Entity-bridge (E) 算子：不同记忆单元共享同一个人/物，但
表面文本完全不同（"小王"/"王总"/"王明"），导致传统检索无法桥接。

实现：
  1. 规范实体表（canonical map）：别名 → 规范 ID，如 "王总"/"小王" → "王明"
  2. 归一化规则：称谓剥离（总/经理/老师/同学）、全半角、大小写
  3. 实体归一化入口 normalize(entity)：统一走规范映射 + 规则
  4. 用于索引期（写入时实体消歧）与检索期（查询实体归一化）双端

设计约束：
  - 纯规则 + 可持久化的映射表（JSON），零 LLM 调用
  - 规范表可热更新：register_alias() / load() / save()
  - 未命中的实体走规则归一化（称谓剥离等），保证幂等
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# 称谓后缀（剥离后归一到主干）
_TITLE_SUFFIXES = (
    "总经理", "总工程师", "董事长", "副经理", "副总经理",
    "经理", "老师", "博士", "医生", "先生", "女士", "主任",
    "部长", "所长", "院长", "主席", "工程师", "同学", "教授",
    "律师", "会计", "设计师", "老板", "客户", "同事",
)

# 别名归一化正则：剥离称谓后缀
_RE_TITLE = re.compile(r"(?:{})".format("|".join(_TITLE_SUFFIXES)))


class CanonicalEntityResolver:
    """规范实体消歧器：别名 → 规范实体 ID。"""

    def __init__(self, data_file: str | None = None) -> None:
        self._alias_to_canonical: dict[str, str] = {}
        self._canonical_to_aliases: dict[str, list[str]] = {}
        self._data_file = data_file
        if data_file and os.path.isfile(data_file):
            self.load(data_file)

    # ── 持久化 ──

    def load(self, path: str) -> None:
        """从 JSON 加载规范表。"""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._alias_to_canonical = {}
            self._canonical_to_aliases = {}
            for canonical, aliases in data.items():
                self._canonical_to_aliases[canonical] = list(aliases)
                for alias in aliases:
                    self._alias_to_canonical[alias] = canonical
            logger.info("CanonicalEntityResolver 加载 %d 个规范实体", len(self._canonical_to_aliases))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("规范表加载失败（非致命）: %s", e)

    def save(self, path: str | None = None) -> None:
        """保存规范表到 JSON。"""
        target = path or self._data_file
        if not target:
            return
        try:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                json.dump(self._canonical_to_aliases, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning("规范表保存失败（非致命）: %s", e)

    # ── 注册 ──

    def register_alias(self, canonical: str, alias: str) -> None:
        """注册别名 → 规范实体映射。"""
        alias = self._normalize(alias)
        canonical = self._normalize(canonical)
        if not alias or not canonical:
            return
        self._alias_to_canonical[alias] = canonical
        if canonical not in self._canonical_to_aliases:
            self._canonical_to_aliases[canonical] = []
        if alias not in self._canonical_to_aliases[canonical]:
            self._canonical_to_aliases[canonical].append(alias)

    def register_group(self, canonical: str, aliases: list[str]) -> None:
        """注册一组别名到同一规范实体。"""
        for alias in aliases:
            self.register_alias(canonical, alias)

    # ── 解析 ──

    def resolve(self, entity: str) -> str:
        """返回实体的规范 ID（未命中则走规则归一化后原样返回）。"""
        if not entity:
            return entity
        norm = self._normalize(entity)
        # 1. 精确别名映射
        if norm in self._alias_to_canonical:
            return self._alias_to_canonical[norm]
        # 2. 去称谓后再查（"王经理" → "王" 再查 "王明"？不，"王"过短不查）
        stripped = _RE_TITLE.sub("", norm)
        if stripped and stripped != norm and stripped in self._alias_to_canonical:
            return self._alias_to_canonical[stripped]
        # 3. 规则归一化后返回
        return stripped or norm

    def get_aliases(self, canonical: str) -> list[str]:
        """返回规范实体的全部别名。"""
        return self._canonical_to_aliases.get(self._normalize(canonical), [])

    def canonical_count(self) -> int:
        return len(self._canonical_to_aliases)

    # ── 批量 ──

    def normalize_entities(self, entities: list[str]) -> list[str]:
        """批量归一化实体列表（去重，保持顺序）。"""
        seen: set[str] = set()
        out: list[str] = []
        for e in entities:
            c = self.resolve(e)
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    # ── 内部 ──

    @staticmethod
    def _normalize(text: str) -> str:
        """规则归一化：全半角/大小写/空白。"""
        if not text:
            return ""
        t = text.strip()
        # 全角 → 半角
        t = "".join(
            chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c
            for c in t
        )
        t = t.replace("\u3000", " ")
        t = t.replace("（", "(").replace("）", ")")
        # 连续空白压缩
        t = re.sub(r"\s+", "", t)
        return t

    def __repr__(self) -> str:
        return f"CanonicalEntityResolver(entries={self.canonical_count()})"


# 默认实例（进程内共享，可被插件替换）
_default_resolver: CanonicalEntityResolver | None = None


def get_default_resolver(data_file: str | None = None) -> CanonicalEntityResolver:
    """获取默认规范实体消歧器（单例）。"""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = CanonicalEntityResolver(data_file)
    return _default_resolver


def normalize_entity(entity: str) -> str:
    """便捷入口：实体归一化。"""
    return get_default_resolver().resolve(entity)
