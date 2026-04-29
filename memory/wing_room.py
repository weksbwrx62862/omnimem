"""WingRoomManager — 宫殿导航：Wing(人/项目) > Hall(类型) > Room(话题)。

参考 MemPalace 的 Wing/Room/Hall 三级结构，实现记忆的空间组织。
Wing: 顶层分类（personal/projects/shared）
Hall: 中层分类（facts/events/preferences/skills/corrections）
Room: 底层话题（根据内容自动检测或手动指定）

改进：话题检测复用 KnowledgeGraph 的 extract_entities，提升精度。
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Wing 映射
# ★ R22修复：补充 team/public/private/secret 映射
# privacy="team"/"public" → wing="shared"（共享空间）
# privacy="personal"/"private" → wing="personal"（个人空间）
# privacy="secret" → wing="personal"（私密归入个人空间）
_SCOPE_TO_WING = {
    "personal": "personal",
    "private": "personal",
    "project": "projects",
    "shared": "shared",
    "team": "shared",
    "public": "shared",
    "secret": "personal",
}

# Hall 映射
_TYPE_TO_HALL = {
    "fact": "facts",
    "preference": "preferences",
    "correction": "corrections",
    "skill": "skills",
    "procedural": "procedures",
    "event": "events",
}

# 停用词表（扩展版）
_STOPWORDS = {
    # 中文停用词
    "这是",
    "那个",
    "这个",
    "什么",
    "怎么",
    "如何",
    "可以",
    "需要",
    "我们",
    "他们",
    "它们",
    "因为",
    "所以",
    "如果",
    "但是",
    "而且",
    "就是",
    "已经",
    "还是",
    "或者",
    "不是",
    "没有",
    "可能",
    "应该",
    "一下",
    "一些",
    "一种",
    "之后",
    "之前",
    "之间",
    "关于",
    "对于",
    "通过",
    "使用",
    "进行",
    "实现",
    "包括",
    "根据",
    "目前",
    "同时",
    # 英文停用词
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "can",
    "had",
    "her",
    "was",
    "one",
    "our",
    "this",
    "that",
    "with",
    "from",
    "have",
    "been",
    "does",
    "will",
    "would",
    "could",
    "should",
}


class WingRoomManager:
    """宫殿导航：Wing(人/项目) > Hall(类型) > Room(话题)。"""

    def __init__(self, palace_dir: Path):
        self._palace_dir = palace_dir
        self._palace_dir.mkdir(parents=True, exist_ok=True)
        # ★ 话题检测缓存：减少高频 add 时的重复解析
        self._topic_cache: Dict[str, str] = {}
        self._topic_cache_max = 1000

    def resolve_wing(self, scope: str) -> str:
        """将 scope 映射为 Wing 名称。"""
        return _SCOPE_TO_WING.get(scope, scope)

    def resolve_hall(self, memory_type: str) -> str:
        """将 memory_type 映射为 Hall 名称。"""
        return _TYPE_TO_HALL.get(memory_type, memory_type)

    def resolve_room(self, content: str, wing: str, memory_type: str = "") -> str:
        """根据内容自动分配 Room（话题检测），带缓存。

        检测策略：
          1. 尝试使用 KnowledgeGraph 的 extract_entities
          2. 回退到增强版正则提取
          3. 最终回退到内容哈希
        """
        # ★ 缓存检查：基于内容前 200 字的 hash
        cache_key = hashlib.md5(content[:200].encode()).hexdigest()[:16]
        cached = self._topic_cache.get(cache_key)
        if cached:
            return cached

        # 缓存满时清空（简单 LRU 策略）
        if len(self._topic_cache) >= self._topic_cache_max:
            self._topic_cache.clear()

        # 尝试提取关键实体
        room = self._detect_topic(content)
        if room:
            result = self._sanitize_name(room)
            self._topic_cache[cache_key] = result
            return result

        # 回退：使用类型 + 哈希
        if memory_type:
            content_hash = hashlib.md5(content.encode()).hexdigest()[:6]
            result = f"{memory_type}-{content_hash}"
            self._topic_cache[cache_key] = result
            return result

        # 最终回退
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        result = f"room-{content_hash}"
        self._topic_cache[cache_key] = result
        return result

    def get_room_path(self, wing: str, hall: str, room: str) -> Path:
        """获取 Room 的完整路径。"""
        path = self._palace_dir / wing / hall / room
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_wings(self) -> list:
        """列出所有 Wing。"""
        if not self._palace_dir.exists():
            return []
        return [
            d.name for d in self._palace_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
        ]

    def list_halls(self, wing: str) -> list:
        """列出指定 Wing 下的所有 Hall。"""
        wing_path = self._palace_dir / wing
        if not wing_path.exists():
            return []
        return [d.name for d in wing_path.iterdir() if d.is_dir() and not d.name.startswith("_")]

    def list_rooms(self, wing: str, hall: str) -> list:
        """列出指定 Hall 下的所有 Room。"""
        hall_path = self._palace_dir / wing / hall
        if not hall_path.exists():
            return []
        return [d.name for d in hall_path.iterdir() if d.is_dir() and not d.name.startswith("_")]

    def _detect_topic(self, content: str) -> Optional[str]:
        """从内容中检测话题。

        策略优先级：
          1. KnowledgeGraph.extract_entities（最精确）
          2. 英文小写技术关键词（react, docker, typescript 等）
          3. CamelCase / ALLCAPS 模式
          4. 中文名词短语（最宽泛，放最后）
        """
        # 策略 1: 尝试使用 KG 的 extract_entities
        try:
            from omnimem.deep.knowledge_graph import extract_entities

            entities = extract_entities(content)
            # 过滤停用词和太短的实体
            valid = [e for e in entities if e.lower() not in _STOPWORDS and len(e) >= 2]
            if valid:
                # 优先纯英文实体（技术术语更精确），再按长度升序
                def _entity_priority(e: str) -> tuple:
                    is_pure_en = bool(re.match(r"^[A-Za-z0-9_.-]+$", e))
                    is_mixed = bool(re.search(r"[A-Za-z]", e) and re.search(r"[\u4e00-\u9fff]", e))
                    return (0 if is_pure_en else (1 if not is_mixed else 2), len(e))

                valid.sort(key=_entity_priority)
                best = valid[0]
                # 纯英文实体统一小写，便于路径规范
                if re.match(r"^[A-Za-z0-9_.-]+$", best):
                    return best.lower()
                return best
        except ImportError:
            pass

        # 策略 2: 英文小写技术关键词（优先级高于中文正则）
        # 不用 \b，因为中英混合时 \b 在英文和中文之间无效
        # 用前后断言防止子串匹配（如 "pythonic" 不应匹配 "python"）
        tech_pattern = r"(?<![a-z])(python|java|go|rust|typescript|react|vue|docker|k8s|redis|mysql|postgresql|mongodb|neo4j|chromadb|sqlite|api|sql|rest|graphql|kubernetes|nginx|flask|django|fastapi|tensorflow|pytorch)(?![a-z])"
        tech_matches = re.findall(tech_pattern, content[:300].lower())
        if tech_matches:
            return tech_matches[0]

        # 策略 3: CamelCase / ALLCAPS 模式
        en_pattern = r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b|\b[A-Z]{2,}\b"
        en_matches = re.findall(en_pattern, content[:300])
        if en_matches:
            for m in en_matches:
                if len(m) >= 3:
                    return m.lower()

        # 策略 4: 中文名词短语（最宽泛，放最后）
        # ★ 优先从内容前 50 字提取，避免从长内容中间提取随机词（如"强调"）
        zh_pattern = r"[\u4e00-\u9fff]{2,6}"
        # 先尝试前 50 字
        zh_matches_head = re.findall(zh_pattern, content[:50])
        for m in zh_matches_head:
            if m.lower() not in _STOPWORDS and len(m) >= 2:
                return m
        # 回退到前 200 字
        zh_matches = re.findall(zh_pattern, content[:200])
        for m in zh_matches:
            if m.lower() not in _STOPWORDS and len(m) >= 2:
                return m

        return None

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """清理名称，使其可安全用于文件路径。"""
        sanitized = re.sub(r"[^\w\u4e00-\u9fff-]", "-", name)
        sanitized = sanitized.strip("-")
        return sanitized[:50] if sanitized else "unnamed"
