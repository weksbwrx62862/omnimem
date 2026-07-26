"""ReflectEngine 核心管线。

职责：
  - 初始化 ReflectEngine 与 SQLite 数据库
  - 执行 reflect 循环（search → recall → expand → observe → synthesize）
  - 公开接口：reflect / get_reflection_history / get_stats / close

LLM 生成、规则归纳、持久化等能力分别挂载自 prompts / synthesis / writer 模块。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from omnimem.deep.reflect.disposition import (
    Disposition,
    ReflectionContext,
    ReflectResult,
    _apply_disposition,
)
from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)


class ReflectEngine:
    """Reflect 工具循环引擎 — L3 深层记忆反思。

    四步 Reflect 循环 (Hindsight-inspired):
      Step 1: search_mental_models — 从 Consolidation 查找已有心智模型
      Step 2: recall_facts — 检索相关事实（外部检索函数或 Consolidation 观察）
      Step 3: expand_context — 从事实中扩展关联上下文（关键词→更多观察）
      Step 4: search_observations — 搜索观察洞察
      Step 5: 综合生成 + Disposition 修饰 → 生成输出

    Disposition 性格系统:
      - skepticism (1-5): 怀疑度，越高越审慎
      - literalness (1-5): 字面度，越高越精确
      - empathy (1-5): 共情度，越高越关注感受

    生成策略:
      优先使用 LLM 推理归纳（通过 llm_fn），LLM 不可用时回退到规则归纳。
      规则归纳基于关键词提取和短语重组，输出质量有限。
      关键词堆砌检测和后处理确保输出为连贯自然语言。

    持久化:
      反思结果存入 SQLite (reflect.db)，支持历史查询和统计。
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        consolidation_engine: Any | None = None,
        default_disposition: Disposition | None = None,
        recall_fn: Callable[..., Any] | None = None,
        llm_fn: Callable[..., Any] | None = None,
        llm_client: Any | None = None,
        governance_store: Any | None = None,
    ):
        """初始化 ReflectEngine。

        Args:
            data_dir: 数据目录，用于持久化反思结果
            consolidation_engine: ConsolidationEngine 实例，用于查询观察/模型
            default_disposition: 默认性格参数
            recall_fn: 外部检索函数，签名: (query, limit) -> List[Dict]
            llm_fn: LLM 调用函数，签名: (prompt: str, system: str, max_tokens: int) -> str
                    接收 prompt + system prompt，返回 LLM 文本响应。
                    为 None 时回退到规则归纳。
            llm_client: LLM 客户端实例，用于直接调用 LLM
        """
        self._data_dir = data_dir
        self._consolidation = consolidation_engine
        self._default_disposition = default_disposition or Disposition()
        self._recall_fn = recall_fn
        self._llm_fn = llm_fn
        self._llm_client = llm_client
        self._conn: sqlite3.Connection | None = None
        self._reflection_count = 0

        # ★ M6-8: 接入 GovernanceStore 统一存储
        if governance_store is not None:
            self._store = governance_store
            self._conn = governance_store.get_write_conn()
            self._lock = governance_store.write_lock
        else:
            self._store = None
            self._lock = threading.RLock()
            if data_dir:
                self._init_db(data_dir)

    def _init_db(self, data_dir: Path) -> None:
        """初始化反思结果数据库。"""
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "reflect.db"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="reflections",
            create_sql="""
                CREATE TABLE IF NOT EXISTS reflections (
                    reflection_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    observation TEXT,
                    mental_model TEXT,
                    confidence REAL,
                    disposition TEXT,
                    source_ids TEXT,
                    created_at TEXT,
                    metadata TEXT
                )
            """,
            migrations=[],
        )
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reflect_query ON reflections(query)
        """)
        self._conn.commit()

    # ─── 公开接口 ─────────────────────────────────────────────

    def reflect(
        self,
        query: str,
        memories: list[dict[str, Any]] | None = None,
        disposition: dict[str, int] | None = None,
    ) -> ReflectResult:
        """执行完整的 Reflect 循环。

        Args:
            query: 反思主题
            memories: 外部提供的记忆列表（如来自检索引擎）
            disposition: 性格参数 (skepticism, literalness, empathy)

        Returns:
            ReflectResult
        """
        # 解析 Disposition
        disp = self._resolve_disposition(disposition)

        # 构建 Reflect 循环上下文
        ctx = ReflectionContext(query=query)

        # Step 1: search_mental_models — 从 Consolidation 查找已有心智模型
        ctx.mental_models = self._search_mental_models(query)

        # Step 2: recall_facts — 检索相关事实
        ctx.facts = self._recall_facts(query, memories)

        # Step 3: expand_context — 从事实中扩展关联
        ctx.expanded = self._expand_context(query, ctx.facts)

        # Step 4: search_observations — 搜索观察洞察
        ctx.observations = self._search_observations(query)

        # Step 5: 综合生成
        result = self._synthesize(query, ctx, disp)

        # 持久化
        self._persist_reflection(result)

        self._reflection_count += 1
        return result

    def get_reflection_history(self, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        """获取反思历史记录。

        Args:
            query: 可选的查询关键词，为空时返回全部
            limit: 最大返回条数

        Returns:
            反思记录列表，按时间倒序排列
        """
        if not self._conn:
            return []
        try:
            if query:
                rows = self._conn.execute(
                    "SELECT * FROM reflections WHERE query LIKE ? ORDER BY created_at DESC LIMIT ?",
                    (f"%{query}%", limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM reflections ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            keys = [
                "reflection_id",
                "query",
                "observation",
                "mental_model",
                "confidence",
                "disposition",
                "source_ids",
                "created_at",
                "metadata",
            ]
            return [dict(zip(keys, row, strict=False)) for row in rows]
        except Exception as e:
            logger.warning("Reflect history query failed: %s", e)
            return []

    def get_stats(self) -> dict[str, Any]:
        """获取反思统计信息，包含总反思次数和持久化数量。"""
        stats = {
            "total_reflections": self._reflection_count,
        }
        if self._conn:
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM reflections").fetchone()
                stats["persisted"] = row[0] if row else 0
            except Exception:
                stats["persisted"] = 0
        return stats

    def close(self) -> None:
        """关闭 SQLite 数据库连接，释放资源。"""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ─── Reflect 循环四步 ─────────────────────────────────────

    def _search_mental_models(self, query: str) -> list[dict[str, Any]]:
        """Step 1: 查找已有的心智模型。"""
        if self._consolidation:
            return self._consolidation.get_mental_models(topic=query, limit=5)  # type: ignore[no-any-return]
        return []

    def _recall_facts(
        self, query: str, memories: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Step 2: 检索相关事实。"""
        # 优先使用外部传入的记忆
        if memories:
            return memories[:20]

        # 使用外部检索函数
        if self._recall_fn:
            try:
                results = self._recall_fn(query, limit=20)
                if results:
                    return results  # type: ignore[no-any-return]
            except Exception as e:
                logger.warning("Recall function failed: %s", e)

        # 从 Consolidation 查询经验事实
        if self._consolidation:
            return self._consolidation.get_observations(topic=query, limit=20)  # type: ignore[no-any-return]
        return []

    def _expand_context(self, query: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Step 3: 从已有事实中扩展关联上下文。

        从事实内容中提取关键词，查找更广泛的观察。
        """
        if not facts or not self._consolidation:
            return []

        # 使用智能关键词提取替代正则切词
        keywords = set()
        for fact in facts[:10]:
            content = fact.get("content", "")
            kws = self._smart_extract_keywords(content, max_keywords=3)
            keywords.update(kws[:3])

        # 用关键词从 Consolidation 检索更多观察
        expanded = []
        for kw in list(keywords)[:5]:
            obs = self._consolidation.get_observations(topic=kw, limit=5)
            for o in obs:
                if o not in expanded and o not in facts:
                    expanded.append(o)
        return expanded[:15]

    def _search_observations(self, query: str) -> list[dict[str, Any]]:
        """Step 4: 搜索观察洞察。"""
        if self._consolidation:
            return self._consolidation.get_observations(topic=query, limit=10)  # type: ignore[no-any-return]
        return []

    # ─── 综合生成 ─────────────────────────────────────────────

    def _synthesize(
        self, query: str, ctx: ReflectionContext, disposition: Disposition
    ) -> ReflectResult:
        """综合 Reflect 循环四步的结果，生成最终反思输出。"""
        source_ids: list[str] = []
        mental_model = ""
        confidence = 0.0
        depth = 0

        # ─── 收集所有可用内容 ───
        all_contents: list[str] = []

        if ctx.mental_models:
            best_model = ctx.mental_models[0]
            mental_model = best_model.get("content", "")
            confidence = best_model.get("confidence", 0.7)
            source_ids.extend(
                best_model.get("source_ids", [])
                if isinstance(best_model.get("source_ids"), list)
                else []
            )
            depth = 3
            all_contents.append(f"[已有心智模型] {mental_model}")

        # 观察内容（无论有无心智模型都收集，供 LLM 深度推理）
        if ctx.observations:
            source_ids.extend(o.get("item_id", "") for o in ctx.observations[:5])
            for o in ctx.observations[:5]:
                all_contents.append(f"[观察] {o.get('content', '')[:200]}")
            depth = max(depth, 2)

        # 事实内容
        if ctx.facts:
            source_ids.extend(f.get("memory_id", f.get("item_id", "")) for f in ctx.facts[:8])
            for f in ctx.facts[:8]:
                all_contents.append(f"[事实] {f.get('content', '')[:200]}")
            depth = max(depth, 1)

        # 扩展上下文
        if ctx.expanded:
            for e in ctx.expanded[:3]:
                all_contents.append(f"[关联] {e.get('content', '')[:150]}")
                source_ids.append(e.get("item_id", ""))

        # ─── 无数据 ───
        if not all_contents:
            return ReflectResult(
                observation=f"没有找到与 '{query}' 相关的记忆来进行反思。",
                mental_model="",
                confidence=0.0,
                sources=[],
                disposition_used=disposition.to_dict(),
                reflection_depth=0,
                query=query,
            )

        # ─── 尝试 LLM 推理归纳 ───
        llm_result = self._generate_with_llm(query, all_contents, disposition)
        if llm_result is not None:
            llm_obs, llm_model, llm_conf = llm_result
            # LLM 成功 → 使用 LLM 输出
            observation = llm_obs or self._rule_based_observation(query, ctx)
            if llm_model:
                mental_model = llm_model
            # ★ confidence 合并：LLM 置信度与事实支撑度取较大值
            # 避免有事实支撑时 confidence=0.0（LLM 可能对异质信息返回低置信度）
            if llm_conf > 0:
                confidence = llm_conf
            elif ctx.facts and confidence < 0.3:
                confidence = 0.3
            depth = max(depth, 2)
        else:
            # LLM 不可用 → 回退到规则归纳
            logger.warning("LLM unavailable, degrading to rule-based synthesis")
            observation, mental_model, confidence = self._rule_based_synthesize(
                query, ctx, confidence
            )

        # ─── 应用 Disposition 修饰 ───
        observation, mental_model = _apply_disposition(observation, mental_model, disposition)

        # ─── 后处理：检测并拒绝关键词堆砌模式 ───
        mental_model = self._post_process_mental_model(mental_model, confidence)

        return ReflectResult(
            observation=observation,
            mental_model=mental_model,
            confidence=confidence,
            sources=source_ids[:20],
            disposition_used=disposition.to_dict(),
            reflection_depth=depth,
            query=query,
        )

    # ─── 内部辅助 ─────────────────────────────────────────────

    def _resolve_disposition(self, disposition: dict[str, int] | None) -> Disposition:
        """解析 Disposition 参数。"""
        if disposition:
            return Disposition(
                skepticism=disposition.get("skepticism", self._default_disposition.skepticism),
                literalness=disposition.get("literalness", self._default_disposition.literalness),
                empathy=disposition.get("empathy", self._default_disposition.empathy),
            ).clamp()
        return self._default_disposition.clamp()


# ─── 挂载 prompts / synthesis / writer 模块方法到 ReflectEngine ───
from omnimem.deep.reflect import prompts as _prompts_module  # noqa: E402
from omnimem.deep.reflect import synthesis as _synthesis_module  # noqa: E402
from omnimem.deep.reflect import writer as _writer_module  # noqa: E402

ReflectEngine._generate_with_llm = _prompts_module._generate_with_llm
ReflectEngine._parse_llm_output = _prompts_module._parse_llm_output

ReflectEngine._smart_extract_keywords = _synthesis_module._smart_extract_keywords
ReflectEngine._extract_content_phrases = _synthesis_module._extract_content_phrases
ReflectEngine._rule_based_observation = _synthesis_module._rule_based_observation
ReflectEngine._rule_based_synthesize = _synthesis_module._rule_based_synthesize
ReflectEngine._generate_model_from_observations = _synthesis_module._generate_model_from_observations
ReflectEngine._generate_observation_from_facts = _synthesis_module._generate_observation_from_facts
ReflectEngine._generate_model_from_facts = _synthesis_module._generate_model_from_facts
ReflectEngine._is_keyword_stuffing = _synthesis_module._is_keyword_stuffing
ReflectEngine._post_process_mental_model = _synthesis_module._post_process_mental_model

ReflectEngine._persist_reflection = _writer_module._persist_reflection
