"""Provider 中间件/钩子：preinject、post-tool-call、periodic tasks、对话同步等。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from omnimem.core.attachment import build_attachments
from omnimem.core.dedup import SemanticDedupService
from omnimem.handlers.schemas import get_tool_schemas as _get_tool_schemas
from omnimem.utils.security import SecurityValidator

logger = logging.getLogger(__name__)


class ProviderMiddlewareMixin:
    """负责 preinject、periodic tasks、对话同步、工具路由等横切逻辑。"""

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """OmniMem 暴露的工具 schema — 委托到 handlers/schemas.py。"""
        return _get_tool_schemas()

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        try:
            return self._tool_router.route(tool_name, args)
        except Exception as e:
            logger.error("OmniMem tool %s failed: %s", tool_name, e)
            return json.dumps({"error": str(e)})

    @staticmethod
    def _strip_system_injections(text: str) -> str:
        """剥离 prefetch 注入的记忆区块，只保留用户原始输入。

        委托 SecurityValidator 实现，支持 Unicode 归一化和编码绕过检测。
        """
        return SecurityValidator.strip_system_injections(text)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """每轮对话后：感知 → 写入 → 治理 — 委托给 SessionManager。"""
        if not self._should_write:
            return
        self._session_manager.sync_turn(user_content, assistant_content)
        # 同步 turn_count 回 provider（其他方法依赖此值）
        self._turn_count = self._session_manager.turn_count

    def _periodic_config_reload(self, turn_number: int) -> None:
        """每 10 轮检查配置文件变更并热重载。"""
        if turn_number % 10 == 0:
            self._config.reload()

    def _periodic_consolidation(self, turn_number: int) -> None:
        """每 15 轮自动触发 Consolidation（长会话不积压）。"""
        if turn_number % 15 == 0 and self._consolidation:
            try:
                processed = self._consolidation.process_pending()
                if processed > 0:
                    logger.info("OmniMem auto-consolidation: %d memories", processed)
            except Exception as e:
                logger.warning("OmniMem auto-consolidation failed: %s", e)

    def _periodic_weight_update(self, turn_number: int) -> None:
        """每 20 轮更新检索来源权重（基于反馈统计）。"""
        if turn_number % 20 == 0 and hasattr(self, "_feedback") and self._feedback:
            try:
                weights = self._feedback.get_source_weights(window=100)
                if weights:
                    self._retriever.set_source_weights(weights)
                    logger.info("Updated source weights from feedback: %s", weights)
            except Exception as e:
                logger.warning("Feedback weight update failed: %s", e)

    def _periodic_sync(self, turn_number: int) -> None:
        """每 5 轮从其他实例拉取变更（changelog 模式）。"""
        if (
            turn_number % 5 == 0
            and hasattr(self, "_sync_engine")
            and self._sync_engine
            and self._sync_engine._config.mode == "changelog"
        ):
            try:
                applied = self._sync_engine.sync_from_others(
                    apply_fn=self._apply_sync_change,
                    get_local_fn=lambda mid: self._store.get(mid),
                )
                if applied > 0:
                    logger.info("OmniMem sync: applied %d changes from other instances", applied)
            except Exception as e:
                logger.warning("OmniMem sync failed: %s", e)

    def _periodic_quality_autotune(self, turn_number: int) -> None:
        """每 30 轮检查质量趋势并应用调优建议。"""
        if (
            turn_number % 30 == 0
            and hasattr(self, "_quality_evaluator")
            and self._quality_evaluator
        ):
            try:
                suggestions = self._quality_evaluator.get_auto_tune_suggestions()
                for s in suggestions.get("suggestions", []):
                    param = s.get("parameter", "")
                    action = s.get("action", "")
                    desc = s.get("description", "")
                    suggested_value = s.get("suggested_value", "")
                    logger.info("质量调优建议: %s → %s (%s)", param, action, desc)
                    if param == "min_rrf" and hasattr(self, "_retriever") and self._retriever:
                        try:
                            new_val = float(suggested_value)
                            self._retriever._rrf._min_rrf = new_val
                            logger.info("已应用调优: min_rrf = %.3f", new_val)
                        except (ValueError, AttributeError) as e:
                            logger.warning("应用 min_rrf 调优失败: %s", e)
                    elif (
                        param == "rrf_vector_weight"
                        and hasattr(self, "_retriever")
                        and self._retriever
                    ):
                        try:
                            new_val = float(suggested_value)
                            if "vector" in self._retriever._channels:
                                retriever, _ = self._retriever._channels["vector"]
                                self._retriever._channels["vector"] = (retriever, new_val)
                                logger.info("已应用调优: vector_weight = %.1f", new_val)
                        except (ValueError, AttributeError) as e:
                            logger.warning("应用 vector_weight 调优失败: %s", e)
                    elif (
                        param == "rrf_bm25_weight"
                        and hasattr(self, "_retriever")
                        and self._retriever
                    ):
                        try:
                            new_val = float(suggested_value)
                            if "bm25" in self._retriever._channels:
                                retriever, _ = self._retriever._channels["bm25"]
                                self._retriever._channels["bm25"] = (retriever, new_val)
                                logger.info("已应用调优: bm25_weight = %.1f", new_val)
                        except (ValueError, AttributeError) as e:
                            logger.warning("应用 bm25_weight 调优失败: %s", e)
            except Exception as e:
                logger.warning("质量评估自动调优失败: %s", e)

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        """每轮开始：配置热重载 + 意图预测 + 预加载 + 分布式同步。"""
        if not self._should_write:
            return
        self._periodic_config_reload(turn_number)
        self._periodic_consolidation(turn_number)
        self._periodic_weight_update(turn_number)
        self._periodic_sync(turn_number)
        self._periodic_quality_autotune(turn_number)
        predicted = self._perception.predict_intent(message)
        if predicted:
            self.queue_prefetch(predicted)

    def wrap_model_call(
        self,
        user_message: str,
        messages: list,
        *,
        session_id: str = "",
    ) -> str:
        """Pre-inject relevant skill details and memory deep-dives.

        Implements the DeepAgents SkillsMiddleware pattern:
        - Analyzes user message for skill-relevant keywords
        - Auto-injects top-1 matching skill's SKILL.md content
        - Pre-injects memory details when the user asks about past context

        Saves the model from needing extra tool-call iterations.
        """
        if not user_message or len(user_message) < 5:
            return ""

        # Per-turn cache: avoid duplicate work within a turn's iterations
        cache_key = f"{session_id}:{user_message[:200]}"
        if cache_key in self._wrap_call_cache:
            return self._wrap_call_cache[cache_key]

        result_parts = []

        # 1. Skill pre-injection
        skill_result = self._try_skill_preinject(user_message)
        if skill_result:
            result_parts.append(skill_result)

        # 2. Memory detail pre-injection
        memory_result = self._try_memory_preinject(user_message)
        if memory_result:
            result_parts.append(memory_result)

        final = "\n\n".join(result_parts) if result_parts else ""
        self._wrap_call_cache[cache_key] = final
        # Keep cache bounded
        if len(self._wrap_call_cache) > 50:
            keys = list(self._wrap_call_cache.keys())
            for k in keys[:25]:
                del self._wrap_call_cache[k]
        return final

    def _build_skill_index(self) -> None:
        """Build lightweight skill keyword index from skills directory."""
        if self._skill_index_built:
            return
        self._skill_index_built = True
        try:

            from agent.skill_commands import (
                _parse_frontmatter,
                get_all_skills_dirs,
                iter_skill_index_files,
            )

            skills_dirs = get_all_skills_dirs()
            seen_names = set()
            for scan_dir in skills_dirs:
                if not scan_dir.exists():
                    continue
                for skill_md in iter_skill_index_files(scan_dir, "SKILL.md"):
                    if any(p in {".git", ".github", ".hub", ".archive"} for p in skill_md.parts):
                        continue
                    try:
                        content = skill_md.read_text(encoding="utf-8")
                        fm, _ = _parse_frontmatter(content)
                        name = fm.get("name") or skill_md.parent.name
                        if name in seen_names:
                            continue
                        seen_names.add(name)
                        desc = fm.get("description", "")
                        # Extract keywords from name + description
                        text = f"{name} {desc}".lower()
                        keywords = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", text))
                        self._skill_index_cache.append(
                            {
                                "name": name,
                                "path": str(skill_md),
                                "description": desc[:200],
                                "keywords": keywords,
                            }
                        )
                    except Exception:
                        continue
        except Exception as e:
            logger.debug("OmniMem skill index build failed: %s", e)

    def _try_skill_preinject(self, user_message: str) -> str:
        """Try to find and pre-inject a matching skill's content."""
        self._build_skill_index()
        if not self._skill_index_cache:
            return ""

        msg_keywords = set(
            re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", user_message.lower())
        )
        if not msg_keywords:
            return ""

        # Score each skill by keyword overlap
        best_score = 0
        best_skill = None
        for skill in self._skill_index_cache:
            overlap = len(msg_keywords & skill["keywords"])
            if overlap > best_score:
                best_score = overlap
                best_skill = skill

        # Require at least 2 keyword overlap to avoid false positives
        if best_score < 2 or not best_skill:
            return ""

        # Load the skill content (compact: first 2000 chars)
        try:
            from pathlib import Path

            skill_path = Path(best_skill["path"])
            if not skill_path.exists():
                return ""
            content = skill_path.read_text(encoding="utf-8")
            # Strip frontmatter
            if content.startswith("---"):
                end = content.find("---", 3)
                if end > 0:
                    content = content[end + 3 :].strip()
            # Truncate to keep it compact
            if len(content) > 2000:
                content = content[:2000] + "\n...(truncated)"
            return (
                f"[Auto-injected skill: {best_skill['name']}]\n"
                f"{content}"
            )
        except Exception as e:
            logger.debug("OmniMem skill preinject failed: %s", e)
            return ""

    def _try_memory_preinject(self, user_message: str) -> str:
        """Pre-inject relevant memory details for recall-type queries."""
        # Detect recall signals: user asking about past context
        recall_signals = [
            "上次",
            "之前",
            "以前",
            "记得",
            "说过的",
            "提到过",
            "last time",
            "previously",
            "remember",
            "mentioned",
            "那个",
            "具体",
            "详情",
            "细节",
            "细节",
        ]
        msg_lower = user_message.lower()
        has_recall_signal = any(s in msg_lower for s in recall_signals)
        if not has_recall_signal:
            return ""

        # Search for relevant memories
        try:
            results = self._retriever.search(user_message, limit=3) if self._retriever else []
        except Exception:
            results = []

        if not results:
            return ""

        # Pre-inject the top result's content (not just summary)
        lines = []
        for r in results[:2]:
            content = r.get("content", "")
            if content and len(content) > 10:
                mtype = r.get("type", "memory")
                # Truncate per entry
                if len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"[{mtype}] {content}")

        if lines:
            return "[Auto-injected memory details]\n" + "\n".join(lines)
        return ""

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """压缩前：构建 Attachment + 紧急保存。"""
        self._store.flush()
        saved_context = self._store_service.emergency_save(messages)

        attachments = build_attachments(messages)

        parts = []
        if saved_context:
            parts.append(saved_context)
        if attachments:
            att_text = "\n".join(f"[{a.kind}] {a.title}: {a.body[:200]}" for a in attachments)
            parts.append(f"### Pre-Compression Attachments\n{att_text}")

        result = "\n\n".join(parts)

        if self._config.get("enable_compression", False) and result:
            result = self._compression_pipeline.compress(result)

        return result

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """内置记忆写入时：冲突检测。"""
        if action == "add":
            conflict = self._conflict_resolver.check(content)
            if conflict.has_conflict:
                logger.warning(
                    "OmniMem: conflict detected with existing memory: %s",
                    conflict.existing_memory,
                )

    @staticmethod
    def _should_store(content: str) -> bool:
        """判断内容是否值得存储，过滤系统注入和递归内容。

        委托 SecurityValidator 实现，支持 Unicode 归一化和编码绕过检测。
        所有存储路径都应经过此检查。
        """
        should_store, reason = SecurityValidator.should_store(content)
        if not should_store and reason:
            logger.warning("SecurityValidator._should_store blocked: %s", reason)
        return should_store

    def _extract_core_fact(self, text: str) -> str:
        """从原文中提取精简核心事实（委托给感知引擎）。"""
        return str(self._perception._extract_core_fact(text))

    @staticmethod
    def _compute_text_similarity(text_a: str, text_b: str) -> float:
        return SemanticDedupService.compute_text_similarity(text_a, text_b)

    def _semantic_dedup(
        self, content: str, memory_type: str, candidates: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        return self._dedup_service.semantic_dedup(content, memory_type, candidates)

    def _unified_candidate_search(self, content: str) -> list[dict[str, Any]]:
        return self._dedup_service.unified_candidate_search(content)

    def _search_candidates(self, content: str) -> list[dict[str, Any]]:
        return self._dedup_service.search_candidates(content)
