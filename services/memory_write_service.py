"""OmniMem 记忆写入 Service。

将 handlers/memorize.py 中的核心业务逻辑下沉：
- 安全扫描与反递归防护
- privacy→wing/room 推导
- 精确去重与语义去重
- 冲突检测
- 主存储 + 索引 + 检索多写
- Saga 协调
- 后台任务（LLM 决策、KG 提取、Consolidation、溯源、遗忘曲线）
- KV Cache 自动预填充
- PluginOrchestrator 事件发布
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, cast

from omnimem.handlers.deps import HandlerDependencies, MemoryWriteResult
from omnimem.memory.wing_room import _PRIVACY_TO_WING
from omnimem.services.memory_service import MemoryService
from omnimem.utils.event_publisher import get_event_publisher
from omnimem.utils.logging import sanitize_for_log
from omnimem.utils.security import SecurityValidator

logger = logging.getLogger(__name__)

# ★ 延迟导入：AtomicFactExtractor 可能在某些环境下不可用
_FactExtractor = None
try:
    from omnimem.perception.fact_extractor import AtomicFactExtractor as _FactExtractor
except ImportError:
    pass


# ★ 后台线程池：非关键路径异步执行，降低主路径延迟
_fallback_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=2, thread_name_prefix="omnimem_mem_bg")
# ★ 修复 C11：保护 _fallback_executor 创建的并发访问锁
_fallback_executor_lock = __import__("threading").Lock()


def shutdown_background_executor(wait: bool = True) -> None:
    """显式关闭模块级后台线程池。"""
    global _fallback_executor
    with _fallback_executor_lock:
        if _fallback_executor is not None:
            _fallback_executor.shutdown(wait=wait)
            _fallback_executor = None


def get_background_executor() -> ThreadPoolExecutor:
    """获取可用的后台线程池；如已关闭则自动重建（线程安全）。"""
    global _fallback_executor
    with _fallback_executor_lock:
        if _fallback_executor is None or _fallback_executor._shutdown:
            _fallback_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="omnimem_mem_bg")
        return _fallback_executor


def _log_bg_error(action: str, memory_id: str, exc: Exception) -> None:
    """统一记录后台任务异常，避免静默吞掉错误。"""
    logger.warning("OmniMem %s 后台失败: memory_id=%s error=%s", action, memory_id, exc)


def _publish_memory_stored_event(session_id: str, result: dict, content: str) -> None:
    """成功写入记忆后，通过 PluginOrchestrator EventBus 通知其他插件。

    例如 deepseek-cache-optimizer 可据此调整缓存策略。
    如果 PluginOrchestrator 未安装，静默跳过。
    """
    if not session_id:
        return
    publisher = get_event_publisher(session_id)
    publisher.publish(
        "memory_stored",
        source_plugin="omnimem",
        session_id=session_id,
        memory_id=result.get("memory_id", ""),
        memory_type=result.get("type", "fact"),
        wing=result.get("wing", ""),
        room=result.get("room", ""),
        privacy=result.get("privacy", ""),
        confidence=result.get("confidence", 3),
        content_preview=sanitize_for_log(content[:80]),
    )


def _generate_summary(content: str, _llm_memory_manager: Any = None) -> str:
    """★ 生成记忆摘要：使用规则提取生成结构化摘要。

    解决 180+ 记忆无摘要的问题。不调用 LLM（避免 memorize 延迟），
    使用 _extract_without_llm 提取 goal/progress/decisions 等字段。
    """
    try:
        from omnimem.compression.llm_summary import _extract_without_llm

        summary_obj = _extract_without_llm(content)
        text = cast(str, summary_obj.to_text()).strip()
        if text:
            return text[:300].replace("\n", " ")
    except Exception as e:
        logger.debug("Summary extraction fallback: %s", e)

    # 最终回退: 内容截断
    return content[:200].replace("\n", " ").replace("\r", " ").replace("\t", " ")


def _extract_entities_for_storage(content: str, max_entities: int = 8) -> list[str]:
    """Extract entities from content for storage in metadata.

    Used by additive fusion entity boost signal.
    """
    try:
        from omnimem.retrieval.entity_extractor import EntityExtractor

        return cast(list[str], EntityExtractor().extract(content, max_entities=max_entities))
    except Exception:
        return []


class MemoryWriteService:
    """记忆写入服务实现。"""

    def __init__(
        self,
        deps: HandlerDependencies,
        llm_memory_manager: Any = None,
        bg_executor: ThreadPoolExecutor | None = None,
        turn_count: int = 0,
    ) -> None:
        self.deps = deps
        self.llm_memory_manager = llm_memory_manager
        self.bg_executor = bg_executor
        self.turn_count = turn_count
        self._memory_service = MemoryService(deps)
        self._fact_extractor = None  # ★ 延迟初始化原子事实提取器

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def handle(self, args: dict[str, Any]) -> MemoryWriteResult:
        """处理 omni_memorize 工具调用，返回结构化结果字典。

        存储流程（安全扫描→去重→存储→索引）:
          1. 转义字符还原
          2. 安全扫描
          3. 反递归防护
          4. privacy→scope 推导
          5. 精确内容去重
          6. 语义去重
          7. 冲突检测
          8. 写入 L2 结构化记忆
          9. 写入三层索引
         10. 写入检索索引
         11. L3: KG 提取 + Consolidation
         12. L4: KV Cache 自动预填充
        """
        content = args["content"]
        dry_run = args.get("dry_run", False)
        user_id = args.get("user_id", "default")

        if self.deps.rbac and not self.deps.rbac.check_permission(user_id, "write"):
            return {"status": "blocked", "reason": f"User '{user_id}' lacks 'write' permission"}

        # ★ 还原转义字符：LLM 传入的 content 可能含字面量 \\n \\t
        content = re.sub(r"\\n(?![a-zA-Z])", "\n", content)
        content = re.sub(r"\\t(?![a-zA-Z])", "\t", content)
        content = re.sub(r"\\r(?![a-zA-Z])", "\r", content)

        # ★ 安全扫描（统一入口）
        scan_error = SecurityValidator.scan_threats(content)
        if scan_error:
            return {"status": "blocked", "reason": scan_error}

        # ★ 原子事实提取：从 content 中提取多条独立事实
        #   过程性/因果性类型不拆句: 步骤顺序与因果链是其核心价值,
        #   拆分会产生失序/断因果的碎片(6条写入膨胀为17条, 见第2轮准确性测试)
        _NO_SPLIT_TYPES = {"correction", "procedural", "skill", "workflow", "reasoning"}
        _mem_type_early = args.get("memory_type", "fact")
        if content and len(content) > 20 and _mem_type_early not in _NO_SPLIT_TYPES:
            facts = self._get_fact_extractor().extract_facts(content) if _FactExtractor else []
            if len(facts) > 1:
                # 多条原子事实：逐条独立写入，各自走完整流程
                results = []
                for i, fact in enumerate(facts):
                    fact_args = dict(args)
                    fact_args["content"] = fact
                    if "metadata" not in fact_args:
                        fact_args["metadata"] = {}
                    fact_args["metadata"]["original_content"] = content
                    fact_args["metadata"]["is_atomic_fact"] = True
                    fact_args["metadata"]["fact_index"] = i
                    result = self.handle(fact_args)
                    results.append(result)
                return results[0] if results else {"status": "error"}

        # ★ 反递归防护
        if self.deps.should_store and not self.deps.should_store(content):
            return {
                "status": "rejected",
                "reason": "Content appears to be a system injection or recursive memory",
            }

        memory_type = args.get("memory_type", "fact")
        confidence = args.get("confidence", 3)
        scope = args.get("scope", "personal")
        privacy = args.get("privacy", "personal")

        # ★ R25修复BUG-1：直接从 privacy 映射到 wing
        wing = self.deps.wing_room.resolve_wing_from_privacy(privacy, memory_type)

        # scope 保留用于 index 存储等需要 scope 字段的场合
        if privacy in _PRIVACY_TO_WING:
            scope = privacy

        # ★ 统一候选搜索
        candidates = self.deps.unified_candidate_search(content) if self.deps.unified_candidate_search else []
        # 补充 FTS5 精确搜索
        if len(candidates) < 5:
            fts_results = self.deps.store.search_by_content(content, limit=5)
            existing_ids = {m.get("memory_id", "") for m in candidates}
            for m in fts_results:
                if m.get("memory_id", "") not in existing_ids:
                    candidates.append(m)

        # ★ 精确内容去重（第一轮）
        for m in candidates:
            if m.get("content", "").strip() == content.strip():
                return {
                    "status": "duplicate",
                    "memory_id": m.get("memory_id", ""),
                    "message": "内容已存在，跳过存储",
                }

        # ★ R46修复：始终执行 FTS5 精确搜索
        fts_results = self.deps.store.search_by_content(content, limit=10)
        existing_ids = {m.get("memory_id", "") for m in candidates}
        for m in fts_results:
            if m.get("memory_id", "") not in existing_ids:
                candidates.append(m)

        # ★ 精确内容去重（第二轮，兼容 meta_store 字段名）
        for m in candidates:
            stored_content = m.get("content_preview", "") or m.get("content", "")
            if stored_content.strip() == content.strip():
                return {
                    "status": "duplicate_skipped",
                    "reason": "Exact content already exists",
                    "existing_id": m.get("memory_id", ""),
                }

        # ★ M7-10: LLM 驱动的 ADD/UPDATE/DELETE 决策（同步，优先于规则去重）
        # 当 LLM 客户端可用且有候选记忆时，用 LLM 语义理解替代纯相似度去重
        dedup_result: dict[str, Any] = {"action": "create"}
        if self.llm_memory_manager and self.llm_memory_manager.is_available and len(candidates) > 0:
            try:
                llm_decision = self.llm_memory_manager.decide(content, memory_type, candidates[:5])
                dedup_result = llm_decision.to_dedup_result()
                logger.info(
                    "LLM memory decision: action=%s, target=%s, reason=%s",
                    llm_decision.action.value, llm_decision.target_memory_id, llm_decision.reason,
                )
            except Exception as e:
                logger.warning("LLM memory decision failed (fallback to rule dedup): %s", e)
                dedup_result = {"action": "create"}

        # ★ 语义去重（规则引擎，LLM 决策未覆盖时回退到此）
        if dedup_result["action"] == "create" and not dedup_result.get("existing_id"):
            dedup_result = (
                self.deps.semantic_dedup(content, memory_type, candidates)
                if self.deps.semantic_dedup
                else {"action": "create"}
            )

        if dedup_result["action"] == "skip":
            existing_id = dedup_result.get("existing_id", "")
            existing_entry = self.deps.store.get(existing_id) if existing_id else {}
            return {
                "status": "duplicate_skipped",
                "reason": dedup_result.get("reason", ""),
                "existing_id": existing_id,
                "wing": existing_entry.get("wing", ""),
                "privacy": existing_entry.get("privacy", ""),
            }

        # ★ M7-10: LLM 决策 UPDATE — 用合并/修正后的内容替换，旧记忆标记为 superseded
        if dedup_result["action"] == "update":
            target_id = dedup_result.get("existing_id", "")
            updated_text = dedup_result.get("updated_content", "")
            if target_id and updated_text:
                content = updated_text  # 用 LLM 合并后的内容替换原内容
                logger.info("M7-10 UPDATE: using merged content, superseding %s", target_id)
            else:
                logger.warning("M7-10 UPDATE: missing target_id or updated_content, falling back to ADD")
                dedup_result = {"action": "create"}

        # ★ M7-10: LLM 决策 DELETE — 仅标记旧记忆为 superseded，不存储新内容
        if dedup_result["action"] == "delete":
            target_id = dedup_result.get("existing_id", "")
            if target_id:
                try:
                    self.deps.store.update_field(target_id, is_superseded=True)
                except Exception as e:
                    logger.warning("M7-10 DELETE: failed to supersede %s: %s", target_id, e)
                if self.deps.audit:
                    self.deps.audit.info("LLM delete decision", memory_id=target_id,
                                         reason=dedup_result.get("reason", ""))
            return {
                "status": "deleted_by_llm",
                "reason": dedup_result.get("reason", ""),
                "deleted_id": target_id,
            }

        # ★ ADD-only 策略：dedup 返回 create + superseded_id 时，记录待标记的旧记忆 ID
        # add_only 模式下不标记旧记忆为 superseded，保留所有记忆
        # M7-10: UPDATE 决策也设置 superseded_id（优先于规则 dedup 的 superseded_id）
        _is_add_only = self.deps.conflict_resolver._strategy == "add_only"
        _dedup_superseded_id = ""
        if not _is_add_only:
            _dedup_superseded_id = (
                dedup_result.get("existing_id", "")  # LLM UPDATE 的 target_id
                or dedup_result.get("superseded_id", "")  # 规则 dedup 的 superseded_id
            )

        # ★ Dry-run 模式
        if dry_run:
            room = self.deps.wing_room.resolve_room(content, wing, memory_type)
            return {
                "status": "dry_run",
                "wing": wing,
                "room": room,
                "dedup_result": dedup_result,
                "content_preview": sanitize_for_log(content[:200]),
            }

        # 治理：冲突检测
        conflict_candidates = list(candidates[:5])
        room = self.deps.wing_room.resolve_room(content, wing, memory_type)
        if wing and room:
            try:
                same_room = self.deps.store.search(wing=wing, memory_type=memory_type, limit=10)
                existing_ids = {m.get("memory_id", "") for m in conflict_candidates}
                for m in same_room:
                    if m.get("memory_id", "") not in existing_ids:
                        conflict_candidates.append(m)
            except (OSError, KeyError) as e:
                logger.warning("OmniMem same_room search failed: %s", e)

        conflict = self.deps.conflict_resolver.check(
            content,
            existing_memories=[
                {"content": m.get("content", ""), "memory_id": m.get("memory_id", "")}
                for m in conflict_candidates[:10]
            ],
        )
        conflict_info = None
        # ★ Task 2: 知识更新标记，保留 resolve 结果中的 is_updated/is_superseded
        update_marker = None
        if conflict.has_conflict:
            resolution = self.deps.conflict_resolver.resolve(content, conflict)
            if resolution.action == "reject":
                return {
                    "status": "conflict_rejected",
                    "reason": resolution.reason,
                    "existing": conflict.existing_memory,
                }
            conflict_info = {
                "conflict_type": conflict.conflict_type,
                "conflicting_with": conflict.existing_id,
                "reason": resolution.reason,
            }
            # ★ Task 2: 记录更新标记
            if resolution.is_updated:
                update_marker = {
                    "is_updated": True,
                    "is_superseded": True,
                    "superseded_id": resolution.superseded_id,
                }

        # 治理：溯源
        provenance = self.deps.provenance.track(content, source=self.deps.session_id, method="tool_call")

        # 写入 L2 结构化记忆
        hall = self.deps.wing_room.resolve_hall(memory_type)
        summary = _generate_summary(content, self.llm_memory_manager)

        # ★ 分布式向量时钟（单机模式下 get_next_vc 返回 None，vc 为空字符串）
        _next_vc = self.deps.get_next_vc() if self.deps.get_next_vc else None
        vc = _next_vc.to_json() if _next_vc is not None else ""
        now = datetime.now(timezone.utc).isoformat()

        # ★ P0方案二：统一 MemoryService Saga 编排写入
        memory_id, saga_result = self._memory_service.add_memory(
            content=content,
            memory_type=memory_type,
            confidence=confidence,
            privacy=privacy,
            wing=wing,
            room=room,
            hall=hall,
            summary=summary,
            scope=scope,
            provenance=provenance,
            vc=vc,
            entities=_extract_entities_for_storage(content),
            stored_at=now,
        )

        if not saga_result.success:
            logger.warning(
                "OmniMem memorize saga partial failure for %s at step '%s': %s",
                memory_id,
                saga_result.failed_step,
                saga_result.error,
            )

        # ★ 异步化：非关键路径提交到后台线程
        executor = self.bg_executor or get_background_executor()

        if self.llm_memory_manager and self.llm_memory_manager.is_available:
            executor.submit(self._bg_llm_decision, content, memory_type, candidates, memory_id)

        if self.deps.knowledge_graph:
            executor.submit(
                self._bg_kg_extract,
                content,
                memory_id,
                confidence / 5.0,
            )

        if self.deps.consolidation:
            executor.submit(self._bg_consolidation_submit, memory_id, content, memory_type)

        if self.deps.provenance:
            executor.submit(self._bg_provenance_record, memory_id, provenance)

        if self.deps.forgetting:
            executor.submit(self._bg_forgetting_record, memory_id)

        # ★ OPT: 记录溯源链 L0 对话 → L1 原子事实
        if self.deps.trace_chain:
            try:
                self.deps.trace_chain.record_derivation(
                    parent_node_ids=[f"conv-{self.deps.session_id}-turn-{self.turn_count}"],
                    child_node_id=memory_id,
                    child_layer="L1",
                    ref_path=str(self.deps.data_dir / "conversations" / f"{self.deps.session_id}.jsonl"),
                )
            except Exception as e:
                logger.warning("TraceChain record failed for %s: %s", memory_id, e)

        # ★ OPT: 通知 PipelineScheduler 有新记忆写入
        if self.deps.pipeline_scheduler:
            self.deps.pipeline_scheduler.on_new_memory(session_key=self.deps.session_id)

        # ★ PluginOrchestrator 集成：发布 memory_stored 事件
        _publish_memory_stored_event(
            self.deps.session_id,
            {
                "memory_id": memory_id,
                "type": memory_type,
                "wing": wing,
                "room": room,
                "privacy": privacy,
                "confidence": confidence,
            },
            content,
        )

        # ★ R25修复Minor-3：写入后确保向量索引就绪
        self.deps.retriever.flush()

        # ★ R24修复EXT-5：写入后创建 event 记录
        _event_worthy_types = {"session", "project", "workflow", "skill", "convention"}
        if confidence >= 3 and memory_type in _event_worthy_types:
            try:
                self.deps.store.add(
                    wing="auto",
                    room=f"event-{memory_id[:8]}",
                    content=sanitize_for_log(f"[create] {content[:120]}"),
                    memory_type="event",
                    confidence=1,
                    privacy="personal",
                    provenance={"trigger": "memorize", "source_memory_id": memory_id},
                )
            except (OSError, KeyError) as e:
                logger.warning("OmniMem event log creation failed: %s", e)

        # ★ 嵌入缓存持久化
        try:
            self.deps.retriever.persist_embedding_cache()
        except Exception as e:
            logger.debug("Embedding cache persist skipped: %s", e)

        # L4: KV Cache 自动预填充
        auto_preloaded = False
        if self.deps.kv_cache:
            auto_preloaded = self.deps.kv_cache.check_and_auto_preload(
                key=memory_id,
                content=content,
                metadata={"type": memory_type, "confidence": confidence, "wing": wing},
                source_memory_ids=[memory_id],
            )

        result: MemoryWriteResult = {
            "status": "stored",
            "memory_id": memory_id,
            "wing": wing,
            "room": room,
            "type": memory_type,
            "privacy": privacy,
            "confidence": confidence,
            "kv_cached": auto_preloaded,
        }

        # ★ Task 2: secret 级记忆透明化加密状态
        if privacy == "secret" and self.deps.privacy is not None:
            try:
                enc_info = self.deps.privacy.encrypt_content_with_status(content)
                if isinstance(enc_info, dict):
                    result["encryption_status"] = enc_info.get("encryption_status", "unknown")
                    if enc_info.get("encryption_status") == "disabled":
                        result["encryption_warning"] = (
                            "encryption_status=disabled; secret-level content is stored with plaintext marker"
                        )
            except Exception as e:
                logger.warning("Failed to obtain encryption status for secret memory %s: %s", memory_id, e)

        # ★ 冲突自动标记
        if conflict_info:
            result["conflict_warning"] = conflict_info
            # ★ ADD-only 模式：降低日志级别，避免大量 WARNING 刷屏
            _is_add_only = self.deps.conflict_resolver._strategy == "add_only"
            if _is_add_only:
                logger.info(
                    "OmniMem ADD-only: stored memory %s alongside existing %s (%s)",
                    memory_id,
                    conflict_info["conflicting_with"],
                    conflict_info["reason"],
                )
            else:
                logger.warning(
                    "OmniMem: stored conflicting memory %s (conflicts with %s: %s)",
                    memory_id,
                    conflict_info["conflicting_with"],
                    conflict_info["reason"],
                )
            conflict_fields = {
                "conflicting_with": conflict_info["conflicting_with"],
                "conflict_type": conflict_info["conflict_type"],
            }
            try:
                self.deps.store.update_field(memory_id, **conflict_fields)
            except Exception as e:
                logger.warning("OmniMem: failed to persist conflict_info to store: %s", e)
            try:
                self.deps.index.update_field(memory_id, immediate=True, **conflict_fields)
            except Exception as e:
                logger.warning("OmniMem: failed to persist conflict_info to index: %s", e)

        # ★ Task 2: 知识更新标记写入
        if update_marker:
            # 新记忆标记为 is_updated
            try:
                self.deps.store.update_field(memory_id, is_updated=True)
            except Exception as e:
                logger.warning("OmniMem: 写入 is_updated 标记到 store 失败: %s", e)
            try:
                self.deps.index.update_field(memory_id, immediate=True, is_updated=True)
            except Exception as e:
                logger.warning("OmniMem: 写入 is_updated 标记到 index 失败: %s", e)
            try:
                self.deps.retriever.update_metadata(memory_id, {"is_updated": True})
            except Exception as e:
                logger.warning("OmniMem: 写入 is_updated 标记到 retriever 失败: %s", e)

            # 旧记忆标记为 is_superseded
            superseded_id = update_marker["superseded_id"]
            if superseded_id:
                try:
                    self.deps.store.update_field(superseded_id, is_superseded=True)
                except Exception as e:
                    logger.warning("OmniMem: 写入 is_superseded 标记到 store 失败: %s", e)
                try:
                    self.deps.index.update_field(superseded_id, immediate=True, is_superseded=True)
                except Exception as e:
                    logger.warning("OmniMem: 写入 is_superseded 标记到 index 失败: %s", e)
                try:
                    self.deps.retriever.update_metadata(
                        superseded_id, {"is_superseded": True}
                    )
                except Exception as e:
                    logger.warning("OmniMem: 写入 is_superseded 标记到 retriever 失败: %s", e)

        # ★ ADD-only 策略：语义去重 create+superseded_id 时，将旧记忆标记为 is_superseded
        if _dedup_superseded_id:
            logger.info(
                "OmniMem ADD-only: 新记忆 %s 创建，旧记忆 %s 标记为 superseded",
                memory_id, _dedup_superseded_id,
            )
            try:
                self.deps.store.update_field(_dedup_superseded_id, is_superseded=True)
            except Exception as e:
                logger.warning("OmniMem: ADD-only 写入 is_superseded 标记到 store 失败: %s", e)
            try:
                self.deps.index.update_field(_dedup_superseded_id, immediate=True, is_superseded=True)
            except Exception as e:
                logger.warning("OmniMem: ADD-only 写入 is_superseded 标记到 index 失败: %s", e)
            try:
                self.deps.retriever.update_metadata(
                    _dedup_superseded_id, {"is_superseded": True}
                )
            except Exception as e:
                logger.warning("OmniMem: ADD-only 写入 is_superseded 标记到 retriever 失败: %s", e)

        if self.deps.audit_logger:
            self.deps.audit_logger.log(
                "memorize",
                memory_id=memory_id,
                details={"wing": wing, "room": room, "type": memory_type, "privacy": privacy},
                result="success",
                instance_id=self.deps.instance_id,
            )

        return result

    # ------------------------------------------------------------------
    # 私有辅助
    # ------------------------------------------------------------------
    def _mark_update_provenance(self, existing_id: str, dedup_result: dict[str, Any]) -> None:
        """语义去重 update 时标记旧记忆的 provenance。"""
        try:
            existing_entry = self.deps.store.get(existing_id) if existing_id else {}
            old_provenance = existing_entry.get("provenance", {})
            if isinstance(old_provenance, str):
                try:
                    old_provenance = json.loads(old_provenance)
                except (json.JSONDecodeError, TypeError):
                    old_provenance = {}
            old_provenance["llm_update_reason"] = dedup_result.get("reason", "")
            old_provenance["replaced_by_llm_decision"] = True
            old_provenance["original_content_preview"] = (existing_entry.get("content", "") or "")[:200]
            try:
                self.deps.store.update_field(
                    existing_id, provenance=json.dumps(old_provenance, ensure_ascii=False)
                )
            except Exception as e:
                logger.warning("OmniMem: 更新 provenance 失败: %s", e)
        except Exception as e:
            logger.warning("OmniMem: UPDATE provenance 标记失败: %s", e)

    def _bg_llm_decision(
        self,
        content: str,
        memory_type: str,
        candidates: list[dict[str, Any]],
        memory_id: str,
    ) -> None:
        """后台执行 LLM 决策，根据结果修正主路径写入。"""
        if self.llm_memory_manager is None:
            return
        try:
            llm_decision = self.llm_memory_manager.decide(content, memory_type, candidates)
            llm_dedup_result = llm_decision.to_dedup_result()
            llm_action = llm_dedup_result.get("action", "")

            if llm_action == "skip":
                self._safe_archive(memory_id)
                self._safe_delete_from_indices(memory_id)
                logger.info("OmniMem LLM 后台决策: SKIP — 归档 %s — %s", memory_id, llm_decision.reason)
            elif llm_action == "update":
                existing_id = llm_dedup_result.get("existing_id", "")
                if existing_id:
                    try:
                        self.deps.forgetting.archive(existing_id)
                    except Exception as e:
                        logger.warning("OmniMem LLM UPDATE 后台归档失败: %s", e)
                logger.info("OmniMem LLM 后台决策: UPDATE target=%s — %s", existing_id, llm_decision.reason)
            elif llm_action == "delete":
                delete_target = llm_dedup_result.get("existing_id", "")
                if delete_target:
                    self._safe_archive(delete_target)
                    self._safe_delete_from_indices(delete_target)
                logger.info("OmniMem LLM 后台决策: DELETE target=%s — %s", delete_target, llm_decision.reason)
            elif llm_action == "create":
                logger.info("OmniMem LLM 后台决策: ADD — %s", llm_decision.reason)
        except Exception as e:
            _log_bg_error("LLM决策", memory_id, e)

    def _safe_archive(self, memory_id: str) -> None:
        """后台安全归档指定记忆。"""
        try:
            self.deps.forgetting.archive(memory_id)
        except Exception as e:
            logger.debug("OmniMem LLM 后台归档失败: %s", e)

    def _safe_delete_from_indices(self, memory_id: str) -> None:
        """后台安全删除指定记忆在各索引中的记录。"""
        for name, client in (
            ("retriever", self.deps.retriever),
            ("index", self.deps.index),
            ("store", self.deps.store),
        ):
            try:
                client.delete(memory_id)
            except Exception as e:
                logger.debug("OmniMem LLM 后台 %s 删除失败: %s", name, e)

    def _bg_kg_extract(
        self,
        content: str,
        memory_id: str,
        confidence: float,
    ) -> None:
        """后台执行 KG 提取，并同步新三元组到时序图谱。"""
        try:
            result = self.deps.knowledge_graph.extract_and_store(
                content, memory_id=memory_id, confidence=confidence
            )
            if self.deps.temporal_kg and result.get("triples_stored", 0) > 0:
                self._sync_new_triples_to_temporal(memory_id)
        except Exception as e:
            _log_bg_error("KG提取", memory_id, e)

    def _sync_new_triples_to_temporal(self, memory_id: str) -> None:
        """将 KG 中指定 memory_id 的三元组同步到时序图谱。"""
        try:
            now = datetime.now(timezone.utc).isoformat()
            triples = self.deps.knowledge_graph._get_all_triples(limit=5000)
            synced = 0
            for t in triples:
                if t.get("source_memory_id") == memory_id:
                    self.deps.temporal_kg.add_triple_from_kg(
                        subject=t["subject"],
                        predicate=t["predicate"],
                        obj=t["object"],
                        valid_at=now,
                        source_memory_id=memory_id,
                        confidence=int(t.get("confidence", 3)),
                    )
                    synced += 1
            if synced > 0:
                logger.info("OmniMem Temporal KG synced %d triples for memory %s", synced, memory_id)
        except Exception as e:
            logger.warning("OmniMem Temporal KG sync failed: %s", e)

    def _bg_consolidation_submit(self, memory_id: str, content: str, memory_type: str) -> None:
        """后台提交 Consolidation。"""
        try:
            self.deps.consolidation.submit(memory_id, content, memory_type=memory_type)
        except Exception as e:
            _log_bg_error("Consolidation", memory_id, e)

    def _bg_provenance_record(self, memory_id: str, provenance_data: Any) -> None:
        """后台记录溯源。"""
        try:
            self.deps.provenance.record(memory_id, provenance_data)
        except Exception as e:
            _log_bg_error("Provenance", memory_id, e)

    def _bg_forgetting_record(self, memory_id: str) -> None:
        """后台记录遗忘曲线。"""
        try:
            self.deps.forgetting.record_access(memory_id)
        except Exception as e:
            _log_bg_error("Forgetting", memory_id, e)

    def _get_fact_extractor(self):
        """延迟初始化原子事实提取器。"""
        if self._fact_extractor is None and _FactExtractor is not None:
            try:
                self._fact_extractor = _FactExtractor()
            except Exception as e:
                logger.warning("原子事实提取器初始化失败: %s", e)
                self._fact_extractor = None
        return self._fact_extractor
