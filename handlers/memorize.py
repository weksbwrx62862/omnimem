"""OmniMem memorize 处理器。

从 provider.py 的 _handle_memorize() 方法提取，通过 provider 参数访问实例组件。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from omnimem.core.saga import SagaStep
from omnimem.utils.security import SecurityValidator

logger = logging.getLogger(__name__)

# ★ R27优化：模块级 privacy→scope 映射，避免每次调用重建字典
_PRIVACY_TO_SCOPE: dict[str, str] = {
    "public": "public",
    "team": "team",
    "personal": "personal",
    "secret": "secret",
}


def handle_memorize(provider: Any, args: dict[str, Any]) -> str:
    """处理 omni_memorize 工具调用。

    存储流程（安全扫描→去重→存储→索引）:
      1. 转义字符还原：将 LLM 传入的字面量 \\n/\\t/\\r 还原为实际控制字符
      2. 安全扫描：调用 SecurityValidator.scan_threats 检查注入攻击
      3. 反递归防护：拒绝存储系统注入内容，防止 prefetch→store→prefetch 循环
      4. privacy→scope 推导：根据 privacy 参数推导 wing 分类
      5. 精确内容去重：先检查是否有完全相同的内容
      6. 语义去重：通过 _semantic_dedup 检查相似记忆，高相似度合并更新
      7. 冲突检测：检查同 room/同 type 记忆的矛盾，冲突严重则拒绝
      8. 写入 L2 结构化记忆（DrawerClosetStore）
      9. 写入三层索引（ThreeLevelIndex）
     10. 写入检索索引（HybridRetriever 向量+BM25）
     11. L3: 提取实体和三元组到知识图谱 + 提交到 Consolidation 队列
     12. L4: 检查 KV Cache 自动预填充触发

    Args:
        provider: OmniMemProvider 实例，用于访问子组件
        args: 工具调用参数，包含 content/memory_type/confidence/scope/privacy

    Returns:
        JSON 字符串，status 可能为:
          stored — 成功存储
          duplicate_skipped — 精确或语义重复，跳过
          conflict_rejected — 冲突严重，拒绝存储
          blocked — 安全扫描拦截
          rejected — 反递归防护拦截
    """
    content = args["content"]
    user_id = args.get("user_id", "default")
    if hasattr(provider, "_rbac") and not provider._rbac.check_permission(user_id, "write"):
        return json.dumps(
            {"status": "blocked", "reason": f"User '{user_id}' lacks 'write' permission"}
        )

    # ★ 还原转义字符：LLM 传入的 content 可能含字面量 \\n \\t
    # 在 JSON 解析后这些变成 \n \t 字面量（两个字符），需要还原为实际控制字符
    # ★ R19修复Minor-1-v2：使用正则替换，避免破坏路径中的\normal、\test等
    #   判断依据：\n/\t 后面紧跟字母的是路径（如C:\new\test），不应还原
    #   \n/\t 后面是空格/标点/行尾/非字母的才是真正的转义字符，应还原
    content = re.sub(r"\\n(?![a-zA-Z])", "\n", content)
    content = re.sub(r"\\t(?![a-zA-Z])", "\t", content)
    content = re.sub(r"\\r(?![a-zA-Z])", "\r", content)

    # ★ 安全扫描（统一入口）
    scan_error = SecurityValidator.scan_threats(content)
    if scan_error:
        return json.dumps({"status": "blocked", "reason": scan_error})

    # ★ 反递归防护：拒绝存储系统注入内容，防止 prefetch → store → prefetch 循环
    if not provider._should_store(content):
        return json.dumps(
            {
                "status": "rejected",
                "reason": "Content appears to be a system injection or recursive memory",
            }
        )

    memory_type = args.get("memory_type", "fact")
    confidence = args.get("confidence", 3)
    scope = args.get("scope", "personal")
    privacy = args.get("privacy", "personal")

    # ★ R22修复：privacy 始终参与 wing 推导
    # 之前仅在 scope=="personal" 时才用 privacy 推导 scope，
    # 导致 LLM 显式传 scope 时 privacy 被忽略，wing 因 type 不同而不一致
    # 现在改为：privacy 始终覆盖 scope，确保同一 privacy 值映射到同一 wing
    if privacy in _PRIVACY_TO_SCOPE:
        scope = _PRIVACY_TO_SCOPE[privacy]

    # ★ R24修复BUG-1：preference/event + team → project scope（映射到 projects wing）
    # 之前所有 team 都映射到 shared，但 preference 和 event 属于项目协作范畴
    if scope == "team" and memory_type in ("preference", "event"):
        scope = "project"

    # ★ 精确内容去重：在语义搜索之前，先检查是否有完全相同的内容
    # 这避免了 ChromaDB 索引延迟导致候选搜索不到的问题
    exact_match = provider._store.search_by_content(content, limit=5)
    for m in exact_match:
        if m.get("content", "").strip() == content.strip():
            return json.dumps(
                {
                    "status": "duplicate_skipped",
                    "reason": "Exact content already exists",
                    "existing_id": m.get("memory_id", ""),
                }
            )

    # ★ 统一搜索：去重和冲突检测共享候选结果，避免重复搜索
    candidates = provider._unified_candidate_search(content)

    # ★ 语义去重：写入前检索已有记忆，高相似度则合并更新
    dedup_result = provider._semantic_dedup(content, memory_type, candidates)
    if dedup_result["action"] == "update":
        existing_id = dedup_result["existing_id"]
        provider._forgetting.archive(existing_id)
        logger.info("OmniMem dedup: archived duplicate %s, storing updated version", existing_id)
    elif dedup_result["action"] == "skip":
        existing_id = dedup_result.get("existing_id", "")
        existing_entry = provider._store.get(existing_id) if existing_id else {}
        return json.dumps(
            {
                "status": "duplicate_skipped",
                "reason": dedup_result["reason"],
                "existing_id": existing_id,
                "wing": existing_entry.get("wing", ""),
                "privacy": existing_entry.get("privacy", ""),
            }
        )

    # 治理：冲突检测
    # ★ 合并候选：语义搜索结果 + 同 room 的记忆（捕捉主题矛盾但语义不相似的情况）
    conflict_candidates = list(candidates[:5])
    # 预计算 wing/room 以查找同 room 记忆
    _wing = provider._wing_room.resolve_wing(scope)
    _room = provider._wing_room.resolve_room(content, _wing, memory_type)
    if _wing and _room:
        try:
            same_room = provider._store.search(wing=_wing, memory_type=memory_type, limit=10)
            existing_ids = {m.get("memory_id", "") for m in conflict_candidates}
            for m in same_room:
                if m.get("memory_id", "") not in existing_ids:
                    conflict_candidates.append(m)
        except (OSError, KeyError) as e:
            logger.debug("OmniMem same_room search failed: %s", e)

    conflict = provider._conflict_resolver.check(
        content,
        existing_memories=[
            {"content": m.get("content", ""), "memory_id": m.get("memory_id", "")}
            for m in conflict_candidates[:10]
        ],
    )
    conflict_info = None  # ★ 记录冲突信息，用于后续标记
    if conflict.has_conflict:
        resolution = provider._conflict_resolver.resolve(content, conflict)
        if resolution.action == "reject":
            return json.dumps(
                {
                    "status": "conflict_rejected",
                    "reason": resolution.reason,
                    "existing": conflict.existing_memory,
                }
            )
        # ★ 冲突被接受时记录信息，写入后标记到记忆
        conflict_info = {
            "conflict_type": conflict.conflict_type,
            "conflicting_with": conflict.existing_id,
            "reason": resolution.reason,
        }

    # 治理：溯源
    provenance = provider._provenance.track(
        content, source=provider._session_id, method="tool_call"
    )

    # 写入 L2 结构化记忆
    wing = provider._wing_room.resolve_wing(scope)
    hall = provider._wing_room.resolve_hall(memory_type)
    room = provider._wing_room.resolve_room(content, wing, memory_type)
    # ★ 分布式向量时钟：为每条记忆附加逻辑时钟
    vc = provider.get_next_vc().to_json()
    memory_id = provider._store.add(
        wing=wing,
        room=room,
        content=content,
        memory_type=memory_type,
        confidence=confidence,
        privacy=privacy,
        provenance=provenance,
        vc=vc,
    )

    # ★ P0方案二：Saga 协调派生数据写入
    # 主存储（store.add）已在上方完成，作为唯一事实来源。
    # index / retriever / knowledge_graph 作为派生数据，通过 Saga 保证最终一致。
    now = datetime.now(timezone.utc).isoformat()
    summary = content[:200].replace("\n", " ").replace("\r", " ").replace("\t", " ")

    saga_result = provider._saga.execute(
        memory_id,
        [
            SagaStep(
                "three_level_index",
                lambda: provider._index.add(
                    memory_id=memory_id,
                    wing=wing,
                    hall=hall,
                    room=room,
                    content=content,
                    summary=summary,
                    type=memory_type,
                    confidence=confidence,
                    privacy=privacy,
                    scope=scope,
                    stored_at=now,
                    provenance=json.dumps(provenance) if provenance else "",
                ),
            ),
            SagaStep(
                "retriever",
                lambda: provider._retriever.add(
                    content,
                    memory_id=memory_id,
                    metadata={
                        "memory_id": memory_id,
                        "type": memory_type,
                        "confidence": confidence,
                        "scope": scope,
                        "privacy": privacy,
                        "wing": wing,
                        "room": room,
                        "stored_at": now,
                    },
                ),
            ),
            SagaStep(
                "knowledge_graph",
                lambda: (
                    provider._knowledge_graph.extract_and_store(
                        content, memory_id=memory_id, confidence=confidence / 5.0
                    )
                    if provider._knowledge_graph
                    else None
                ),
            ),
        ],
    )

    if not saga_result.success:
        logger.warning(
            "OmniMem memorize saga partial failure for %s at step '%s': %s",
            memory_id,
            saga_result.failed_step,
            saga_result.error,
        )

    # 记录溯源
    provider._provenance.record(memory_id, provenance)

    # 记录遗忘状态
    provider._forgetting.record_access(memory_id)

    # ★ R24修复EXT-5：写入后创建 event 记录，供 omni_detail(events) 查询
    try:
        provider._store.add(
            wing="auto",
            room=f"event-{memory_id[:8]}",
            content=f"[create] {content[:120]}",
            memory_type="event",
            confidence=1,
            privacy="personal",
            provenance={"trigger": "memorize", "source_memory_id": memory_id},
        )
    except (OSError, KeyError) as e:
        logger.debug("OmniMem event log creation failed: %s", e)

    # L3: 从 Saga 结果中获取知识图谱统计（避免重复执行）
    kg_stats = saga_result.step_results.get("knowledge_graph") or {}
    if not kg_stats and provider._knowledge_graph:
        # Fallback：Saga 未执行 kg 步骤时（如 kg 为 None 被跳过）单独提取
        try:
            kg_stats = provider._knowledge_graph.extract_and_store(
                content, memory_id=memory_id, confidence=confidence / 5.0
            )
        except (ValueError, RuntimeError) as e:
            logger.debug("KnowledgeGraph extraction failed: %s", e)

    # L3: 提交到 Consolidation 队列
    if provider._consolidation:
        provider._consolidation.submit(memory_id, content, memory_type=memory_type)

    # ★ 写入后主动矛盾扫描：检查同 type 下的记忆是否有矛盾
    post_conflict_info = None
    try:
        same_type = provider._store.search(memory_type=memory_type, limit=20)
        same_type = [m for m in same_type if m.get("memory_id", "") != memory_id][:10]
        if same_type:
            post_conflict = provider._conflict_resolver.check(
                content,
                existing_memories=[
                    {"content": m.get("content", ""), "memory_id": m.get("memory_id", "")}
                    for m in same_type
                ],
            )
            if post_conflict.has_conflict:
                post_conflict_info = {
                    "conflict_type": post_conflict.conflict_type,
                    "conflicting_with": post_conflict.existing_id,
                    "reason": f"Post-write conflict detected: {post_conflict.conflict_type}",
                }
                # 优先用 post_conflict_info（比写入前的更准确）
                conflict_info = post_conflict_info
    except (OSError, KeyError) as e:
        logger.debug("OmniMem post-write conflict scan failed: %s", e)

    # L4: 检查 KV Cache 自动预填充触发
    auto_preloaded = False
    if provider._kv_cache:
        auto_preloaded = provider._kv_cache.check_and_auto_preload(
            key=memory_id,
            content=content,
            metadata={"type": memory_type, "confidence": confidence, "wing": wing},
            source_memory_ids=[memory_id],
        )

    result = {
        "status": "stored",
        "memory_id": memory_id,
        "wing": wing,
        "room": room,
        "type": memory_type,
        "privacy": privacy,
        "kv_cached": auto_preloaded,
    }
    # ★ 冲突自动标记：写入存在冲突的记忆时，返回中包含冲突警告
    if conflict_info:
        result["conflict_warning"] = conflict_info
        logger.warning(
            "OmniMem: stored conflicting memory %s (conflicts with %s: %s)",
            memory_id,
            conflict_info["conflicting_with"],
            conflict_info["reason"],
        )

    provider._audit_logger.log(
        "memorize",
        memory_id=memory_id,
        details={"wing": wing, "room": room, "type": memory_type, "privacy": privacy},
        result="success",
        instance_id=getattr(provider, "_instance_id", None),
    )

    return json.dumps(result, ensure_ascii=False)
