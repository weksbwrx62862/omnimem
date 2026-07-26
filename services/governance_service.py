"""OmniMem 记忆治理 Service。

将 handlers/govern.py 中的核心业务逻辑下沉：
- 冲突扫描与解决
- 遗忘调度（archive / reactivate / status）
- 隐私审计（set_privacy / provenance / audit_log）
- 导入导出
- RBAC / KMS / 备份 / 索引重建等治理动作
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any

from omnimem.governance.conflict import ConflictResolver
from omnimem.handlers.deps import HandlerDependencies

logger = logging.getLogger(__name__)

# ★ Task 3.6: MinHash/LSH 粗筛
try:
    from datasketch import MinHash, MinHashLSH

    _HAS_DATASKETCH = True
except Exception:  # pragma: no cover
    _HAS_DATASKETCH = False
    MinHash = None  # type: ignore[misc, assignment]
    MinHashLSH = None  # type: ignore[misc, assignment]

_NEGATION_INDICATORS: tuple[str, ...] = (
    "不是",
    "不对",
    "并非",
    "不再",
    "改为",
    "而不是",
    "不用",
    "改用",
    "不要",
    "无法",
    "没能",
    "错误",
    "纠正",
    "not",
    "no longer",
    "instead of",
    "rather than",
)

# ★ R27优化：预编译正则
_CONFLICT_KEYWORD_RE = re.compile(r"[\u4e00-\u9fff]{2,4}|[a-zA-Z]{3,}")


class ActionRegistry:
    """治理动作注册表。"""

    def __init__(self) -> None:
        self._actions: dict[str, Callable[..., str]] = {}

    def register(self, name: str, handler: Callable[..., str]) -> None:
        self._actions[name] = handler

    def unregister(self, name: str) -> None:
        self._actions.pop(name, None)

    def execute(self, name: str, **kwargs: Any) -> str:
        handler = self._actions.get(name)
        if handler is None:
            return json.dumps(
                {"error": f"Unknown governance action: {name}", "available_actions": list(self._actions.keys())}
            )
        return handler(**kwargs)

    def list_actions(self) -> list[str]:
        return list(self._actions.keys())


def _shingle_text(text: str, k: int = 3) -> list[str]:
    """从文本生成字符级 shingles，用于 MinHash 签名。"""
    text = re.sub(r"\s+", "", text.lower())
    if len(text) <= k:
        return [text] if text else []
    return [text[i : i + k] for i in range(len(text) - k + 1)]


def _make_minhash(text: str, num_perm: int = 128) -> Any | None:
    """为文本构建 MinHash 签名。"""
    if not _HAS_DATASKETCH or MinHash is None:
        return None
    shingles = _shingle_text(text)
    if not shingles:
        return None
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def _lsh_candidate_pairs(
    memories: list[dict[str, Any]],
    threshold: float = 0.3,
    num_perm: int = 128,
) -> set[tuple[str, str]]:
    """使用 MinHash/LSH 粗筛潜在相似记忆对，避免 O(n²) 比较。"""
    if not _HAS_DATASKETCH or MinHashLSH is None or len(memories) < 2:
        return set()

    signatures: dict[str, Any] = {}
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    with lsh.insertion_session() as session:
        for m in memories:
            mid = m.get("memory_id", "")
            if not mid or mid in signatures:
                continue
            mh = _make_minhash(m.get("content", ""), num_perm=num_perm)
            if mh is None:
                continue
            signatures[mid] = mh
            session.insert(mid, mh)

    pairs: set[tuple[str, str]] = set()
    for mid, mh in signatures.items():
        for candidate in lsh.query(mh):
            if candidate == mid:
                continue
            pair = tuple(sorted((mid, candidate)))  # type: ignore[assignment]
            pairs.add(pair)  # type: ignore[arg-type]
    return pairs


def _scan_memory_conflicts(deps: HandlerDependencies) -> list[dict[str, Any]]:
    """主动扫描所有记忆，检测同主题的矛盾对。

    策略：对所有 fact/preference/correction 类型的记忆，
    用内容关键词做粗筛分组；组内使用 MinHash/LSH 做相似性粗筛，
    仅对候选对执行精细的冲突检测，避免 O(n²) 两两比较。
    """
    config = deps.config or {}
    all_memories = deps.store.search(limit=500)
    checkable = [
        m for m in all_memories if m.get("type", "") in ("fact", "preference", "correction")
    ]
    if len(checkable) < 2:
        return []

    groups: dict[str, list[dict[str, Any]]] = {}
    if len(checkable) <= 4:
        groups["_all"] = list(checkable)
    else:
        for m in checkable:
            content = m.get("content", "")
            keywords = _CONFLICT_KEYWORD_RE.findall(content)
            key = "|".join(keywords[:2]) if keywords else "_other"
            groups.setdefault(key, []).append(m)

    conflicts = []
    resolver = ConflictResolver(strategy="latest")
    max_size = 50
    try:
        raw_max = config.get("conflict_scan_max_group_size", 50) if config else 50
        if isinstance(raw_max, (int, float)) and not isinstance(raw_max, bool) or isinstance(raw_max, str):
            max_size = int(raw_max)
    except Exception:
        max_size = 50

    lsh_small_group_threshold = 8

    for key, group in groups.items():
        if len(group) < 2:
            continue
        if len(group) > max_size:
            group = sorted(
                group,
                key=lambda m: m.get("stored_at", m.get("memory_id", "")),
                reverse=True,
            )[:max_size]

        candidate_pairs: set[tuple[int, int]]
        if len(group) <= lsh_small_group_threshold or not _HAS_DATASKETCH:
            candidate_pairs = {
                (i, j) for i in range(len(group)) for j in range(i + 1, len(group))
            }
        else:
            lsh_pairs = _lsh_candidate_pairs(group)
            mid_to_idx = {m.get("memory_id", ""): i for i, m in enumerate(group)}
            candidate_pairs = set()
            for mid_a, mid_b in lsh_pairs:
                idx_a = mid_to_idx.get(mid_a)
                idx_b = mid_to_idx.get(mid_b)
                if idx_a is not None and idx_b is not None:
                    i, j = sorted((idx_a, idx_b))
                    candidate_pairs.add((i, j))
            if not candidate_pairs:
                candidate_pairs = {
                    (i, j) for i in range(len(group)) for j in range(i + 1, len(group))
                }

        for i, j in candidate_pairs:
            a, b = group[i], group[j]
            ca = a.get("content", "").lower()
            cb = b.get("content", "").lower()
            result = resolver.check(
                ca,
                existing_memories=[{"content": cb, "memory_id": b.get("memory_id", "")}],
            )
            if result.has_conflict:
                a_has_neg = any(ni in ca for ni in _NEGATION_INDICATORS)
                any(ni in cb for ni in _NEGATION_INDICATORS)
                overlap = ConflictResolver._compute_overlap(ca, cb)
                conflicts.append(
                    {
                        "memory_a": {
                            "memory_id": a.get("memory_id", ""),
                            "content": a.get("content", "")[:100],
                            "type": a.get("type", ""),
                        },
                        "memory_b": {
                            "memory_id": b.get("memory_id", ""),
                            "content": b.get("content", "")[:100],
                            "type": b.get("type", ""),
                        },
                        "overlap": round(overlap, 2),
                        "negation_in": "a" if a_has_neg else "b",
                        "conflict_type": result.conflict_type,
                    }
                )
    return conflicts


class GovernanceService:
    """记忆治理服务实现。"""

    def __init__(self, deps: HandlerDependencies) -> None:
        self.deps = deps
        self._registry = ActionRegistry()
        self._register_actions()

    def _register_actions(self) -> None:
        """注册所有治理动作。"""
        registry = self._registry
        registry.register("resolve_conflict", self._action_resolve_conflict)
        registry.register("scan_conflicts", self._action_scan_conflicts)
        registry.register("set_privacy", self._action_set_privacy)
        registry.register("archive", self._action_archive)
        registry.register("reactivate", self._action_reactivate)
        registry.register("provenance", self._action_provenance)
        registry.register("forgetting_status", self._action_forgetting_status)
        registry.register("lora_train", self._action_lora_train)
        registry.register("export_training_data", self._action_export_training_data)
        registry.register("register_adapter", self._action_register_adapter)
        registry.register("shade_switch", self._action_shade_switch)
        registry.register("shade_list", self._action_shade_list)
        registry.register("kv_cache_stats", self._action_kv_cache_stats)
        registry.register("consolidation_stats", self._action_consolidation_stats)
        registry.register("sync_status", self._action_sync_status)
        registry.register("sync_instances", self._action_sync_instances)
        registry.register("export_memories", self._action_export_memories)
        registry.register("import_memories", self._action_import_memories)
        registry.register("audit_log", self._action_audit_log)
        registry.register("assign_role", self._action_assign_role)
        registry.register("revoke_role", self._action_revoke_role)
        registry.register("add_role", self._action_add_role)
        registry.register("check_permission", self._action_check_permission)
        registry.register("get_permissions", self._action_get_permissions)
        registry.register("configure_kms", self._action_configure_kms)
        registry.register("rotate_key", self._action_rotate_key)
        registry.register("reencrypt", self._action_reencrypt)
        registry.register("kms_status", self._action_kms_status)
        registry.register("rebuild_index", self._action_rebuild_index)
        registry.register("purge_test_data", self._action_purge_test_data)
        registry.register("wiki_upgrade", self._action_wiki_upgrade)
        registry.register("mark_wiki", self._action_mark_wiki)
        registry.register("forgetting_heat", self._action_forgetting_heat)
        registry.register("tree", self._action_tree)
        registry.register("grep_rooms", self._action_grep_rooms)
        registry.register("count", self._action_count)
        registry.register("backup", self._action_backup)

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def handle(self, args: dict[str, Any]) -> str:
        """处理 omni_govern 工具调用，返回 JSON 字符串。"""
        action = args["action"]
        target = args.get("target", "") or args.get("target_id", "")
        params = args.get("params", {})

        combined = {k: v for k, v in args.items() if k not in ("params", "action")}
        combined.update(params)
        combined["target"] = target
        combined["_original_params"] = params

        result = self._registry.execute(action, params=combined)
        if isinstance(result, dict):
            return json.dumps(result)
        return result

    # ------------------------------------------------------------------
    # 动作实现
    # ------------------------------------------------------------------
    def _action_resolve_conflict(self, params: dict[str, Any]) -> str:
        """冲突解决：无 target 时全局扫描，有 target 时检查指定记忆。"""
        target = params.get("target", "")
        if not target or not target.strip():
            scan_results = _scan_memory_conflicts(self.deps)
            if scan_results:
                archived_ids = []
                for pair in scan_results[:5]:
                    old_id = pair.get("memory_b", {}).get("memory_id") or pair.get("memory_a", {}).get(
                        "memory_id"
                    )
                    if old_id:
                        self.deps.forgetting.archive(old_id)
                        archived_ids.append(old_id)
                return json.dumps(
                    {
                        "status": "conflicts_found",
                        "action_taken": "archived_old_entries",
                        "reason": f"Found {len(scan_results)} conflicting pairs, archived {len(archived_ids)} old entries",
                        "conflicts": scan_results[:5],
                        "archived": archived_ids,
                    }
                )
            return json.dumps(
                {
                    "status": "no_conflict",
                    "reason": "No conflicting memories found (global scan)",
                }
            )

        target_entry = self.deps.store.get(target)
        if not target_entry:
            return json.dumps({"status": "error", "reason": f"Memory {target} not found"})

        target_content = target_entry.get("content", "")

        # 优先使用 conflict_warning 中记录的冲突对
        conflict_with_id = target_entry.get("conflicting_with", "") or ""
        if not conflict_with_id:
            index_entry = self.deps.index.get(target)
            if index_entry and isinstance(index_entry, dict):
                conflict_with_id = index_entry.get("conflicting_with", "") or ""
        if conflict_with_id:
            conflict_entry = self.deps.store.get(conflict_with_id)
            if conflict_entry:
                conflict = self.deps.conflict_resolver.check(
                    target_content,
                    existing_memories=[
                        {"content": conflict_entry.get("content", ""), "memory_id": conflict_with_id}
                    ],
                )
                if conflict.has_conflict:
                    resolution = self.deps.conflict_resolver.resolve(target_content, conflict)
                    old_id = conflict.existing_id
                    if old_id and old_id != target:
                        self.deps.forgetting.archive(old_id)
                    return json.dumps(
                        {
                            "status": "resolved",
                            "action_taken": resolution.action,
                            "reason": resolution.reason,
                            "target": target,
                            "conflicting_with": old_id,
                            "source": "conflict_warning_direct",
                        }
                    )

        # Fallback：语义搜索路径
        try:
            semantic_candidates = []
            if self.deps.unified_candidate_search:
                try:
                    semantic_candidates = self.deps.unified_candidate_search(target_content)
                except Exception:
                    logger.warning("Govern: unified candidate search failed, using empty list", exc_info=True)
                    semantic_candidates = []
            all_memories = self.deps.store.search(limit=100)
            simple_candidates = [
                m
                for m in all_memories
                if m.get("memory_id", "") != target
                and m.get("type", "") in ("fact", "preference", "correction")
            ]
            seen_ids = set()
            merged_candidates = []
            for m in semantic_candidates + simple_candidates:
                mid = m.get("memory_id", "")
                if mid and mid != target and mid not in seen_ids:
                    seen_ids.add(mid)
                    merged_candidates.append({"content": m.get("content", ""), "memory_id": mid})

            conflict = self.deps.conflict_resolver.check(
                target_content, existing_memories=merged_candidates
            )

            if conflict.has_conflict:
                resolution = self.deps.conflict_resolver.resolve(target_content, conflict)
                old_id = conflict.existing_id
                if old_id and old_id != target:
                    self.deps.forgetting.archive(old_id)
                    logger.warning("OmniMem resolve_conflict: archived old entry %s", old_id)
                return json.dumps(
                    {
                        "status": "resolved",
                        "action_taken": resolution.action,
                        "reason": resolution.reason,
                        "conflicting_with": old_id,
                        "conflict_type": conflict.conflict_type,
                        "archived_old": old_id if old_id and old_id != target else None,
                    }
                )
        except (ValueError, KeyError) as e:
            logger.warning("OmniMem conflict detection failed: %s", e)

        # fallback: 全局否定词扫描
        scan_results = _scan_memory_conflicts(self.deps)
        target_conflicts = [
            c
            for c in scan_results
            if c.get("memory_a", {}).get("memory_id") == target
            or c.get("memory_b", {}).get("memory_id") == target
        ]
        if target_conflicts:
            return json.dumps(
                {
                    "status": "conflicts_found",
                    "action_taken": "pending",
                    "reason": f"Found {len(target_conflicts)} conflicting pairs",
                    "conflicts": target_conflicts[:3],
                }
            )

        return json.dumps(
            {
                "status": "no_conflict",
                "reason": "No conflicting memories found for this target",
                "memory_id": target,
            }
        )

    def _action_scan_conflicts(self, params: dict[str, Any]) -> str:
        """主动扫描矛盾记忆。"""
        conflicts = _scan_memory_conflicts(self.deps)
        return json.dumps(
            {
                "status": "scanned",
                "conflict_count": len(conflicts),
                "conflicts": conflicts[:10],
            },
            ensure_ascii=False,
        )

    def _action_set_privacy(self, params: dict[str, Any]) -> str:
        """设置隐私级别：同步更新 index/store/wing。"""
        target = params.get("target", "")
        level = params.get("level", params.get("privacy", "personal"))
        existing = self.deps.store.get(target)
        existing_type = existing.get("type", existing.get("memory_type", "")) if existing else ""
        new_wing = self.deps.wing_room.resolve_wing_from_privacy(level, existing_type)
        self.deps.privacy.set(target, level, new_wing=new_wing)
        self.deps.index.update_privacy(target, level)
        self.deps.store.update_privacy(target, level, new_wing=new_wing)
        self.deps.index.update_field(target, wing=new_wing, immediate=True)
        # 同步更新检索索引中的 metadata
        if existing and self.deps.retriever:
            try:
                content = existing.get("content", "")
                if content:
                    self.deps.retriever.update_metadata(
                        target,
                        {
                            "_content": content,
                            "memory_id": target,
                            "type": existing_type,
                            "wing": new_wing,
                            "privacy": level,
                        },
                    )
            except Exception as e:
                logger.warning("OmniMem set_privacy retriever sync failed: %s", e)
        # 验证
        verify = self.deps.store.get(target)
        actual_privacy = verify.get("privacy", "personal") if verify else "unknown"
        actual_wing = verify.get("wing", "personal") if verify else "unknown"
        if actual_wing != new_wing:
            logger.warning(
                "govern set_privacy wing mismatch: expected=%s actual=%s for %s",
                new_wing,
                actual_wing,
                target,
            )
        if self.deps.audit_logger:
            self.deps.audit_logger.log(
                "govern_set_privacy",
                memory_id=target,
                details={"privacy": actual_privacy, "wing": actual_wing},
                result="success",
                instance_id=self.deps.instance_id,
            )
        return json.dumps(
            {
                "status": "updated",
                "memory_id": target,
                "privacy": actual_privacy,
                "wing": actual_wing,
            }
        )

    def _action_archive(self, params: dict[str, Any]) -> str:
        """封存记忆。"""
        target = params.get("target", "")
        dry_run = params.get("dry_run", False)
        if dry_run:
            return json.dumps(
                {
                    "status": "dry_run",
                    "action": "archive",
                    "target": target,
                    "hint": "Set dry_run=false to actually execute",
                }
            )
        self.deps.forgetting.archive(target)
        result = {"status": "sealed", "memory_id": target, "note": "数据保留，召回时降权显示"}
        if self.deps.audit_logger:
            self.deps.audit_logger.log(
                "govern_archive", memory_id=target, result="sealed", instance_id=self.deps.instance_id
            )
        return json.dumps(result)

    def _action_reactivate(self, params: dict[str, Any]) -> str:
        """重新激活已归档的记忆。"""
        target = params.get("target", "")
        dry_run = params.get("dry_run", False)
        if dry_run:
            return json.dumps(
                {
                    "status": "dry_run",
                    "action": "reactivate",
                    "target": target,
                    "hint": "Set dry_run=false to actually execute",
                }
            )
        self.deps.forgetting.reactivate(target)
        if self.deps.audit_logger:
            self.deps.audit_logger.log(
                "govern_reactivate",
                memory_id=target,
                result="success",
                instance_id=self.deps.instance_id,
            )
        return json.dumps({"status": "reactivated", "memory_id": target})

    def _action_provenance(self, params: dict[str, Any]) -> str:
        """查询记忆溯源信息。"""
        target = params.get("target", "")
        prov = self.deps.provenance.lookup(target)
        return json.dumps({"status": "found", "provenance": prov})

    def _action_forgetting_status(self, params: dict[str, Any]) -> str:
        """查看遗忘曲线状态并返回中文报告。"""
        status = self.deps.forgetting.get_status()
        stages = status.get("stages", {})
        heat = status.get("heat", {})
        upgrade_candidates = status.get("upgrade_candidates", [])
        total_memories = sum(stages.values())

        report_lines = [
            "📊 OmniMem 遗忘曲线审计报告",
            "=" * 50,
            "",
            "📋 记忆阶段统计",
            "-" * 30,
            f"  活跃 (active): {stages.get('active', 0)} 条",
            f"  巩固中 (consolidating): {stages.get('consolidating', 0)} 条",
            f"  已归档 (archived): {stages.get('archived', 0)} 条",
            f"  已遗忘 (forgotten): {stages.get('forgotten', 0)} 条",
            f"  总计: {total_memories} 条",
            "",
            "🔥 热度分布",
            "-" * 30,
            f"  热门 (hot): {heat.get('hot', 0)} 条",
            f"  温热 (warm): {heat.get('warm', 0)} 条",
            f"  中性 (neutral): {heat.get('neutral', 0)} 条",
            f"  冷却 (cold): {heat.get('cold', 0)} 条",
            "",
            "⬆️ 升级候选",
            "-" * 30,
            f"  候选数量: {len(upgrade_candidates)} 条",
        ]

        if upgrade_candidates:
            report_lines.append("  Top 5 高频记忆:")
            for i, candidate in enumerate(upgrade_candidates[:5], 1):
                mem_id = candidate.get("memory_id", "unknown")
                recall_count = candidate.get("recall_count", 0)
                heat_level = candidate.get("heat", "unknown")
                report_lines.append(
                    f"    {i}. {mem_id[:8]}... | 召回 {recall_count} 次 | 热度: {heat_level}"
                )

        report_lines.append("")
        report_lines.append("=" * 50)

        report_text = "\n".join(report_lines)
        return json.dumps({"status": "ok", "forgetting": status, "report": report_text}, ensure_ascii=False)

    def _action_lora_train(self, params: dict[str, Any]) -> str:
        """L4: 触发 LoRA 训练。"""
        if not self.deps.lora_trainer:
            return json.dumps({"error": "LoRA trainer not available"})
        try:
            shade = params.get("shade", "default")
            epochs = params.get("epochs", 3)
            result = self.deps.lora_trainer.train(shade=shade, epochs=epochs)
            return json.dumps(result)
        except (RuntimeError, AttributeError) as e:
            logger.warning("OmniMem lora_train failed: %s", e)
            return json.dumps({"status": "error", "reason": f"LoRA train failed: {e}"})

    def _action_export_training_data(self, params: dict[str, Any]) -> str:
        """★ M4: L4 三段式第 1 段 — 导出训练数据为 alpaca JSONL。"""
        if not self.deps.lora_trainer:
            return json.dumps({"error": "LoRA trainer not available"})
        try:
            result = self.deps.lora_trainer.export_training_jsonl(
                output_path=params.get("output_path") or None,
                shade=params.get("shade", "all"),
                include_used=bool(params.get("include_used", True)),
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.warning("OmniMem export_training_data failed: %s", e)
            return json.dumps({"status": "error", "reason": f"export failed: {e}"})

    def _action_reencrypt(self, params: dict[str, Any]) -> str:
        """★ M8-18: 批量将 secret 记忆密文升级为 V2 AES-GCM 格式。

        params:
            dry_run: 仅统计待升级数量，不实际重写
            limit: 扫描索引的最大条数（默认 10000）
        """
        dry_run = bool(params.get("dry_run", False))
        limit = int(params.get("limit", 10000) or 10000)
        try:
            entries = self.deps.index.search_l1(limit=limit)
        except Exception as e:
            return json.dumps({"status": "error", "reason": f"index scan failed: {e}"})
        secret_ids = [e["memory_id"] for e in entries if e.get("privacy") == "secret"]
        stats: dict[str, Any] = {
            "total_secret": len(secret_ids),
            "upgraded": 0, "already_current": 0, "skipped": 0, "failed": 0,
        }
        if dry_run:
            return json.dumps({"status": "dry_run", **stats}, ensure_ascii=False)
        for mid in secret_ids:
            try:
                r = self.deps.store.reencrypt_secret(mid)
            except Exception as e:
                logger.warning("reencrypt %s failed: %s", mid, e)
                r = "error"
            if r == "upgraded":
                stats["upgraded"] += 1
            elif r == "already_current":
                stats["already_current"] += 1
            elif r in ("not_secret", "not_found"):
                stats["skipped"] += 1
            else:
                stats["failed"] += 1
        # 审计：重加密属敏感批量操作
        if getattr(self.deps, "audit_logger", None):
            self.deps.audit_logger.log("reencrypt", details=stats)
        return json.dumps({"status": "ok", **stats}, ensure_ascii=False)

    def _action_register_adapter(self, params: dict[str, Any]) -> str:
        """★ M4: L4 三段式第 3 段 — 回注外部训练完成的 LoRA 适配器。"""
        if not self.deps.lora_trainer:
            return json.dumps({"error": "LoRA trainer not available"})
        adapter_path = params.get("adapter_path", "") or params.get("target", "")
        if not adapter_path:
            return json.dumps({"status": "error", "reason": "adapter_path is required"})
        try:
            result = self.deps.lora_trainer.register_external_adapter(
                adapter_path=adapter_path,
                shade=params.get("shade", "default"),
                base_model=params.get("base_model", ""),
                training_samples=int(params.get("training_samples", 0) or 0),
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.warning("OmniMem register_adapter failed: %s", e)
            return json.dumps({"status": "error", "reason": f"register failed: {e}"})

    def _action_shade_switch(self, params: dict[str, Any]) -> str:
        """L4: 切换 LoRA shade。"""
        target = params.get("target", "")
        dry_run = params.get("dry_run", False)
        if dry_run:
            shade_name = params.get("shade") or target or "default"
            return json.dumps(
                {
                    "status": "dry_run",
                    "action": "shade_switch",
                    "target": shade_name,
                    "hint": "Set dry_run=false to actually execute",
                }
            )
        if not self.deps.lora_trainer:
            return json.dumps({"status": "error", "reason": "LoRA trainer not available"})
        try:
            shade_name = params.get("shade") or target or "default"
            available_shades = [s["name"] for s in self.deps.lora_trainer.get_shades()]
            if shade_name not in available_shades:
                self.deps.lora_trainer.register_shade(shade_name, f"自定义模式：{shade_name}")
            result = self.deps.lora_trainer.switch_shade(shade_name)
            if result is None:
                result = {"status": "switched", "shade": shade_name}
            verify_shade = self.deps.lora_trainer.active_shade
            result["verified_active"] = verify_shade
            if verify_shade != shade_name:
                result["status"] = "error"
                result["message"] = f"Switch failed: expected {shade_name}, got {verify_shade}"
            return json.dumps(result)
        except Exception as e:
            logger.warning("OmniMem shade_switch failed: %s", e)
            return json.dumps({"status": "error", "reason": f"Shade switch failed: {e}"})

    def _action_shade_list(self, params: dict[str, Any]) -> str:
        """L4: 列出所有可用 shade。"""
        if not self.deps.lora_trainer:
            return json.dumps({"error": "LoRA trainer not available"})
        shades = self.deps.lora_trainer.get_shades()
        return json.dumps({"status": "ok", "shades": shades})

    def _action_kv_cache_stats(self, params: dict[str, Any]) -> str:
        """L4: 查看 KV Cache 统计。"""
        if not self.deps.kv_cache:
            return json.dumps({"error": "KV Cache not available"})
        stats = self.deps.kv_cache.get_stats()
        return json.dumps({"status": "ok", "kv_cache": stats})

    def _action_consolidation_stats(self, params: dict[str, Any]) -> str:
        """L3: 查看 Consolidation 统计。"""
        if not self.deps.consolidation:
            return json.dumps({"error": "Consolidation not available"})
        stats = self.deps.consolidation.get_stats()
        return json.dumps({"status": "ok", "consolidation": stats})

    def _action_sync_status(self, params: dict[str, Any]) -> str:
        """查看同步引擎状态。"""
        if not self.deps.sync_engine:
            return json.dumps({"error": "Sync engine not available"})
        info = self.deps.sync_engine.get_instance_info()
        return json.dumps({"status": "ok", "sync": info})

    def _action_sync_instances(self, params: dict[str, Any]) -> str:
        """查看活跃同步实例。"""
        if not self.deps.sync_engine:
            return json.dumps({"error": "Sync engine not available"})
        instances = self.deps.sync_engine.get_active_instances()
        return json.dumps({"status": "ok", "instances": instances})

    def _action_export_memories(self, params: dict[str, Any]) -> str:
        """导出记忆到文件。默认使用 Fernet+SHA-256 加密，未配置密钥时拒绝导出。"""
        from omnimem.core.import_export import MemoryExporter

        output_path = params.get("output_path", "")
        if not output_path:
            return json.dumps({"error": "output_path is required for export_memories"})

        # 优先级：params.encryption_key > config.export_key > OMNIMEM_EXPORT_KEY 环境变量
        export_key = (
            params.get("encryption_key")
            or (self.deps.config.get("export_key") if self.deps.config else None)
            or os.environ.get("OMNIMEM_EXPORT_KEY", "")
        )
        if not export_key:
            return json.dumps(
                {
                    "error": (
                        "未配置导出密钥：请在 config 中设置 export_key "
                        "或设置环境变量 OMNIMEM_EXPORT_KEY"
                    )
                }
            )

        fmt = params.get("format", "json")
        exporter = MemoryExporter(self.deps.store, self.deps.index, self.deps.store.meta_store)
        try:
            if fmt == "markdown":
                count = exporter.export_markdown(output_path, wing=params.get("wing"))
            else:
                count = exporter.export_json(
                    output_path,
                    wing=params.get("wing"),
                    memory_type=params.get("memory_type"),
                    encryption_key=export_key,
                )
            return json.dumps({"status": "exported", "count": count, "path": str(output_path)})
        except Exception as e:
            return json.dumps({"error": f"Export failed: {e}"})

    def _action_import_memories(self, params: dict[str, Any]) -> str:
        """从文件导入记忆。调用 Fernet+SHA-256 解密/校验路径，HMAC 失败则拒绝导入。"""
        from omnimem.core.import_export import MemoryImporter

        input_path = params.get("input_path", "")
        if not input_path:
            return json.dumps({"error": "input_path is required for import_memories"})

        # 优先级：params.encryption_key > config.export_key > OMNIMEM_EXPORT_KEY 环境变量
        export_key = (
            params.get("encryption_key")
            or (self.deps.config.get("export_key") if self.deps.config else None)
            or os.environ.get("OMNIMEM_EXPORT_KEY", "")
        )

        skip_dup = params.get("skip_duplicates", True)
        resolve_conf = params.get("resolve_conflicts", True)
        importer = MemoryImporter(
            self.deps.store,
            self.deps.index,
            self.deps.retriever,
            self.deps.dedup,
            self.deps.conflict_resolver,
            self.deps.forgetting,
        )
        try:
            result = importer.import_json(
                input_path,
                skip_duplicates=skip_dup,
                resolve_conflicts=resolve_conf,
                encryption_key=export_key,
            )
            return json.dumps({"status": "imported", **result})
        except Exception as e:
            return json.dumps({"error": f"Import failed: {e}"})

    def _action_audit_log(self, params: dict[str, Any]) -> str:
        """查询审计日志。"""
        target = params.get("target", "")
        operation = params.get("operation")
        memory_id = params.get("memory_id") or target or None
        from_time = params.get("from_time")
        to_time = params.get("to_time")
        limit = params.get("limit", 100)
        try:
            entries = self.deps.audit_logger.query(
                operation=operation,
                memory_id=memory_id,
                from_time=from_time,
                to_time=to_time,
                limit=limit,
            )
            return json.dumps({"status": "ok", "count": len(entries), "entries": entries}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Audit log query failed: {e}"})

    def _action_assign_role(self, params: dict[str, Any]) -> str:
        """为用户分配角色。"""
        caller_id = params.get("caller_id", "default")
        if self.deps.rbac is not None:
            if not self.deps.rbac.check_permission(caller_id, "govern"):
                return json.dumps({"status": "blocked", "reason": "govern permission required"})
        user_id = params.get("user_id", "default")
        role_name = params.get("role_name", "")
        if not role_name:
            return json.dumps({"error": "role_name is required"})
        self.deps.rbac.assign_role(user_id, role_name)
        return json.dumps({"status": "assigned", "user_id": user_id, "role": role_name})

    def _action_revoke_role(self, params: dict[str, Any]) -> str:
        """撤销用户角色。"""
        caller_id = params.get("caller_id", "default")
        if self.deps.rbac is not None:
            if not self.deps.rbac.check_permission(caller_id, "govern"):
                return json.dumps({"status": "blocked", "reason": "govern permission required"})
        user_id = params.get("user_id", "default")
        role_name = params.get("role_name", "")
        if not role_name:
            return json.dumps({"error": "role_name is required"})
        self.deps.rbac.revoke_role(user_id, role_name)
        return json.dumps({"status": "revoked", "user_id": user_id, "role": role_name})

    def _action_add_role(self, params: dict[str, Any]) -> str:
        """创建新角色。"""
        caller_id = params.get("caller_id", "default")
        if self.deps.rbac is not None:
            if not self.deps.rbac.check_permission(caller_id, "govern"):
                return json.dumps({"status": "blocked", "reason": "govern permission required"})
        role_name = params.get("role_name", "")
        permissions = params.get("permissions", [])
        if not role_name:
            return json.dumps({"error": "role_name is required"})
        self.deps.rbac.add_role(role_name, permissions)
        return json.dumps({"status": "created", "role": role_name, "permissions": permissions})

    def _action_check_permission(self, params: dict[str, Any]) -> str:
        """检查用户权限。"""
        user_id = params.get("user_id", "default")
        permission = params.get("permission", "")
        if not permission:
            return json.dumps({"error": "permission is required"})
        allowed = self.deps.rbac.check_permission(user_id, permission)
        return json.dumps(
            {"status": "ok", "user_id": user_id, "permission": permission, "allowed": allowed}
        )

    def _action_get_permissions(self, params: dict[str, Any]) -> str:
        """获取用户权限列表。"""
        user_id = params.get("user_id", "default")
        permissions = self.deps.rbac.get_user_permissions(user_id)
        return json.dumps({"status": "ok", "user_id": user_id, "permissions": permissions})

    def _action_configure_kms(self, params: dict[str, Any]) -> str:
        """配置 KMS 提供商。"""
        provider_name = params.get("provider", "local")
        raw_params = params.get("_original_params", params)
        config_kwargs = {k: v for k, v in raw_params.items() if k != "provider"}
        try:
            self.deps.kms.configure_provider(provider_name, **config_kwargs)
            if self.deps.audit_logger:
                self.deps.audit_logger.log(
                    "govern_configure_kms",
                    details={"provider": provider_name},
                    result="success",
                    instance_id=self.deps.instance_id,
                )
            return json.dumps({"status": "configured", "provider": provider_name})
        except ValueError as e:
            return json.dumps({"error": str(e)})

    def _action_rotate_key(self, params: dict[str, Any]) -> str:
        """轮换密钥。"""
        key_id = params.get("key_id", "default")
        self.deps.kms.rotate_key(key_id)
        if self.deps.audit_logger:
            self.deps.audit_logger.log(
                "govern_rotate_key",
                details={"key_id": key_id},
                result="success",
                instance_id=self.deps.instance_id,
            )
        return json.dumps({"status": "rotated", "key_id": key_id})

    def _action_kms_status(self, params: dict[str, Any]) -> str:
        """查看 KMS 状态。"""
        return json.dumps(
            {
                "status": "ok",
                "provider": self.deps.kms.provider,
                "config": self.deps.kms._config,
            }
        )

    def _action_rebuild_index(self, params: dict[str, Any]) -> str:
        """全量重建向量+BM25检索索引。"""
        try:
            entries = self.deps.index.search_all_for_retrieval(limit=5000)
            if not entries:
                return json.dumps({"status": "no_data", "reason": "No entries in index to rebuild"})
            stats = self.deps.retriever.rebuild_all_from_entries(entries)
            return json.dumps(
                {
                    "status": "rebuilt",
                    "entries_processed": len(entries),
                    "vector_rebuilt": stats.get("vector", 0),
                    "bm25_rebuilt": stats.get("bm25", 0),
                }
            )
        except Exception as e:
            logger.warning("OmniMem rebuild_index failed: %s", e)
            return json.dumps({"status": "error", "reason": str(e)})

    def _action_purge_test_data(self, params: dict[str, Any]) -> str:
        """清理测试残留数据。"""
        dry_run = params.get("dry_run", True)
        test_patterns = params.get("patterns", ["test_", "QUAL-", "BUG-", "zzzzz", "mock"])
        purged = []
        all_entries = self.deps.index.search_all_for_retrieval(limit=5000)
        for entry in all_entries:
            content = entry.get("content", "")
            memory_id = entry.get("memory_id", "")
            if any(p in content for p in test_patterns):
                purged.append({"memory_id": memory_id, "content_preview": content[:80]})
                if not dry_run:
                    try:
                        self.deps.forgetting.archive(memory_id)
                        self.deps.index.delete(memory_id)
                        if self.deps.retriever:
                            self.deps.retriever.update_metadata(
                                memory_id, {"_content": "", "memory_id": memory_id}
                            )
                    except Exception as e:
                        logger.warning("purge_test_data delete failed for %s: %s", memory_id, e)
        return json.dumps(
            {
                "status": "dry_run" if dry_run else "purged",
                "test_entries_found": len(purged),
                "entries": purged[:20],
                "hint": "Set dry_run=false to actually delete" if dry_run else None,
            },
            ensure_ascii=False,
        )

    def _action_wiki_upgrade(self, params: dict[str, Any]) -> str:
        """获取升级候选并执行升级。"""
        candidates = self.deps.forgetting.get_upgrade_candidates()
        if not candidates:
            return json.dumps({"status": "no_candidates", "message": "No memories eligible for Wiki upgrade."})
        return json.dumps(
            {
                "status": "candidates",
                "count": len(candidates),
                "candidates": candidates[:10],
                "hint": "Use llm-wiki skill to create Wiki pages for these memories.",
            },
            ensure_ascii=False,
        )

    def _action_mark_wiki(self, params: dict[str, Any]) -> str:
        """标记记忆已升级到 Wiki。"""
        mid = params.get("memory_id") or params.get("target", "")
        path = params.get("wiki_page_path", "")
        if not mid:
            return json.dumps({"status": "error", "message": "memory_id required"})
        try:
            self.deps.forgetting.mark_upgraded_to_wiki(mid, path)
            return json.dumps({"status": "marked", "memory_id": mid, "wiki_page_path": path})
        except Exception as e:
            return json.dumps({"status": "error", "reason": str(e)})

    def _action_forgetting_heat(self, params: dict[str, Any]) -> str:
        """返回热度分类统计。"""
        status = self.deps.forgetting.get_status()
        heat = status.get("heat", {})
        return json.dumps({"status": "ok", "heat": heat})

    def _action_tree(self, params: dict[str, Any]) -> str:
        """展示记忆目录树。"""
        wing = params.get("wing", "")
        hall = params.get("hall", "")
        tree_data = self.deps.wing_room.tree(wing=wing, hall=hall)
        return json.dumps({"status": "ok", "tree": tree_data}, ensure_ascii=False)

    def _action_grep_rooms(self, params: dict[str, Any]) -> str:
        """搜索 Room 名称。"""
        pattern = params.get("pattern", "")
        if not pattern:
            return json.dumps({"status": "error", "message": "pattern is required"})
        rooms = self.deps.wing_room.grep_rooms(pattern)
        return json.dumps({"status": "ok", "rooms": rooms}, ensure_ascii=False)

    def _action_count(self, params: dict[str, Any]) -> str:
        """统计各目录记忆数量。"""
        wing = params.get("wing", "")
        hall = params.get("hall", "")
        room = params.get("room", "")
        counts = self.deps.wing_room.count_memories(wing=wing, hall=hall, room=room)
        return json.dumps({"status": "ok", "counts": counts}, ensure_ascii=False)

    def _action_backup(self, params: dict[str, Any]) -> str:
        """手动触发数据目录备份。"""
        try:
            backup_path, backup_size = self.deps.create_backup() if self.deps.create_backup else (None, 0)
            backup_max_copies = self.deps.config.get("backup_max_copies", 3)
            if self.deps.cleanup_old_backups:
                self.deps.cleanup_old_backups(backup_max_copies)
            return json.dumps(
                {
                    "status": "ok",
                    "backup_path": backup_path,
                    "backup_size_kb": round(backup_size / 1024, 1),
                }
            )
        except Exception as e:
            logger.warning("OmniMem 手动备份失败: %s", e)
            return json.dumps({"status": "error", "reason": str(e)})
