"""★ P1-1: 以磁盘为事实来源（SSOT）的 index.db 对账。

线上实况（2026-09-29 报告）：3 135 个 drawer / 712 行 index / 交集 539，
即 **2 596 条记忆有抽屉、无索引行** —— 关键词与语义检索都召不回它们，而系统里
没有任何机制会发现这种缺口：写抽屉成功、写 index 失败被 `except` 吞掉了（P1-3/P1-4
已止血，存量还得靠对账清）。

为什么不复用 Auditor：``governance/auditor.py`` 以 **MetaStore** 为 SSOT
（``run_full_audit`` 先取 ``meta_store.get_all()``）。MetaStore 自身也是并行双写的
一方，降级路径下同样会缺行，拿它当基准就修不出「磁盘有、哪都缺」的那批。
drawer 文件是原始内容的冷备份，文件在记忆就在，所以这里以文件名为准 ——
``drawer/<12hex>.md`` 的 stem 就是 ``memory_id``。

★ 例外（P1-10）：抽屉在磁盘上 ≠ 该被召回。遗忘曲线归档/遗忘一条记忆时只改
``forgetting_state.stage``，抽屉留待后续物理删除，而 ``GovernanceAuditor`` 每次开机会把
它们的 index 行删掉。对账必须把这批排除，否则就是"把用户忘掉的东西重新召回"+ 与审计器
拉锯。排除依据读 ``governance/forgetting.db`` 的 ``forgetting_state``；读不到时不排除，
但会在 summary 里如实说明（见 :func:`archived_memory_ids`）。

★ 例外（合成评测语料）：``provenance.source`` 形如 ``api-<hash>`` 的抽屉来自 LongMemEval
式评测会话，内容是 harness 提示词而非用户记忆。评测跑一次就多一批抽屉，磁盘上不会自己
消失，所以每次对账都要重新筛掉（见 :func:`is_synthetic_source`）—— 只在事后清 index.db
是不够的，下一次 ``--apply`` 会把同一批原样灌回来。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omnimem.memory.markdown_store import MarkdownStore

logger = logging.getLogger(__name__)

# index.db 全表扫描用，够覆盖单机记忆量（search_all_for_retrieval 是 LIMIT 查询）
_FULL_SCAN_LIMIT = 10_000_000

# ★ 合成评测会话的 provenance.source 前缀。LongMemEval 式评测跑出来的会话 id 形如
# ``api-<hash>``（state.db 里 2 648 个，source 全是 api_server），真实用户会话一律是
# ``<日期>_<序号>_<hash>``。这类会话的"记忆"是 harness 提示词本身
# （``Please answer yes if the response contains the correct answer`` 单句重复 603 次），
# 召回到对话里只会污染上下文。2026-09-30 一次 ``reconcile_index_from_disk --apply``
# 把 1 791 条这种抽屉灌进 index.db，曲线覆盖率从 99.9% 掉到 23%。
SYNTHETIC_SOURCE_PREFIXES = ("api-",)


def is_synthetic_source(data: dict[str, Any]) -> bool:
    """drawer front matter 的 ``provenance.source`` 是否属于合成评测会话。

    读不出来就判 False：对账宁可对单个抽屉多索引一次，也不要在判据模糊时静默丢弃。
    """
    provenance = data.get("provenance")
    if isinstance(provenance, str):
        try:
            provenance = json.loads(provenance)
        except ValueError:
            return False
    if not isinstance(provenance, dict):
        return False
    return str(provenance.get("source", "") or "").startswith(SYNTHETIC_SOURCE_PREFIXES)


def scan_drawers(palace_dir: Path) -> dict[str, Path]:
    """枚举磁盘抽屉：``palace/<wing>/<hall>/<room>/drawer/<memory_id>.md``。

    只收 drawer 目录下的文件（closet 是同一条记忆的摘要指针，会重复计数），
    并跳过 ``_`` 前缀的保留文件。
    """
    drawers: dict[str, Path] = {}
    if not palace_dir.exists():
        return drawers
    for path in sorted(palace_dir.rglob("*.md")):
        if path.parent.name != "drawer" or path.name.startswith("_"):
            continue
        drawers[path.stem] = path
    return drawers


def disk_memory_ids(palace_dir: Path | None) -> set[str] | None:
    """磁盘抽屉的 memory_id 集合；**无法可靠核验时返回 None**，调用方据此拒绝删除。

    ★ P1-7/P1-8/P1-9 的共同原语：index.db / MetaStore / BM25 都只是并行双写的镜像，
    降级时各自都会缺行，谁都不能当"这条记忆不存在"的判据。唯一事实来源是抽屉文件。
    把"扫不到抽屉"一律判为 None（而不是空集合），是为了让挂载缺失、权限不足、路径写错
    这类事故退化成"什么都不做"，而不是"把索引当成全没了清理一遍"。
    """
    if palace_dir is None or not palace_dir.exists():
        return None
    ids = set(scan_drawers(palace_dir))
    if not ids:
        logger.warning("磁盘抽屉扫描为空（%s）——按无法核验处理，不以任何镜像为准做删除", palace_dir)
        return None
    return ids


@dataclass
class ReconcileReport:
    """一次对账的结果。``repaired``/``pruned`` 只在非 dry_run 时非零。"""

    dry_run: bool = True
    palace_dir: Path | None = None
    disk_total: int = 0
    index_total: int = 0
    missing_in_index: list[str] = field(default_factory=list)
    ghosts_in_index: list[str] = field(default_factory=list)
    skipped_archived: list[str] = field(default_factory=list)
    skipped_synthetic: list[str] = field(default_factory=list)
    repaired: int = 0
    pruned: int = 0
    failed: list[str] = field(default_factory=list)
    vector_pending: int = 0

    @property
    def recallable_total(self) -> int:
        """磁盘上"本该可召回"的记忆数：抽屉总数减去已归档/已遗忘与合成评测的那批。"""
        return self.disk_total - len(self.skipped_archived) - len(self.skipped_synthetic)

    @property
    def coverage_before(self) -> float:
        """对账前的有效检索覆盖率（index 命中数 / 应可召回的抽屉数）。"""
        base = self.recallable_total
        if not base:
            return 1.0
        return (base - len(self.missing_in_index)) / base

    def summary(self) -> str:
        excluded = len(self.skipped_archived) + len(self.skipped_synthetic)
        lines = [
            f"磁盘抽屉 {self.disk_total} 条 / index.db {self.index_total} 行",
            f"覆盖率（对账前）: {self.coverage_before * 100:.1f}%"
            + (f"（分母已剔除 {excluded} 条不该召回的）" if excluded else ""),
            f"缺失索引行: {len(self.missing_in_index)}",
            f"幽灵索引行（有 index 无 drawer）: {len(self.ghosts_in_index)}",
            f"待回填向量队列: {self.vector_pending} 条",
        ]
        if self.skipped_archived:
            lines.append(
                f"已归档/已遗忘（抽屉还在但**不该**重新索引，已跳过）: {len(self.skipped_archived)}"
            )
        if self.skipped_synthetic:
            lines.append(
                f"合成评测会话 provenance=api-*（**不该**索引，已跳过）: {len(self.skipped_synthetic)}"
            )
        if self.dry_run:
            lines.append("模式: dry-run（未写入）")
        else:
            lines.append(f"模式: apply — 已补 {self.repaired} 行，已清 {self.pruned} 行")
        if self.failed:
            lines.append(f"失败: {len(self.failed)} 条，例如 {self.failed[:5]}")
        return "\n  ".join(lines)


def _entry_from_disk(
    memory_id: str, path: Path, reader: MarkdownStore, palace_dir: Path
) -> dict[str, Any] | None:
    """把 drawer 文件还原成 index.add() 需要的字段。

    字段来源与写路径一致（``_write_drawer`` 的 front matter 只有 type/confidence/
    privacy/stored_at/provenance/vc/entities，**没有 wing/hall/room**），
    位置信息只能从目录布局取：``palace/<wing>/<hall>/<room>/drawer/<id>.md``。

    secret 级：index.db 的 content 与 FTS 表都是明文存储，所以沿用写入路径的约定
    存占位串，不把密文（更不把解密后的明文）灌进检索索引。
    """
    data = reader.read(path)
    if not data:
        return None
    privacy = str(data.get("privacy", "personal")) or "personal"
    content = str(data.get("content", "") or "")
    location = _location_from_path(path, palace_dir)
    memory_type = str(data.get("type", "fact")) or "fact"
    summary = str(data.get("summary", "") or "")
    if not summary:
        # 与 retry_index_add 一致：摘要取原文前 200 字符，并把换行/制表符压平，
        # 否则 summary 列里的换行会破坏 BM25 分词边界
        summary = content[:200].replace("\n", " ").replace("\r", " ").replace("\t", " ")
    provenance = data.get("provenance")
    return {
        "memory_id": memory_id,
        "wing": location["wing"] or "personal",
        "hall": location["hall"] or memory_type,
        "room": location["room"] or "default",
        "content": "[加密记忆]" if privacy == "secret" else content,
        "summary": summary,
        "type": memory_type,
        "confidence": int(data.get("confidence", 3) or 3),
        "privacy": privacy,
        "scope": privacy,
        "stored_at": str(data.get("stored_at", "") or ""),
        "provenance": json.dumps(provenance, ensure_ascii=False) if provenance else "",
    }


def _location_from_path(path: Path, palace_dir: Path) -> dict[str, str]:
    """从 ``palace/<wing>/<hall>/<room>/drawer/<id>.md`` 解析位置三元组。"""
    try:
        parts = path.relative_to(palace_dir).parts
    except ValueError:
        parts = path.parts
    if len(parts) < 5:
        return {"wing": "", "hall": "", "room": ""}
    wing, hall, room = parts[-5], parts[-4], parts[-3]
    return {"wing": wing, "hall": hall, "room": room}


def archived_memory_ids(data_dir: Path) -> set[str] | None:
    """只读取出遗忘库里 ``stage IN ('archived','forgotten')`` 的 memory_id。

    ★ P1-10：抽屉在磁盘上 ≠ 这条记忆还该被召回。归档/遗忘只改 ``forgetting_state.stage``
    （物理删抽屉是后续动作），而 index 行的删除由 ``GovernanceAuditor`` 在每次开机执行。
    对账如果只看抽屉，就会把这批"用户/曲线判定要忘掉"的记忆重新索引 —— 既违背遗忘语义，
    又和审计器形成拉锯（对账加、开机删、下次对账再加）。副本实测：某次差额 440 行里
    **440 行都属于已归档/已遗忘**。

    返回 ``None`` 表示没有可用的遗忘状态基准（库不存在 / 表缺失 / 打不开）。调用方按
    "无归档"处理（补索引是不可逆性较低的一侧），但必须让操作员知道这层保护没生效。
    """
    candidates = [data_dir / "governance" / "forgetting.db", data_dir / "forgetting.db"]
    db_path = next((p for p in candidates if p.exists()), None)
    if db_path is None:
        logger.warning("未找到 forgetting.db，本次对账不排除已归档记忆（%s）", data_dir)
        return None
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    except sqlite3.Error as e:
        logger.warning("只读打开 forgetting.db 失败（本次不排除已归档记忆）: %s", e)
        return None
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT memory_id FROM forgetting_state WHERE stage IN ('archived', 'forgotten')"
            )
        }
    except sqlite3.Error as e:
        logger.warning("只读扫描 forgetting_state 失败（本次不排除已归档记忆）: %s", e)
        return None
    finally:
        conn.close()


def _read_only_index_ids(index_dir: Path) -> set[str] | None:
    """以只读 URI 打开 index.db 取全部 memory_id。

    dry-run 也必须能被安全地用在生产目录上：构造 ThreeLevelIndex 会建目录、跑
    schema 初始化，因此这里绕过它直接用 ``mode=ro``。数据库不存在视为空集
    （首次运行），真正打不开则返回 None 让调用方回退到常规连接。
    """
    db_path = index_dir / "index.db"
    if not db_path.exists():
        return set()
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    except sqlite3.Error as e:
        logger.warning("只读打开 index.db 失败（回退常规连接）: %s", e)
        return None
    try:
        return {row[0] for row in conn.execute("SELECT memory_id FROM memory_index")}
    except sqlite3.Error as e:
        logger.warning("只读扫描 memory_index 失败（回退常规连接）: %s", e)
        return None
    finally:
        conn.close()


def reconcile_index(
    data_dir: Path,
    *,
    apply: bool = False,
    prune_ghosts: bool = False,
    index: Any | None = None,
) -> ReconcileReport:
    """对账 ``data_dir``：磁盘抽屉 ↔ index.db。

    Args:
        data_dir: OmniMem 数据根目录（含 ``palace/`` 与 ``index/``）
        apply: False（默认）只报告差额，不写任何文件
        prune_ghosts: 仅在 apply 时生效，删除有 index 行但磁盘无抽屉的幽灵行
        index: 已构造好的 ThreeLevelIndex（进程内复用）；不传则按 data_dir 新建
    """
    data_dir = Path(data_dir)
    palace_dir = data_dir / "palace"
    drawers = scan_drawers(palace_dir)

    index_ids: set[str] | None = None
    if index is not None:
        entries = index.search_all_for_retrieval(limit=_FULL_SCAN_LIMIT)
        index_ids = {e.get("memory_id", "") for e in entries if e.get("memory_id")}
    elif not apply:
        index_ids = _read_only_index_ids(data_dir / "index")

    # 只有真要写（apply）或只读打不开时，才构造会建目录/初始化 schema 的 ThreeLevelIndex
    owns_index = index is None and index_ids is None
    if owns_index:
        from omnimem.memory.index import ThreeLevelIndex

        index = ThreeLevelIndex(data_dir / "index")

    try:
        if index_ids is None:
            entries = index.search_all_for_retrieval(limit=_FULL_SCAN_LIMIT)
            index_ids = {e.get("memory_id", "") for e in entries if e.get("memory_id")}

        # ★ P1-10：已归档/已遗忘的记忆不该被重新索引（抽屉只是尚未物理删除的冷备份）
        archived = archived_memory_ids(data_dir) or set()
        all_missing = set(drawers) - index_ids
        skipped_archived = sorted(all_missing & archived)
        missing = sorted(all_missing - archived)
        ghosts = sorted(index_ids - set(drawers))

        # ★ 合成评测会话（provenance.source = api-*）同样不该进检索链路。抽屉是写路径
        # 留下的冷备份，评测语料在磁盘上会一直在，所以每次对账都要重新筛掉 ——
        # 只清 index.db 是不够的。
        reader = MarkdownStore(palace_dir)
        skipped_synthetic: list[str] = []
        real_missing: list[str] = []
        for memory_id in missing:
            if is_synthetic_source(reader.read(drawers[memory_id]) or {}):
                skipped_synthetic.append(memory_id)
            else:
                real_missing.append(memory_id)
        missing = real_missing

        report = ReconcileReport(
            dry_run=not apply,
            palace_dir=palace_dir,
            disk_total=len(drawers),
            index_total=len(index_ids),
            missing_in_index=missing,
            ghosts_in_index=ghosts,
            skipped_archived=skipped_archived,
            skipped_synthetic=skipped_synthetic,
            vector_pending=count_vector_pending(data_dir),
        )

        if not apply:
            return report

        for memory_id in missing:
            try:
                entry = _entry_from_disk(memory_id, drawers[memory_id], reader, palace_dir)
                if entry is None:
                    report.failed.append(memory_id)
                    logger.warning("对账：无法读取抽屉 %s，跳过", drawers[memory_id])
                    continue
                index.add(**entry)
                report.repaired += 1
            except Exception as e:
                report.failed.append(memory_id)
                logger.warning("对账：补写 index 行失败 %s: %s", memory_id, e)

        if prune_ghosts:
            for memory_id in ghosts:
                try:
                    if index.delete(memory_id):
                        report.pruned += 1
                    else:
                        report.failed.append(memory_id)
                except Exception as e:
                    report.failed.append(memory_id)
                    logger.warning("对账：清理幽灵行失败 %s: %s", memory_id, e)

        index.flush()
        return report
    finally:
        if owns_index:
            index.close()


def count_vector_pending(data_dir: Path) -> int:
    """待回填向量队列的积压条数（不构造检索引擎，只数 jsonl 行）。"""
    from omnimem.retrieval.vector_store import vector_pending_path

    path = vector_pending_path(Path(data_dir) / "retrieval" / "chroma")
    if not path.exists():
        return 0
    try:
        with path.open(encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except Exception as e:
        logger.warning("统计待回填向量队列失败: %s", e)
        return 0
