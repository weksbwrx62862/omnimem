"""P1-11 回归：向量覆盖率门控必须按**记忆**去比，不能拿 chunk 行数当覆盖数。

``WarmupManager._backfill_vectors``（P1-2 引入的判据）原先比的是 ``vector_count()``
—— 那是向量**行**数。一条长记忆在 ``retrieval/vector.py`` 的 ``_prepare_batch`` 里会被
切成 ``f"{memory_id}_chunk{hash}"`` 多行写入，所以行数恒 ≥ 记忆数，"vectors >= indexed"
只要分块存在就永远成立：真实缺口报不出来，门控形同虚设。线上实测 3 217 行 / 2 742 条
活跃记忆，判据给出"覆盖率 100%"是巧合而非验证。

现在的判据是集合差 ``uncovered = 活跃 id − 已向量化 id``；另外补一次孤儿向量清扫
（重建走 upsert，永不回收"记忆已经不存在"的行）。孤儿的定义是**抽屉 ∪ index 全表**
都查不到 —— 副本实测 584 个「index 查不到」的向量 id 里 440 个磁盘抽屉还在，只按 index
判就会把这些还能召回的记忆从向量通道抹掉。护栏沿用 P1-7 的教训：拿不到抽屉基线、
或一次性孤儿占比过高时**拒绝删除**。
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from omnimem.core.warmup_manager import _FULL_SCAN_LIMIT, WarmupManager
from omnimem.retrieval.vector import VectorRetriever

# 生产量级（2026-09-30 线上活跃索引 2 742 条）
ACTIVE = 2742
# 反例：80 条长记忆切成 40 chunk = 3 200 行，旧判据（行数 >= 活跃数 - 容忍）必然放行
CHUNKED_MEMORIES = 80
ROWS_FROM_CHUNKS = CHUNKED_MEMORIES * 40


class _IdAwareIndex:
    """search_l1 = 活跃条目（is_superseded=0）；search_all_for_retrieval = 全表含 superseded。"""

    def __init__(self, active: int, superseded: int = 0) -> None:
        self._active = [{"memory_id": f"m{i:012d}", "content": f"c{i}"} for i in range(active)]
        self._all = self._active + [
            {"memory_id": f"s{i:012d}", "content": f"sup{i}"} for i in range(superseded)
        ]
        self.requested_limits: list[int] = []

    def search_l1(self, wing: str = "", type: str = "", limit: int = 50) -> list[dict[str, Any]]:
        self.requested_limits.append(limit)
        return self._active[:limit]

    def search_all_for_retrieval(self, limit: int = 1000) -> list[dict[str, Any]]:
        return self._all[:limit]


class _IdAwareRetriever:
    """按 id 集合驱动的检索器 fake：``rows`` 与 ``covered`` 可以故意不一致。"""

    def __init__(
        self,
        covered: set[str] | None,
        rows: int,
        stored: set[str] | None = None,
        pending: int = 0,
    ) -> None:
        self._covered = covered
        self._rows = rows
        self._stored = stored if stored is not None else {f"{m}_chunkdeadbeef" for m in covered or set()}
        self._pending = pending
        self.drained = False
        self.rebuilt = False
        self.rebuild_input: list[dict[str, Any]] | None = None
        self.deleted: list[str] | None = None

    def _check_vector_health(self) -> dict[str, Any]:
        return {"vector_count": self._rows, "breaker_state": "closed"}

    def vector_count(self) -> int:
        return self._rows

    def count_vector_pending(self) -> int:
        return 0 if self.drained else self._pending

    def drain_vector_pending(self, limit: int = 2000) -> dict[str, int]:
        self.drained = True
        return {"pending": self._pending, "replayed": self._pending, "deferred": 0, "failed": 0}

    def rebuild_all_from_entries(self, entries: list[dict[str, Any]]) -> dict[str, int]:
        self.rebuilt = True
        self.rebuild_input = entries
        return {"vector": len(entries)}

    # ── ★ P1-11 新增接口 ──────────────────────────────────────────
    def vector_covered_memory_ids(self) -> set[str] | None:
        return self._covered

    def vector_stored_ids(self) -> set[str] | None:
        return set(self._stored)

    def vector_orphan_ids(self, known_memory_ids: set[str]) -> list[str]:
        return sorted(v for v in self._stored if v.split("_chunk")[0] not in known_memory_ids)

    def delete_vectors_by_ids(self, vector_ids: list[str]) -> int:
        self.deleted = list(vector_ids)
        for v in vector_ids:
            self._stored.discard(v)
        return len(vector_ids)


def _palace(tmp_path: Any, ids: set[str]) -> Path:
    """按 palace 目录布局落抽屉文件（``<wing>/<hall>/<room>/drawer/<id>.md``），
    给清扫流程一个"记忆确实不存在"的反证基线。"""
    palace = tmp_path / "palace"
    for mid in ids:
        d = palace / "w" / "h" / "r" / "drawer"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{mid}.md").write_text(f"# {mid}\n", encoding="utf-8")
    return palace


class _FakeStore:
    def __init__(self, palace_dir: Path | None) -> None:
        self.meta_store = SimpleNamespace(palace_dir=palace_dir)


def _manager(index: Any, retriever: Any, store: Any | None = None) -> WarmupManager:
    manager = object.__new__(WarmupManager)
    manager._index = index
    manager._retriever = retriever
    manager._store = store if store is not None else _FakeStore(None)
    manager._retrieval = type("R", (), {"warmup": lambda self: None})()
    return manager


def _covered_of(active: int, missing: int = 0) -> set[str]:
    return {f"m{i:012d}" for i in range(missing, active)}


def _rows_for(covered: set[str], chunks: int = 1) -> int:
    return len(covered) * chunks


# ── 门控：chunk 不能再骗过判据 ─────────────────────────────────────────

def test_chunk_inflated_row_count_cannot_mask_a_real_gap():
    """★ 反例本体：80 条记忆 × 40 chunk = 3 200 行 > 活跃 2 742 条。

    旧判据 ``rows >= active - tolerance``（3 200 >= 2 605）会放行，而实际只有 80 条有向量；
    新判据必须触发全量重建。
    """
    covered = {f"m{i:012d}" for i in range(CHUNKED_MEMORIES)}
    assert ACTIVE - max(ACTIVE // 20, 5) <= ROWS_FROM_CHUNKS, "反例失效：旧判据这里本该放行"

    index = _IdAwareIndex(ACTIVE)
    retriever = _IdAwareRetriever(covered=covered, rows=ROWS_FROM_CHUNKS)
    _manager(index, retriever)._backfill_vectors()

    assert retriever.rebuilt, "chunk 虚高的行数仍在掩盖真实缺口"
    assert retriever.rebuild_input is not None and len(retriever.rebuild_input) == ACTIVE


def test_full_memory_level_coverage_does_not_rebuild():
    """每条活跃记忆都有向量 ⇒ 不重建。"""
    covered = _covered_of(ACTIVE)
    retriever = _IdAwareRetriever(covered=covered, rows=_rows_for(covered))
    _manager(_IdAwareIndex(ACTIVE), retriever)._backfill_vectors()

    assert not retriever.rebuilt


def test_chunked_full_coverage_does_not_rebuild():
    """覆盖率达标 + 分块 ⇒ 不因行数远大于记忆数而反复全量重建。"""
    covered = _covered_of(ACTIVE)
    retriever = _IdAwareRetriever(covered=covered, rows=_rows_for(covered, chunks=40))
    _manager(_IdAwareIndex(ACTIVE), retriever)._backfill_vectors()

    assert not retriever.rebuilt


def test_gap_beyond_tolerance_triggers_rebuild():
    """缺口 500 条 > 容忍 137 ⇒ 重建。"""
    covered = _covered_of(ACTIVE, missing=500)
    retriever = _IdAwareRetriever(covered=covered, rows=_rows_for(covered))
    _manager(_IdAwareIndex(ACTIVE), retriever)._backfill_vectors()

    assert retriever.rebuilt


def test_gap_within_tolerance_does_not_rebuild():
    """容忍度沿用 5%（下限 5）：100 条 < 137 ⇒ 不重建，避免开机反复全量重建。"""
    covered = _covered_of(ACTIVE, missing=100)
    retriever = _IdAwareRetriever(covered=covered, rows=_rows_for(covered))
    _manager(_IdAwareIndex(ACTIVE), retriever)._backfill_vectors()

    assert not retriever.rebuilt


def test_healthy_coverage_leaves_an_audit_trail(caplog):
    """判据通过也要留痕：事后要能区分"核对过且全覆盖"与"核对根本没跑"。"""
    caplog.set_level(logging.INFO)
    covered = _covered_of(ACTIVE)
    retriever = _IdAwareRetriever(covered=covered, rows=_rows_for(covered, chunks=40))
    _manager(_IdAwareIndex(ACTIVE), retriever)._backfill_vectors()

    assert "向量覆盖核对通过" in caplog.text


def test_backfill_scan_uses_full_table_limit_not_5000():
    """★ P1-8 的同一形状：判据的输入必须是全量活跃条目，limit=5000 会漏掉老记忆。"""
    index = _IdAwareIndex(6000)
    retriever = _IdAwareRetriever(covered=set(), rows=0)
    _manager(index, retriever)._backfill_vectors()

    assert index.requested_limits == [_FULL_SCAN_LIMIT]
    assert retriever.rebuild_input is not None and len(retriever.rebuild_input) == 6000


# ── 后端无法枚举 id：退回标量口径，但不能静默 ─────────────────────────

def test_scalar_fallback_still_catches_catastrophic_loss(caplog):
    retriever = _IdAwareRetriever(covered=None, rows=100)
    _manager(_IdAwareIndex(ACTIVE), retriever)._backfill_vectors()

    assert retriever.rebuilt
    assert "不支持枚举 id" in caplog.text


def test_scalar_fallback_reports_unusable_judgement(caplog):
    """行数达标时放行，但必须告警"这个判据不可信"，不能报成覆盖率 100%。"""
    retriever = _IdAwareRetriever(covered=None, rows=ROWS_FROM_CHUNKS)
    _manager(_IdAwareIndex(ACTIVE), retriever)._backfill_vectors()

    assert not retriever.rebuilt
    assert "虚高" in caplog.text


# ── 孤儿向量清扫：回收 upsert 永不删除的残留（基线 = 抽屉 ∪ index）────────

def test_orphan_vectors_are_swept_when_coverage_is_healthy(tmp_path):
    """覆盖率达标时也要清掉「抽屉与 index 都查不到」的残留向量（副本实测 144 条）。"""
    covered = _covered_of(1500)
    stored = {f"{m}_chunkdeadbeef" for m in covered}
    stored |= {f"o{i:012d}_chunkdeadbeef" for i in range(144)}
    retriever = _IdAwareRetriever(covered=covered, rows=_rows_for(covered), stored=stored)
    store = _FakeStore(_palace(tmp_path, covered))

    _manager(_IdAwareIndex(1500), retriever, store)._backfill_vectors()

    assert retriever.deleted is not None and len(retriever.deleted) == 144
    assert all(v.startswith("o") for v in retriever.deleted)


def test_vectors_of_memories_missing_from_index_but_still_on_disk_are_kept(tmp_path):
    """★ 副本实测形状：584 个「index 查不到」的向量 id 里 440 个抽屉还在 —— 只按 index
    判孤儿就会抹掉这 440 条还能召回的记忆，等于自己造一个 P1-1。只删两边都没有的 144 个。
    """
    covered = _covered_of(1000)
    ghost_index_rows = {f"m{i:012d}" for i in range(1000, 1440)}  # 抽屉在、index 没有
    truly_gone = {f"g{i:012d}" for i in range(144)}               # 抽屉与 index 都没有
    stored = {f"{m}_chunkdeadbeef" for m in covered | ghost_index_rows | truly_gone}
    retriever = _IdAwareRetriever(covered=covered, rows=len(stored), stored=stored)
    drawers = covered | ghost_index_rows  # 磁盘上仍有 1000+440 条抽屉
    store = _FakeStore(_palace(tmp_path, drawers))

    _manager(_IdAwareIndex(1000), retriever, store)._backfill_vectors()

    assert retriever.deleted is not None
    assert sorted(v.split("_chunk")[0] for v in retriever.deleted) == sorted(truly_gone)
    assert len(retriever.deleted) == 144, "误删了抽屉仍在（索引缺行）的记忆向量"


def test_superseded_memory_vectors_are_not_swept(tmp_path):
    """被替换的记忆仍在索引表里 ⇒ 不是孤儿；删它属于对账/遗忘流程，这里不越权。"""
    covered = _covered_of(300) | {f"s{i:012d}" for i in range(30)}
    stored = {f"{m}_chunkdeadbeef" for m in covered}
    retriever = _IdAwareRetriever(covered=covered, rows=_rows_for(covered), stored=stored)
    store = _FakeStore(_palace(tmp_path, _covered_of(300)))

    _manager(_IdAwareIndex(300, superseded=30), retriever, store)._backfill_vectors()

    assert retriever.deleted is None


def test_sweep_refuses_without_a_disk_baseline(tmp_path):
    """拿不到抽屉基线（palace 未挂载 / 路径写错）⇒ 一律不删，退化成"什么都不做"。"""
    covered = _covered_of(300)
    stored = {f"{m}_chunkdeadbeef" for m in covered} | {
        f"o{i:012d}_chunkdeadbeef" for i in range(10)
    }
    retriever = _IdAwareRetriever(covered=covered, rows=len(stored), stored=stored)

    _manager(_IdAwareIndex(300), retriever, _FakeStore(None))._backfill_vectors()
    assert retriever.deleted is None

    empty_palace = _palace(tmp_path, set())  # 目录存在但没有抽屉 ⇒ 视为无法核验
    retriever2 = _IdAwareRetriever(covered=covered, rows=len(stored), stored=stored)
    _manager(_IdAwareIndex(300), retriever2, _FakeStore(empty_palace))._backfill_vectors()
    assert retriever2.deleted is None


def test_sweep_refuses_when_orphans_are_a_large_share_of_the_store(tmp_path):
    """★ 护栏：一次检出 2 000 个孤儿（占库 >10%）⇒ 基线可疑（接错数据目录等），拒绝删。"""
    covered = _covered_of(500)
    stored = {f"{m}_chunkdeadbeef" for m in covered}
    stored |= {f"x{i:012d}_chunkdeadbeef" for i in range(2000)}
    retriever = _IdAwareRetriever(covered=covered, rows=len(stored), stored=stored)
    store = _FakeStore(_palace(tmp_path, covered))

    _manager(_IdAwareIndex(500), retriever, store)._backfill_vectors()

    assert retriever.deleted is None, "删除基线可疑时仍然执行了批量删除"


def test_sweep_falls_back_to_drawer_baseline_when_index_cannot_scan(tmp_path):
    """旧索引对象只有 search_l1：抽屉单独也够判孤儿，但缺 index 侧信息时更保守 —— 仍按抽屉删。"""
    covered = _covered_of(200)
    stored = {f"{m}_chunkdeadbeef" for m in covered} | {"gone0000000_chunkdeadbeef"}
    retriever = _IdAwareRetriever(covered=covered, rows=len(stored), stored=stored)
    index = _IdAwareIndex(200)
    index.search_all_for_retrieval = None  # type: ignore[assignment]
    store = _FakeStore(_palace(tmp_path, covered))

    _manager(index, retriever, store)._backfill_vectors()

    assert retriever.deleted == ["gone0000000_chunkdeadbeef"]


# ── 同一单位错误的第二个现场：auditor 的 ChromaDB 偏差检查 ──────────────

class _AuditMetaStore:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids
        self.palace_dir = None  # 无抽屉基准：本次只测向量口径

    def get_all(self, limit: int = 1000) -> list[dict[str, Any]]:
        return [{"memory_id": mid} for mid in self._ids[:limit]]

    def count(self) -> int:
        return len(self._ids)


class _AuditStore:
    def __init__(self, ids: list[str]) -> None:
        self.meta_store = _AuditMetaStore(ids)

    def get(self, mid: str) -> dict[str, Any] | None:
        return {"memory_id": mid, "content": "x", "wing": "w", "type": "fact", "room": "r"}


class _AuditIndex:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def search_all_for_retrieval(self, limit: int = 1000) -> list[dict[str, Any]]:
        return [{"memory_id": mid} for mid in self._ids[:limit]]


class _AuditRetriever:
    bm25_document_count = 0

    def __init__(self, rows: int, covered: set[str] | None) -> None:
        self._rows = rows
        self._covered = covered

    def vector_count(self) -> int:
        return self._rows

    def vector_covered_memory_ids(self) -> set[str] | None:
        return self._covered


class _ScalarOnlyRetriever(_AuditRetriever):
    """后端不支持枚举 id：接口存在但不可调用，auditor 必须退回标量口径。"""

    vector_covered_memory_ids = None  # type: ignore[assignment]


def _auditor(ids: list[str], rows: int, covered: set[str] | None) -> Any:
    from omnimem.governance.auditor import GovernanceAuditor

    retriever: Any = (
        _ScalarOnlyRetriever(rows, None)
        if covered is None
        else _AuditRetriever(rows, covered)
    )
    return GovernanceAuditor(
        store=_AuditStore(ids),
        index=_AuditIndex(ids),
        retriever=retriever,
        forgetting=type("F", (), {"get_archived_ids": lambda self, limit=1000: []})(),
    )


META_IDS = [f"m{i:012d}" for i in range(200)]


def test_auditor_ignores_legitimately_inflated_row_count():
    """旧判据 ``abs(meta - rows) > threshold`` 在 100% 覆盖时误报偏差（chunk 让行数虚高）。"""
    audit = _auditor(META_IDS, rows=8000, covered=set(META_IDS)).run_full_audit(limit=2000)

    assert max(200 // 20, 5) < 8000 - 200, "反例失效：旧判据这里本该误报"
    assert audit["chroma_degraded"] is False
    assert audit["chroma_uncovered"] == 0


def test_auditor_reports_memory_level_gap():
    """只有 20 条有向量、缺口 180 > 阈值 10 ⇒ 必须判为退化并报出缺口数。"""
    audit = _auditor(META_IDS, rows=8000, covered=set(META_IDS[:20])).run_full_audit(limit=2000)

    assert audit["chroma_degraded"] is True
    assert audit["chroma_uncovered"] == 180


def test_auditor_scalar_fallback_only_flags_shortfall():
    """不能枚举 id 时：行数虚高不再报警，但行数明显低于记忆数仍要报警。"""
    inflated = _auditor(META_IDS, rows=8000, covered=None).run_full_audit(limit=2000)
    assert inflated["chroma_degraded"] is False
    assert inflated["chroma_uncovered"] is None

    collapsed = _auditor(META_IDS, rows=10, covered=None).run_full_audit(limit=2000)
    assert collapsed["chroma_degraded"] is True


# ── VectorRetriever 层的 id 口径 ─────────────────────────────────────

class _FakeCollection:
    """钉住 chromadb collection 的调用形状：``get(include=[])["ids"]``。"""

    def __init__(self, ids: list[str], reject_include_kw: bool = False) -> None:
        self._ids = ids
        self._reject_include_kw = reject_include_kw
        self.calls: list[dict[str, Any]] = []

    def get(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self._reject_include_kw and "include" in kwargs:
            raise TypeError("unexpected keyword argument 'include'")
        return {"ids": list(self._ids), "metadatas": None, "documents": None}


def test_chroma_store_all_ids_pins_the_collection_call_shape():
    from omnimem.retrieval.vector_store import ChromaDBStore

    ids = ["aabbccddeeff_chunk1a2b3c4d", "112233445566"]
    store = object.__new__(ChromaDBStore)
    store._initialized = True
    store._collection = _FakeCollection(ids)

    assert store.all_ids() == ids
    assert store._collection.calls == [{"include": []}]


def test_chroma_store_all_ids_falls_back_when_include_unsupported():
    from omnimem.retrieval.vector_store import ChromaDBStore

    store = object.__new__(ChromaDBStore)
    store._initialized = True
    store._collection = _FakeCollection(["m1"], reject_include_kw=True)

    assert store.all_ids() == ["m1"]


def test_chroma_store_all_ids_empty_on_unavailable_collection():
    from omnimem.retrieval.vector_store import ChromaDBStore

    store = object.__new__(ChromaDBStore)
    store._initialized = True
    store._collection = None

    assert store.all_ids() == []


def test_retriever_falls_back_to_raw_collection_when_store_lacks_all_ids():
    """旧 store 对象没有 all_ids()：从 ``_collection`` 直接取，取不到才退回标量口径。"""
    retriever = object.__new__(VectorRetriever)
    retriever._store = type("Legacy", (), {"_collection": _FakeCollection(["m1_chunkabcd"])})()
    retriever._ensure_initialized = lambda: None  # type: ignore[method-assign]

    assert retriever.covered_memory_ids() == {"m1"}


def test_retriever_reports_none_when_backend_cannot_enumerate(caplog):
    """Qdrant 的 point id 是派生 UUID，枚举不出来 ⇒ 必须返回 None（"不知道"），不能返回空集。"""
    retriever = object.__new__(VectorRetriever)
    retriever._store = object()
    retriever._ensure_initialized = lambda: None  # type: ignore[method-assign]

    assert retriever.covered_memory_ids() is None
    assert "不支持枚举 id" in caplog.text


def test_chunk_suffix_stripping():
    assert VectorRetriever._strip_chunk_suffix("abc123def456_chunk1a2b3c4d") == "abc123def456"
    assert VectorRetriever._strip_chunk_suffix("abc123def456") == "abc123def456"


def test_orphan_query_refuses_an_empty_baseline():
    """基线为空 ⇒ 返回空列表：绝不因为"查不到"就把整库判成孤儿。"""
    stored_ids = [f"m{i}_chunkdeadbeef" for i in range(5)]
    retriever = object.__new__(VectorRetriever)
    retriever._store = type("S", (), {"all_ids": lambda self: list(stored_ids)})()
    retriever._ensure_initialized = lambda: None  # type: ignore[method-assign]

    assert retriever.stored_vector_ids() == set(stored_ids)
    assert retriever.orphan_vector_ids(set()) == []
    assert retriever.orphan_vector_ids({"m0", "m1"}) == [
        "m2_chunkdeadbeef",
        "m3_chunkdeadbeef",
        "m4_chunkdeadbeef",
    ]
