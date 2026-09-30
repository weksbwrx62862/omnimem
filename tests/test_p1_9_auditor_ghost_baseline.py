"""P1-9 回归：``GovernanceAuditor`` 的删除判定不得以 MetaStore 为准。

审计器原先把「index 有、MetaStore 无」直接判成幽灵，``repair()`` 再据此
``index.delete(mid)``。MetaStore 只是并行双写的一方，降级时自己就缺行（副本实测
877 行 vs 磁盘 3 135 个抽屉），于是：

  ``quick_health_check`` 用 |meta - index| 判活 → 这套数据**永远**不健康
  → 每次开机跑 ``run_full_audit`` → ``repair`` 把 2 000 行里所有"meta 没有"的
    合法索引行删掉。

这是 §5.9 的同一个错误前提的第二条、也是更宽的路径：sync_from_index 只在 warmup
里删一次，审计器是设计上就要"持续修正"。删除判据必须是磁盘抽屉。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omnimem.governance.auditor import GovernanceAuditor


class _FakeMetaStore:
    def __init__(self, ids: list[str], palace_dir: Path | None) -> None:
        self._ids = ids
        self.palace_dir = palace_dir

    def get_all(self, limit: int = 5000) -> list[dict[str, Any]]:
        return [{"memory_id": mid} for mid in self._ids[:limit]]

    def count(self) -> int:
        return len(self._ids)


class _FakeStore:
    def __init__(self, meta_ids: list[str], palace_dir: Path | None) -> None:
        self.meta_store = _FakeMetaStore(meta_ids, palace_dir)

    def get(self, mid: str) -> dict[str, Any] | None:
        return {"memory_id": mid, "content": "x", "wing": "w", "type": "fact", "room": "r"}


class _FakeIndex:
    def __init__(self, ids: list[str]) -> None:
        self.ids = set(ids)
        self.deleted: list[str] = []
        self.added: list[str] = []

    def search_all_for_retrieval(self, limit: int = 1000) -> list[dict[str, Any]]:
        return [{"memory_id": mid} for mid in sorted(self.ids)[:limit]]

    def delete(self, mid: str) -> bool:
        self.deleted.append(mid)
        self.ids.discard(mid)
        return True

    def add(self, memory_id: str, **kwargs: Any) -> None:
        self.added.append(memory_id)
        self.ids.add(memory_id)


class _FakeRetriever:
    bm25_document_count = 100

    def __init__(self) -> None:
        self.vector_count_value = 100
        self.added: list[str] = []

    def vector_count(self) -> int:
        return self.vector_count_value

    def add(self, content: str, memory_id: str = "", metadata: dict | None = None) -> None:
        self.added.append(memory_id)


class _FakeForgetting:
    def __init__(self, archived: list[str] | None = None) -> None:
        self._archived = archived or []

    def get_archived_ids(self, limit: int = 1000) -> list[str]:
        return list(self._archived)


def _palace(tmp_path: Path, ids: list[str]) -> Path:
    palace = tmp_path / "palace"
    for mid in ids:
        d = palace / "w" / "h" / "r" / "drawer"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{mid}.md").write_text(f"# {mid}\n", encoding="utf-8")
    return palace


def _auditor(
    tmp_path: Path,
    *,
    index_ids: list[str],
    meta_ids: list[str],
    drawer_ids: list[str],
    palace: bool = True,
) -> tuple[GovernanceAuditor, _FakeIndex]:
    palace_dir = _palace(tmp_path, drawer_ids) if palace else None
    index = _FakeIndex(index_ids)
    auditor = GovernanceAuditor(
        store=_FakeStore(meta_ids, palace_dir),
        index=index,
        retriever=_FakeRetriever(),
        forgetting=_FakeForgetting(),
    )
    return auditor, index


def test_drawer_backed_index_rows_are_not_ghosts(tmp_path):
    """★ 核心：MetaStore 缺行、但磁盘抽屉在 → 不是幽灵，不该被报出来、更不该被删。"""
    ids = [f"m{i:012d}" for i in range(300)]
    auditor, index = _auditor(
        tmp_path, index_ids=ids, meta_ids=["m00000000000"], drawer_ids=ids
    )

    audit = auditor.run_full_audit(limit=2000)

    assert audit["ghost_in_index"] == []
    auditor.repair(audit)
    assert index.deleted == []


def test_rows_absent_from_every_mirror_and_disk_are_ghosts(tmp_path):
    """meta 无、index 有、磁盘也没有 —— 三处都缺才是幽灵。"""
    auditor, index = _auditor(
        tmp_path,
        index_ids=["real00000000", "ghost0000000"],
        meta_ids=["real00000000"],
        drawer_ids=["real00000000"],
    )

    audit = auditor.run_full_audit(limit=2000)

    assert audit["ghost_in_index"] == ["ghost0000000"]
    auditor.repair(audit)
    assert index.deleted == ["ghost0000000"]


def test_no_disk_baseline_means_no_deletion_advice(tmp_path):
    """拿不到抽屉基准（未绑定 palace）时不产生任何幽灵判定。"""
    auditor, index = _auditor(
        tmp_path,
        index_ids=["a00000000000", "b00000000000"],
        meta_ids=[],
        drawer_ids=[],
        palace=False,
    )

    audit = auditor.run_full_audit(limit=2000)

    assert audit["ghost_in_index"] == []
    assert auditor.quick_health_check()["healthy"] is True


def test_repair_from_metastore_keeps_drawer_backed_rows(tmp_path):
    """``_repair_from_metastore`` 的清理分支同样不得删有抽屉的行。"""
    ids = [f"m{i:012d}" for i in range(50)]
    auditor, index = _auditor(
        tmp_path, index_ids=ids, meta_ids=["m00000000000"], drawer_ids=ids
    )

    auditor._repair_from_metastore(limit=2000)

    assert index.deleted == []


def test_health_gate_uses_disk_coverage_not_metastore(tmp_path):
    """★ 门控本身：meta 稀疏不该让整套数据永远"不健康"并每开机触发一轮全量修复。"""
    ids = [f"m{i:012d}" for i in range(300)]
    auditor, _index = _auditor(
        tmp_path, index_ids=ids, meta_ids=ids[:3], drawer_ids=ids
    )

    health = auditor.quick_health_check()

    assert health["baseline"] == "disk"
    assert health["meta_count"] == 3
    assert health["index_count"] == 300
    assert health["healthy"] is True, "index 与磁盘一致却被 MetaStore 的缺口判成不健康"


def test_health_gate_still_flags_real_coverage_gap(tmp_path):
    """磁盘有、index 没有（P1-1 那批）必须仍报不健康 —— 门控没有变成常亮绿灯。"""
    ids = [f"m{i:012d}" for i in range(300)]
    auditor, _index = _auditor(
        tmp_path, index_ids=ids[:20], meta_ids=ids[:20], drawer_ids=ids
    )

    health = auditor.quick_health_check()

    assert health["healthy"] is False
