"""★ P1-1 回归：以磁盘抽屉为事实来源对账 index.db。

覆盖 2026-09-29 报告里最刺眼的一条：3 135 个 drawer / 712 行 index，2 596 条记忆
「有抽屉、无索引行」，关键词与语义检索都召不回，且系统里没有任何机制会发现它。
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pytest

from omnimem.governance.reconciler import (
    ReconcileReport,
    count_vector_pending,
    reconcile_index,
    scan_drawers,
)
from omnimem.memory.index import ThreeLevelIndex
from omnimem.retrieval.vector_store import vector_pending_path


def _write_drawer(palace: Path, wing: str, hall: str, room: str, memory_id: str,
                  content: str, *, privacy: str = "personal", memory_type: str = "fact",
                  confidence: int = 3, stored_at: str = "2026-09-01T00:00:00+00:00",
                  provenance: str = "") -> Path:
    """按生产布局 palace/<wing>/<hall>/<room>/drawer/<id>.md 写一条抽屉。"""
    path = palace / wing / hall / room / "drawer" / f"{memory_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = (
        f"memory_id: {memory_id}\n"
        f"type: {memory_type}\n"
        f"confidence: {confidence}\n"
        f"privacy: {privacy}\n"
        f"stored_at: '{stored_at}'\n"
    )
    if provenance:
        fm += f"provenance: {provenance}\n"
    path.write_text(f"---\n{fm}---\n\n{content}\n", encoding="utf-8")
    return path


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """一个「磁盘 4 条抽屉、index 只有 1 条 + 1 条幽灵」的最小现场。"""
    palace = tmp_path / "palace"
    _write_drawer(palace, "personal", "fact", "k8s", "aaaa11112222", "kubectl 排查 Pod 重启")
    _write_drawer(palace, "personal", "reasoning", "方向全", "bbbb11112222",
                  "缩量涨停的样本结论", memory_type="reasoning", confidence=5,
                  provenance="{method: tool_call}")
    _write_drawer(palace, "team", "preference", "称呼", "cccc11112222", "老板喜欢叫 AI 小兰",
                  memory_type="preference")
    _write_drawer(palace, "personal", "fact", "secret", "dddd11112222", "密码明文不能进索引",
                  privacy="secret")

    index = ThreeLevelIndex(tmp_path / "index")
    index.add(memory_id="aaaa11112222", wing="personal", hall="fact", room="k8s",
              content="kubectl 排查 Pod 重启", summary="kubectl", type="fact")
    index.add(memory_id="eeeee11111111", wing="personal", hall="fact", room="gone",
              content="磁盘上已经没有抽屉了", summary="ghost", type="fact")
    index.flush()
    index.close()
    return tmp_path


def _index_rows(data_dir: Path) -> dict[str, sqlite3.Row]:
    conn = sqlite3.connect(data_dir / "index" / "index.db")
    try:
        rows = conn.execute("SELECT * FROM memory_index").fetchall()
        cols = [c[0] for c in conn.execute("SELECT * FROM memory_index LIMIT 0").description]
        return {dict(zip(cols, r))["memory_id"]: dict(zip(cols, r)) for r in rows}
    finally:
        conn.close()


class TestScanDrawers:
    def test_scans_only_drawer_dirs(self, data_dir: Path):
        """closet 是同一条记忆的摘要指针，不能重复计数。"""
        palace = data_dir / "palace"
        pointer = palace / "personal" / "fact" / "k8s" / "closet"
        pointer.mkdir(parents=True, exist_ok=True)
        (pointer / "aaaa11112222.md").write_text("---\nsummary: 指针\n---\n\n摘要\n", encoding="utf-8")

        assert set(scan_drawers(palace)) == {
            "aaaa11112222", "bbbb11112222", "cccc11112222", "dddd11112222",
        }

    def test_skips_reserved_files_and_missing_dir(self, data_dir: Path):
        palace = data_dir / "palace"
        (palace / "personal" / "fact" / "k8s" / "drawer" / "_index.md").write_text("x", encoding="utf-8")

        assert "_index" not in scan_drawers(palace)
        assert scan_drawers(palace.parent / "nope") == {}


class TestDryRunIsReadOnly:
    def test_dry_run_reports_gap_without_writing(self, data_dir: Path):
        """默认只报差额：缺 3 条、幽灵 1 条，且不新增任何 index 行。"""
        before = set(_index_rows(data_dir))
        report = reconcile_index(data_dir)

        assert report.dry_run is True
        assert report.disk_total == 4
        assert report.index_total == 2
        assert report.missing_in_index == ["bbbb11112222", "cccc11112222", "dddd11112222"]
        assert report.ghosts_in_index == ["eeeee11111111"]
        assert report.repaired == 0
        assert set(_index_rows(data_dir)) == before

    def test_dry_run_does_not_create_index_dir(self, tmp_path: Path):
        """只读路径不得为了「看一眼差额」就建目录、初始化 schema。"""
        palace = tmp_path / "palace"
        _write_drawer(palace, "personal", "fact", "r", "aaaa00000001", "内容")

        report = reconcile_index(tmp_path)

        assert report.missing_in_index == ["aaaa00000001"]
        assert not (tmp_path / "index").exists()

    def test_coverage_before(self, data_dir: Path):
        report = reconcile_index(data_dir)
        assert report.coverage_before == pytest.approx(0.25)
        assert "25.0%" in report.summary()


class TestApplyRepairs:
    def test_apply_adds_missing_rows_with_path_derived_location(self, data_dir: Path):
        report = reconcile_index(data_dir, apply=True)
        rows = _index_rows(data_dir)

        assert report.repaired == 3
        assert report.pruned == 0
        assert report.dry_run is False
        assert "bbbb11112222" in rows
        # wing/hall/room 只能来自目录布局（drawer front matter 里没有这几个键）
        assert rows["bbbb11112222"]["wing"] == "personal"
        assert rows["bbbb11112222"]["hall"] == "reasoning"
        assert rows["bbbb11112222"]["room"] == "方向全"
        assert rows["bbbb11112222"]["type"] == "reasoning"
        assert rows["bbbb11112222"]["confidence"] == 5
        assert rows["bbbb11112222"]["scope"] == "personal"
        assert "tool_call" in rows["bbbb11112222"]["provenance"]

    def test_apply_never_repairs_twice(self, data_dir: Path):
        reconcile_index(data_dir, apply=True)
        second = reconcile_index(data_dir, apply=True)

        assert second.missing_in_index == []
        assert second.repaired == 0

    def test_secret_content_is_placeholder_not_plaintext(self, data_dir: Path):
        """secret 级：index.content 与 FTS 都是明文存储，只能放占位串。"""
        reconcile_index(data_dir, apply=True)
        row = _index_rows(data_dir)["dddd11112222"]

        assert row["content"] == "[加密记忆]"
        assert row["privacy"] == "secret"

    def test_summary_falls_back_to_flattened_content(self, data_dir: Path):
        palace = data_dir / "palace"
        _write_drawer(palace, "personal", "fact", "multi", "ffff11112222",
                      "第一行\n第二行\t制表\n" + "长" * 300)

        reconcile_index(palace.parent, apply=True)
        row = _index_rows(palace.parent)["ffff11112222"]

        assert "\n" not in row["summary"] and "\t" not in row["summary"]
        assert len(row["summary"]) <= 200

    def test_ghosts_survive_without_prune_flag(self, data_dir: Path):
        reconcile_index(data_dir, apply=True)

        assert "eeeee11111111" in _index_rows(data_dir)

    def test_prune_ghosts_removes_row_without_drawer(self, data_dir: Path):
        report = reconcile_index(data_dir, apply=True, prune_ghosts=True)

        assert report.pruned == 1
        assert "eeeee11111111" not in _index_rows(data_dir)
        # 真实抽屉一条都不能被误删
        assert "aaaa11112222" in _index_rows(data_dir)

    def test_unreadable_drawer_is_reported_as_failed_not_silently_ok(self, data_dir: Path):
        drawer = data_dir / "palace" / "personal" / "fact" / "k8s" / "drawer" / "gggg11112222.md"
        drawer.write_bytes(b"\xff\xfe broken yaml ---\n")

        report = reconcile_index(data_dir, apply=True)

        assert report.failed
        assert report.repaired == 3


class TestReconcilerWithInjectedIndex:
    def test_reuses_provided_index(self, data_dir: Path):
        index = ThreeLevelIndex(data_dir / "index")
        try:
            report = reconcile_index(data_dir, apply=True, index=index)
            assert report.repaired == 3
        finally:
            index.close()


class TestVectorPendingAccounting:
    def test_counts_pending_lines(self, data_dir: Path):
        path = vector_pending_path(data_dir / "retrieval" / "chroma")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"id":"a"}\n\n{"id":"b"}\n', encoding="utf-8")

        assert count_vector_pending(data_dir) == 2
        assert reconcile_index(data_dir).vector_pending == 2

    def test_missing_pending_file_is_zero(self, data_dir: Path):
        assert count_vector_pending(data_dir) == 0


class TestSummary:
    def test_empty_disk_reports_full_coverage(self, tmp_path: Path):
        report = reconcile_index(tmp_path)

        assert report.disk_total == 0
        assert isinstance(report, ReconcileReport)
        assert report.coverage_before == 1.0


class TestCliScript:
    @staticmethod
    def _main():
        """scripts/ 不是包，按文件路径加载 CLI。"""
        import importlib.util

        script = Path(__file__).resolve().parent.parent / "scripts" / "reconcile_index_from_disk.py"
        spec = importlib.util.spec_from_file_location("reconcile_cli", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main

    def test_default_is_dry_run(self, data_dir: Path):
        rc = self._main()(["--data-dir", str(data_dir)])
        assert rc == 0

    def test_prune_ghosts_requires_apply(self, data_dir: Path):
        assert self._main()(["--data-dir", str(data_dir), "--prune-ghosts"]) == 1

    def test_missing_palace_dir_fails_loudly(self, tmp_path: Path):
        assert self._main()(["--data-dir", str(tmp_path)]) == 1

    def test_explicit_data_dir_beats_env_default(self, data_dir: Path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(data_dir.parent / "nonexistent"))
        assert self._main()(["--data-dir", str(data_dir)]) == 0


class TestDoctorReconcileSubcommand:
    def test_dry_run_exit_zero(self, data_dir: Path, capsys):
        from omnimem.doctor import _cmd_reconcile

        _cmd_reconcile(argparse.Namespace(data_dir=data_dir, apply=False, prune_ghosts=False))
        out = capsys.readouterr().out

        assert "缺失索引行: 3" in out
        assert "dry-run" in out

    def test_apply_then_reconcile_reports_no_gap(self, data_dir: Path, capsys):
        from omnimem.doctor import _cmd_reconcile

        _cmd_reconcile(argparse.Namespace(data_dir=data_dir, apply=True, prune_ghosts=True))
        capsys.readouterr()
        _cmd_reconcile(argparse.Namespace(data_dir=data_dir, apply=False, prune_ghosts=False))
        out = capsys.readouterr().out

        assert "缺失索引行: 0" in out
        assert "幽灵索引行（有 index 无 drawer）: 0" in out

    def test_bad_combination_exits_1(self, data_dir: Path):
        from omnimem.doctor import _cmd_reconcile

        with pytest.raises(SystemExit) as exc:
            _cmd_reconcile(argparse.Namespace(data_dir=data_dir, apply=False, prune_ghosts=True))
        assert exc.value.code == 1

    def test_missing_palace_exits_1(self, tmp_path: Path):
        from omnimem.doctor import _cmd_reconcile

        with pytest.raises(SystemExit) as exc:
            _cmd_reconcile(argparse.Namespace(data_dir=tmp_path, apply=False, prune_ghosts=False))
        assert exc.value.code == 1
