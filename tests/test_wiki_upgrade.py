"""Tests for governance.wiki_upgrade — OmniMem → Wiki page upgrade pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from governance.wiki_upgrade import WikiUpgradePipeline, _slugify

# ── _slugify ────────────────────────────────────────────────────────────


def test_slugify_lowercase() -> None:
    assert _slugify("Hello World") == "hello-world"


def test_slugify_chinese_preserved() -> None:
    s = _slugify("记忆系统")
    assert "记忆系统" in s


def test_slugify_special_chars_removed() -> None:
    s = _slugify("Hello, World! (2024)")
    assert "," not in s
    assert "!" not in s
    assert "(" not in s
    assert ")" not in s


def test_slugify_multiple_spaces_collapsed() -> None:
    assert _slugify("a   b") == "a-b"


def test_slugify_leading_trailing_hyphens_stripped() -> None:
    assert _slugify("  hello  ") == "hello"
    assert _slugify("-hello-") == "hello"


def test_slugify_truncates_to_80() -> None:
    s = _slugify("a" * 200)
    assert len(s) <= 80


def test_slugify_empty() -> None:
    assert _slugify("") == ""


def test_slugify_underscores_kept() -> None:
    assert _slugify("hello_world") == "hello_world"


# ── constructor ─────────────────────────────────────────────────────────


def test_constructor_expands_path(tmp_path) -> None:
    p = WikiUpgradePipeline(wiki_path=str(tmp_path))
    assert p._wiki_path == tmp_path


def test_constructor_stores_deps(tmp_path) -> None:
    store = object()
    forgetting = object()
    llm = lambda x: x  # noqa: E731
    p = WikiUpgradePipeline(wiki_path=tmp_path, store=store, forgetting=forgetting, llm_call=llm)
    assert p._store is store
    assert p._forgetting is forgetting
    assert p._llm_call is llm


def test_constructor_defaults_none(tmp_path) -> None:
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    assert p._store is None
    assert p._forgetting is None
    assert p._llm_call is None


# ── _get_existing_pages ─────────────────────────────────────────────────


def test_get_existing_pages_empty(tmp_path) -> None:
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    assert p._get_existing_pages() == []


def test_get_existing_pages_scans_subdirs(tmp_path) -> None:
    (tmp_path / "entities").mkdir()
    (tmp_path / "entities" / "foo.md").write_text("# foo", encoding="utf-8")
    (tmp_path / "concepts").mkdir()
    (tmp_path / "concepts" / "bar.md").write_text("# bar", encoding="utf-8")
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    pages = p._get_existing_pages()
    assert "[entities/foo]" in pages
    assert "[concepts/bar]" in pages


def test_get_existing_pages_ignores_missing_subdirs(tmp_path) -> None:
    (tmp_path / "entities").mkdir()
    (tmp_path / "entities" / "x.md").write_text("# x", encoding="utf-8")
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    pages = p._get_existing_pages()
    assert pages == ["[entities/x]"]


def test_get_existing_pages_ignores_non_md(tmp_path) -> None:
    (tmp_path / "entities").mkdir()
    (tmp_path / "entities" / "x.txt").write_text("not md", encoding="utf-8")
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    assert p._get_existing_pages() == []


# ── _update_index ───────────────────────────────────────────────────────


def test_update_index_no_file_noop(tmp_path) -> None:
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    p._update_index("concept", "foo", "Foo", "summary")  # should not raise


def test_update_index_inserts_under_section(tmp_path) -> None:
    idx = tmp_path / "index.md"
    idx.write_text(
        "# Wiki\n\n## Entities\n\n## Concepts\n\n## Comparisons\n\nTotal pages: 0\nLast updated: 2024-01-01\n",
        encoding="utf-8",
    )
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    p._update_index("concept", "foo", "Foo", "a summary")
    text = idx.read_text(encoding="utf-8")
    assert "[[foo]] — a summary" in text
    assert "Total pages: 1" in text


def test_update_index_unknown_type_defaults_concepts(tmp_path) -> None:
    idx = tmp_path / "index.md"
    idx.write_text("## Concepts\n\nTotal pages: 0\n", encoding="utf-8")
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    p._update_index("weird", "bar", "Bar", "s")
    text = idx.read_text(encoding="utf-8")
    assert "[[bar]] — s" in text


def test_update_index_entity_section(tmp_path) -> None:
    idx = tmp_path / "index.md"
    idx.write_text("## Entities\n\n## Concepts\n\nTotal pages: 0\n", encoding="utf-8")
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    p._update_index("entity", "alice", "Alice", "person")
    text = idx.read_text(encoding="utf-8")
    assert "[[alice]] — person" in text


def test_update_index_updates_last_updated(tmp_path) -> None:
    idx = tmp_path / "index.md"
    idx.write_text("## Concepts\n\nTotal pages: 0\nLast updated: 2020-01-01\n", encoding="utf-8")
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    p._update_index("concept", "x", "X", "s")
    text = idx.read_text(encoding="utf-8")
    assert "Last updated: 2020-01-01" not in text


def test_update_index_no_total_noop(tmp_path) -> None:
    idx = tmp_path / "index.md"
    idx.write_text("## Concepts\n", encoding="utf-8")
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    p._update_index("concept", "x", "X", "s")
    text = idx.read_text(encoding="utf-8")
    assert "[[x]]" in text


# ── _update_log ─────────────────────────────────────────────────────────


def test_update_log_no_file_noop(tmp_path) -> None:
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    p._update_log("promote", "Foo", ["concepts/foo.md"])  # should not raise


def test_update_log_appends(tmp_path) -> None:
    log = tmp_path / "log.md"
    log.write_text("# Log\n", encoding="utf-8")
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    p._update_log("promote", "Foo", ["concepts/foo.md"])
    text = log.read_text(encoding="utf-8")
    assert "promote | Foo" in text
    assert "- concepts/foo.md" in text


def test_update_log_multiple_files(tmp_path) -> None:
    log = tmp_path / "log.md"
    log.write_text("# Log\n", encoding="utf-8")
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    p._update_log("promote", "Foo", ["a.md", "b.md", "c.md"])
    text = log.read_text(encoding="utf-8")
    assert text.count("- ") >= 3


# ── upgrade_memory ──────────────────────────────────────────────────────


def test_upgrade_memory_missing_wiki_path(tmp_path) -> None:
    p = WikiUpgradePipeline(wiki_path=tmp_path / "nonexistent")
    result = p.upgrade_memory("m1", "content")
    assert result["status"] == "error"
    assert "does not exist" in result["reason"]


def test_upgrade_memory_no_llm(tmp_path) -> None:
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    result = p.upgrade_memory("m1", "content")
    assert result["status"] == "error"
    assert "No LLM" in result["reason"]


def test_upgrade_memory_llm_exception(tmp_path) -> None:
    def boom(prompt):
        raise RuntimeError("llm down")

    p = WikiUpgradePipeline(wiki_path=tmp_path, llm_call=boom)
    result = p.upgrade_memory("m1", "content")
    assert result["status"] == "error"
    assert "LLM call failed" in result["reason"]


def test_upgrade_memory_invalid_json(tmp_path) -> None:
    p = WikiUpgradePipeline(wiki_path=tmp_path, llm_call=lambda _: "not json at all")
    result = p.upgrade_memory("m1", "content")
    assert result["status"] == "error"
    assert "not valid JSON" in result["reason"]


def test_upgrade_memory_json_in_block(tmp_path) -> None:
    payload = {
        "page_type": "concept",
        "title": "Foo",
        "filename": "foo",
        "content": "# Foo\nSome text with [[bar]] and [[baz]] links.",
    }
    llm_out = f"Here is the JSON:\n```json\n{json.dumps(payload)}\n```"

    def llm(prompt):
        return llm_out

    p = WikiUpgradePipeline(wiki_path=tmp_path, llm_call=llm)
    result = p.upgrade_memory("m1", "content")
    # json.loads fails on wrapper text → regex extracts flat JSON with "page_type" key
    assert result["status"] == "ok"
    assert result["title"] == "Foo"


def test_upgrade_memory_success_concept(tmp_path) -> None:
    payload = {
        "page_type": "concept",
        "title": "Memory System",
        "filename": "memory-system",
        "content": "# Memory System\n\nSome [[foo]] content.",
    }
    p = WikiUpgradePipeline(wiki_path=tmp_path, llm_call=lambda _: json.dumps(payload))
    result = p.upgrade_memory("m1", "content here")
    assert result["status"] == "ok"
    assert result["page_type"] == "concept"
    assert result["title"] == "Memory System"
    assert result["filename"] == "memory-system"
    assert Path(result["path"]).exists()
    assert Path(result["path"]).read_text(encoding="utf-8").startswith("# Memory System")


def test_upgrade_memory_creates_subdir(tmp_path) -> None:
    payload = {
        "page_type": "entity",
        "title": "Alice",
        "filename": "alice",
        "content": "# Alice",
    }
    p = WikiUpgradePipeline(wiki_path=tmp_path, llm_call=lambda _: json.dumps(payload))
    result = p.upgrade_memory("m1", "content")
    assert (tmp_path / "entities" / "alice.md").exists()
    assert result["path"].endswith("entities/alice.md")


def test_upgrade_memory_unknown_type_defaults_concepts(tmp_path) -> None:
    payload = {
        "page_type": "weird",
        "title": "X",
        "filename": "x",
        "content": "# X",
    }
    p = WikiUpgradePipeline(wiki_path=tmp_path, llm_call=lambda _: json.dumps(payload))
    result = p.upgrade_memory("m1", "c")
    assert (tmp_path / "concepts" / "x.md").exists()
    assert result["status"] == "ok"


def test_upgrade_memory_missing_filename_falls_back_to_slug(tmp_path) -> None:
    payload = {
        "page_type": "concept",
        "title": "Hello World!",
        "content": "# HW",
    }
    p = WikiUpgradePipeline(wiki_path=tmp_path, llm_call=lambda _: json.dumps(payload))
    result = p.upgrade_memory("m1", "c")
    assert result["filename"] == "hello-world"
    assert (tmp_path / "concepts" / "hello-world.md").exists()


def test_upgrade_memory_marks_forgetting(tmp_path) -> None:
    calls = []

    class _FakeForgetting:
        def mark_upgraded_to_wiki(self, mid, rel):
            calls.append((mid, rel))

    payload = {
        "page_type": "concept",
        "title": "Foo",
        "filename": "foo",
        "content": "# Foo",
    }
    p = WikiUpgradePipeline(wiki_path=tmp_path, llm_call=lambda _: json.dumps(payload), forgetting=_FakeForgetting())
    p.upgrade_memory("m42", "c")
    assert calls == [("m42", "concepts/foo.md")]


def test_upgrade_memory_no_forgetting_noop(tmp_path) -> None:
    payload = {
        "page_type": "concept",
        "title": "Foo",
        "filename": "foo",
        "content": "# Foo",
    }
    p = WikiUpgradePipeline(wiki_path=tmp_path, llm_call=lambda _: json.dumps(payload))
    result = p.upgrade_memory("m1", "c")
    assert result["status"] == "ok"


def test_upgrade_memory_comparison_subdir(tmp_path) -> None:
    payload = {
        "page_type": "comparison",
        "title": "A vs B",
        "filename": "a-vs-b",
        "content": "# A vs B",
    }
    p = WikiUpgradePipeline(wiki_path=tmp_path, llm_call=lambda _: json.dumps(payload))
    result = p.upgrade_memory("m1", "c")
    assert (tmp_path / "comparisons" / "a-vs-b.md").exists()
    assert result["page_type"] == "comparison"


# ── batch_upgrade ───────────────────────────────────────────────────────


def test_batch_upgrade_empty(tmp_path) -> None:
    p = WikiUpgradePipeline(wiki_path=tmp_path)
    assert p.batch_upgrade([], {}) == []


def test_batch_upgrade_missing_content(tmp_path) -> None:
    payload = {
        "page_type": "concept",
        "title": "X",
        "filename": "x",
        "content": "# X",
    }
    p = WikiUpgradePipeline(wiki_path=tmp_path, llm_call=lambda _: json.dumps(payload))
    results = p.batch_upgrade(["m1", "m2"], {"m1": "content"})
    assert len(results) == 2
    assert results[0]["status"] == "ok"
    assert results[1]["status"] == "error"
    assert results[1]["reason"] == "Content not found"


def test_batch_upgrade_all_ok(tmp_path) -> None:
    payload = {
        "page_type": "concept",
        "title": "X",
        "filename": "x",
        "content": "# X",
    }
    p = WikiUpgradePipeline(wiki_path=tmp_path, llm_call=lambda _: json.dumps(payload))
    results = p.batch_upgrade(["m1", "m2"], {"m1": "c1", "m2": "c2"})
    assert len(results) == 2
    assert all(r["status"] == "ok" for r in results)
    assert results[0]["memory_id"] == "m1"
    assert results[1]["memory_id"] == "m2"


def test_batch_upgrade_overwrites_same_filename(tmp_path) -> None:
    payload = {
        "page_type": "concept",
        "title": "X",
        "filename": "x",
        "content": "# X",
    }
    p = WikiUpgradePipeline(wiki_path=tmp_path, llm_call=lambda _: json.dumps(payload))
    results = p.batch_upgrade(["m1", "m2"], {"m1": "c1", "m2": "c2"})
    # Both write to same file; last one wins
    assert (tmp_path / "concepts" / "x.md").exists()
    assert len(results) == 2
