"""改进项 #8 导入器单测（纯函数，离线，无需 store/LLM）。"""
from __future__ import annotations

import json
from datetime import datetime

from omnimem.importers import SUPPORTED_SOURCES, _base, convert_file, convert_source


def _is_iso(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False


def test_supported_sources_present():
    for s in ("mem0", "letta", "zep", "graphiti", "cognee", "auto"):
        assert s in SUPPORTED_SOURCES


def test_mem0_results_container():
    data = {"results": [
        {"id": "a1", "memory": "User prefers PostgreSQL", "created_at": "2026-01-02T03:04:05Z", "metadata": {"categories": ["preference"]}},
        {"id": "a2", "memory": "Deployed to k8s", "created_at": 1700000000},
    ]}
    env = convert_source(data, source="mem0", wing="team")
    assert env["source"] == "mem0"
    assert env["count"] == 2
    first = env["memories"][0]
    assert first["content"] == "User prefers PostgreSQL"
    assert first["type"] == "preference"
    assert first["wing"] == "team"
    assert _is_iso(first["created_at"])
    # epoch 秒也应被规整为 ISO
    assert _is_iso(env["memories"][1]["created_at"])


def test_letta_core_blocks_and_passages():
    data = {
        "memory_blocks": [{"label": "persona", "value": "I am a helpful coding assistant."}],
        "passages": [{"id": "p1", "text": "The repo uses Python 3.13", "tags": ["stack"]}],
    }
    env = convert_source(data, source="letta")
    texts = [m["content"] for m in env["memories"]]
    assert "I am a helpful coding assistant." in texts
    assert "The repo uses Python 3.13" in texts
    pref = next(m for m in env["memories"] if m["content"].startswith("I am"))
    assert pref["type"] == "preference"


def test_zep_facts():
    data = {"facts": [{"id": "f1", "text": "Alice leads the backend team", "type": "entity"}]}
    env = convert_source(data, source="zep")
    assert env["memories"][0]["content"] == "Alice leads the backend team"
    assert env["memories"][0]["type"] == "entity"


def test_graphiti_edges_and_nodes():
    data = {
        "edges": [{"uuid": "e1", "fact": "Bob joined Acme in 2024", "valid_at": "2024-05-01T00:00:00Z", "name": "employment"}],
        "nodes": [{"uuid": "n1", "name": "Acme", "summary": "A software company", "labels": ["org"]}],
    }
    env = convert_source(data, source="graphiti")
    contents = {m["content"] for m in env["memories"]}
    assert "Bob joined Acme in 2024" in contents
    assert "A software company" in contents
    edge = next(m for m in env["memories"] if m["content"].startswith("Bob"))
    assert edge["created_at"].startswith("2024-05-01")
    node = next(m for m in env["memories"] if m["content"].startswith("A software"))
    assert node["type"] == "entity"


def test_cognee_data_points():
    data = {"data_points": [{"id": "c1", "text": "Migration finished on 2026-06", "type": "event"}]}
    env = convert_source(data, source="cognee")
    assert env["memories"][0]["content"] == "Migration finished on 2026-06"
    assert env["memories"][0]["type"] == "event"


def test_empty_and_missing_content_skipped():
    data = {"results": [{"memory": "ok one"}, {"note": ""}, {}, "plain string fact"]}
    env = convert_source(data, source="mem0")
    contents = [m["content"] for m in env["memories"]]
    assert "ok one" in contents
    assert "plain string fact" in contents
    assert "" not in contents


def test_native_envelope_is_importer_compatible():
    env = convert_source({"results": [{"memory": "hello"}]}, source="mem0")
    assert env["version"] == "2.0"
    assert env["encrypted"] is False
    assert isinstance(env["memories"], list) and env["memories"]
    rec = env["memories"][0]
    for key in ("memory_id", "content", "type", "wing", "room", "privacy", "confidence", "created_at"):
        assert key in rec


def test_convert_file_roundtrip(tmp_path):
    src = tmp_path / "in.json"
    src.write_text(json.dumps({"results": [{"memory": "A fact from file"}]}), encoding="utf-8")
    out = tmp_path / "native.json"
    env = convert_file(src, source="mem0", output_path=out, wing="imported")
    assert out.exists()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["count"] == env["count"] == 1
    assert on_disk["memories"][0]["wing"] == "imported"


def test_unknown_source_raises():
    try:
        convert_source({"results": []}, source="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown source")


def test_infer_type_keywords():
    pref = _base._infer_type("user prefers dark mode", "", [])
    event = _base._infer_type("we held a meeting yesterday", "", ["event"])
    fact = _base._infer_type("the capital is Paris", "", [])
    assert pref == "preference"
    assert event == "event"
    assert fact == "fact"
