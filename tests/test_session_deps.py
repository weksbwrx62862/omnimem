"""core/session_deps.py SessionDependencies 默认值单测。"""
from __future__ import annotations

from dataclasses import MISSING, fields

from omnimem.core.session_deps import SessionDependencies

_REQUIRED = (
    "config", "perception", "store_service", "retriever", "bg_executor",
    "forgetting", "consolidation", "kv_cache", "lora_trainer", "store",
    "index", "auditor", "saga", "prefetch_executor",
)


def _required_kwargs() -> dict:
    return {name: i for i, name in enumerate(_REQUIRED)}


def test_all_required_fields_are_declared():
    actual_required = {
        f.name for f in fields(SessionDependencies)
        if f.default is MISSING and f.default_factory is MISSING
    }
    assert actual_required == set(_REQUIRED)


def test_optional_defaults():
    d = SessionDependencies(**_required_kwargs())
    assert d.pipeline_scheduler is None
    assert d.distillation_engine is None
    assert d.session_id == ""
    assert d.should_write is True
    for fn_attr in (
        "distill_init_fn", "strip_system_injections_fn", "should_store_fn",
        "handle_memorize_fn", "retry_index_add_fn", "retry_retriever_add_fn",
        "retry_kg_extract_fn", "create_backup_fn", "cleanup_old_backups_fn",
    ):
        assert getattr(d, fn_attr) is None


def test_optional_overrides():
    d = SessionDependencies(**_required_kwargs(), session_id="s1", should_write=False)
    assert d.session_id == "s1"
    assert d.should_write is False
