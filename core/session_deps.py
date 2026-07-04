"""SessionManager 依赖封装 — 将长参数列表聚合为 dataclass。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class SessionDependencies:
    """SessionManager 运行所需的全部外部依赖。

    将原本 __init__ 的二十余个位置/关键字参数收敛到单一对象，
    降低调用方出错概率并提升可维护性。
    """

    config: Any
    perception: Any
    store_service: Any
    retriever: Any
    bg_executor: Any
    forgetting: Any
    consolidation: Any
    kv_cache: Any
    lora_trainer: Any
    store: Any
    index: Any
    auditor: Any
    saga: Any
    prefetch_executor: Any
    pipeline_scheduler: Any | None = None
    distill_init_fn: Callable[[], None] | None = None
    distillation_engine: Any = None
    session_id: str = ""
    should_write: bool = True
    strip_system_injections_fn: Callable[[str], str] | None = None
    should_store_fn: Callable[[str], bool] | None = None
    handle_memorize_fn: Callable[[dict[str, Any]], str] | None = None
    retry_index_add_fn: Callable[[str], None] | None = None
    retry_retriever_add_fn: Callable[[str], None] | None = None
    retry_kg_extract_fn: Callable[[str], None] | None = None
    create_backup_fn: Callable[[], tuple[str, int]] | None = None
    cleanup_old_backups_fn: Callable[[int], None] | None = None
