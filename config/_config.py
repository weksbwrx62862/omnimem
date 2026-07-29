"""OmniMem 配置管理。"""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_SCHEMA = {
    "save_interval": {"type": (int, float), "min": 1, "max": 3600, "default": 15},
    "retrieval_mode": {"type": str, "choices": ["rag", "hybrid", "vector", "bm25"], "default": "rag"},
    "vector_backend": {"type": str, "choices": ["chromadb", "qdrant", "faiss"], "default": "chromadb"},
    "vector_store_backend": {"type": str, "choices": ["chromadb", "qdrant", "faiss"], "default": "chromadb"},
    "qdrant_url": {"type": str, "default": "localhost:6333"},
    "max_prefetch_tokens": {"type": int, "min": 10, "max": 100000, "default": 300},
    "budget_tokens": {"type": int, "min": 100, "max": 100000, "default": 4000},
    "fact_threshold": {"type": int, "min": 1, "max": 1000, "default": 10},
    "enable_reranker": {"type": bool, "default": False},
    "conflict_strategy": {"type": str, "choices": ["latest", "add_only", "merge", "reject"], "default": "latest"},
    "conflict_scan_max_group_size": {"type": int, "min": 2, "max": 1000, "default": 50},
    "default_privacy": {"type": str, "choices": ["public", "team", "personal", "secret"], "default": "personal"},
    "auto_memorize": {"type": bool, "default": True},
    "kv_cache_threshold": {"type": int, "min": 1, "max": 1000, "default": 10},
    "kv_cache_max": {"type": int, "min": 10, "max": 10000, "default": 100},
    "lora_base_model": {"type": str, "default": "Qwen2.5-7B"},
    "lora_rank": {"type": int, "min": 1, "max": 256, "default": 16},
    "lora_alpha": {"type": int, "min": 1, "max": 512, "default": 32},
    "sync_mode": {"type": str, "choices": ["none", "file_lock", "changelog"], "default": "none"},
    "sync_interval": {"type": (int, float), "min": 1, "max": 3600, "default": 30},
    "sync_conflict_resolution": {"type": str, "choices": ["latest_wins", "manual", "merge"], "default": "latest_wins"},
    "forgetting_active_days": {"type": (int, float), "min": 1, "max": 365, "default": 7},
    "forgetting_consolidating_days": {"type": (int, float), "min": 1, "max": 365, "default": 30},
    "forgetting_archived_days": {"type": (int, float), "min": 1, "max": 3650, "default": 90},
    "enable_compression": {"type": bool, "default": False},
    "llm_backend": {"type": str, "choices": ["openai", "ollama", "anthropic"], "default": "openai"},
    "ollama_model": {"type": str, "default": "llama3"},
    "ollama_base_url": {"type": str, "default": "http://localhost:11434"},
    "anthropic_model": {"type": str, "default": "claude-3-haiku-20240307"},
    # OPT-1: LLM 蒸馏引擎 — 定期将 auto-captured raw facts 喂给 LLM 提炼
    "distill_enabled": {"type": bool, "default": True},
    "distill_model": {"type": str, "default": ""},  # 空=使用主模型; 可指定自定义模型名
    "distill_interval": {"type": int, "min": 5, "max": 1000, "default": 15},
    # ★ M7-13: 统一 LLM 模型配置来源（替代 AsyncLLMClient 硬编码 glm-5.1）
    "llm_model": {"type": str, "default": "glm-5.1"},  # 空="" 表示从 hermes config 自动匹配
    # ★ M7-14: Reranker 设备可配（cpu/cuda/cuda:0/mps 等）
    "reranker_device": {"type": str, "default": "cpu"},
    # ★ M6-7: FTS5 替代 BM25 灰度开关（需 use_unified_index=True 才生效）
    # ★ M6-5: UnifiedMemoryIndex 灰度开关（合并 ThreeLevelIndex+MetaStore 为单库）
    "use_unified_index": {"type": bool, "default": False},
    # ★ P1: 事实抽取模式（rule=纯规则 / hybrid=LLM 精炼 / llm=纯LLM）
    "extraction_mode": {"type": str, "choices": ["rule", "hybrid", "llm"], "default": "hybrid"},
    "use_fts5": {"type": bool, "default": False},
    # ★ OPT: 检索超时降级 — recall 整体超时 + 策略切换
    "recall_timeout_ms": {"type": int, "min": 100, "max": 30000, "default": 5000},
    "recall_strategy": {"type": str, "choices": ["hybrid", "keyword", "embedding"], "default": "hybrid"},
    # archived 记忆召回策略: downweight=降权保留(sealed), exclude=彻底排除
    "archive_recall_policy": {"type": str, "choices": ["downweight", "exclude"], "default": "downweight"},
    # 项目召回严格隔离: True 时指定 project 查询仅返回同名 project 记忆(空标签也排除)
    "project_recall_strict": {"type": bool, "default": False},
    # ★ OPT: Pipeline 调度器 — L2/L3 自动触发
    "pipeline_every_n_conversations": {"type": int, "min": 1, "max": 100, "default": 5},
    "pipeline_enable_warmup": {"type": bool, "default": True},
    "pipeline_l2_delay_after_l1_seconds": {"type": int, "min": 10, "max": 3600, "default": 90},
    "persona_trigger_every_n": {"type": int, "min": 5, "max": 1000, "default": 15},
    "persona_min_interval_seconds": {"type": int, "min": 60, "max": 86400, "default": 300},
    # ★ OPT: Mermaid 符号化压缩
    "mermaid_tool_log_patterns": {"type": list, "default": []},
    "max_refs_age_days": {"type": int, "min": 1, "max": 365, "default": 30},
    # ★ OPT: OpenViking 目录递归检索
    "enable_catalog": {"type": bool, "default": True},
    "catalog_weight": {"type": float, "min": 0.1, "max": 10.0, "default": 2.0},
    # ★ OPT: 三层渐进式披露
    "max_overview_chars": {"type": int, "min": 50, "max": 1000, "default": 200},
    # ★ OPT: 可视化检索轨迹
    "enable_trace_by_default": {"type": bool, "default": False},
    # ★ OPT: prefetch 是否记录访问到遗忘曲线（驱动热度分类）
    "prefetch_record_access": {"type": bool, "default": True},
    "backup_interval_hours": {"type": int, "min": 1, "max": 720, "default": 24},
    "backup_max_copies": {"type": int, "min": 1, "max": 100, "default": 3},
    "debug_mode": {"type": bool, "default": False, "description": "调试模式"},
    "enable_encryption": {"type": bool, "default": True},
    "audit_log_max_rows": {"type": int, "min": 1000, "max": 10000000, "default": 100000},
    "audit_log_retention_days": {"type": int, "min": 1, "max": 3650, "default": 90},
    "health_check_interval": {"type": int, "min": 1, "max": 1000, "default": 10},
    "query_cache_ttl": {"type": (int, float), "min": 0, "max": 3600, "default": 60},
    # REST API / MCP 安全相关配置
    # api_key 默认在运行时生成 32 字节随机 hex，禁止空字符串启用生产服务
    "api_key": {"type": str, "default": ""},
    "admin_token": {"type": str, "default": ""},
    "mcp_require_api_key": {"type": bool, "default": False},
    "api_rate_limit_per_minute": {"type": int, "min": 1, "max": 10000, "default": 60},
    "cors_allowed_origins": {"type": list, "default": []},
    # 导出/备份加密密钥：未配置时默认拒绝无密钥导出
    "export_key": {"type": str, "default": ""},
    "write_buffer_threshold": {"type": int, "min": 1, "max": 200, "default": 20},
    "audit_interval_turns": {"type": int, "min": 5, "max": 500, "default": 50},
    # Task 3: 检索参数可配置化
    "rrf_k": {"type": int, "min": 1, "max": 1000, "default": 60},
    "rrf_min_score": {"type": float, "min": 0.0, "max": 1.0, "default": 0.035},
    # ★ 缺陷2: 绝对语义相关性地板(向量余弦下限), 低于且无词法命中→零结果; 0 表示关闭
    "min_relevance_score": {"type": float, "min": 0.0, "max": 1.0, "default": 0.35},
    # ★ 缺陷3: 偏好记忆查询相关性门控开关
    "preference_relevance_gate": {"type": bool, "default": True},
    "circuit_breaker_threshold": {"type": int, "min": 1, "max": 100, "default": 3},
    "circuit_breaker_cooldown_seconds": {"type": (int, float), "min": 1, "max": 3600, "default": 60},
    "max_sync_turn_entries": {"type": int, "min": 10, "max": 100000, "default": 1000},
    # Task 3: 索引重建并行化参数
    "rebuild_batch_size": {"type": int, "min": 1, "max": 1024, "default": 32},
    "rebuild_max_workers": {"type": int, "min": 1, "max": 64, "default": 4},
    # Task 2 & Task 5: embedding / vector_store 后端可配置化
    "embedding.provider": {
        "type": str,
        "choices": ["sentence_transformers", "openai", "onnx"],
        "default": "sentence_transformers",
    },
    "embedding.model_name": {"type": str, "default": "all-MiniLM-L6-v2"},
    # 本地嵌入模型目录(优先于 model_name); 换 bge-m3 等本地模型时指向模型目录
    "embedding_model_path": {"type": str, "default": ""},
    "embedding.api_key": {"type": str, "default": ""},
    "embedding.base_url": {"type": str, "default": ""},
    "vector_store.provider": {
        "type": str,
        "choices": ["chroma", "milvus"],
        "default": "chroma",
    },
    "vector_store.collection_name": {"type": str, "default": "omnimem"},
    "vector_store.persist_dir": {"type": str, "default": "/tmp/omnimem/storage/chroma"},
    "vector_store.uri": {"type": str, "default": "http://localhost:19530"},
    "vector_store.token": {"type": str, "default": ""},
    "vector_store.embedding_dimension": {"type": int, "min": 1, "max": 10000, "default": 384},
    "vector_store.metric_type": {"type": str, "default": "COSINE"},
    "vector_store.consistency_level": {"type": str, "default": "Bounded"},
    # Task 5: 分布式锁后端可配置化
    "lock.backend": {
        "type": str,
        "choices": ["file", "redis"],
        "default": "file",
    },
    "lock.redis_url": {"type": str, "default": "redis://localhost:6379/0"},
}

DEFAULTS = {k: v["default"] for k, v in _CONFIG_SCHEMA.items()}


def _flatten_dict(nested: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """将嵌套字典展平为点分键，支持 YAML 嵌套配置。"""
    items: list[tuple[str, Any]] = []
    for k, v in nested.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


class OmniMemConfig:
    """OmniMem 配置管理器（支持热重载）。"""

    def __init__(self, config_dir: Path):
        self._config_dir = config_dir
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._config_path = self._config_dir / "config.yaml"
        self._values: dict[str, Any] = dict(DEFAULTS)
        self._last_mtime: float = 0.0
        self._load()
        # 禁止空 api_key：未配置或显式置空时强制生成随机 32 字节 hex
        if not self._values.get("api_key"):
            self._values["api_key"] = secrets.token_hex(32)
            self.save()

    def reload(self, force: bool = False) -> bool:
        """检测配置文件是否变更，若变更则重新加载。返回是否发生重载。"""
        if not self._config_path.exists():
            return False
        try:
            mtime = os.path.getmtime(self._config_path)
            if not force and mtime <= self._last_mtime:
                return False
            self._last_mtime = mtime
            self._load()
            logger.info("OmniMemConfig reloaded from %s", self._config_path)
            return True
        except Exception as e:
            logger.warning("Config reload failed: %s", e)
            return False

    def _validate(self, key: str, value: Any) -> None:
        schema = _CONFIG_SCHEMA.get(key)
        if not schema:
            return
        expected_type = schema["type"]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Config '{key}' expects {expected_type}, got {type(value).__name__}"
            )
        if "min" in schema and value < schema["min"]:
            raise ValueError(
                f"Config '{key}' must be >= {schema['min']}, got {value}"
            )
        if "max" in schema and value > schema["max"]:
            raise ValueError(
                f"Config '{key}' must be <= {schema['max']}, got {value}"
            )
        if "choices" in schema and value not in schema["choices"]:
            raise ValueError(
                f"Config '{key}' must be one of {schema['choices']}, got {value}"
            )
        # 安全关键配置禁止空字符串
        if key == "api_key" and not value:
            raise ValueError("Config 'api_key' 不能为空")

    def _load(self) -> None:
        """从配置文件加载。"""
        if not self._config_path.exists():
            return
        try:
            import yaml

            with open(self._config_path, encoding="utf-8") as f:
                file_values = yaml.safe_load(f) or {}
            # 支持嵌套配置（如 embedding.provider）
            flat_values = _flatten_dict(file_values)
            for k, v in flat_values.items():
                try:
                    self._validate(k, v)
                except (TypeError, ValueError) as e:
                    logger.warning("Invalid config '%s': %s — using default", k, e)
                    continue
                self._values[k] = v
            self._last_mtime = os.path.getmtime(self._config_path)
        except ImportError:
            logger.warning("yaml not available — using defaults")
        except Exception as e:
            logger.warning("Config load failed: %s", e)

    def save(self, values: dict[str, Any] | None = None) -> None:
        """保存配置到文件。"""
        if values:
            self._values.update(values)
        try:
            import yaml

            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._values, f, allow_unicode=True, default_flow_style=False)
        except ImportError:
            logger.warning("yaml not available — config not saved")
        except Exception as e:
            logger.warning("Config save failed: %s", e)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值。"""
        return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置配置值。"""
        self._validate(key, value)
        self._values[key] = value

    @property
    def values(self) -> dict[str, Any]:
        """返回所有配置值。"""
        return dict(self._values)
