"""OmniMem 配置管理。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 默认配置
DEFAULTS = {
    "save_interval": 15,
    "retrieval_mode": "rag",
    "vector_backend": "chromadb",
    "max_prefetch_tokens": 300,
    "budget_tokens": 4000,
    "fact_threshold": 10,  # Consolidation 触发阈值
    "enable_reranker": False,
    "conflict_strategy": "latest",
    "default_privacy": "personal",
    "auto_memorize": True,
    "kv_cache_threshold": 10,  # KV Cache 自动预填充阈值
    "kv_cache_max": 100,  # KV Cache 最大条目数
    "lora_base_model": "Qwen2.5-7B",  # LoRA 基座模型
    "lora_rank": 16,  # LoRA 秩
    "lora_alpha": 32,  # LoRA alpha
    "sync_mode": "none",  # 同步模式: none / file_lock / changelog
    "sync_interval": 30,  # 同步间隔(秒)
    "sync_conflict_resolution": "latest_wins",  # 同步冲突解决策略
}


class OmniMemConfig:
    """OmniMem 配置管理器（支持热重载）。"""

    def __init__(self, config_dir: Path):
        self._config_dir = config_dir
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._config_path = self._config_dir / "config.yaml"
        self._values: dict[str, Any] = dict(DEFAULTS)
        self._last_mtime: float = 0.0
        self._load()

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
            logger.debug("Config reload failed: %s", e)
            return False

    def _load(self) -> None:
        """从配置文件加载。"""
        if not self._config_path.exists():
            return
        try:
            import yaml

            with open(self._config_path, encoding="utf-8") as f:
                file_values = yaml.safe_load(f) or {}
            self._values.update(file_values)
            self._last_mtime = os.path.getmtime(self._config_path)
        except ImportError:
            logger.debug("yaml not available — using defaults")
        except Exception as e:
            logger.debug("Config load failed: %s", e)

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
            logger.debug("yaml not available — config not saved")
        except Exception as e:
            logger.debug("Config save failed: %s", e)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值。"""
        return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置配置值。"""
        self._values[key] = value

    @property
    def values(self) -> dict[str, Any]:
        """返回所有配置值。"""
        return dict(self._values)
