"""
Plur 集成配置
"""

import os
from typing import Any


class PlurConfig:
    """Plur 配置管理"""

    # 默认配置
    DEFAULT_CONFIG = {
        "plur": {
            "endpoint": "http://localhost:8080",
            "api_version": "v1",
            "timeout": 30,
            "retry_count": 3,
            "retry_delay": 5
        },
        "sync": {
            "auto_sync_enabled": True,
            "sync_interval_seconds": 300,
            "batch_size": 50,
            "conflict_strategy": "merge",  # local, remote, newest, highest_confidence, merge
            "max_conflict_queue_size": 100
        },
        "federation": {
            "enabled": True,
            "max_instances": 10,
            "query_timeout": 60,
            "aggregation_strategy": "confidence_weighted"
        },
        "cache": {
            "local_cache_size": 1000,
            "remote_cache_size": 5000,
            "cache_ttl_seconds": 3600
        }
    }

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or os.path.expanduser("~/.hermes/plugins/omnimem/plur_config.json")
        self.config = self.DEFAULT_CONFIG.copy()
        self._load_config()

    def _load_config(self):
        """加载配置"""
        try:
            if os.path.exists(self.config_path):
                import json
                with open(self.config_path, encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self._deep_update(self.config, loaded_config)
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")

    def _deep_update(self, base: dict, update: dict):
        """深度更新字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def save_config(self):
        """保存配置"""
        try:
            import json
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save config to {self.config_path}: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值，支持点号路径"""
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """设置配置值，支持点号路径"""
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    @property
    def plur_endpoint(self) -> str:
        """获取 Plur 端点"""
        return self.get("plur.endpoint", "http://localhost:8080")

    @plur_endpoint.setter
    def plur_endpoint(self, value: str):
        """设置 Plur 端点"""
        self.set("plur.endpoint", value)

    @property
    def sync_interval(self) -> int:
        """获取同步间隔"""
        return self.get("sync.sync_interval_seconds", 300)

    @sync_interval.setter
    def sync_interval(self, value: int):
        """设置同步间隔"""
        self.set("sync.sync_interval_seconds", value)

    @property
    def auto_sync_enabled(self) -> bool:
        """获取自动同步状态"""
        return self.get("sync.auto_sync_enabled", True)

    @auto_sync_enabled.setter
    def auto_sync_enabled(self, value: bool):
        """设置自动同步状态"""
        self.set("sync.auto_sync_enabled", value)

    @property
    def conflict_strategy(self) -> str:
        """获取冲突解决策略"""
        return self.get("sync.conflict_strategy", "merge")

    @conflict_strategy.setter
    def conflict_strategy(self, value: str):
        """设置冲突解决策略"""
        self.set("sync.conflict_strategy", value)


# 全局配置实例
_config: PlurConfig | None = None


def get_config() -> PlurConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = PlurConfig()
    return _config


def reload_config():
    """重新加载配置"""
    global _config
    _config = None
    return get_config()
