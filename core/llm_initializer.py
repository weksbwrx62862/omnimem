"""LLM 客户端初始化器 — 负责 LLM 客户端的多级凭证回退和 Provider 匹配。

从 core/tool_router.py 拆分而来，解决 LLMClientManager 与 tool_router 之间的循环依赖。

职责:
  1. 多级凭证回退（env → hermes_env → hermes_config）
  2. Provider 匹配（读取 ~/.hermes/config.yaml 的 providers 列表）
  3. 模型选择策略（匹配 base_url 对应的 provider models）
  4. 凭证有效性检测
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Reflect 缓存 TTL（秒）— init_llm_client 创建客户端时使用，call_llm_for_reflect 也复用
_REFLECT_CACHE_TTL = 60.0


def init_llm_client(config: Any) -> Any:
    """初始化 LLM 客户端，含多级凭证回退和 Provider 匹配。

    凭证回退顺序:
      1. 环境变量（load_credentials_from_env）
      2. Hermes 环境变量（load_credentials_from_hermes_env）
      3. Hermes 配置文件（load_credentials_from_hermes_config）

    模型选择策略:
      1. 优先用匹配到 base_url 的 provider models
      2. 其次用 config_creds 的 models（仅当 base_url 也来自 config 时）
      3. 最后用 default_model
    """
    from omnimem.utils.llm_client import AsyncLLMClient

    creds = AsyncLLMClient.load_credentials_from_env()
    if not creds.get("api_key") or not creds.get("base_url"):
        creds.update(AsyncLLMClient.load_credentials_from_hermes_env())
    config_creds = AsyncLLMClient.load_credentials_from_hermes_config()
    if not creds.get("base_url"):
        creds["base_url"] = config_creds.get("base_url", "")
    if not creds.get("api_key"):
        creds["api_key"] = config_creds.get("api_key", "")
    # ★ R25修复ARCH-1 + P0-fix：model 选择策略
    # 必须匹配实际 base_url 对应的 provider 的 models 列表，
    # 否则会把 mimo-v2.5-pro 发给 deepseek API（400错误）
    actual_base_url = creds.get("base_url", "")
    default_model = config_creds.get("model") or config.get("default", "glm-5.1")

    # 从 config providers 中找到与实际 base_url 匹配的 provider，取其 models
    matched_models: list[str] = []
    try:
        from pathlib import Path

        import yaml
        cfg_file = Path.home() / ".hermes" / "config.yaml"
        if cfg_file.exists():
            cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            for _pname, pval in (cfg.get("providers") or {}).items():
                if isinstance(pval, dict) and pval.get("base_url") == actual_base_url:
                    matched_models = pval.get("models", [])
                    if matched_models:
                        logger.info("Matched provider '%s' for base_url %s, models: %s",
                                     _pname, actual_base_url, matched_models)
                        break
    except Exception as e:
        logger.warning("ToolRouter init_llm_client provider match failed: %s", e)

    # 优先用匹配到的 provider models，其次用 config_creds 的 models，最后用 default
    if matched_models:
        model = matched_models[0]
    elif config_creds.get("models"):
        # config_creds.models 来自第一个 provider，可能与 base_url 不匹配
        # 只在 base_url 也来自 config（未被 env 覆盖）时使用
        config_base_url = config_creds.get("base_url", "")
        if config_base_url and config_base_url == actual_base_url:
            model = config_creds["models"][0]
        else:
            model = default_model
    else:
        model = default_model
    logger.warning("Selected model=%s (actual_base_url=%s, default=%s)", model, actual_base_url, default_model)

    # ★ R25修复ARCH-1：凭证有效性检测
    has_api_key = bool(creds.get("api_key", "").strip())
    has_base_url = bool(creds.get("base_url", "").strip())
    if not has_api_key or not has_base_url:
        logger.warning(
            "AsyncLLMClient: LLM 凭证不完整 (api_key=%s, base_url=%s), "
            "Reflect/Recall 的 LLM 功能将不可用，回退到规则归纳",
            "有" if has_api_key else "缺失",
            "有" if has_base_url else "缺失",
        )

    llm_client = AsyncLLMClient(
        api_key=creds.get("api_key", ""),
        base_url=creds.get("base_url", ""),
        model=model,
        max_concurrent=3,
        timeout=30.0,
        cache_ttl=_REFLECT_CACHE_TTL,
    )
    logger.warning("AsyncLLMClient initialized: model=%s, has_creds=%s", model, has_api_key and has_base_url)
    return llm_client
