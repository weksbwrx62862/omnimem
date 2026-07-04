"""日志工具：PII 脱敏与日志安全。

为关键路径提供统一的日志内容脱敏函数，避免在日志中泄露敏感信息。
"""

from __future__ import annotations

import re
from typing import Any

# 常见的敏感字段名（不区分大小写）
_SENSITIVE_KEYS: tuple[str, ...] = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "cookie",
    "session",
    "private_key",
    "access_key",
    "accesskey",
    "key",
)

# 用于匹配 JSON/dict 中的敏感键值对
_SENSITIVE_PATTERN = re.compile(
    rf'("(?:{"|".join(_SENSITIVE_KEYS)})"\s*:\s*)"[^"]*"',
    re.IGNORECASE,
)


def sanitize_for_log(value: Any, max_length: int = 2000) -> str:
    """对日志内容进行脱敏处理。

    规则：
      - Fernet 令牌替换为 [加密令牌]
      - OpenAI / Anthropic 风格 API Key 替换为 sk-***
      - JSON/dict 中敏感键对应的值替换为 ***
      - key / token / secret / password 赋值替换为 ***
      - 超长十六进制随机串替换为 ***
      - 截断超长内容到 max_length
      - 非字符串输入会先使用 str() 转换

    Args:
        value: 待脱敏的内容
        max_length: 最大长度限制

    Returns:
        脱敏后的字符串
    """
    text = str(value)

    # Fernet 令牌（gAAAA...）
    text = re.sub(r"\bgAAAA[A-Za-z0-9_=-]{10,}", "[加密令牌]", text)

    # OpenAI / Anthropic 风格 API Key
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{10,}", "sk-***", text)

    # JSON/dict 中的敏感键值对
    text = _SENSITIVE_PATTERN.sub(r'\1"***"', text)

    # 通用 key / token / secret / password 赋值
    text = re.sub(
        r'(?i)(api[_-]?key|apikey|token|secret|password)\s*[:=]\s*["\']?[A-Za-z0-9_\-]{8,}["\']?',
        r"\1=***",
        text,
    )

    # 长十六进制随机串（>=32 字符），可能为密钥/哈希
    text = re.sub(r"\b[A-Fa-f0-9]{32,}\b", "***", text)

    if len(text) > max_length:
        text = text[:max_length] + "...[已截断]"
    return text
