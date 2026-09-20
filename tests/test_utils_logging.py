"""utils/logging.py sanitize_for_log 补充单测（离线，避开 Fernet/加密依赖）。

聚焦 test_security.py 未覆盖的分支：JSON 键值掩码、赋值掩码、
长十六进制串、非字符串输入、以及无敏感内容时的原样透传。
"""
from __future__ import annotations

from omnimem.utils.logging import sanitize_for_log


def test_json_dict_sensitive_key_masked():
    out = sanitize_for_log('{"token": "mysecretvalue", "name": "bob"}')
    assert "mysecretvalue" not in out
    assert '"token": "***"' in out
    assert "bob" in out  # 非敏感键保留


def test_assignment_style_masked():
    out = sanitize_for_log("password=SuperSecret123")
    assert "SuperSecret123" not in out
    assert "password=***" in out


def test_long_hex_string_masked():
    hex_str = "deadbeef" * 4  # 32 位十六进制
    out = sanitize_for_log(f"digest={hex_str}")
    assert hex_str not in out
    assert "***" in out


def test_non_string_input_coerced():
    assert sanitize_for_log(12345) == "12345"
    assert sanitize_for_log(None) == "None"


def test_clean_text_passthrough():
    text = "hello world, nothing secret here"
    assert sanitize_for_log(text) == text


def test_truncation_marker():
    out = sanitize_for_log("x" * 300, max_length=50)
    assert out.endswith("...[已截断]")
    assert len(out) <= 50 + len("...[已截断]")
