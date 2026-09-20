#!/usr/bin/env bash
# 本地全量测试入口（**兜底路径**）。
#
# 首选做法已变成直接用一个健康的 venv，不需要本脚本：
#     .venv/bin/python -m pytest tests/
# `.venv` 于 2026-09-20 重建（uv + CPython 3.13.13），其内 cryptography 50.0.1 /
# packaging 26.3 均来自 venv 本地，且 include-system-site-packages = false，
# /usr/lib/python3/dist-packages 根本不进 sys.path，遮蔽问题不复存在。
# 已实测：env -u PYTHONPATH .venv/bin/python -m pytest tests/ -q
#         → 2680 passed in 267.45s。
# 本脚本保留的价值：在 venv 缺失/损坏时，仅靠系统解释器也能复现全量测试。
#
# 它要绕开的病因见 docs/pending_work.md §1：
# 本机 PYTHONPATH 长期含 /usr/lib/python3/dist-packages，会让 Debian 版
# cryptography 46.0.5（其 _rust 是空目录 → namespace package）遮蔽 pip 版，
# 表现为 ImportError: cannot import name 'exceptions' from
# 'cryptography.hazmat.bindings._rust'。
# 该路径又不能删：packaging 曾只存在于此处，pytest 的 _checkversion() 启动即需，
# 且发生在任何 conftest 加载之前，故 conftest 里重排 sys.path 无效。
# 唯一可行的语义是"append 到 sys.path 末尾"，PYTHONPATH 表达不了，只能内联 python。
#
# 用法：
#   ./run_tests.sh                      # 全量
#   ./run_tests.sh tests/test_trust.py  # 覆盖默认目标
#   ./run_tests.sh tests/ -k fusion     # 追加任意 pytest 参数
set -euo pipefail
cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ANONYMIZED_TELEMETRY=False
export CHROMA_TELEMETRY=False
export POSTHOG_DISABLED=1

if [ "$#" -eq 0 ]; then
  target=("tests/")
else
  target=("$@")
fi

exec env PYTHONPATH=/home/xxh/.hermes/plugins${PYTHONPATH:+:$PYTHONPATH} \
  /home/xxh/.local/bin/python3.13 -c '
import sys
sys.path.append("/usr/lib/python3/dist-packages")
import pytest
sys.exit(pytest.main(sys.argv[1:]))
' -q --tb=short "${target[@]}"
