"""pytest 共享 fixture 配置。"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# 禁用 ChromaDB/posthog 遥测：离线/弱网环境下其 socket 连接会无限阻塞，导致测试挂死
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "1")
# HF offline: avoid SentenceTransformer online revision check hanging on slow network
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Mock agent.memory_provider 模块（Hermes 框架依赖）
_mock_agent = MagicMock()
_mock_agent.memory_provider = MagicMock()
_mock_agent.memory_provider.MemoryProvider = object
sys.modules.setdefault("agent", _mock_agent)
sys.modules.setdefault("agent.memory_provider", _mock_agent.memory_provider)


@pytest.fixture
def omni_tmp_path() -> Generator[Path, None, None]:
    """提供临时目录路径。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=False)
def skip_if_no_chromadb():
    try:
        import chromadb  # noqa: F401
    except ImportError:
        pytest.skip("chromadb not installed")


@pytest.fixture(autouse=False)
def skip_if_no_sentence_transformers():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pytest.skip("sentence-transformers not installed")
