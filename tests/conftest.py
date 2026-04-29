"""pytest 共享 fixture 配置。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path() -> Path:
    """提供临时目录路径。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
