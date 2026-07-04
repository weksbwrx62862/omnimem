"""备份管理器 — 负责数据目录的 tar.gz 备份和旧备份清理。"""

from __future__ import annotations

import logging
import tarfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class BackupManager:
    """管理 OmniMem 数据目录的备份与清理。

    职责:
      1. 将 omnimem 数据目录打包为 tar.gz 备份
      2. 清理旧备份，保留最近 N 个
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._last_backup_time: float = 0.0

    @property
    def last_backup_time(self) -> float:
        return self._last_backup_time

    @last_backup_time.setter
    def last_backup_time(self, value: float) -> None:
        self._last_backup_time = value

    def create_backup(self) -> tuple[str, int]:
        """将 omnimem 数据目录打包为 tar.gz 备份。

        备份路径: ~/.hermes/omnimem.bak/YYYYMMDD_HHMMSS.tar.gz
        返回: (备份路径, 备份字节数)
        """
        backup_dir = Path.home() / ".hermes" / "omnimem.bak"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{timestamp}.tar.gz"

        with tarfile.open(str(backup_path), "w:gz") as tar:
            tar.add(str(self._data_dir), arcname=self._data_dir.name)

        size = backup_path.stat().st_size
        self._last_backup_time = __import__("time").time()
        logger.info("OmniMem 备份完成: %s (%.1f KB)", backup_path, size / 1024)
        return str(backup_path), size

    def cleanup_old_backups(self, max_copies: int = 3) -> None:
        """清理旧备份，保留最近 max_copies 个。"""
        backup_dir = Path.home() / ".hermes" / "omnimem.bak"
        if not backup_dir.exists():
            return

        backups = sorted(backup_dir.glob("*.tar.gz"))
        if len(backups) > max_copies:
            for old in backups[:-max_copies]:
                old.unlink()
                logger.info("OmniMem 清理旧备份: %s", old)
