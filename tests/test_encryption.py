"""Governance Encryption 模块测试。

覆盖: MemoryEncryption 初始化、加密/解密、降级、种子缺失、加密状态透明化
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest
from omnimem.config import OmniMemConfig
from omnimem.governance.encryption import (
    _UNENCRYPTED_PREFIX,
    EncryptionUnavailableError,
    MemoryEncryption,
)
from omnimem.governance.privacy import PrivacyManager
from omnimem.memory.drawer_closet import DrawerClosetStore


class TestMemoryEncryptionInit:
    def test_with_session_seed(self):
        enc = MemoryEncryption(session_seed="test-seed-123")
        assert enc._encryption_enabled is True

    def test_without_seed_no_env_var(self):
        """无种子且无环境变量时，加密降级为不可用。"""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OMNIMEM_ENCRYPTION_KEY", None)
            enc = MemoryEncryption(session_seed="")
            assert enc._encryption_enabled is False

    def test_env_var_fallback(self):
        """有环境变量时应启用加密。"""
        with patch.dict(os.environ, {"OMNIMEM_ENCRYPTION_KEY": "env-key-123"}):
            enc = MemoryEncryption(session_seed="")
            assert enc._encryption_enabled is True

    def test_kms_manager_provided(self):
        """KMS 管理器提供的 key 应被使用。"""
        kms = MagicMock()
        kms.get_encryption_key.return_value = b"a" * 32
        MemoryEncryption(kms_manager=kms)
        kms.get_encryption_key.assert_called_once()


class TestMemoryEncryptionEncryptDecrypt:
    def test_encrypt_decrypt_roundtrip(self):
        """加密后解密应恢复原文。"""
        enc = MemoryEncryption(session_seed="roundtrip-test")
        if not enc.is_available():
            pytest.skip("cryptography not installed")
        plaintext = "secret memory content"
        encrypted = enc.encrypt(plaintext)
        assert encrypted != plaintext
        decrypted = enc.decrypt(encrypted)
        assert decrypted == plaintext

    def test_encrypt_empty_string(self):
        enc = MemoryEncryption(session_seed="empty-test")
        if not enc.is_available():
            pytest.skip("cryptography not installed")
        result = enc.encrypt("")
        assert result == ""

    def test_decrypt_empty_string(self):
        enc = MemoryEncryption(session_seed="empty-decrypt")
        result = enc.decrypt("")
        assert result == ""

    def test_disabled_mode_raises_error(self):
        """加密不可用时，encrypt 应抛出 EncryptionUnavailableError。"""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OMNIMEM_ENCRYPTION_KEY", None)
            enc = MemoryEncryption(session_seed="")
            assert enc._encryption_enabled is False
            assert enc.is_available() is False
            with pytest.raises(EncryptionUnavailableError):
                enc.encrypt("sensitive data")

    def test_encrypt_with_status_enabled(self):
        """cryptography 可用时，encrypt_with_status 返回 enabled。"""
        enc = MemoryEncryption(session_seed="status-enabled")
        if not enc.is_available():
            pytest.skip("cryptography not installed")
        result = enc.encrypt_with_status("sensitive data")
        assert result["encryption_status"] == "enabled"
        assert not result["content"].startswith(_UNENCRYPTED_PREFIX)

    def test_encrypt_with_status_disabled_raises(self, monkeypatch):
        """cryptography 不可用时，encrypt_with_status 抛出 EncryptionUnavailableError。"""
        monkeypatch.setitem(sys.modules, "cryptography", None)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OMNIMEM_ENCRYPTION_KEY", None)
            enc = MemoryEncryption(session_seed="")
            with pytest.raises(EncryptionUnavailableError):
                enc.encrypt_with_status("sensitive data")


class TestMemoryEncryptionIsAvailable:
    def test_available_with_seed(self):
        enc = MemoryEncryption(session_seed="avail-test")
        # depends on cryptography package
        result = enc.is_available()
        assert isinstance(result, bool)

    def test_unavailable_when_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OMNIMEM_ENCRYPTION_KEY", None)
            enc = MemoryEncryption(session_seed="")
            assert enc.is_available() is False


class TestMemoryEncryptionIsEncrypted:
    def test_fernet_token_detected(self):
        enc = MemoryEncryption(session_seed="detect-test")
        assert enc.is_encrypted("gAAAAABh") is True

    def test_unencrypted_prefix_detected(self):
        enc = MemoryEncryption(session_seed="detect-test")
        assert enc.is_encrypted(f"{_UNENCRYPTED_PREFIX}data") is True

    def test_plain_text_not_detected(self):
        enc = MemoryEncryption(session_seed="detect-test")
        assert enc.is_encrypted("just plain text") is False

    def test_empty_string(self):
        enc = MemoryEncryption(session_seed="detect-test")
        assert enc.is_encrypted("") is False


class TestSDKMemorizeEncryptionStatus:
    def test_sdk_memorize_secret_returns_encryption_status(self):
        """SDK memorize secret 级记忆时返回结果应包含 encryption_status。"""
        pytest.importorskip("chromadb", reason="chromadb not installed")

        from omnimem.sdk import OmniMemSDK

        tmpdir = tempfile.mkdtemp(prefix=f"omnimem_enc_{uuid.uuid4().hex[:8]}_")
        try:
            sdk = OmniMemSDK(storage_dir=tmpdir)
            result = sdk.memorize("my secret token", memory_type="fact", privacy="secret")
            assert result["status"] == "stored"
            assert result["privacy"] == "secret"
            assert "encryption_status" in result
            assert result["encryption_status"] == "enabled"
            sdk.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestEncryptionDefaultsAndStoreIntegration:
    """Task 3: 加密默认启用与 secret 写入拒绝策略。"""

    def test_enable_encryption_default_true(self, tmp_path):
        """默认配置中 enable_encryption 应为 True。"""
        config = OmniMemConfig(tmp_path)
        assert config.get("enable_encryption") is True

    def test_secret_write_rejected_without_key(self, tmp_path):
        """加密启用但无有效密钥时，secret 级写入应被拒绝。"""
        config = OmniMemConfig(tmp_path)
        store = DrawerClosetStore(tmp_path / "palace", config=config)
        with pytest.raises(EncryptionUnavailableError, match="未配置有效加密密钥"):
            store.add(
                wing="personal",
                room="r1",
                content="top secret",
                memory_type="fact",
                privacy="secret",
            )

    def test_secret_write_accepted_with_valid_key(self, tmp_path):
        """配置启用加密并绑定有效 PrivacyManager 时，secret 级写入应成功。"""
        config = OmniMemConfig(tmp_path)
        store = DrawerClosetStore(tmp_path / "palace", config=config)
        pm = PrivacyManager(session_id="test-session")
        store.bind_privacy_manager(pm)
        mid = store.add(
            wing="personal",
            room="r1",
            content="top secret",
            memory_type="fact",
            privacy="secret",
        )
        entry = store.get(mid)
        assert entry is not None
        assert entry["content"] == "top secret"
        assert entry["privacy"] == "secret"

    def test_non_secret_write_not_affected(self, tmp_path):
        """加密启用不影响非 secret 级记忆写入。"""
        config = OmniMemConfig(tmp_path)
        store = DrawerClosetStore(tmp_path / "palace", config=config)
        mid = store.add(
            wing="personal",
            room="r1",
            content="public info",
            memory_type="fact",
            privacy="public",
        )
        entry = store.get(mid)
        assert entry is not None
        assert entry["content"] == "public info"


class TestSecretMemoryIndexProtection:
    """Task 2: secret 级记忆在内存索引中不得保留明文。"""

    @pytest.fixture
    def store_with_pm(self, tmp_path):
        """创建默认启用加密并绑定 PrivacyManager 的 DrawerClosetStore。"""
        config = OmniMemConfig(tmp_path)
        store = DrawerClosetStore(tmp_path / "palace", config=config)
        pm = PrivacyManager(session_id="task2-test-session")
        store.bind_privacy_manager(pm)
        return store, pm

    def test_secret_index_content_is_not_plaintext(self, store_with_pm):
        """secret 写入后，_closet_index 中不应出现明文 content。"""
        store, _pm = store_with_pm
        plaintext = "my secret password is 123456"
        mid = store.add(
            wing="personal",
            room="r1",
            content=plaintext,
            memory_type="fact",
            privacy="secret",
        )
        index_entry = store._closet_index[mid]
        assert index_entry["privacy"] == "secret"
        assert index_entry["content"] != plaintext
        assert plaintext not in index_entry["content"]
        assert index_entry["summary"] == "[加密记忆]"

    def test_non_secret_index_content_remains_plaintext(self, store_with_pm):
        """非 secret 级记忆 content 仍保留明文，便于快速读取。"""
        store, _pm = store_with_pm
        plaintext = "ordinary public content"
        mid = store.add(
            wing="personal",
            room="r1",
            content=plaintext,
            memory_type="fact",
            privacy="public",
        )
        assert store._closet_index[mid]["content"] == plaintext

    def test_get_decrypts_secret_content_from_index(self, store_with_pm):
        """get() 应对 secret 级索引中的密文按需解密并返回明文。"""
        store, _pm = store_with_pm
        plaintext = "decrypt me if you can"
        mid = store.add(
            wing="personal",
            room="r1",
            content=plaintext,
            memory_type="fact",
            privacy="secret",
        )
        entry = store.get(mid)
        assert entry is not None
        assert entry["content"] == plaintext
        assert entry["privacy"] == "secret"

    def test_get_decrypts_secret_content_after_flush_and_eviction(self, store_with_pm):
        """flush 并淘汰内存索引后，get() 仍能从磁盘解密返回明文。"""
        store, _pm = store_with_pm
        plaintext = "secret after eviction"
        mid = store.add(
            wing="personal",
            room="r1",
            content=plaintext,
            memory_type="fact",
            privacy="secret",
        )
        store.flush()
        # 强制淘汰内存索引，模拟冷数据回退
        store._closet_index.pop(mid, None)
        entry = store.get(mid)
        assert entry is not None
        assert entry["content"] == plaintext

    def test_search_by_content_does_not_leak_secret_plaintext(self, store_with_pm):
        """按内容搜索不应返回 secret 记忆的明文。"""
        store, _pm = store_with_pm
        plaintext = "unique-secret-token-xyz"
        mid = store.add(
            wing="personal",
            room="r1",
            content=plaintext,
            memory_type="fact",
            privacy="secret",
        )
        store.flush()
        results = store.search_by_content(plaintext)
        for r in results:
            assert r["memory_id"] != mid or plaintext not in (r.get("content") or "")
            if r["memory_id"] == mid:
                assert r.get("_encrypted") is True
                assert "加密" in (r.get("content") or "")

    def test_search_does_not_leak_secret_plaintext(self, store_with_pm):
        """按 wing 搜索返回的 secret 结果应为占位符，不含明文。"""
        store, _pm = store_with_pm
        plaintext = "another secret token"
        mid = store.add(
            wing="personal",
            room="r1",
            content=plaintext,
            memory_type="fact",
            privacy="secret",
        )
        store.flush()
        results = store.search(wing="personal")
        secret_results = [r for r in results if r["memory_id"] == mid]
        assert secret_results, "应能通过 wing 检索到 secret 元数据"
        for r in secret_results:
            assert plaintext not in (r.get("content") or "")
            assert plaintext not in (r.get("content_preview") or "")
            assert r.get("_encrypted") is True
