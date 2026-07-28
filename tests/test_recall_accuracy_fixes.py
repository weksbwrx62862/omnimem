"""召回准确率三缺陷修复的回归测试。

覆盖:
  缺陷1 中文分片破碎 —— ASCII 句点不再切碎 deploy.yml / 7.2 等 token;
  缺陷3 偏好召回噪声 —— 偏好记忆查询相关性门控。
(缺陷2 零结果地板依赖嵌入模型的端到端验证, 见基准脚本; 此处覆盖可单元化的门控逻辑。)
"""

from __future__ import annotations

from omnimem.perception.fact_extractor import AtomicFactExtractor
from omnimem.retrieval.fusion import FusionMixin


class TestSentenceSplitProtectsDots:
    """缺陷1: 分句正则保护 ASCII 句点内的 token(文件名/小数/域名/版本号)。"""

    def _fallback(self, content: str) -> list[str]:
        return AtomicFactExtractor()._extract_with_fallback(content)

    def test_filename_dot_not_split(self) -> None:
        content = "CI/CD 流水线使用 GitHub Actions，部署脚本在 deploy.yml，触发条件是 push 到 main 分支"
        facts = self._fallback(content)
        assert any("deploy.yml" in f for f in facts), facts
        assert not any(f.strip().startswith("yml") for f in facts), facts

    def test_relative_path_dot_not_split(self) -> None:
        # 二轮修复: "./manage.py" 开头的点(后跟 /)也不作句末符
        content = "在 Windows 上运行 Django 迁移时不能用 ./manage.py，必须用 python manage.py，因为 Windows 不识别 shebang"
        facts = self._fallback(content)
        assert len(facts) == 1, facts
        assert "./manage.py" in facts[0] and "shebang" in facts[0], facts

    def test_decimal_dot_not_split(self) -> None:
        content = "Redis 版本是 7.2，用于缓存和会话存储的核心组件"
        facts = self._fallback(content)
        assert any("7.2" in f for f in facts), facts
        assert not any(f.strip().startswith("2") for f in facts), facts

    def test_real_sentence_period_still_splits(self) -> None:
        content = "第一个句子已经完整结束了. 请记得更新那个 deploy.yml 配置文件"
        facts = self._fallback(content)
        assert len(facts) >= 2, facts
        assert any("deploy.yml" in f for f in facts), facts


class TestSentenceSplitDotSequence:
    """第2轮: 点序列(go test ./... / 省略号)不作句末切分。"""

    def _fallback(self, content: str) -> list[str]:
        return AtomicFactExtractor()._extract_with_fallback(content)

    def test_go_test_ellipsis_path_not_split(self) -> None:
        content = "在 Windows git-bash 中运行 Go 测试时必须用 go test ./... 而非 make test。因为 make 在 MSYS 下路径转换会出错。"
        facts = self._fallback(content)
        assert any("go test ./..." in f for f in facts), facts


class TestPreferenceGate:
    """缺陷3: 偏好记忆查询相关性门控。"""

    def setup_method(self) -> None:
        self.gate = FusionMixin()._gate_preferences
        self.pref = {"type": "preference", "content": "用户偏好使用中文进行交互", "memory_id": "p1"}
        self.fact = {"type": "fact", "content": "Aurora 项目使用 PostgreSQL 数据库", "memory_id": "f1"}

    def test_unrelated_query_drops_preference(self) -> None:
        out = self.gate("Aurora 项目用什么数据库", [self.fact, self.pref])
        ids = {r["memory_id"] for r in out}
        assert "f1" in ids
        assert "p1" not in ids

    def test_preference_intent_query_keeps_preference(self) -> None:
        out = self.gate("用户喜欢什么交互方式", [self.pref])
        assert any(r["memory_id"] == "p1" for r in out)

    def test_term_overlap_keeps_preference(self) -> None:
        out = self.gate("中文编码问题", [self.pref])
        assert any(r["memory_id"] == "p1" for r in out)

    def test_non_preference_untouched(self) -> None:
        out = self.gate("任意无关查询xyz", [self.fact])
        assert out == [self.fact]

    def test_empty_results(self) -> None:
        assert self.gate("anything", []) == []

class TestPrefIntentSkipsTypeBoost:
    """偏好意图查询跳过类型加权: boost 不得让 reasoning 翻转压过 preference。"""

    def test_pref_intent_no_boost(self) -> None:
        results = [
            {"type": "preference", "score": 0.0902, "content": "用户偏好使用 Neovim"},
            {"type": "reasoning", "score": 0.0888, "content": "选择 Kotlin 因为协程"},
        ]
        out = FusionMixin.apply_type_boost([dict(r) for r in results], query="用户喜欢什么编辑器")
        out.sort(key=lambda r: r.get("score", 0), reverse=True)
        assert out[0]["type"] == "preference", out

    def test_normal_query_boost_kept(self) -> None:
        results = [{"type": "reasoning", "score": 0.05, "content": "x"}]
        out = FusionMixin.apply_type_boost([dict(r) for r in results], query="部署流程")
        assert out[0].get("type_boost") == 1.3, out


class TestExportKeyDerivation:
    """export_key 任意口令经 PBKDF2 派生, 不再因非 base64 口令崩溃。"""

    def test_passphrase_derived(self) -> None:
        from cryptography.fernet import Fernet
        from omnimem.core.import_export import _get_export_key

        key = _get_export_key("simple-passphrase-123")
        assert key is not None
        Fernet(key)  # 不抛 = 合法 Fernet key
        assert key == _get_export_key("simple-passphrase-123")  # 派生确定性

    def test_fernet_key_passthrough(self) -> None:
        from cryptography.fernet import Fernet
        from omnimem.core.import_export import _get_export_key

        raw = Fernet.generate_key().decode()
        assert _get_export_key(raw) == raw.encode()

class TestBoostGapGuard:
    """分差护栏: type_boost 只做平票裁决, 预分差 >10% 禁止翻转。"""

    def test_large_gap_no_flip(self) -> None:
        rs = [
            {"type": "fact", "score": 0.0902, "content": "a"},
            {"type": "reasoning", "score": 0.0726, "content": "b"},
        ]
        out = FusionMixin.apply_type_boost([dict(r) for r in rs], query="部署流程")
        assert out[0]["type"] == "fact", out
        assert out[1].get("boost_capped") is True, out

    def test_small_gap_boost_wins(self) -> None:
        rs = [
            {"type": "fact", "score": 0.0750, "content": "a"},
            {"type": "reasoning", "score": 0.0740, "content": "b"},
        ]
        out = FusionMixin.apply_type_boost([dict(r) for r in rs], query="部署流程")
        assert out[0]["type"] == "reasoning", out


class TestLifecycleSharedHelper:
    """_apply_lifecycle 为同步/异步 validate 唯一判定实现(防双实现漂移)。"""

    def _svc(self, stage: str):
        from types import SimpleNamespace

        from omnimem.services.recall_service import RecallService

        svc = RecallService.__new__(RecallService)
        forgetting = SimpleNamespace(get_stage=lambda mid: stage)
        svc.deps = SimpleNamespace(store=None, forgetting=forgetting)
        return svc

    def test_forgotten_dropped(self) -> None:
        svc = self._svc("forgotten")
        r = {"score": 1.0}
        assert svc._apply_lifecycle(r, "m1", {"content": "x"}) is False

    def test_archived_sealed_downweight(self) -> None:
        svc = self._svc("archived")
        r = {"score": 1.0}
        assert svc._apply_lifecycle(r, "m1", {"content": "x"}) is True
        assert r["sealed"] is True and r["score"] == 0.3

    def test_missing_entry_dropped(self) -> None:
        svc = self._svc("active")
        assert svc._apply_lifecycle({"score": 1.0}, "m1", None) is False

    def test_async_reuses_shared_helper(self) -> None:
        # 结构性断言: 异步 validate 源码必须调用共享 _apply_lifecycle
        import inspect

        from omnimem.services.recall_service import RecallService

        async_src = inspect.getsource(RecallService._async_validate_store_entries)
        assert "_apply_lifecycle" in async_src
        fb_async = inspect.getsource(RecallService._async_fallback_if_few)
        assert "_admit_fts_fallback" in fb_async and "_admit_store_fallback" in fb_async

class TestRecallExplain:
    """项A: refine_recall_results 的 explain opt-in 与 _explain 内容。"""

    def test_build_explain_collects_present_fields(self) -> None:
        from omnimem.context.manager import ContextManager

        r = {
            "score": 0.15, "_source": "fusion", "rrf_score": 0.09,
            "type_boost": 1.3, "boost_capped": True, "decay_factor": None,
        }
        ex = ContextManager._build_explain(r)
        assert ex["final_score"] == 0.15
        assert ex["source"] == "fusion"
        assert ex["rrf_score"] == 0.09
        assert ex["type_boost"] == 1.3
        assert ex["boost_capped"] is True
        assert "decay_factor" not in ex  # None 值不收集

    def test_explain_flag_controls_field(self) -> None:
        raw = [{"content": "orion 用 Rust", "type": "fact", "memory_id": "m1",
                "score": 0.2, "_source": "fusion", "rrf_score": 0.1}]
        from omnimem.context.manager import ContextManager

        base = ContextManager().refine_recall_results(raw)
        assert all("_explain" not in it for it in base)
        with_ex = ContextManager().refine_recall_results(raw, explain=True)
        assert with_ex and "_explain" in with_ex[0]
        assert with_ex[0]["_explain"]["source"] == "fusion"
