"""context/manager.py 深度离线测试。

覆盖 test_context.py 未触及的表面：
  - ContextBudget / RefinedItem 数据类
  - refine_content 的 8 个结构化前缀、6 个压缩模板、句子提取、边界截断、换行归一化
  - refine_overview 的信号词打分与排序
  - _tokenize_chinese / _normalize_word / _load_synonym_map
  - _cached_fingerprint / _cached_similarity 三级相似度分层
  - _is_duplicate 快慢双路径
  - _embedding_similarity / _trim_embedding_cache
  - refine_prefetch_results 预算与条数上限
  - _build_explain / refine_recall_results 字段透传
  - get_detail_for / get_injected_items
  - reset_for_new_turn / add_persistent_fingerprint
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from omnimem.context.manager import ContextBudget, ContextManager, RefinedItem

# ─── 脚手架 ────────────────────────────────────────────────


class _FakeStore:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = data or {}
        self.calls: list[str] = []

    def get(self, memory_id: str) -> Any:
        self.calls.append(memory_id)
        return self.data.get(memory_id)


def _mgr(**kw: Any) -> ContextManager:
    return ContextManager(**kw)


# ─── 数据类 ────────────────────────────────────────────────


class TestDataclasses:
    def test_budget_defaults(self):
        b = ContextBudget()
        assert b.max_prefetch_tokens == 300
        assert b.max_summary_chars == 60
        assert b.max_overview_chars == 200
        assert b.max_prefetch_items == 8
        assert b.max_detail_chars == 500
        assert b.dedup_similarity_threshold == 0.7
        assert b.mmd_max_token_ratio == 0.2
        assert b.mild_offload_ratio == 0.5
        assert b.aggressive_compress_ratio == 0.85

    def test_budget_override(self):
        b = ContextBudget(max_prefetch_tokens=42, max_summary_chars=7)
        assert b.max_prefetch_tokens == 42
        assert b.max_summary_chars == 7

    def test_refined_item_defaults(self):
        it = RefinedItem(summary="s", memory_id="m1", memory_type="fact", confidence=0.9)
        assert it.source_type == ""
        assert it.overview == ""
        assert it.trace_node_id == ""
        assert it.ref_path == ""

    def test_max_summary_chars_property(self):
        m = _mgr(budget=ContextBudget(max_summary_chars=33))
        assert m.max_summary_chars == 33

    def test_default_budget_when_none(self):
        assert _mgr(budget=None)._budget.max_prefetch_tokens == 300


# ─── refine_content ────────────────────────────────────────


class TestRefineContent:
    def test_short_content_passthrough(self):
        assert ContextManager.refine_content("用户喜欢猫", 60) == "用户喜欢猫"

    def test_strips_whitespace(self):
        assert ContextManager.refine_content("   短内容   ", 60) == "短内容"

    @pytest.mark.parametrize(
        ("raw", "expected_prefix"),
        [
            ("CORRECTION: 用户其实叫老板而不是徐先生这个称呼需要记住", "纠正: "),
            ("correction: 用户其实叫老板而不是徐先生这个称呼需要记住", "纠正: "),
            ("REINFORCED: 用户其实叫老板而不是徐先生这个称呼需要记住", "确认: "),
            ("纠正: 用户其实叫老板而不是徐先生这个称呼需要记住", "纠正: "),
            ("确认: 用户其实叫老板而不是徐先生这个称呼需要记住", "确认: "),
            ("[Pre-compression emergency save] 用户其实叫老板而不是徐先生这个称呼", "用户其实叫老板"),
            ("[Emergency save] 用户其实叫老板而不是徐先生这个称呼需要记住才行", "用户其实叫老板"),
            ("[Turn 12] 用户其实叫老板而不是徐先生这个称呼需要记住一下", "用户其实叫老板"),
            ("[Checkpoint at turn 3] 用户其实叫老板而不是徐先生这个称呼", "用户其实叫老板"),
        ],
    )
    def test_structured_prefix_stripped(self, raw, expected_prefix):
        out = ContextManager.refine_content(raw, 60)
        assert out.startswith(expected_prefix)
        assert "CORRECTION" not in out and "[Turn" not in out

    def test_only_first_prefix_replaced(self):
        raw = "CORRECTION: 纠正: 用户其实叫老板而不是徐先生这个称呼需要记住"
        out = ContextManager.refine_content(raw, 60)
        assert out == "纠正: 纠正: 用户其实叫老板而不是徐先生这个称呼需要记住"

    def test_template_correction_drops_delimiter(self):
        raw = "纠正: 部署分支应该是 main，其余内容非常长用来占位"
        expected = "纠正: 部署分支应该是 main其余内容非常长用来占位"
        out = ContextManager.refine_content(raw, max_chars=len(expected))
        assert out == expected

    def test_template_confirm_drops_delimiter(self):
        raw = "确认: 部署已经完成了，其余内容非常长用来占位"
        expected = "确认: 部署已经完成了其余内容非常长用来占位"
        out = ContextManager.refine_content(raw, max_chars=len(expected))
        assert out == expected

    def test_template_remember_collapses_whitespace(self):
        raw = "记住   项目根目录，其余内容非常长用来占位"
        expected = "记住: 项目根目录其余内容非常长用来占位"
        out = ContextManager.refine_content(raw, max_chars=len(expected))
        assert out == expected

    def test_non_shortening_template_falls_through(self):
        # "用户否定: " 与 "用户不喜欢X，" 等长，压缩不会缩短 → 走截断路径
        raw = "用户不喜欢吃辣的食物，尤其是川菜和湘菜这类重口味的菜系"
        out = ContextManager.refine_content(raw, max_chars=15)
        assert out.startswith("用户不喜欢")

    def test_sentence_extraction_picks_short_sentence(self):
        raw = (
            "这是一个非常非常长的开头句子用来把整体长度顶到超过最大字符限制"
            "。短句子在这里。后面还有一段同样很长的收尾内容用来填充字符数量"
        )
        out = ContextManager.refine_content(raw, 20)
        assert out == "短句子在这里"

    def test_boundary_truncation_at_punctuation(self):
        raw = "第一段内容\n第二段内容\n第三段内容"
        out = ContextManager.refine_content(raw, 8)
        assert out == "第一段内容"

    def test_newlines_normalized_before_split(self):
        raw = "路径 C:\nnew 目录已经确认"
        out = ContextManager.refine_content(raw, 10)
        assert "\n" not in out

    def test_tabs_and_cr_normalized(self):
        raw = "甲\t乙\r丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
        out = ContextManager.refine_content(raw, 12)
        assert "\t" not in out and "\r" not in out

    def test_no_truncation_when_punctuation_too_early(self):
        raw = "，abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ"
        out = ContextManager.refine_content(raw, 20)
        assert len(out) == 20

    def test_decimal_not_split_mid_token(self):
        raw = "版本号 3.12 与部署文件 deploy.yml 都已经同步更新完毕了"
        out = ContextManager.refine_content(raw, 60)
        assert "3.12" in out


# ─── refine_overview ───────────────────────────────────────


class TestRefineOverview:
    def test_short_passthrough(self):
        assert ContextManager.refine_overview("很短", 200) == "很短"

    def test_signal_words_ranked_first(self):
        raw = (
            "第一段是平淡的陈述没有任何信号词只是用来占据字符空间而已。"
            "但是如果忽略这一条会导致部署失败所以必须注意。"
            "第三段同样是平淡的陈述没有信号词只是继续占据字符空间。"
            "第四段还是平淡的陈述继续占据字符空间用来顶满长度限制。"
        )
        out = ContextManager.refine_overview(raw, 40)
        assert out.startswith("但是如果忽略这一条会导致部署失败所以必须注意")

    def test_joined_with_chinese_period(self):
        raw = ("必须配置甲。" * 3) + ("平淡陈述乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉。" * 3)
        out = ContextManager.refine_overview(raw, 60)
        assert out.endswith("。")

    def test_fallback_truncation_when_no_sentence_fits(self):
        raw = "字" * 500
        out = ContextManager.refine_overview(raw, 30)
        assert len(out) <= 30

    def test_newlines_flattened(self):
        raw = "第一行内容\n第二行内容\n" + ("填充内容用来顶满长度限制。" * 10)
        out = ContextManager.refine_overview(raw, 40)
        assert "\n" not in out

    def test_template_compression_applies(self):
        raw = "纠正: " + "内" * 176 + "，" + "容" * 20
        assert len(raw) == 201
        out = ContextManager.refine_overview(raw, 200)
        assert out == "纠正: " + "内" * 176 + "容" * 20

    def test_english_signal_words(self):
        raw = (
            "plain statement one without any signal words at all here ok. "
            "however this one matters because it blocks the deploy pipeline. "
            "plain statement two without any signal words at all here ok. "
            "plain statement three without any signal words at all here ok."
        )
        out = ContextManager.refine_overview(raw, 80)
        assert "however" in out.lower()


# ─── 分词与归一化 ──────────────────────────────────────────


class TestTokenize:
    def test_empty_string(self):
        assert ContextManager._tokenize_chinese("") == []

    def test_english_lowercased(self):
        assert ContextManager._tokenize_chinese("Python And Rust") == ["python", "and", "rust"]

    def test_single_letter_english_dropped(self):
        assert ContextManager._tokenize_chinese("a b cd") == ["cd"]

    def test_digits_only_segment_yields_nothing(self):
        assert ContextManager._tokenize_chinese("12345") == []

    def test_stopwords_filtered(self):
        assert ContextManager._tokenize_chinese("的，了，啊") == []

    def test_single_stopword_char_filtered(self):
        assert ContextManager._tokenize_chinese("的") == []
        assert ContextManager._tokenize_chinese("啊") == []

    def test_dict_word_matched(self):
        tokens = ContextManager._tokenize_chinese("猫咪")
        assert "猫咪" in tokens

    def test_punctuation_splits_segments(self):
        tokens = ContextManager._tokenize_chinese("猫咪，狗狗")
        assert "猫咪" in tokens and "狗狗" in tokens

    def test_dict_words_sorted_by_length_desc(self):
        words = ContextManager._get_dict_words()
        lengths = [len(w) for w in words]
        assert lengths == sorted(lengths, reverse=True)

    def test_dict_set_matches_words(self):
        assert ContextManager._get_dict_set() == set(ContextManager._get_dict_words())


class TestNormalizeWord:
    def test_unknown_word_returns_itself(self, monkeypatch):
        monkeypatch.setattr(ContextManager, "_SYNONYM_MAP", {"甲": "乙"})
        assert ContextManager._normalize_word("丙") == "丙"

    def test_string_mapping(self, monkeypatch):
        monkeypatch.setattr(ContextManager, "_SYNONYM_MAP", {"甲": "乙"})
        assert ContextManager._normalize_word("甲") == "乙"

    def test_list_mapping_takes_first(self, monkeypatch):
        monkeypatch.setattr(ContextManager, "_SYNONYM_MAP", {"甲": ["乙", "丙"]})
        assert ContextManager._normalize_word("甲") == "乙"

    def test_empty_list_mapping_returns_word(self, monkeypatch):
        monkeypatch.setattr(ContextManager, "_SYNONYM_MAP", {"甲": []})
        assert ContextManager._normalize_word("甲") == "甲"


class TestLoadSynonymMap:
    def _point_at(self, monkeypatch, tmp_path):
        ctx_dir = tmp_path / "context"
        (ctx_dir).mkdir(parents=True, exist_ok=True)
        (tmp_path / "config").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "omnimem.context.manager.__file__", str(ctx_dir / "manager.py")
        )
        return tmp_path / "config" / "synonyms.json"

    def test_flattens_one_to_many(self, monkeypatch, tmp_path):
        cfg = self._point_at(monkeypatch, tmp_path)
        cfg.write_text(
            json.dumps({"宠物": ["猫", "狗"], "颜色": "红色"}, ensure_ascii=False),
            encoding="utf-8",
        )
        out = ContextManager._load_synonym_map()
        assert out["宠物"] == "宠物"
        assert out["猫"] == "宠物"
        assert out["狗"] == "宠物"
        assert out["红色"] == "颜色"

    def test_non_string_value_skipped(self, monkeypatch, tmp_path):
        cfg = self._point_at(monkeypatch, tmp_path)
        cfg.write_text(json.dumps({"数字": 123, "列表": [1, "真"]}), encoding="utf-8")
        out = ContextManager._load_synonym_map()
        assert out["数字"] == "数字"
        assert out["真"] == "列表"
        assert 1 not in out

    def test_first_mapping_wins(self, monkeypatch, tmp_path):
        cfg = self._point_at(monkeypatch, tmp_path)
        cfg.write_text(
            json.dumps({"甲": ["共"], "乙": ["共"]}, ensure_ascii=False), encoding="utf-8"
        )
        out = ContextManager._load_synonym_map()
        assert out["共"] == "甲"

    def test_missing_file_returns_empty(self, monkeypatch, tmp_path):
        self._point_at(monkeypatch, tmp_path)
        assert ContextManager._load_synonym_map() == {}

    def test_malformed_json_returns_empty(self, monkeypatch, tmp_path):
        cfg = self._point_at(monkeypatch, tmp_path)
        cfg.write_text("{ not json", encoding="utf-8")
        assert ContextManager._load_synonym_map() == {}

    def test_non_dict_json_returns_empty(self, monkeypatch, tmp_path):
        cfg = self._point_at(monkeypatch, tmp_path)
        cfg.write_text("[1, 2, 3]", encoding="utf-8")
        assert ContextManager._load_synonym_map() == {}

    def test_real_config_loads(self):
        out = ContextManager._load_synonym_map()
        assert isinstance(out, dict)
        assert out


# ─── 指纹与相似度 ──────────────────────────────────────────


class TestFingerprint:
    def test_stopword_only_yields_empty(self):
        assert ContextManager._content_fingerprint("的") == ""
        assert ContextManager._content_fingerprint("的，了，啊") == ""

    def test_deterministic(self):
        a = ContextManager._content_fingerprint("用户喜欢喝美式咖啡")
        b = ContextManager._content_fingerprint("用户喜欢喝美式咖啡")
        assert a == b and a

    def test_sorted_pipe_joined(self):
        fp = ContextManager._content_fingerprint("alpha beta gamma")
        assert fp == "|".join(sorted(fp.split("|")))

    def test_low_info_words_removed(self):
        assert ContextManager._content_fingerprint("好 alpha") == "alpha"

    def test_synonyms_collapse(self):
        assert ContextManager._content_fingerprint("柯基") == "宠物"
        assert ContextManager._content_fingerprint("柴犬") == "宠物"
        assert ContextManager._content_fingerprint("用户喜欢柯基") == (
            ContextManager._content_fingerprint("用户喜欢柴犬")
        )


class TestFingerprintSimilarity:
    def test_empty_returns_zero(self):
        assert ContextManager._fingerprint_similarity("", "a") == 0.0
        assert ContextManager._fingerprint_similarity("a", "") == 0.0

    def test_only_separators_returns_zero(self):
        assert ContextManager._fingerprint_similarity("|", "a") == 0.0

    def test_identical_returns_one(self):
        assert ContextManager._fingerprint_similarity("a|b", "a|b") == 1.0

    def test_subset_high_coverage(self):
        assert ContextManager._fingerprint_similarity("a|b", "a|b|c|d") == 0.85

    def test_subset_medium_coverage(self):
        assert ContextManager._fingerprint_similarity("a", "a|b|c") == 0.75

    def test_subset_low_coverage_multi_word(self):
        assert ContextManager._fingerprint_similarity("a|b", "a|b|c|d|e|f|g") == 0.72

    def test_subset_low_coverage_single_word_falls_to_jaccard(self):
        assert ContextManager._fingerprint_similarity("a", "a|b|c|d|e") == pytest.approx(0.2)

    def test_reverse_subset_high_coverage(self):
        assert ContextManager._fingerprint_similarity("a|b|c|d", "a|b") == 0.85

    def test_jaccard_above_threshold(self):
        fp1 = "a|b|c|d|e|f|g"
        fp2 = "a|b|c|d|e|f|x"
        assert ContextManager._fingerprint_similarity(fp1, fp2) == pytest.approx(0.75)

    def test_loose_coverage_beats_jaccard(self):
        fp1 = "a|b|c|x"
        fp2 = "a|b|c|d|e"
        assert ContextManager._fingerprint_similarity(fp1, fp2) == pytest.approx(0.675)

    def test_loose_coverage_needs_two_overlap(self):
        fp1 = "a|b|x|y"
        fp2 = "a|c|d|e"
        assert ContextManager._fingerprint_similarity(fp1, fp2) == pytest.approx(1 / 7)

    def test_disjoint_returns_zero(self):
        assert ContextManager._fingerprint_similarity("a|b", "x|y") == 0.0

    def test_symmetric(self):
        assert ContextManager._fingerprint_similarity("p|q", "q|r") == (
            ContextManager._fingerprint_similarity("q|r", "p|q")
        )


# ─── 去重 ──────────────────────────────────────────────────


def _item(summary: str, mid: str = "m1") -> RefinedItem:
    return RefinedItem(summary=summary, memory_id=mid, memory_type="fact", confidence=1.0)


class TestIsDuplicate:
    def test_empty_fingerprint_never_duplicate(self):
        m = _mgr()
        m._injected_fingerprints.add("的")
        assert m._is_duplicate(_item("的")) is False

    def test_exact_match_is_duplicate(self):
        m = _mgr()
        summary = "用户喜欢喝美式咖啡不加糖"
        m._injected_fingerprints.add(ContextManager._content_fingerprint(summary))
        assert m._is_duplicate(_item(summary)) is True

    def test_unrelated_is_not_duplicate(self):
        m = _mgr()
        m._injected_fingerprints.add(ContextManager._content_fingerprint("alpha beta gamma"))
        assert m._is_duplicate(_item("delta epsilon zeta")) is False

    def test_empty_fingerprint_set(self):
        assert _mgr()._is_duplicate(_item("用户喜欢喝美式咖啡")) is False

    def test_embedding_slow_path_confirms(self, monkeypatch):
        monkeypatch.setattr(
            ContextManager, "_content_fingerprint", classmethod(lambda cls, c: "a|b|c|x")
        )
        m = _mgr(embedding_fn=lambda t: [1.0, 0.0])
        m._injected_fingerprints.add("a|b|c|d|e")
        m._fp_to_summary["a|b|c|d|e"] = "existing summary"
        assert m._is_duplicate(_item("new summary")) is True

    def test_embedding_slow_path_rejects(self, monkeypatch):
        monkeypatch.setattr(
            ContextManager, "_content_fingerprint", classmethod(lambda cls, c: "a|b|c|x")
        )
        m = _mgr(embedding_fn=lambda t: [1.0, 0.0] if t == "new summary" else [0.0, 1.0])
        m._injected_fingerprints.add("a|b|c|d|e")
        m._fp_to_summary["a|b|c|d|e"] = "existing summary"
        assert m._is_duplicate(_item("new summary")) is False

    def test_slow_path_skipped_without_embedding_fn(self, monkeypatch):
        monkeypatch.setattr(
            ContextManager, "_content_fingerprint", classmethod(lambda cls, c: "a|b|c|x")
        )
        m = _mgr()
        m._injected_fingerprints.add("a|b|c|d|e")
        assert m._is_duplicate(_item("new summary")) is False

    def test_missing_summary_mapping_does_not_crash(self, monkeypatch):
        monkeypatch.setattr(
            ContextManager, "_content_fingerprint", classmethod(lambda cls, c: "a|b|c|x")
        )
        calls: list[str] = []

        def _fn(t: str) -> list[float]:
            calls.append(t)
            return [1.0, 0.0]

        m = _mgr(embedding_fn=_fn)
        m._injected_fingerprints.add("a|b|c|d|e")
        assert m._is_duplicate(_item("new summary")) is False
        assert calls == []


class TestEmbeddingSimilarity:
    def test_no_fn_returns_zero(self):
        assert _mgr()._embedding_similarity("a", "b") == 0.0

    def test_empty_text_returns_zero(self):
        m = _mgr(embedding_fn=lambda t: [1.0])
        assert m._embedding_similarity("", "b") == 0.0
        assert m._embedding_similarity("a", "") == 0.0

    def test_identical_vectors(self):
        m = _mgr(embedding_fn=lambda t: [1.0, 2.0, 3.0])
        assert m._embedding_similarity("a", "b") == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        m = _mgr(embedding_fn=lambda t: [1.0, 0.0] if t == "a" else [0.0, 1.0])
        assert m._embedding_similarity("a", "b") == pytest.approx(0.0)

    def test_zero_norm_returns_zero(self):
        m = _mgr(embedding_fn=lambda t: [0.0, 0.0])
        assert m._embedding_similarity("a", "b") == 0.0

    def test_empty_vector_returns_zero(self):
        m = _mgr(embedding_fn=lambda t: [])
        assert m._embedding_similarity("a", "b") == 0.0
        assert m._embedding_cache == {}

    def test_cache_avoids_recompute(self):
        calls: list[str] = []

        def _fn(t: str) -> list[float]:
            calls.append(t)
            return [1.0, 0.0]

        m = _mgr(embedding_fn=_fn)
        m._embedding_similarity("a", "b")
        m._embedding_similarity("a", "b")
        assert calls == ["a", "b"]

    def test_uneven_length_vectors_zip_shortest(self):
        m = _mgr(embedding_fn=lambda t: [1.0, 1.0, 1.0] if t == "a" else [1.0, 1.0])
        assert m._embedding_similarity("a", "b") == pytest.approx(2 / 6**0.5)

    def test_trim_removes_oldest(self):
        m = _mgr()
        m._embedding_cache_max_size = 2
        m._embedding_cache = {"a": [1.0], "b": [2.0], "c": [3.0]}
        m._trim_embedding_cache()
        assert list(m._embedding_cache) == ["b", "c"]

    def test_trim_noop_under_limit(self):
        m = _mgr()
        m._embedding_cache_max_size = 5
        m._embedding_cache = {"a": [1.0]}
        m._trim_embedding_cache()
        assert m._embedding_cache == {"a": [1.0]}

    def test_trim_triggered_on_insert(self):
        m = _mgr(embedding_fn=lambda t: [1.0, 0.0])
        m._embedding_cache_max_size = 2
        for t in ("a", "b", "c"):
            m._embedding_similarity(t, t)
        assert len(m._embedding_cache) == 2


# ─── 注入状态 ──────────────────────────────────────────────


class TestInjectionState:
    def test_reset_clears_turn_items(self):
        m = _mgr()
        m._injected_items.append(_item("s"))
        m._injected_fingerprints.add("fp")
        m.reset_for_new_turn()
        assert m._injected_items == []
        assert m._injected_fingerprints == set()

    def test_reset_preserves_persistent(self):
        m = _mgr()
        m.add_persistent_fingerprint("keep")
        m.reset_for_new_turn()
        assert "keep" in m._injected_fingerprints

    def test_add_persistent_ignores_empty(self):
        m = _mgr()
        m.add_persistent_fingerprint("")
        assert m._persistent_fingerprints == set()

    def test_add_persistent_updates_both_sets(self):
        m = _mgr()
        m.add_persistent_fingerprint("fp")
        assert "fp" in m._persistent_fingerprints
        assert "fp" in m._injected_fingerprints

    def test_persistent_cap_at_500(self):
        m = _mgr()
        for i in range(600):
            m.add_persistent_fingerprint(f"fp-{i}")
        assert len(m._persistent_fingerprints) == 500

    def test_get_injected_fingerprints_returns_copy(self):
        m = _mgr()
        m.add_persistent_fingerprint("fp")
        snap = m.get_injected_fingerprints()
        snap.add("other")
        assert "other" not in m._injected_fingerprints


# ─── prefetch ──────────────────────────────────────────────


class TestRefinePrefetch:
    def test_empty_returns_empty_string(self):
        assert _mgr().refine_prefetch_results([]) == ""

    def test_missing_content_skipped(self):
        assert _mgr().refine_prefetch_results([{"memory_id": "m1"}]) == ""

    def test_format(self):
        out = _mgr().refine_prefetch_results(
            [{"content": "用户姓名: 徐信豪", "memory_id": "m1", "type": "fact"}]
        )
        assert out == "### Relevant Memories\n- [fact] 用户姓名: 徐信豪"

    def test_default_type_is_fact(self):
        out = _mgr().refine_prefetch_results([{"content": "alpha fact one"}])
        assert out.splitlines()[1] == "- [fact] alpha fact one"

    def test_multiple_items_joined(self):
        out = _mgr(budget=ContextBudget(max_prefetch_tokens=5000)).refine_prefetch_results(
            [
                {"content": "alpha fact one", "type": "fact"},
                {"content": "beta note two", "type": "preference"},
            ]
        )
        lines = out.splitlines()
        assert lines[0] == "### Relevant Memories"
        assert lines[1] == "- [fact] alpha fact one"
        assert lines[2] == "- [preference] beta note two"

    def test_duplicates_skipped(self):
        out = _mgr().refine_prefetch_results(
            [{"content": "alpha fact one"}, {"content": "alpha fact one"}]
        )
        assert len(out.splitlines()) == 2

    def test_budget_break(self):
        out = _mgr(budget=ContextBudget(max_prefetch_tokens=50)).refine_prefetch_results(
            [{"content": "alpha fact one"}, {"content": "beta note two"}]
        )
        assert out.splitlines() == ["### Relevant Memories", "- [fact] alpha fact one"]

    def test_item_cap(self):
        budget = ContextBudget(max_prefetch_tokens=10000, max_prefetch_items=2)
        out = _mgr(budget=budget).refine_prefetch_results(
            [
                {"content": "alpha one"},
                {"content": "beta two"},
                {"content": "gamma three"},
                {"content": "delta four"},
            ]
        )
        assert len(out.splitlines()) == 3

    def test_registers_fingerprint_and_item(self):
        m = _mgr()
        m.refine_prefetch_results([{"content": "alpha fact one", "memory_id": "m1"}])
        assert len(m._injected_fingerprints) == 1
        assert len(m._injected_items) == 1
        assert m._injected_items[0].memory_id == "m1"
        assert m._fp_to_summary

    def test_second_call_dedups_across_calls(self):
        m = _mgr()
        m.refine_prefetch_results([{"content": "alpha fact one"}])
        assert m.refine_prefetch_results([{"content": "alpha fact one"}]) == ""

    def test_summary_refined_to_budget_chars(self):
        budget = ContextBudget(max_summary_chars=10, max_prefetch_tokens=5000)
        raw = "这是一段非常非常长的原始记忆内容需要被精炼成摘要才行"
        out = _mgr(budget=budget).refine_prefetch_results([{"content": raw}])
        summary = out.splitlines()[1].split("] ", 1)[1]
        assert len(summary) <= 10

    def test_confidence_and_source_type_captured(self):
        m = _mgr()
        m.refine_prefetch_results(
            [{"content": "alpha fact one", "confidence": 0.42, "source_type": "bm25"}]
        )
        it = m._injected_items[0]
        assert it.confidence == 0.42
        assert it.source_type == "bm25"


# ─── explain ───────────────────────────────────────────────


class TestBuildExplain:
    def test_minimal(self):
        assert ContextManager._build_explain({"score": 0.5}) == {
            "final_score": 0.5,
            "source": "",
        }

    def test_source_passthrough(self):
        ex = ContextManager._build_explain({"_source": "vector"})
        assert ex["source"] == "vector"

    def test_none_values_skipped(self):
        ex = ContextManager._build_explain({"score": 1.0, "rrf_score": None})
        assert "rrf_score" not in ex

    def test_false_values_kept(self):
        ex = ContextManager._build_explain({"score": 1.0, "boost_capped": False})
        assert ex["boost_capped"] is False

    def test_leading_underscore_stripped(self):
        ex = ContextManager._build_explain(
            {"score": 1.0, "_priming_boost": 0.2, "_temporal_weight": 0.8, "_channels": ["bm25"]}
        )
        assert ex["priming_boost"] == 0.2
        assert ex["temporal_weight"] == 0.8
        assert ex["channels"] == ["bm25"]
        assert "_priming_boost" not in ex

    def test_unknown_fields_ignored(self):
        ex = ContextManager._build_explain({"score": 1.0, "whatever": 9})
        assert "whatever" not in ex

    def test_all_known_fields_collected(self):
        raw: dict[str, Any] = {"score": 1.0, "_source": "s"}
        for f in (
            "rrf_score", "additive_score", "type_boost", "updated_boost",
            "entity_boost", "boost_capped", "rerank_score", "_priming_boost",
            "_temporal_weight", "_temporal_reranked", "decay_factor", "_channels",
        ):
            raw[f] = 1
        ex = ContextManager._build_explain(raw)
        assert len(ex) == 14


# ─── recall ────────────────────────────────────────────────


class TestRefineRecall:
    def test_empty_returns_empty_list(self):
        assert _mgr().refine_recall_results([]) == []

    def test_missing_content_skipped(self):
        assert _mgr().refine_recall_results([{"memory_id": "m1"}]) == []

    def test_summary_and_original_kept(self):
        raw = "这是一段非常长的原始记忆内容" * 20
        out = _mgr().refine_recall_results([{"content": raw}])
        assert out[0]["original_content"] == raw
        assert len(out[0]["content"]) <= 100

    def test_duplicates_removed(self):
        out = _mgr().refine_recall_results(
            [{"content": "alpha fact one"}, {"content": "alpha fact one"}]
        )
        assert len(out) == 1

    def test_similar_duplicates_removed(self):
        out = _mgr().refine_recall_results(
            [
                {"content": "用户喜欢喝美式咖啡不加糖"},
                {"content": "用户喜欢喝美式咖啡不加糖不加奶"},
            ]
        )
        assert len(out) == 1

    def test_distinct_kept(self):
        out = _mgr().refine_recall_results(
            [{"content": "alpha beta gamma"}, {"content": "delta epsilon zeta"}]
        )
        assert len(out) == 2

    def test_field_defaults(self):
        out = _mgr().refine_recall_results([{"content": "alpha beta gamma"}])
        r = out[0]
        assert r["type"] == "fact"
        assert r["confidence"] is None
        assert r["memory_id"] == ""
        assert r["wing"] is None
        assert r["room"] is None
        assert r["stored_at"] is None
        assert r["entities"] == []
        assert r["provenance"] == ""
        assert r["scope"] == ""
        assert r["_evidence_enriched"] is False
        assert r["_group_start"] is False
        assert r["_group_size"] == 0
        assert r["_conflict"] is None
        assert r["sealed"] is False
        assert r["score"] is None
        assert r["_source"] == ""

    def test_fields_passthrough(self):
        raw = {
            "content": "alpha beta gamma",
            "type": "preference",
            "confidence": 0.8,
            "memory_id": "m1",
            "wing": "w",
            "room": "r",
            "stored_at": "2026-01-01",
            "entities": ["e"],
            "provenance": "p",
            "scope": "s",
            "_evidence_enriched": True,
            "_group_start": True,
            "_group_size": 3,
            "_conflict": {"a": 1},
            "sealed": True,
            "score": 0.9,
            "_source": "vector",
        }
        out = _mgr().refine_recall_results([raw])
        r = out[0]
        for k in ("type", "confidence", "memory_id", "wing", "room", "stored_at",
                  "entities", "provenance", "scope", "_evidence_enriched",
                  "_group_start", "_group_size", "_conflict", "sealed", "score", "_source"):
            assert r[k] == raw[k]

    def test_explain_off_by_default(self):
        out = _mgr().refine_recall_results([{"content": "alpha beta gamma", "score": 0.5}])
        assert "_explain" not in out[0]

    def test_explain_enabled(self):
        out = _mgr().refine_recall_results(
            [{"content": "alpha beta gamma", "score": 0.5, "rrf_score": 0.3}], explain=True
        )
        assert out[0]["_explain"]["final_score"] == 0.5
        assert out[0]["_explain"]["rrf_score"] == 0.3

    def test_empty_fingerprint_items_both_kept(self):
        out = _mgr().refine_recall_results([{"content": "的"}, {"content": "的"}])
        assert len(out) == 2


# ─── lazy detail ───────────────────────────────────────────


class TestGetDetailFor:
    def test_injected_hit(self):
        m = _mgr()
        m._injected_items.append(
            RefinedItem(summary="摘要", memory_id="m1", memory_type="preference", confidence=0.7)
        )
        store = _FakeStore(
            {"m1": {"content": "完整内容", "privacy": "public", "metadata": {"k": "v"}}}
        )
        out = m.get_detail_for("m1", store)
        assert out == {
            "status": "found",
            "memory_id": "m1",
            "summary": "摘要",
            "full_content": "完整内容",
            "type": "preference",
            "confidence": 0.7,
            "privacy": "public",
            "metadata": {"k": "v"},
        }

    def test_injected_hit_default_privacy(self):
        m = _mgr()
        m._injected_items.append(
            RefinedItem(summary="s", memory_id="m1", memory_type="fact", confidence=0.0)
        )
        out = m.get_detail_for("m1", _FakeStore({"m1": {"content": "c"}}))
        assert out["privacy"] == "personal"
        assert out["metadata"] == {}

    def test_injected_but_store_missing(self):
        m = _mgr()
        m._injected_items.append(
            RefinedItem(summary="s", memory_id="m1", memory_type="fact", confidence=0.0)
        )
        store = _FakeStore()
        out = m.get_detail_for("m1", store)
        assert out == {"status": "not_found", "memory_id": "m1"}
        assert store.calls == ["m1", "m1"]

    def test_not_injected_falls_back_to_store(self):
        m = _mgr()
        store = _FakeStore(
            {"m2": {"content": "完整内容二", "type": "action", "confidence": 0.3}}
        )
        out = m.get_detail_for("m2", store)
        assert out["status"] == "found"
        assert out["summary"] == "完整内容二"
        assert out["type"] == "action"
        assert out["confidence"] == 0.3

    def test_not_injected_defaults(self):
        out = _mgr().get_detail_for("m3", _FakeStore({"m3": {"content": "c"}}))
        assert out["type"] == "fact"
        assert out["confidence"] == 0
        assert out["privacy"] == "personal"

    def test_not_found(self):
        assert _mgr().get_detail_for("nope", _FakeStore()) == {
            "status": "not_found",
            "memory_id": "nope",
        }

    def test_first_matching_injected_item_wins(self):
        m = _mgr()
        m._injected_items.append(
            RefinedItem(summary="first", memory_id="m1", memory_type="fact", confidence=0.0)
        )
        m._injected_items.append(
            RefinedItem(summary="second", memory_id="m1", memory_type="fact", confidence=0.0)
        )
        out = m.get_detail_for("m1", _FakeStore({"m1": {"content": "c"}}))
        assert out["summary"] == "first"


class TestGetInjectedItems:
    def test_empty(self):
        assert _mgr().get_injected_items() == []

    def test_shape(self):
        m = _mgr()
        m._injected_items.append(
            RefinedItem(summary="s", memory_id="m1", memory_type="fact", confidence=0.0)
        )
        assert m.get_injected_items() == [
            {"memory_id": "m1", "summary": "s", "type": "fact"}
        ]

    def test_filters_empty_memory_id(self):
        m = _mgr()
        m._injected_items.append(
            RefinedItem(summary="s", memory_id="", memory_type="fact", confidence=0.0)
        )
        m._injected_items.append(
            RefinedItem(summary="t", memory_id="m2", memory_type="fact", confidence=0.0)
        )
        assert [i["memory_id"] for i in m.get_injected_items()] == ["m2"]
