"""HMS Entity-bridge (E) 算子离线测试。

覆盖 `omnimem.retrieval.canonical_entity`：
  - 归一化：全半角、括号、空白压缩
  - resolve：精确别名、称谓剥离、未命中回退
  - register_alias / register_group：去重、空值保护
  - load/save：JSON 往返、非法输入非致命
  - normalize_entities：批量 + 顺序去重
  - get_default_resolver / normalize_entity：模块级单例
"""

from __future__ import annotations

import json

import pytest
from omnimem.retrieval import canonical_entity as ce
from omnimem.retrieval.canonical_entity import (
    CanonicalEntityResolver,
    get_default_resolver,
    normalize_entity,
)


@pytest.fixture
def resolver() -> CanonicalEntityResolver:
    return CanonicalEntityResolver()


# ── _normalize ──


def test_normalize_handles_full_width_and_parens() -> None:
    # 全角 ＡＢＣ（） → 半角 ABC()
    assert CanonicalEntityResolver._normalize("ＡＢＣ（测试）") == "ABC(测试)"


def test_normalize_strips_all_whitespace_including_ideographic() -> None:
    assert CanonicalEntityResolver._normalize("王\u3000 明 abc") == "王明abc"


def test_normalize_empty_returns_empty() -> None:
    assert CanonicalEntityResolver._normalize("") == ""
    assert CanonicalEntityResolver._normalize("   ") == ""


# ── register_alias / resolve ──


def test_register_alias_maps_to_canonical(resolver: CanonicalEntityResolver) -> None:
    resolver.register_alias("王明", "小王")
    assert resolver.resolve("小王") == "王明"


def test_register_alias_is_idempotent(resolver: CanonicalEntityResolver) -> None:
    resolver.register_alias("王明", "小王")
    resolver.register_alias("王明", "小王")
    assert resolver.get_aliases("王明") == ["小王"]


def test_register_alias_skips_empty_alias(resolver: CanonicalEntityResolver) -> None:
    resolver.register_alias("王明", "")
    assert resolver.canonical_count() == 0


def test_register_alias_skips_empty_canonical(resolver: CanonicalEntityResolver) -> None:
    resolver.register_alias("", "小王")
    assert resolver.canonical_count() == 0


def test_register_alias_normalizes_inputs(resolver: CanonicalEntityResolver) -> None:
    # 全角别名与半角规范 → 归一化后仍可查询
    resolver.register_alias("ＡＢＣ", "小王")
    assert resolver.resolve("小王") == "ABC"
    assert resolver.resolve("  小王  ") == "ABC"


def test_register_group_batches(resolver: CanonicalEntityResolver) -> None:
    resolver.register_group("王明", ["小王", "王总", "王经理"])
    for alias in ("小王", "王总", "王经理"):
        assert resolver.resolve(alias) == "王明"
    assert resolver.canonical_count() == 1
    assert sorted(resolver.get_aliases("王明")) == sorted(["小王", "王总", "王经理"])


# ── resolve 3-step ──


def test_resolve_exact_alias_hit(resolver: CanonicalEntityResolver) -> None:
    resolver.register_alias("王明", "王总")
    assert resolver.resolve("王总") == "王明"


def test_resolve_title_stripped_fallback(resolver: CanonicalEntityResolver) -> None:
    # 表中登记了 "王"，查询 "王经理" → 剥离称谓 → "王" → 命中
    resolver.register_alias("王明", "王")
    assert resolver.resolve("王经理") == "王明"


def test_resolve_no_hit_returns_stripped_when_title_present(
    resolver: CanonicalEntityResolver,
) -> None:
    # 未命中映射，剥离后返回主干（"李经理" → "李"）
    assert resolver.resolve("李经理") == "李"


def test_resolve_no_hit_returns_normalized_input(resolver: CanonicalEntityResolver) -> None:
    assert resolver.resolve("张三") == "张三"


def test_resolve_empty_returns_empty(resolver: CanonicalEntityResolver) -> None:
    assert resolver.resolve("") == ""


def test_resolve_idempotent_for_canonical(resolver: CanonicalEntityResolver) -> None:
    resolver.register_alias("王明", "小王")
    # 对规范名再次 resolve 应保持稳定（不进入 alias 表时返回归一化输入）
    assert resolver.resolve("王明") == "王明"
    assert resolver.resolve("王明") == "王明"


# ── get_aliases / canonical_count ──


def test_get_aliases_unknown_returns_empty(resolver: CanonicalEntityResolver) -> None:
    assert resolver.get_aliases("不存在") == []


def test_canonical_count_tracks_unique_canonicals(resolver: CanonicalEntityResolver) -> None:
    resolver.register_group("A", ["a1", "a2"])
    resolver.register_group("B", ["b1"])
    assert resolver.canonical_count() == 2


# ── normalize_entities batch ──


def test_normalize_entities_dedup_preserves_order(
    resolver: CanonicalEntityResolver,
) -> None:
    resolver.register_group("王明", ["小王", "王总"])
    out = resolver.normalize_entities(["小王", "张三", "王总", "张三", ""])
    assert out == ["王明", "张三"]


def test_normalize_entities_empty_list(resolver: CanonicalEntityResolver) -> None:
    assert resolver.normalize_entities([]) == []


# ── load / save ──


def test_save_load_round_trip(resolver: CanonicalEntityResolver, tmp_path) -> None:
    path = tmp_path / "canon.json"
    resolver.register_group("王明", ["小王", "王总"])
    resolver.register_group("李雷", ["小李"])
    resolver.save(str(path))

    fresh = CanonicalEntityResolver()
    fresh.load(str(path))
    assert fresh.resolve("小王") == "王明"
    assert fresh.resolve("小李") == "李雷"
    assert fresh.canonical_count() == 2


def test_load_missing_file_is_non_fatal(tmp_path) -> None:
    r = CanonicalEntityResolver()
    r.load(str(tmp_path / "nope.json"))
    assert r.canonical_count() == 0


def test_load_invalid_json_is_non_fatal(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json at all", encoding="utf-8")
    r = CanonicalEntityResolver()
    r.load(str(bad))
    assert r.canonical_count() == 0


def test_constructor_auto_loads(tmp_path) -> None:
    path = tmp_path / "canon.json"
    path.write_text(
        json.dumps({"王明": ["小王", "王总"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    r = CanonicalEntityResolver(data_file=str(path))
    assert r.resolve("小王") == "王明"


def test_save_without_path_and_no_data_file_is_noop(
    resolver: CanonicalEntityResolver,
) -> None:
    resolver.register_alias("A", "a")
    resolver.save()  # 应静默返回，不抛异常


# ── 模块级单例 ──


def test_get_default_resolver_is_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ce, "_default_resolver", None)
    r1 = get_default_resolver()
    r2 = get_default_resolver()
    assert r1 is r2


def test_normalize_entity_convenience(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ce, "_default_resolver", None)
    resolver = get_default_resolver()
    resolver.register_alias("王明", "王总")
    assert normalize_entity("王总") == "王明"


def test_repr_reports_entry_count(resolver: CanonicalEntityResolver) -> None:
    resolver.register_group("A", ["a1"])
    assert repr(resolver) == "CanonicalEntityResolver(entries=1)"
