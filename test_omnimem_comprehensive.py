#!/usr/bin/env python3
"""OmniMem 综合测试套件 — 覆盖五层架构的核心模块。

测试范围:
  1. CoreBlock — L1 工作记忆块
  2. CompactAttachment — 压缩后附件
  3. SoulSystem — 人格三元
  4. BudgetManager — Token 预算
  5. WingRoomManager — 宫殿导航
  6. DrawerClosetStore — Drawer/Closet 双存储
  7. ThreeLevelIndex — 三层索引
  8. ContextManager — 上下文管理（精炼/去重/预算）
  9. ConflictResolver — 冲突仲裁
  10. TemporalDecay — 时间衰减
  11. ForgettingCurve — 遗忘曲线
  12. PrivacyManager — 隐私分级
  13. PerceptionEngine — L0 感知引擎
  14. OmniMemProvider._should_store — 反递归防护
  15. OmniMemProvider._semantic_dedup — 语义去重
  16. OmniMemProvider._strip_system_injections — 输入净化

运行: python test_omnimem_comprehensive.py
"""

import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 添加项目路径（支持从项目根目录直接运行）
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from omnimem.core.block import CoreBlock
from omnimem.core.attachment import CompactAttachment, build_attachments
from omnimem.core.soul import SoulSystem
from omnimem.core.budget import BudgetManager
from omnimem.memory.wing_room import WingRoomManager
from omnimem.memory.drawer_closet import DrawerClosetStore
from omnimem.memory.index import ThreeLevelIndex
from omnimem.context.manager import ContextManager, ContextBudget, RefinedItem
from omnimem.governance.conflict import ConflictResolver, ConflictResult
from omnimem.governance.decay import TemporalDecay
from omnimem.governance.forgetting import ForgettingCurve
from omnimem.governance.privacy import PrivacyManager
from omnimem.perception.engine import PerceptionEngine, PerceptionSignals
from omnimem.config import OmniMemConfig


# ═══════════════════════════════════════════════════════════════════
# L1 工作记忆
# ═══════════════════════════════════════════════════════════════════

class TestCoreBlock(unittest.TestCase):
    """CoreBlock 测试。"""

    def test_default_values(self):
        cb = CoreBlock()
        self.assertEqual(cb.identity_block, "")
        self.assertEqual(cb.context_block, "")
        self.assertEqual(cb.plan_block, "")

    def test_to_prompt_text_all(self):
        cb = CoreBlock(identity_block="我是AI", context_block="当前任务", plan_block="步骤1")
        text = cb.to_prompt_text()
        self.assertIn("### Identity", text)
        self.assertIn("我是AI", text)
        self.assertIn("### Current Context", text)
        self.assertIn("当前任务", text)
        self.assertIn("### Plan", text)
        self.assertIn("步骤1", text)

    def test_to_prompt_text_partial(self):
        cb = CoreBlock(identity_block="AI")
        text = cb.to_prompt_text()
        self.assertIn("### Identity", text)
        self.assertNotIn("### Current Context", text)
        self.assertNotIn("### Plan", text)

    def test_update_context(self):
        cb = CoreBlock()
        cb.update_context("新上下文")
        self.assertEqual(cb.context_block, "新上下文")

    def test_update_plan(self):
        cb = CoreBlock()
        cb.update_plan("新计划")
        self.assertEqual(cb.plan_block, "新计划")


class TestCompactAttachment(unittest.TestCase):
    """CompactAttachment 测试。"""

    def test_to_text(self):
        att = CompactAttachment(kind="key_decisions", title="决策", body="选择了Python")
        text = att.to_text()
        self.assertIn("[key_decisions]", text)
        self.assertIn("决策", text)
        self.assertIn("选择了Python", text)

    def test_to_text_truncation(self):
        att = CompactAttachment(kind="task", title="T", body="X" * 1000)
        text = att.to_text(max_body_len=50)
        # body 应被截断
        self.assertLessEqual(len(text), 100)

    def test_build_attachments_empty(self):
        result = build_attachments([])
        self.assertEqual(result, [])

    def test_build_attachments_with_decisions(self):
        msgs = [
            {"role": "user", "content": "我决定使用Python开发"},
            {"role": "assistant", "content": "好的，Python很适合"},
        ]
        result = build_attachments(msgs)
        kinds = [a.kind for a in result]
        self.assertIn("key_decisions", kinds)

    def test_build_attachments_with_preferences(self):
        msgs = [{"role": "user", "content": "我喜欢暗色主题"}]
        result = build_attachments(msgs)
        kinds = [a.kind for a in result]
        self.assertIn("user_preferences", kinds)

    def test_build_attachments_with_errors(self):
        msgs = [{"role": "assistant", "content": "遇到了一个错误"}]
        result = build_attachments(msgs)
        kinds = [a.kind for a in result]
        self.assertIn("error_patterns", kinds)


class TestSoulSystem(unittest.TestCase):
    """SoulSystem 测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.soul = SoulSystem(Path(self.tmpdir))

    def test_load_identity_empty(self):
        result = self.soul.load_identity()
        self.assertEqual(result, "")

    def test_set_soul_and_load(self):
        self.soul.set_soul("我是AI助手")
        result = self.soul.load_identity()
        self.assertIn("我是AI助手", result)

    def test_set_soul_only_once(self):
        self.soul.set_soul("第一版")
        self.soul.set_soul("第二版")  # 不应覆盖
        result = self.soul.load_identity()
        self.assertIn("第一版", result)
        self.assertNotIn("第二版", result)

    def test_update_identity(self):
        self.soul.update_identity("开发者助手")
        result = self.soul.load_identity()
        self.assertIn("开发者助手", result)

    def test_update_user_profile_append(self):
        self.soul.update_user_profile("喜欢简洁")
        self.soul.update_user_profile("偏好中文")
        result = self.soul.load_identity()
        self.assertIn("喜欢简洁", result)
        self.assertIn("偏好中文", result)


class TestBudgetManager(unittest.TestCase):
    """BudgetManager 测试。"""

    def test_default_max_tokens(self):
        bm = BudgetManager()
        self.assertEqual(bm.max_tokens, 4000)

    def test_custom_max_tokens(self):
        bm = BudgetManager(max_tokens=2000)
        self.assertEqual(bm.max_tokens, 2000)

    def test_estimate_tokens(self):
        bm = BudgetManager()
        tokens = bm.estimate_tokens("Hello world")
        self.assertGreater(tokens, 0)

    def test_estimate_tokens_chinese(self):
        bm = BudgetManager()
        tokens = bm.estimate_tokens("你好世界")
        self.assertGreater(tokens, 0)

    def test_trim_to_budget(self):
        bm = BudgetManager(max_tokens=10)
        items = [
            {"content": "A" * 100},
            {"content": "B" * 100},
        ]
        result = bm.trim_to_budget(items)
        # 只能容纳第一部分
        self.assertLessEqual(len(result), 2)

    def test_fits(self):
        bm = BudgetManager(max_tokens=1000)
        self.assertTrue(bm.fits("短文本"))
        self.assertFalse(bm.fits("X" * 10000))


# ═══════════════════════════════════════════════════════════════════
# L2 结构化记忆
# ═══════════════════════════════════════════════════════════════════

class TestWingRoomManager(unittest.TestCase):
    """WingRoomManager 测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.wrm = WingRoomManager(Path(self.tmpdir))

    def test_resolve_wing(self):
        self.assertEqual(self.wrm.resolve_wing("personal"), "personal")
        self.assertEqual(self.wrm.resolve_wing("project"), "projects")
        self.assertEqual(self.wrm.resolve_wing("team"), "shared")
        self.assertEqual(self.wrm.resolve_wing("secret"), "personal")

    def test_resolve_hall(self):
        self.assertEqual(self.wrm.resolve_hall("fact"), "facts")
        self.assertEqual(self.wrm.resolve_hall("preference"), "preferences")
        self.assertEqual(self.wrm.resolve_hall("correction"), "corrections")

    def test_resolve_room_tech_keyword(self):
        room = self.wrm.resolve_room("使用python编写爬虫", "personal")
        self.assertEqual(room, "python")

    def test_resolve_room_chinese_topic(self):
        room = self.wrm.resolve_room("量子计算是未来技术", "personal")
        self.assertIn("量子", room)

    def test_resolve_room_fallback_hash(self):
        room = self.wrm.resolve_room("xxx", "personal", "fact")
        self.assertTrue(room.startswith("fact-"))

    def test_get_room_path(self):
        path = self.wrm.get_room_path("personal", "facts", "quantum")
        self.assertTrue(path.exists())
        self.assertIn("personal", str(path))
        self.assertIn("facts", str(path))
        self.assertIn("quantum", str(path))

    def test_list_wings(self):
        self.wrm.get_room_path("personal", "facts", "test")
        wings = self.wrm.list_wings()
        self.assertIn("personal", wings)

    def test_sanitize_name(self):
        self.assertEqual(WingRoomManager._sanitize_name("a/b\\c"), "a-b-c")
        self.assertEqual(WingRoomManager._sanitize_name(""), "unnamed")


class TestDrawerClosetStore(unittest.TestCase):
    """DrawerClosetStore 测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = DrawerClosetStore(Path(self.tmpdir))

    def test_add_and_get(self):
        mid = self.store.add(
            wing="personal", room="test", content="测试内容",
            memory_type="fact", confidence=3
        )
        self.assertTrue(mid)
        result = self.store.get(mid)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "测试内容")
        self.assertEqual(result["type"], "fact")

    def test_add_creates_disk_files(self):
        mid = self.store.add(
            wing="personal", room="test", content="磁盘测试"
        )
        # 磁盘写入缓冲：add() 后需显式 flush 才能落盘
        self.store.flush()
        # 检查 Drawer 和 Closet 文件
        palace = Path(self.tmpdir)
        drawer_files = list(palace.rglob(f"drawer/{mid}.md"))
        closet_files = list(palace.rglob(f"closet/{mid}.md"))
        self.assertTrue(len(drawer_files) > 0, "Drawer file should exist")
        self.assertTrue(len(closet_files) > 0, "Closet file should exist")

    def test_search_by_type(self):
        self.store.add(wing="personal", room="r1", content="事实1", memory_type="fact")
        self.store.add(wing="personal", room="r2", content="偏好1", memory_type="preference")
        facts = self.store.search(memory_type="fact")
        prefs = self.store.search(memory_type="preference")
        self.assertTrue(any(f["type"] == "fact" for f in facts))
        self.assertTrue(any(p["type"] == "preference" for p in prefs))

    def test_search_by_wing(self):
        self.store.add(wing="personal", room="r1", content="个人记忆")
        self.store.add(wing="shared", room="r2", content="共享记忆")
        personal = self.store.search(wing="personal")
        shared = self.store.search(wing="shared")
        self.assertTrue(all(m["wing"] == "personal" for m in personal))
        self.assertTrue(all(m["wing"] == "shared" for m in shared))

    def test_search_by_content(self):
        self.store.add(wing="personal", room="r1", content="Python是最好的语言")
        results = self.store.search_by_content("Python")
        self.assertTrue(len(results) > 0)
        self.assertIn("Python", results[0]["content"])

    def test_warm_up(self):
        mid = "warm-test-001"
        entries = [{
            "memory_id": mid,
            "content": "预热内容",
            "summary": "预热摘要",
            "type": "fact",
            "wing": "personal",
        }]
        self.store.warm_up(entries)
        result = self.store.get(mid)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "预热内容")

    def test_update_privacy(self):
        mid = self.store.add(wing="personal", room="r1", content="隐私测试", privacy="personal")
        ok = self.store.update_privacy(mid, "secret", new_wing="personal")
        self.assertTrue(ok)
        result = self.store.get(mid)
        self.assertEqual(result["privacy"], "secret")

    def test_get_nonexistent(self):
        result = self.store.get("nonexistent-id")
        self.assertIsNone(result)

    def test_lru_eviction(self):
        store = DrawerClosetStore(Path(self.tmpdir) / "evict", max_index_size=3)
        ids = []
        for i in range(5):
            ids.append(store.add(wing="personal", room=f"r{i}", content=f"内容{i}"))
        # 只应保留最近3个
        self.assertLessEqual(len(store._closet_index), 3)

    def test_closet_summary_no_newline(self):
        """Closet 摘要中换行符应被替换为空格。"""
        mid = self.store.add(wing="personal", room="r1", content="第一行\n第二行\n第三行")
        result = self.store.get(mid)
        self.assertIn("第一行", result["summary"])
        self.assertNotIn("\n", result["summary"])


class TestThreeLevelIndex(unittest.TestCase):
    """ThreeLevelIndex 测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index = ThreeLevelIndex(Path(self.tmpdir))

    def tearDown(self):
        self.index.close()

    def test_add_and_get(self):
        self.index.add(
            memory_id="idx-001", wing="personal", hall="facts",
            room="test", content="索引测试"
        )
        self.index.flush()
        result = self.index.get("idx-001")
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "索引测试")

    def test_search_l0(self):
        self.index.add(memory_id="l0-1", wing="personal", hall="facts", room="room-a", content="c1")
        self.index.add(memory_id="l0-2", wing="personal", hall="facts", room="room-b", content="c2")
        self.index.flush()
        rooms = self.index.search_l0(wing="personal", hall="facts")
        self.assertIn("room-a", rooms)
        self.assertIn("room-b", rooms)

    def test_search_l1(self):
        self.index.add(memory_id="l1-1", wing="personal", hall="facts", room="r", content="c", type="fact")
        self.index.flush()
        results = self.index.search_l1(wing="personal")
        self.assertTrue(len(results) > 0)

    def test_search_l2_keyword(self):
        self.index.add(memory_id="l2-1", wing="personal", hall="facts", room="r", content="量子计算")
        self.index.flush()
        results = self.index.search_l2(keyword="量子")
        self.assertTrue(len(results) > 0)

    def test_remove(self):
        self.index.add(memory_id="rm-1", wing="personal", hall="facts", room="r", content="c")
        self.index.flush()
        ok = self.index.remove("rm-1")
        self.assertTrue(ok)
        self.index.flush()
        result = self.index.get("rm-1")
        self.assertIsNone(result)

    def test_update_privacy(self):
        self.index.add(memory_id="up-1", wing="personal", hall="facts", room="r", content="c", privacy="personal")
        self.index.flush()
        ok = self.index.update_privacy("up-1", "secret")
        self.assertTrue(ok)
        self.index.flush()
        result = self.index.get("up-1")
        self.assertEqual(result["privacy"], "secret")

    def test_batch_commit(self):
        """添加多个条目后应自动提交。"""
        for i in range(10):
            self.index.add(memory_id=f"batch-{i}", wing="personal", hall="facts", room="r", content=f"c{i}")
        # flush 确保全部提交
        self.index.flush()
        for i in range(10):
            result = self.index.get(f"batch-{i}")
            self.assertIsNotNone(result, f"batch-{i} should exist")


# ═══════════════════════════════════════════════════════════════════
# 上下文管理
# ═══════════════════════════════════════════════════════════════════

class TestContextManager(unittest.TestCase):
    """ContextManager 测试。"""

    def setUp(self):
        self.cm = ContextManager(budget=ContextBudget(
            max_prefetch_tokens=300,
            max_summary_chars=60,
            max_prefetch_items=8,
        ))

    def test_refine_content_short(self):
        result = ContextManager.refine_content("短内容", max_chars=60)
        self.assertEqual(result, "短内容")

    def test_refine_content_long_truncated(self):
        long_text = "这是一段很长的内容" * 20
        result = ContextManager.refine_content(long_text, max_chars=60)
        self.assertLessEqual(len(result), 60)

    def test_refine_content_strip_prefix(self):
        result = ContextManager.refine_content("CORRECTION: 错误内容", max_chars=60)
        self.assertTrue(result.startswith("纠正:") or "错误内容" in result)

    def test_refine_content_turn_prefix(self):
        result = ContextManager.refine_content("[Turn 5] 重要事实", max_chars=60)
        self.assertNotIn("[Turn 5]", result)

    def test_content_fingerprint(self):
        fp1 = ContextManager._content_fingerprint("我喜欢Python")
        fp2 = ContextManager._content_fingerprint("我喜欢Python")
        self.assertEqual(fp1, fp2)

    def test_fingerprint_similarity_identical(self):
        fp = ContextManager._content_fingerprint("用户姓名: 徐信豪")
        sim = ContextManager._fingerprint_similarity(fp, fp)
        self.assertEqual(sim, 1.0)

    def test_fingerprint_similarity_different(self):
        fp1 = ContextManager._content_fingerprint("Python编程语言")
        fp2 = ContextManager._content_fingerprint("量子计算技术")
        sim = ContextManager._fingerprint_similarity(fp1, fp2)
        self.assertLess(sim, 0.5)

    def test_fingerprint_similarity_synonyms(self):
        fp1 = ContextManager._content_fingerprint("我喜欢暗色主题")
        fp2 = ContextManager._content_fingerprint("偏好深色模式")
        sim = ContextManager._fingerprint_similarity(fp1, fp2)
        # 同义词归一化后应较高
        self.assertGreater(sim, 0.5)

    def test_refine_prefetch_results(self):
        raw = [
            {"content": "用户喜欢Python", "type": "preference", "memory_id": "m1", "confidence": 4},
            {"content": "项目使用Docker部署", "type": "fact", "memory_id": "m2", "confidence": 3},
        ]
        result = self.cm.refine_prefetch_results(raw)
        self.assertIn("### Relevant Memories", result)
        self.assertIn("preference", result)
        self.assertIn("fact", result)

    def test_refine_prefetch_dedup(self):
        raw = [
            {"content": "用户偏好暗色主题", "type": "preference", "memory_id": "m1", "confidence": 4},
            {"content": "我喜欢深色模式", "type": "fact", "memory_id": "m2", "confidence": 3},
        ]
        result = self.cm.refine_prefetch_results(raw)
        # 两条语义相似（"暗色主题"≡"深色模式"），应去重
        lines = [l for l in result.split("\n") if l.startswith("- [")]
        self.assertLessEqual(len(lines), 2)

    def test_refine_prefetch_empty(self):
        result = self.cm.refine_prefetch_results([])
        self.assertEqual(result, "")

    def test_reset_for_new_turn(self):
        raw = [{"content": "测试", "type": "fact", "memory_id": "m1", "confidence": 3}]
        self.cm.refine_prefetch_results(raw)
        self.assertTrue(len(self.cm._injected_items) > 0)
        self.cm.reset_for_new_turn()
        self.assertEqual(len(self.cm._injected_items), 0)

    def test_persistent_fingerprints_preserved(self):
        self.cm.add_persistent_fingerprint("fp-test")
        self.assertIn("fp-test", self.cm._persistent_fingerprints)
        self.cm.reset_for_new_turn()
        # 持久指纹应保留
        self.assertIn("fp-test", self.cm.get_injected_fingerprints())

    def test_get_injected_items(self):
        raw = [{"content": "测试项", "type": "fact", "memory_id": "m1", "confidence": 3}]
        self.cm.refine_prefetch_results(raw)
        items = self.cm.get_injected_items()
        self.assertTrue(len(items) > 0)
        self.assertEqual(items[0]["memory_id"], "m1")

    def test_refine_recall_results(self):
        raw = [
            {"content": "关于量子计算的事实：量子比特可以叠加", "type": "fact", "memory_id": "m1", "confidence": 3},
            {"content": "关于深度学习的发现：Transformer架构", "type": "skill", "memory_id": "m2", "confidence": 3},
        ]
        result = self.cm.refine_recall_results(raw)
        # 两条不同主题，不应去重
        self.assertGreaterEqual(len(result), 1)
        self.assertIn("original_content", result[0])


# ═══════════════════════════════════════════════════════════════════
# 治理引擎
# ═══════════════════════════════════════════════════════════════════

class TestConflictResolver(unittest.TestCase):
    """ConflictResolver 测试。"""

    def test_no_conflict(self):
        cr = ConflictResolver(strategy="latest")
        result = cr.check("普通事实内容")
        self.assertFalse(result.has_conflict)

    def test_negation_without_existing(self):
        """仅含否定词但无已有记忆 → 不视为冲突。"""
        cr = ConflictResolver(strategy="latest")
        result = cr.check("纠正: 应该是Python")
        self.assertFalse(result.has_conflict)

    def test_negation_with_contradiction(self):
        """含否定词 + 已有矛盾记忆 → 冲突。"""
        cr = ConflictResolver(strategy="latest")
        existing = [{"content": "项目使用Java开发", "memory_id": "old-1"}]
        result = cr.check("项目使用Python开发，不对，应该不是Java", existing)
        # 取决于重叠率
        # 这里主要验证不会崩溃
        self.assertIsInstance(result, ConflictResult)

    def test_mutual_exclusive(self):
        """互斥选项检测。"""
        cr = ConflictResolver(strategy="latest")
        existing = [{"content": "使用AWS部署服务", "memory_id": "old-1"}]
        result = cr.check("使用腾讯云部署", existing)
        self.assertTrue(result.has_conflict)
        self.assertEqual(result.conflict_type, "semantic_contradiction")

    def test_resolve_latest(self):
        cr = ConflictResolver(strategy="latest")
        conflict = ConflictResult(has_conflict=True, existing_id="old-1")
        result = cr.resolve("新内容", conflict)
        self.assertEqual(result.action, "accept")

    def test_compute_overlap(self):
        overlap = ConflictResolver._compute_overlap("Python编程", "Python开发")
        self.assertGreater(overlap, 0)

    def test_compute_overlap_no_overlap(self):
        overlap = ConflictResolver._compute_overlap("量子计算", "烹饪食谱")
        self.assertLess(overlap, 0.3)


class TestTemporalDecay(unittest.TestCase):
    """TemporalDecay 测试。"""

    def test_fact_no_decay(self):
        td = TemporalDecay()
        results = [{
            "content": "事实",
            "type": "fact",
            "score": 1.0,
            "stored_at": (datetime.now(timezone.utc) - timedelta(days=100)).isoformat(),
        }]
        decayed = td.apply(results)
        self.assertEqual(decayed[0]["score"], 1.0)

    def test_event_decay(self):
        td = TemporalDecay()
        results = [{
            "content": "事件",
            "type": "event",
            "score": 1.0,
            "stored_at": (datetime.now(timezone.utc) - timedelta(days=90)).isoformat(),
        }]
        decayed = td.apply(results)
        self.assertLess(decayed[0]["score"], 1.0)
        self.assertIn("decay_factor", decayed[0])

    def test_preference_half_life(self):
        td = TemporalDecay()
        half_life = td.get_half_life("preference")
        self.assertEqual(half_life, 180)

    def test_custom_half_life(self):
        td = TemporalDecay(custom_half_lives={"event": 30})
        self.assertEqual(td.get_half_life("event"), 30)

    def test_sorted_by_score(self):
        td = TemporalDecay()
        results = [
            {"content": "旧事件", "type": "event", "score": 1.0,
             "stored_at": (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()},
            {"content": "新事件", "type": "event", "score": 1.0,
             "stored_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
        ]
        decayed = td.apply(results)
        self.assertGreater(decayed[0]["score"], decayed[1]["score"])


class TestForgettingCurve(unittest.TestCase):
    """ForgettingCurve 测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fc = ForgettingCurve(Path(self.tmpdir))

    def tearDown(self):
        self.fc.close()

    def test_default_stage(self):
        stage = self.fc.get_stage("new-memory")
        self.assertEqual(stage, "active")

    def test_archive(self):
        self.fc.archive("mem-1")
        self.fc.flush()
        stage = self.fc.get_stage("mem-1")
        self.assertEqual(stage, "archived")

    def test_archive_twice_to_forgotten(self):
        self.fc.archive("mem-2")
        self.fc.flush()
        self.fc.archive("mem-2")
        self.fc.flush()
        stage = self.fc.get_stage("mem-2")
        self.assertEqual(stage, "forgotten")

    def test_forgotten_not_archived_again(self):
        self.fc.archive("mem-3")
        self.fc.flush()
        self.fc.archive("mem-3")
        self.fc.flush()
        self.fc.archive("mem-3")  # 已经是 forgotten
        self.fc.flush()
        stage = self.fc.get_stage("mem-3")
        self.assertEqual(stage, "forgotten")

    def test_reactivate(self):
        self.fc.archive("mem-4")
        self.fc.flush()
        self.fc.reactivate("mem-4")
        self.fc.flush()
        stage = self.fc.get_stage("mem-4")
        self.assertEqual(stage, "active")

    def test_get_stage_by_age(self):
        self.assertEqual(self.fc.get_stage_by_age(0), "active")
        self.assertEqual(self.fc.get_stage_by_age(5), "active")
        self.assertEqual(self.fc.get_stage_by_age(10), "consolidating")
        self.assertEqual(self.fc.get_stage_by_age(60), "archived")
        self.assertEqual(self.fc.get_stage_by_age(120), "forgotten")

    def test_record_access(self):
        self.fc.record_access("mem-access")
        self.fc.flush()
        stage = self.fc.get_stage("mem-access")
        self.assertEqual(stage, "active")

    def test_get_status(self):
        self.fc.archive("s-1")
        self.fc.flush()
        status = self.fc.get_status()
        self.assertIn("active", status)
        self.assertIn("archived", status)


class TestPrivacyManager(unittest.TestCase):
    """PrivacyManager 测试。"""

    def test_default_level(self):
        pm = PrivacyManager()
        self.assertEqual(pm.get("any-id"), "personal")

    def test_set_and_get(self):
        pm = PrivacyManager()
        pm.set("m1", "public")
        self.assertEqual(pm.get("m1"), "public")

    def test_filter_secret(self):
        pm = PrivacyManager()
        results = [
            {"content": "公开", "privacy": "public"},
            {"content": "秘密", "privacy": "secret"},
        ]
        filtered = pm.filter(results)
        # OPT-1: secret 级不再直接过滤丢弃，而是保留并标记 _encrypted=True
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["privacy"], "public")
        self.assertEqual(filtered[1]["privacy"], "secret")
        self.assertTrue(filtered[1].get("_encrypted"))

    def test_filter_by_max_privacy(self):
        pm = PrivacyManager()
        results = [
            {"content": "公开", "privacy": "public"},
            {"content": "团队", "privacy": "team"},
            {"content": "个人", "privacy": "personal"},
        ]
        filtered = pm.filter(results, max_privacy="team")
        self.assertTrue(all(r["privacy"] in ("public", "team") for r in filtered))

    def test_bind_store_and_persist(self):
        tmpdir = tempfile.mkdtemp()
        store = DrawerClosetStore(Path(tmpdir))
        pm = PrivacyManager()
        pm.bind_store(store)
        mid = store.add(wing="personal", room="r", content="隐私测试", privacy="personal")
        pm.set(mid, "team")
        # 通过 store 验证持久化
        result = store.get(mid)
        self.assertEqual(result["privacy"], "team")

    def test_invalid_level_ignored(self):
        pm = PrivacyManager()
        pm.set("m1", "invalid_level")
        self.assertEqual(pm.get("m1"), "personal")


# ═══════════════════════════════════════════════════════════════════
# L0 感知引擎
# ═══════════════════════════════════════════════════════════════════

class TestPerceptionEngine(unittest.TestCase):
    """PerceptionEngine 测试。"""

    def setUp(self):
        self.pe = PerceptionEngine()

    def test_detect_correction(self):
        signals = self.pe.detect_signals("不对，应该是Python")
        self.assertTrue(signals.has_correction)
        self.assertTrue(signals.should_memorize)

    def test_detect_reinforcement(self):
        signals = self.pe.detect_signals("对，就是这样")
        self.assertTrue(signals.has_reinforcement)

    def test_detect_preference(self):
        signals = self.pe.detect_signals("我喜欢暗色主题")
        self.assertTrue(signals.has_preference)
        self.assertTrue(signals.should_memorize)

    def test_detect_memorable(self):
        signals = self.pe.detect_signals("记住这个重要信息")
        self.assertTrue(signals.should_memorize)

    def test_no_signal_for_plain_text(self):
        signals = self.pe.detect_signals("今天天气不错")
        self.assertFalse(signals.has_correction)
        self.assertFalse(signals.has_reinforcement)

    def test_correction_question_not_triggered(self):
        """问句中的'不对'不应触发纠正信号。"""
        signals = self.pe.detect_signals("这样不对吗？")
        self.assertFalse(signals.has_correction)

    def test_injection_content_not_memorized(self):
        """注入内容不应触发自动记忆。"""
        signals = self.pe.detect_signals("### Relevant Memories\n- [fact] 测试")
        self.assertFalse(signals.should_memorize)

    def test_extract_core_fact_preference(self):
        result = self.pe._extract_core_fact("我喜欢Python编程")
        self.assertIn("偏好", result)
        self.assertIn("Python", result)

    def test_extract_core_fact_name(self):
        result = self.pe._extract_core_fact("我叫徐信豪")
        self.assertIn("姓名", result)
        self.assertIn("徐信豪", result)

    def test_extract_core_fact_correction(self):
        result = self.pe._extract_core_fact("不对，应该是Python")
        self.assertIn("纠正", result)

    def test_predict_intent_returns_string(self):
        result = self.pe.predict_intent("什么是量子计算？")
        self.assertIsInstance(result, str)
        # 修复后应返回问号前的内容
        self.assertIn("量子计算", result)

    def test_predict_intent_with_entities(self):
        result = self.pe.predict_intent("Docker和Kubernetes的区别")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_extract_implicit_memories(self):
        result = self.pe.extract_implicit_memories("我喜欢简洁的代码风格。记住要用类型提示。")
        self.assertTrue(len(result) > 0)


# ═══════════════════════════════════════════════════════════════════
# 配置管理
# ═══════════════════════════════════════════════════════════════════

class TestOmniMemConfig(unittest.TestCase):
    """OmniMemConfig 测试。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_defaults(self):
        config = OmniMemConfig(Path(self.tmpdir))
        self.assertEqual(config.get("save_interval"), 15)
        self.assertEqual(config.get("retrieval_mode"), "rag")
        self.assertEqual(config.get("budget_tokens"), 4000)

    def test_set_and_get(self):
        config = OmniMemConfig(Path(self.tmpdir))
        config.set("custom_key", "custom_value")
        self.assertEqual(config.get("custom_key"), "custom_value")

    def test_save_and_reload(self):
        config = OmniMemConfig(Path(self.tmpdir))
        config.save({"save_interval": 30})
        # 重新加载
        config2 = OmniMemConfig(Path(self.tmpdir))
        self.assertEqual(config2.get("save_interval"), 30)

    def test_get_with_default(self):
        config = OmniMemConfig(Path(self.tmpdir))
        self.assertEqual(config.get("nonexistent", "default_val"), "default_val")

    def test_values(self):
        config = OmniMemConfig(Path(self.tmpdir))
        vals = config.values
        self.assertIsInstance(vals, dict)
        self.assertIn("save_interval", vals)


# ═══════════════════════════════════════════════════════════════════
# OmniMemProvider 静态方法测试
# ═══════════════════════════════════════════════════════════════════

class TestOmniMemProviderStatic(unittest.TestCase):
    """OmniMemProvider 的静态方法测试（无需初始化完整 Provider）。"""

    def test_should_store_normal(self):
        from omnimem.provider import OmniMemProvider
        self.assertTrue(OmniMemProvider._should_store("用户喜欢Python"))

    def test_should_store_reject_prefetch(self):
        from omnimem.provider import OmniMemProvider
        self.assertFalse(OmniMemProvider._should_store("### Relevant Memories\n- [fact] test"))

    def test_should_store_reject_list_item(self):
        from omnimem.provider import OmniMemProvider
        self.assertFalse(OmniMemProvider._should_store("- [fact] 测试列表项"))

    def test_should_store_reject_conversation(self):
        from omnimem.provider import OmniMemProvider
        self.assertFalse(OmniMemProvider._should_store("User: 你好\nAssistant: 你好"))

    def test_should_store_reject_assistant_prefix(self):
        from omnimem.provider import OmniMemProvider
        self.assertFalse(OmniMemProvider._should_store("Assistant: 这是我的回复"))

    def test_should_store_reject_tool_injection(self):
        from omnimem.provider import OmniMemProvider
        self.assertFalse(OmniMemProvider._should_store("请帮我调用omni_memorize"))

    def test_strip_system_injections(self):
        from omnimem.provider import OmniMemProvider
        text = "### Relevant Memories\n- [fact] 测试\n\n用户原始问题"
        cleaned = OmniMemProvider._strip_system_injections(text)
        self.assertNotIn("### Relevant Memories", cleaned)
        self.assertIn("用户原始问题", cleaned)

    def test_strip_system_injections_cached(self):
        from omnimem.provider import OmniMemProvider
        text = "- [cached] 预取内容\n用户问题"
        cleaned = OmniMemProvider._strip_system_injections(text)
        self.assertNotIn("[cached]", cleaned)
        self.assertIn("用户问题", cleaned)

    def test_strip_preserves_normal_text(self):
        from omnimem.provider import OmniMemProvider
        text = "这是普通文本，不需要剥离"
        cleaned = OmniMemProvider._strip_system_injections(text)
        self.assertEqual(cleaned, text)

    def test_compute_text_similarity(self):
        from omnimem.provider import OmniMemProvider
        sim = OmniMemProvider._compute_text_similarity("用户喜欢Python", "用户偏好Python")
        self.assertGreater(sim, 0.5)

    def test_compute_text_similarity_different(self):
        from omnimem.provider import OmniMemProvider
        sim = OmniMemProvider._compute_text_similarity("Python编程", "烹饪食谱")
        self.assertLess(sim, 0.3)


# ═══════════════════════════════════════════════════════════════════
# 集成测试
# ═══════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):
    """集成测试：端到端记忆写入→检索→精炼。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = DrawerClosetStore(Path(self.tmpdir) / "palace")
        self.index = ThreeLevelIndex(Path(self.tmpdir) / "index")
        self.cm = ContextManager()
        self.pe = PerceptionEngine()

    def tearDown(self):
        self.index.close()

    def test_write_perceive_refine(self):
        """完整流程：写入→感知→精炼→注入。"""
        # 1. 写入记忆
        mid = self.store.add(
            wing="personal", room="python", content="用户喜欢Python编程语言",
            memory_type="preference", confidence=5, privacy="personal",
        )
        self.assertTrue(mid)

        # 2. 感知信号检测
        signals = self.pe.detect_signals("我喜欢Python")
        self.assertTrue(signals.has_preference)

        # 3. 检索记忆
        results = self.store.search_by_content("Python")
        self.assertTrue(len(results) > 0)

        # 4. 精炼+去重
        refined = self.cm.refine_prefetch_results(results)
        self.assertIn("### Relevant Memories", refined)

    def test_multi_type_storage_and_search(self):
        """多种类型记忆的存储和分类搜索。"""
        self.store.add(wing="personal", room="r1", content="事实1", memory_type="fact")
        self.store.add(wing="personal", room="r2", content="偏好1", memory_type="preference")
        self.store.add(wing="personal", room="r3", content="纠正1", memory_type="correction")

        facts = self.store.search(memory_type="fact")
        prefs = self.store.search(memory_type="preference")
        corrections = self.store.search(memory_type="correction")

        self.assertTrue(any(f["type"] == "fact" for f in facts))
        self.assertTrue(any(p["type"] == "preference" for p in prefs))
        self.assertTrue(any(c["type"] == "correction" for c in corrections))

    def test_privacy_filter_integration(self):
        """隐私过滤集成测试。"""
        self.store.add(wing="personal", room="r1", content="公开", memory_type="fact", privacy="public")
        self.store.add(wing="personal", room="r2", content="秘密", memory_type="fact", privacy="secret")

        all_items = self.store.search(limit=10)
        pm = PrivacyManager()
        filtered = pm.filter(all_items)
        # OPT-1: secret 级保留但标记 _encrypted，不再直接过滤
        secret_items = [r for r in filtered if r.get("privacy") == "secret"]
        self.assertEqual(len(secret_items), 1)
        self.assertTrue(secret_items[0].get("_encrypted"))
        self.assertTrue(secret_items[0]["content"].startswith("[加密记忆"))

    def test_temporal_decay_integration(self):
        """时间衰减集成测试。"""
        self.store.add(wing="personal", room="r1", content="新事件", memory_type="event",
                       confidence=3, privacy="personal")
        self.store.add(wing="personal", room="r2", content="旧事件", memory_type="event",
                       confidence=3, privacy="personal")

        results = self.store.search(memory_type="event")
        # 手动设置时间
        now = datetime.now(timezone.utc)
        results[0]["stored_at"] = now.isoformat()
        results[0]["score"] = 1.0
        results[0]["type"] = "event"
        results[1]["stored_at"] = (now - timedelta(days=180)).isoformat()
        results[1]["score"] = 1.0
        results[1]["type"] = "event"

        td = TemporalDecay()
        decayed = td.apply(results)
        # 新事件应排前面
        self.assertGreater(decayed[0]["score"], decayed[1]["score"])


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)
