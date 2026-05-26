"""
OmniMem 遗忘曲线系统测试套件

覆盖:
- FSRS 引擎测试
- 记忆强度评估测试
- 语义重要性测试
- 遗忘曲线集成测试
- API 接口测试
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone


# ── FSRS 引擎测试 ──────────────────────────────────────────────────────────

class TestFSRSEngine:
    """FSRS 引擎测试"""

    def setup_method(self):
        from governance.fsrs_engine import FSRSEngine
        self.engine = FSRSEngine()

    def test_forgetting_curve(self):
        """测试遗忘曲线计算"""
        # t=0 时保持率应为 1
        assert self.engine.forgetting_curve(0, 10) == 1.0

        # t>0 时保持率应 < 1
        retention = self.engine.forgetting_curve(1, 10)
        assert 0 < retention < 1

        # 稳定性越高，保持率越高
        retention_low = self.engine.forgetting_curve(7, 5)
        retention_high = self.engine.forgetting_curve(7, 20)
        assert retention_high > retention_low

    def test_next_interval(self):
        """测试间隔计算"""
        # 稳定性=10，目标保持率=0.9
        interval = self.engine.next_interval(10, 0.9)
        assert interval >= 1

        # 稳定性越高，间隔越长
        interval_low = self.engine.next_interval(5, 0.9)
        interval_high = self.engine.next_interval(20, 0.9)
        assert interval_high > interval_low

    def test_init_stability(self):
        """测试初始化稳定性"""
        # rating=1 (Again) 应有最低稳定性
        stability = self.engine.init_stability(1)
        assert stability > 0

        # rating=4 (Easy) 应有最高稳定性
        stability_easy = self.engine.init_stability(4)
        stability_again = self.engine.init_stability(1)
        assert stability_easy > stability_again

    def test_review(self):
        """测试复习模拟"""
        from governance.fsrs_engine import FSRSItem

        item = FSRSItem()
        now = datetime.now()

        # 第一次复习
        item = self.engine.review(item, 3, now)
        assert item.reps == 1
        assert item.stability > 0
        assert item.last_review == now

        # 第二次复习
        item = self.engine.review(item, 4, now + timedelta(days=3))
        assert item.reps == 2
        assert item.stability > 0

    def test_predict_retention(self):
        """测试保持率预测"""
        from governance.fsrs_engine import FSRSItem

        item = FSRSItem(stability=10, last_review=datetime.now())
        now = datetime.now()

        # 刚复习完，保持率应接近 1
        retention = self.engine.predict_retention(item, now)
        assert retention > 0.95

        # 10天后，保持率应下降
        future = now + timedelta(days=10)
        retention_future = self.engine.predict_retention(item, future)
        assert retention_future < retention


# ── 记忆强度评估测试 ──────────────────────────────────────────────────────

class TestMemoryStrengthEvaluator:
    """记忆强度评估器测试"""

    def setup_method(self):
        from governance.memory_strength import MemoryStrengthEvaluator
        self.evaluator = MemoryStrengthEvaluator()

    def test_calculate_strength(self):
        """测试强度计算"""
        strength = self.evaluator.calculate_strength(
            memory_id="test_001",
            recall_count=5,
            fsrs_retention=0.85,
            fsrs_stability=10.0,
        )

        assert strength.stability == 10.0
        assert strength.retrievability == 0.85
        assert strength.frequency == 5

    def test_calculate_score(self):
        """测试综合评分"""
        from governance.memory_strength import MemoryStrengthVector

        strength = MemoryStrengthVector(
            stability=10.0,
            retrievability=0.85,
            frequency=5,
        )

        score = self.evaluator.calculate_score(strength)
        assert 0 <= score <= 100

    def test_score_to_grade(self):
        """测试等级划分"""
        assert self.evaluator._score_to_grade(95) == "S"
        assert self.evaluator._score_to_grade(85) == "A"
        assert self.evaluator._score_to_grade(65) == "B"
        assert self.evaluator._score_to_grade(45) == "C"
        assert self.evaluator._score_to_grade(25) == "D"

    def test_evaluate_memory(self):
        """测试单个记忆评估"""
        result = self.evaluator.evaluate_memory(
            memory_id="test_001",
            recall_count=5,
            fsrs_retention=0.85,
            fsrs_stability=10.0,
        )

        assert "memory_id" in result
        assert "score" in result
        assert "grade" in result
        assert result["memory_id"] == "test_001"


# ── 语义重要性测试 ──────────────────────────────────────────────────────

class TestSemanticImportanceEvaluator:
    """语义重要性评估器测试"""

    def setup_method(self):
        from governance.semantic_importance import SemanticImportanceEvaluator
        self.evaluator = SemanticImportanceEvaluator()

    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]

        # 相同向量相似度应为 1
        sim = self.evaluator._cosine_similarity(vec1, vec2)
        assert abs(sim - 1.0) < 0.001

        # 正交向量相似度应为 0
        sim = self.evaluator._cosine_similarity(vec1, vec3)
        assert abs(sim) < 0.001

    def test_calculate_content_richness(self):
        """测试内容丰富度计算"""
        # 短内容
        richness_short = self.evaluator.calculate_content_richness("Hello")

        # 长内容
        richness_long = self.evaluator.calculate_content_richness(
            "This is a much longer content with many words and punctuation marks."
        )

        # 长内容应有更高的丰富度
        assert richness_long >= richness_short

    def test_evaluate(self):
        """测试语义评估"""
        features = self.evaluator.evaluate("test_001")

        assert hasattr(features, 'vector_centrality')
        assert hasattr(features, 'connection_density')
        assert hasattr(features, 'graph_importance')
        assert hasattr(features, 'content_richness')
        assert hasattr(features, 'uniqueness')


# ── 遗忘曲线集成测试 ──────────────────────────────────────────────────────

class TestForgettingCurveIntegration:
    """遗忘曲线集成测试"""

    def setup_method(self):
        from governance.forgetting import ForgettingCurve
        self.tmpdir = tempfile.mkdtemp()
        self.curve = ForgettingCurve(Path(self.tmpdir) / "governance")

    def teardown_method(self):
        self.curve.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_access(self):
        """测试访问记录"""
        self.curve.record_access("test_001", memory_type="fact")

        stage = self.curve.get_stage("test_001")
        assert stage == "active"

    def test_heat_calculation(self):
        """测试热度计算"""
        # 创建记忆
        self.curve.record_access("test_001", memory_type="fact")

        # 多次访问
        for _ in range(5):
            self.curve.record_access("test_001")

        # 运行筛选
        self.curve.run_first_screening()

        # 验证热度
        heat = self.curve.get_heat("test_001")
        assert heat in ["hot", "warm", "neutral", "cold"]

    def test_fsrs_retention(self):
        """测试 FSRS 保持率计算"""
        self.curve.record_access("test_001", memory_type="fact")

        retention = self.curve.calculate_fsrs_retention("test_001")
        assert 0 <= retention <= 1

    def test_evaluate_memory_strength(self):
        """测试记忆强度评估"""
        self.curve.record_access("test_001", memory_type="fact")

        result = self.curve.evaluate_memory_strength("test_001")
        assert "memory_id" in result
        assert "score" in result
        assert "grade" in result

    def test_archive_cycle(self):
        """测试归档周期"""
        # 创建多个记忆
        for i in range(5):
            self.curve.record_access(f"test_{i:03d}", memory_type="fact")

        # 运行归档周期
        archived = self.curve.run_archive_cycle()
        assert archived >= 0

    def test_get_status(self):
        """测试状态获取"""
        self.curve.record_access("test_001", memory_type="fact")

        status = self.curve.get_status()
        assert "stages" in status
        assert "heat" in status


# ── API 接口测试 ──────────────────────────────────────────────────────

class TestMemoryAPI:
    """API 接口测试"""

    def setup_method(self):
        from governance.api import MemoryAPI
        self.tmpdir = tempfile.mkdtemp()
        self.api = MemoryAPI(Path(self.tmpdir) / "governance")

    def teardown_method(self):
        self.api.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_system_stats(self):
        """测试系统统计"""
        stats = self.api.get_system_stats()

        assert "total_memories" in stats
        assert "stages" in stats
        assert "heat" in stats

    def test_run_archive_cycle(self):
        """测试归档周期"""
        result = self.api.run_archive_cycle()
        assert "archived" in result

    def test_generate_dashboard(self):
        """测试仪表盘生成"""
        result = self.api.generate_dashboard(output_dir=self.tmpdir)

        assert "filepath" in result
        assert os.path.exists(result["filepath"])


# ── 运行测试 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
