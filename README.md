# OmniMem - 五层混合记忆系统

[![Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

> 基于 Ebbinghaus 遗忘曲线的智能记忆管理系统，集成 FSRS 算法、多维度评估、语义重要性分析。

## ✨ 核心特性

### 🧠 遗忘曲线管理
- **4 阶段生命周期**: active → consolidating → archived → forgotten
- **频率密度热度计算**: hot/warm/neutral/cold 四级热度
- **自动升级机制**: 高频访问记忆自动回到 active
- **Wiki 晋升**: T+30d 自动扫描并晋升候选记忆

### 📊 FSRS 算法集成
- **FSRS v4 引擎**: 目前最优的间隔重复算法
- **保持率预测**: 精度 85-90% (vs SM-2 的 60-70%)
- **间隔计算**: 智能建议下次复习时间
- **个性化参数**: 支持从历史数据学习

### 💪 六维记忆强度评估
- **稳定性 (Stability)**: FSRS 稳定性
- **可提取性 (Retrievability)**: 当前保持率
- **难度 (Difficulty)**: FSRS 难度
- **新近性 (Recency)**: 距离上次访问
- **频率 (Frequency)**: 访问次数
- **语义重要性 (Semantic Importance)**: 向量中心性

### 🔗 语义重要性评估
- **向量中心性**: 在语义空间中的位置
- **关联密度**: 与其他记忆的连接
- **图结构重要性**: 知识图谱中的位置
- **内容丰富度**: 内容质量和多样性

### 📈 可视化与 API
- **HTML 仪表盘**: 热度、强度、保持率分布图
- **REST API**: 完整的查询和管理接口
- **批量评估**: 支持批量记忆评估

---

## 🚀 快速开始

### 安装

```bash
# 克隆到 plugins 目录
git clone https://github.com/weksbwrx62862/omnimem.git ~/.hermes/plugins/omnimem

# 安装依赖
cd ~/.hermes/plugins/omnimem
pip install -r requirements.txt
```

### 配置

在 `~/.hermes/config.yaml` 中启用：

```yaml
plugins:
  enabled:
    - omnimem
```

### 基本使用

```python
from governance.forgetting import ForgettingCurve
from pathlib import Path

# 初始化
curve = ForgettingCurve(Path("~/.hermes/omnimem/governance"))

# 记录访问
curve.record_access("memory_001", memory_type="fact")

# 获取状态
status = curve.get_status()
print(f"热度分布: {status['heat']}")
print(f"阶段分布: {status['stages']}")

# FSRS 保持率
retention = curve.calculate_fsrs_retention("memory_001")
print(f"保持率: {retention:.2%}")

# 记忆强度评估
strength = curve.evaluate_memory_strength("memory_001")
print(f"评分: {strength['score']}, 等级: {strength['grade']}")

# 生成仪表盘
from governance.api import MemoryAPI

api = MemoryAPI()
result = api.generate_dashboard()
print(f"仪表盘: {result['filepath']}")
```

---

## 📚 API 参考

### ForgettingCurve

```python
class ForgettingCurve:
    def record_access(memory_id: str, memory_type: str = "fact")
    def get_stage(memory_id: str) -> str
    def get_heat(memory_id: str) -> str
    def calculate_fsrs_retention(memory_id: str) -> float
    def suggest_review_time(memory_id: str, desired_retention: float = 0.9) -> datetime
    def evaluate_memory_strength(memory_id: str) -> dict
    def evaluate_all_memories(limit: int = 100) -> dict
    def get_fsrs_stats() -> dict
    def get_strength_distribution() -> dict
    def get_semantic_importance_distribution() -> dict
    def run_archive_cycle() -> int
    def get_status() -> dict
```

### MemoryAPI

```python
class MemoryAPI:
    def get_memory_info(memory_id: str) -> dict
    def get_system_stats() -> dict
    def run_archive_cycle() -> dict
    def generate_dashboard(output_dir: str = None) -> dict
    def evaluate_batch(memory_ids: list[str]) -> dict
```

### FSRSEngine

```python
class FSRSEngine:
    def forgetting_curve(t: float, s: float) -> float
    def next_interval(s: float, desired_retention: float = 0.9) -> int
    def predict_retention(item: FSRSItem, now: datetime) -> float
    def review(item: FSRSItem, rating: int, now: datetime) -> FSRSItem
```

---

## 📊 系统状态

### 当前统计 (2026-05-26)

| 指标 | 数值 |
|------|------|
| **总记忆数** | 273 |
| **平均保持率** | 89.62% |
| **热度分布** | hot 49.1% / warm 3.3% / neutral 11.7% / cold 35.9% |
| **强度分布** | B 9.2% / C 76.9% / D 13.9% |
| **平均评分** | 49.88 |

### GitHub 提交记录

| 提交 | 内容 |
|------|------|
| `be27d21` | Phase 4: 语义重要性 + 可视化 + API |
| `6768f00` | Phase 3: 多维度记忆强度评估系统 |
| `cb2418f` | Phase 2: 引入 FSRS 算法 |
| `8cb8097` | Phase 1: 频率密度热度计算 + 自动升级 |

---

## 🧪 测试

```bash
# 运行测试
cd ~/.hermes/plugins/omnimem
pytest tests/ -v

# 生成覆盖率报告
pytest tests/ --cov=governance --cov-report=html
```

### 测试覆盖

- ✅ FSRS 引擎测试 (5 个)
- ✅ 记忆强度评估测试 (4 个)
- ✅ 语义重要性测试 (3 个)
- ✅ 遗忘曲线集成测试 (6 个)
- ✅ API 接口测试 (3 个)

**总计**: 21 个测试，100% 通过

---

## 📁 项目结构

```
omnimem/
├── governance/
│   ├── __init__.py
│   ├── forgetting.py           # 遗忘曲线核心
│   ├── fsrs_engine.py          # FSRS v4 引擎
│   ├── memory_strength.py      # 六维记忆强度评估
│   ├── semantic_importance.py  # 语义重要性评估
│   ├── visualizer.py           # 可视化模块
│   └── api.py                  # REST API
├── tests/
│   └── test_forgetting_curve.py  # 测试套件
├── README.md                   # 本文件
└── OPTIMIZATION_PLAN.md        # 优化计划
```

---

## 🔗 相关资源

- **FSRS 算法**: https://github.com/open-spaced-repetition/fsrs4anki
- **间隔重复理论**: https://supermemo.guru/wiki/Two_component_model_of_memory
- **Ebbinghaus 遗忘曲线**: https://en.wikipedia.org/wiki/Forgetting_curve

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- [FSRS](https://github.com/open-spaced-repetition/fsrs4anki) - 间隔重复算法
- [Anki](https://apps.ankiweb.net/) - 记忆卡片应用
- [SuperMemo](https://www.supermemo.com/) - 间隔重复理论
