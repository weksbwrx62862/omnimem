# OmniMem 项目维护计划

## 项目概述

OmniMem 是一个为 AI Agent 设计的多层混合记忆系统，采用五层架构（感知→工作→结构化→深层→内化），配备完整的治理引擎。

- **当前版本**: 1.0.0
- **代码规模**: 约 647 个函数/方法，79 个类，53 个 Python 文件
- **类型注解覆盖率**: 约 61.7% (399/647 个函数有返回类型注解)
- **文档字符串覆盖率**: 较高，核心模块均有文档

---

## 一、已发现的可维护项

### 🔴 高优先级

#### 1. 缺少标准化的依赖管理文件
- **问题**: 项目没有 `requirements.txt`、`pyproject.toml` 或 `setup.py`
- **影响**: 
  - 用户无法通过标准方式安装依赖
  - 无法发布到 PyPI
  - 依赖版本无法锁定，存在兼容性问题
- **建议**: 
  - 创建 `pyproject.toml`（现代 Python 标准）
  - 或至少创建 `requirements.txt` 和 `requirements-dev.txt`
  - 在 `pyproject.toml` 中定义可选依赖组（如 `crypto`, `lora`, `dev`）

#### 2. 缺少 CI/CD 流水线
- **问题**: `.github/workflows/` 目录不存在
- **影响**:
  - 无自动化测试
  - 无代码质量检查（lint、format）
  - 无自动发布流程
- **建议**:
  - 创建 GitHub Actions 工作流：
    - `ci.yml`: 运行测试、代码检查（ruff/black/mypy）
    - `release.yml`: 自动发布到 PyPI
  - 添加 pre-commit 配置

#### 3. 测试结构不完整
- **问题**: 
  - 测试文件位于项目根目录（`test_omnimem_comprehensive.py`、`test_qual2_fix.py`），未放入 `tests/` 目录
  - 缺少 `tests/__init__.py`
  - 测试导入路径硬编码了 `plugins.memory.omnimem`，在独立运行时可能失败
- **建议**:
  - 创建 `tests/` 目录结构：
    ```
    tests/
    ├── __init__.py
    ├── conftest.py          # pytest 共享 fixture
    ├── test_core/
    ├── test_memory/
    ├── test_governance/
    ├── test_retrieval/
    ├── test_compression/
    └── test_integration.py
    ```
  - 修复测试导入路径，支持独立运行
  - 添加 pytest 配置到 `pyproject.toml`

---

### 🟡 中优先级

#### 4. 类型注解覆盖率可提升
- **当前**: 约 61.7% 的函数有返回类型注解
- **缺失类型注解的文件**:
  - `retrieval/rrf.py` (1/2)
  - `retrieval/reranker.py` (1/3)
  - `handlers/govern.py` (2/2 - 但参数类型不全)
  - `handlers/memorize.py` (1/1)
  - `handlers/recall.py` (2/2)
  - `core/async_provider.py` (7/9)
  - `compression/` 模块整体较低
- **建议**: 逐步为所有公共 API 添加类型注解，启用 `mypy` 检查

#### 5. 代码质量工具缺失
- **问题**: 无 lint、format、type check 配置
- **建议**: 
  - 添加 `ruff` 配置（替代 flake8 + black + isort）
  - 添加 `mypy` 配置
  - 示例 `pyproject.toml` 配置：
    ```toml
    [tool.ruff]
    line-length = 100
    select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]
    
    [tool.mypy]
    python_version = "3.10"
    strict = true
    warn_return_any = true
    ```

#### 6. GitHub 仓库配置可优化
- **当前 `.github/settings.yml` 配置**:
  - `has_projects: false` — 建议开启用于路线图管理
  - `has_discussions: false` — 建议开启用于社区问答
  - `has_wiki: false` — 可保持关闭，文档放在 README/docs
- **建议**:
  - 开启 Discussions 作为社区支持渠道
  - 开启 Projects 用于路线图跟踪
  - 添加 Issue 模板（Bug 报告、功能请求）
  - 添加 PR 模板

#### 7. 缺少 CHANGELOG
- **问题**: 无版本变更记录
- **建议**: 
  - 创建 `CHANGELOG.md`，遵循 [Keep a Changelog](https://keepachangelog.com/) 格式
  - 或使用 GitHub Releases 自动生成

---

### 🟢 低优先级

#### 8. README 中的安装说明需更新
- **问题**: 
  - 提到 `pip install -r requirements.txt` 但文件不存在
  - 提到 `pytest tests/` 但 `tests/` 目录不存在
- **建议**: 修复 README 中的安装和测试说明

#### 9. 路线图项目未更新
- **当前路线图状态**:
  - [x] 五层记忆架构
  - [x] 混合检索引擎
  - [x] 完整治理引擎
  - [x] Saga 事务协调
  - [x] 多实例同步
  - [x] 内置 memory 工具兼容
  - [ ] 分布式部署支持
  - [ ] Web 管理界面
  - [ ] 记忆可视化（知识图谱渲染）
  - [ ] 自动 LoRA 训练流水线
- **建议**: 
  - 为未完成的项创建 GitHub Issues
  - 添加预估优先级和时间线

#### 10. 安全相关
- **当前**: `.gitignore` 已正确排除敏感文件（`.env`, `config.yaml`, `*.key` 等）
- **建议**: 
  - 添加 `SECURITY.md` 说明如何报告安全漏洞
  - 考虑添加依赖漏洞扫描（Dependabot）

#### 11. 贡献指南可完善
- **当前**: README 中有简单的开发流程
- **建议**: 
  - 创建 `CONTRIBUTING.md` 详细说明：
    - 开发环境搭建
    - 代码规范（Black、PEP 8）
    - 测试要求
    - PR 流程

---

## 二、维护计划执行步骤

### 阶段 1: 基础设施（高优先级）
1. **创建 `pyproject.toml`**
   - 定义项目元数据
   - 定义依赖和可选依赖
   - 配置构建工具（setuptools/flit/hatch）
   - 配置 pytest、mypy、ruff

2. **创建 `requirements.txt` 和 `requirements-dev.txt`**
   - 锁定核心依赖版本
   - 分离开发依赖（pytest、mypy、ruff、black）

3. **创建 GitHub Actions CI 工作流**
   - `.github/workflows/ci.yml`
   - 运行测试矩阵（Python 3.10, 3.11, 3.12）
   - 运行代码检查（ruff、black --check、mypy）

4. **重构测试目录**
   - 创建 `tests/` 目录结构
   - 移动现有测试文件
   - 修复导入路径
   - 添加 `conftest.py` 和共享 fixture

### 阶段 2: 代码质量（中优先级）
5. **添加 pre-commit 配置**
   - `.pre-commit-config.yaml`
   - 配置 ruff、black、mypy 钩子

6. **提升类型注解覆盖率**
   - 优先为公共 API 添加类型注解
   - 启用 mypy 严格模式

7. **添加 Issue/PR 模板**
   - `.github/ISSUE_TEMPLATE/bug_report.md`
   - `.github/ISSUE_TEMPLATE/feature_request.md`
   - `.github/pull_request_template.md`

### 阶段 3: 文档与社区（低优先级）
8. **创建 `CHANGELOG.md`**
9. **创建 `CONTRIBUTING.md`**
10. **创建 `SECURITY.md`**
11. **更新 README**
    - 修复安装说明
    - 添加 CI 状态徽章
    - 添加 PyPI 版本徽章

12. **开启 GitHub Discussions 和 Projects**
    - 更新 `.github/settings.yml`

---

## 三、代码健康度评估

| 指标 | 状态 | 评分 |
|:---:|:---:|:---:|
| 代码结构 | 模块化良好，分层清晰 | ⭐⭐⭐⭐⭐ |
| 文档字符串 | 核心模块均有文档 | ⭐⭐⭐⭐⭐ |
| 类型注解 | 61.7%，有提升空间 | ⭐⭐⭐ |
| 测试覆盖 | 有综合测试，但结构需整理 | ⭐⭐⭐ |
| 依赖管理 | 完全缺失 | ⭐ |
| CI/CD | 完全缺失 | ⭐ |
| 社区工具 | Issue/PR 模板缺失 | ⭐⭐ |
| 安全实践 | .gitignore 完善 | ⭐⭐⭐⭐ |

**总体健康度**: ⭐⭐⭐ (中等，基础设施需要加强)

---

## 四、立即行动项

如果只能做 3 件事，建议优先：

1. **创建 `pyproject.toml`** — 解决依赖管理和构建问题
2. **创建 GitHub Actions CI** — 确保代码质量和测试自动化
3. **重构测试目录** — 使测试可维护、可扩展

这三项完成后，项目的可维护性将显著提升。
