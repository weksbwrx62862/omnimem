# OmniMem GitHub 仓库背景设定维护计划

## 目标

维护 GitHub 仓库"背景设定"（Settings）中的各项配置，使仓库功能完整、社区友好、自动化程度高。

---

## 当前状态分析

### 已有配置（.github/settings.yml）

| 项目 | 当前值 | 状态 |
|------|--------|------|
| 仓库名称 | omnimem | ✅ |
| 描述 | 五层混合记忆系统... | ✅ |
| Topics | 13 个标签 | ✅ |
| Issues | true | ✅ |
| Projects | false | ⚠️ 建议开启 |
| Wiki | false | ✅ |
| Discussions | false | ⚠️ 建议开启 |
| 默认分支 | main | ✅ |
| Squash Merge | true | ✅ |
| Merge Commit | true | ✅ |
| Rebase Merge | true | ✅ |
| Delete branch on merge | true | ✅ |

### 截图中可见的未配置项

1. **社交预览** — 未上传图片
2. **发行作品（Releases）** — 未启用"发布不可变性"
3. **议题（Issues）** — 已启用但未配置模板
4. **GitHub 应用** — 需手动在网页端安装

---

## 维护计划

### 阶段 1: 更新 .github/settings.yml（高优先级）

#### 1.1 开启 Projects 和 Discussions
```yaml
has_projects: true      # 用于路线图管理
has_discussions: true   # 用于社区问答
```

#### 1.2 添加更多仓库配置
```yaml
# 启用 Releases 的不可变性
enable_release_notes: true

# 允许自动删除已合并分支（已有）
delete_branch_on_merge: true

# 允许 fork
allow_forking: true

# 允许 issues 和 PR 的空白模板
blank_issues_enabled: true
```

#### 1.3 添加分支保护规则
```yaml
branches:
  - name: main
    protection:
      required_pull_request_reviews:
        required_approving_review_count: 1
      required_status_checks:
        strict: true
        contexts:
          - "Lint & Format Check"
          - "Test Python 3.10"
          - "Test Python 3.11"
          - "Test Python 3.12"
      enforce_admins: false
      required_linear_history: false
      allow_force_pushes: false
      allow_deletions: false
```

### 阶段 2: 社交预览图片（中优先级）

为仓库上传社交预览图片（用于 Twitter/Facebook 等分享时显示）：
- 尺寸：1280×640px（最低 640×320px）
- 内容：项目 Logo + 简短描述
- 格式：PNG 或 JPG

**建议设计元素**：
- 背景色：深色（#1a1a2e 或类似）
- 标题：OmniMem
- 副标题：五层混合记忆系统
- 图标：大脑/记忆相关的抽象图形

### 阶段 3: 配置议题模板（中优先级）

已在之前的维护中创建了：
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`

需确认：
- 模板是否正确渲染
- 是否需要添加 config.yml 配置模板选择器

### 阶段 4: GitHub 应用安装（需手动在网页端操作）

以下应用需要在 GitHub 网页端手动安装：

#### 4.1 pre-commit.ci
- 访问：https://pre-commit.ci
- 登录 GitHub 账号
- 在设置中启用 `weksbwrx62862/omnimem`
- 效果：每次 PR 自动运行 pre-commit 检查

#### 4.2 Codecov
- 访问：https://codecov.io
- 登录 GitHub 账号
- 添加 `weksbwrx62862/omnimem` 仓库
- 效果：PR 中显示覆盖率报告

#### 4.3 Dependabot（已配置，自动生效）
- 配置文件：`.github/dependabot.yml`
- 效果：自动创建依赖更新 PR

### 阶段 5: 发布第一个 Release（低优先级）

创建 v1.0.0 Release：
- 基于当前 main 分支
- 使用 CHANGELOG.md 中的内容
- 标记为 "Latest"
- 启用"发布不可变性"

---

## 执行步骤

### 步骤 1: 更新 .github/settings.yml
1. 修改 `has_projects: true`
2. 修改 `has_discussions: true`
3. 添加分支保护配置
4. 提交并推送

### 步骤 2: 创建社交预览图片
1. 设计 1280×640px 的预览图
2. 上传到 GitHub 仓库设置页面

### 步骤 3: 验证议题模板
1. 在 GitHub 上创建测试 Issue
2. 确认模板正确渲染

### 步骤 4: 手动安装 GitHub 应用
1. 安装 pre-commit.ci
2. 安装 Codecov
3. 验证 Dependabot 已生效

### 步骤 5: 创建 Release
1. 在 GitHub 上创建 v1.0.0 Release
2. 填写发布说明
3. 启用不可变性

---

## 预期结果

| 项目 | 维护前 | 维护后 |
|------|--------|--------|
| Projects | ❌ 关闭 | ✅ 开启 |
| Discussions | ❌ 关闭 | ✅ 开启 |
| 分支保护 | ❌ 无 | ✅ 有 |
| 社交预览 | ❌ 无 | ✅ 有 |
| pre-commit.ci | ❌ 未安装 | ✅ 已安装 |
| Codecov | ❌ 未安装 | ✅ 已安装 |
| Dependabot | ✅ 已配置 | ✅ 已生效 |
| Release | ❌ 无 | ✅ v1.0.0 |
