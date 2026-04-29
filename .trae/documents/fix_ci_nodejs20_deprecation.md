# 修复 CI 中 Node.js 20 弃用警告的计划

## 问题背景

GitHub Actions 运行日志中出现以下弃用警告：

> Node.js 20 actions are deprecated. The following actions are running on Node.js 20 and may not work as expected: `actions/checkout@v4`, `actions/setup-python@v5`, `codecov/codecov-action@v4`. Actions will be forced to run with Node.js 24 by default starting June 2nd, 2026. Node.js 20 will be removed from the runner on September 16th, 2026.

受影响的工作流文件：[`.github/workflows/ci.yml`](file:///g:/omnimem/.github/workflows/ci.yml)

## 修复方案

### 步骤 1：升级 `actions/checkout` 到 v5
- 当前版本：`actions/checkout@v4`
- 目标版本：`actions/checkout@v5`
- 修改位置：ci.yml 第 18 行和第 48 行

### 步骤 2：升级 `actions/setup-python` 到 v6
- 当前版本：`actions/setup-python@v5`
- 目标版本：`actions/setup-python@v6`
- 修改位置：ci.yml 第 21 行和第 51 行

### 步骤 3：升级 `codecov/codecov-action` 到 v5
- 当前版本：`codecov/codecov-action@v4`
- 目标版本：`codecov/codecov-action@v5`
- 修改位置：ci.yml 第 65 行

## 验证方式

- 修改后提交 PR，观察 GitHub Actions 运行日志，确认不再出现 Node.js 20 弃用警告。
- 确保所有 job（lint、test 3.10/3.11/3.12）均正常运行。
