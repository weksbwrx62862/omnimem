# OmniMem Dashboard（改进项 #5）

零构建、零第三方依赖的本地可视化面板：检索 / 写入 / 列表 / 治理 / 系统健康与
Prometheus 指标。纯静态前端（`index.html` + `app.js` + `styles.css`）+ 一个仅用
标准库的反向代理启动器 `scripts/omni_dashboard.py`。

## 为什么是反向代理

面板与 OmniMem REST 后端同源，避免 CORS，且 **不改动** `api_fastapi.py` /
`rest_api.py` 任何代码 —— 纯新增。启动器把 `/api/*` 与 `/metrics` 透传到后端，
并原样转发 `Authorization` / `X-Admin-Token` 请求头。

## 用法

```bash
# 1) 先起 REST 后端（默认 127.0.0.1:8765）
python -m omnimem.api_fastapi

# 2) 起面板（默认 http://127.0.0.1:8790，代理到 8765）
python scripts/omni_dashboard.py
#   自定义： python scripts/omni_dashboard.py --port 9000 --api http://127.0.0.1:8765
```

浏览器打开面板后，在顶栏填入后端地址 + API key（导出/导入再填 admin token），
点“保存”（存 localStorage）。安全说明：默认只绑定 `127.0.0.1`（fail-closed）。

## 面板能力

| 标签 | 调用 | 说明 |
|:---|:---|:---|
| Recall | `POST /api/recall` | hybrid/rag/vector/bm25，可开 explain 查看评分链路 |
| Memorize | `POST /api/memorize` | 类型 / privacy / confidence |
| Detail | `POST /api/detail {action:list}` | 记忆列表 |
| Govern | `POST /api/govern` | 常用动作下拉 + 附加 JSON 参数 |
| System | `GET /api/health` `/api/tools` `/metrics` | 健康 / 工具 / Prometheus 指标 |

## 测试

`tests/test_dashboard.py` 用 mock 上游服务离线验证代理转发、鉴权头透传、
上游错误中继（500）、后端不可达（502）、静态托管与目录穿越防护（纯标准库，
不需要 omnimem 运行时）。
