# @omnimem/sdk (TypeScript)

Dependency-free TypeScript client for the [OmniMem](../README.md) hybrid memory
REST API. Works in Node 18+ (uses the global `fetch`). This mirrors the Python
`OmniMemSDK` surface so JS/TS agents can read and write the same long-term
memory store.

> Status: **preview**. The transport and auth model track the running OmniMem
> server; field names match `~/.hermes/plugins/omnimem/api_fastapi.py`.

## Install

```bash
npm install @omnimem/sdk
```

Start the OmniMem REST server (from the plugin root):

```bash
python -m omnimem.api_fastapi            # binds 127.0.0.1:8765, prints api_key
# or embed / run via your host, then point the client at it
```

## Quick start

```ts
import { OmniMemClient } from "@omnimem/sdk";

const mem = new OmniMemClient({
  baseUrl: "http://127.0.0.1:8765",
  apiKey: process.env.OMNIMEM_API_KEY,
  adminToken: process.env.OMNIMEM_ADMIN_TOKEN, // only for export/import
});

// Write
await mem.memorize({
  content: "User prefers PostgreSQL over MySQL.",
  memory_type: "preference",
  privacy: "personal",
});

// Read
const { results } = await mem.recall({ query: "which database?", mode: "hybrid" });
console.log(results?.[0]?.content);

// Reflect / govern / health
await mem.reflect({ query: "summarize the user's data-store opinions" });
await mem.govern({ action: "health" });
const health = await mem.health();
```

## API surface

| Method | Server endpoint | Notes |
|:---|:---|:---|
| `memorize(input)` | `POST /api/memorize` | `content`, `memory_type`, `confidence`, `privacy`, `scope` |
| `recall(input)` | `POST /api/recall` | `query`, `mode`, `max_tokens`, `type_filter`, `explain` |
| `reflect(input)` | `POST /api/reflect` | `query` |
| `govern(input)` | `POST /api/govern` | `action` + action-specific kwargs |
| `compact(input?)` | `POST /api/compact` | `budget` |
| `detail({memory_id})` | `POST /api/detail` | sets `action:"get"` |
| `detailList(input?)` | `POST /api/detail` | sets `action:"list"` |
| `detailEvents(input?)` | `POST /api/detail` | sets `action:"events"` |
| `exportMemories(input?)` | `POST /api/export` | admin token required |
| `importMemories(input)` | `POST /api/import` | admin token required |
| `health()` | `GET /api/health` | |
| `tools()` | `GET /api/tools` | |
| `metrics()` | `GET /metrics` | raw Prometheus text |

Unknown body keys pass straight through to the handler (`**kwargs`), so newer
server options work without an SDK upgrade.

## Auth

Every request carries `Authorization: Bearer <apiKey>` (a raw key is upgraded to
`Bearer <key>` automatically; pass a full `Basic ...` value to override).
`export` / `import` additionally send `X-Admin-Token`. The server defaults to
`127.0.0.1` (fail-closed) and rate-limits per minute.

## Errors

`OmniMemError` is the base; the client throws typed subclasses you can branch on:
`AuthError` (401), `ForbiddenError` (403), `NotFoundError` (404),
`RateLimitError` (429, exposes `retryAfterSeconds`), `OmniMemServerError` (5xx /
network), and `TimeoutError` (client-side `timeoutMs`). Soft errors returned
inside a 200 JSON body (`{status:"error", reason}` / `{error}`) also throw
`OmniMemError`.

## Development

```bash
npm install
npm run typecheck   # tsc --noEmit over src + test + examples
npm test            # typecheck, then node --test
npm run build       # emit dist/ (ESM + .d.ts)
```

## Layout

```
sdk-typescript/
  src/
    index.ts     barrel export
    client.ts    OmniMemClient (fetch, auth, timeout, error mapping)
    types.ts     request/response types
    errors.ts    typed error hierarchy
  test/          node:test suite (fetch injected, no live server needed)
  examples/      basic.ts
```
