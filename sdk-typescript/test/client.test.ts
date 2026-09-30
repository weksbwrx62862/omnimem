import { test } from "node:test";
import assert from "node:assert/strict";

import { OmniMemClient } from "../src/index.ts";
import { AuthError, OmniMemError, OmniMemServerError, RateLimitError } from "../src/index.ts";
import type { FetchLike } from "../src/index.ts";

interface Recorded {
  url: string;
  init: RequestInit;
}

function mockFetch(
  responder: (url: string, init: RequestInit) => Response | Promise<Response>,
  sink: Recorded[],
): FetchLike {
  return async (url: string, init: RequestInit = {}) => {
    sink.push({ url, init });
    return responder(url, init);
  };
}

function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

function headersOf(record: Recorded | undefined): Headers {
  const raw = record?.init.headers as Record<string, string> | Headers | undefined;
  return raw instanceof Headers ? raw : new Headers(raw ?? {});
}

test("memorize posts to /api/memorize with bearer auth and body", async () => {
  const sink: Recorded[] = [];
  const client = new OmniMemClient({
    baseUrl: "http://example.test/",
    apiKey: "secret-key",
    fetch: mockFetch(() => jsonResponse({ status: "ok", memory_id: "m1" }), sink),
  });

  const res = await client.memorize({ content: "user likes PostgreSQL", memory_type: "preference" });

  assert.equal(sink[0]?.url, "http://example.test/api/memorize");
  assert.equal(sink[0]?.init.method, "POST");
  assert.equal(headersOf(sink[0]).get("Authorization"), "Bearer secret-key");
  assert.equal(headersOf(sink[0]).get("Content-Type"), "application/json");
  assert.deepEqual(JSON.parse(String(sink[0]?.init.body)), {
    content: "user likes PostgreSQL",
    memory_type: "preference",
  });
  assert.equal(res.memory_id, "m1");
});

test("recall returns parsed results and defaults base url", async () => {
  const sink: Recorded[] = [];
  const client = new OmniMemClient({
    fetch: mockFetch(() => jsonResponse({ results: [{ id: "a", content: "x", score: 0.9 }], count: 1 }), sink),
  });
  const out = await client.recall({ query: "db", mode: "hybrid", explain: true });
  assert.equal(sink[0]?.url, "http://127.0.0.1:8765/api/recall");
  assert.equal(out.count, 1);
  assert.equal(out.results?.[0]?.id, "a");
});

test("detail() injects action=get and preserves memory_id", async () => {
  const sink: Recorded[] = [];
  const client = new OmniMemClient({ fetch: mockFetch(() => jsonResponse({ status: "ok" }), sink) });
  await client.detail({ memory_id: "m42" });
  assert.deepEqual(JSON.parse(String(sink[0]?.init.body)), { action: "get", memory_id: "m42" });
});

test("exportMemories sends X-Admin-Token when provided", async () => {
  const sink: Recorded[] = [];
  const client = new OmniMemClient({
    apiKey: "k",
    adminToken: "admin-1",
    fetch: mockFetch(() => jsonResponse({ status: "ok", path: "/tmp/e.json" }), sink),
  });
  await client.exportMemories({ output_path: "/tmp/e.json" });
  assert.equal(headersOf(sink[0]).get("X-Admin-Token"), "admin-1");
});

test("401 maps to AuthError", async () => {
  const client = new OmniMemClient({
    apiKey: "bad",
    fetch: mockFetch(() => jsonResponse({ error: "invalid api key" }, 401), []),
  });
  await assert.rejects(() => client.recall({ query: "x" }), AuthError);
});

test("429 maps to RateLimitError with retryAfterSeconds", async () => {
  const client = new OmniMemClient({
    fetch: mockFetch(() => jsonResponse({ error: "slow down" }, 429, { "Retry-After": "7" }), []),
  });
  await assert.rejects(
    () => client.memorize({ content: "hi" }),
    (err: unknown) => {
      assert.ok(err instanceof RateLimitError);
      assert.equal(err.retryAfterSeconds, 7);
      return true;
    },
  );
});

test("soft error inside a 200 body throws OmniMemError", async () => {
  const client = new OmniMemClient({
    fetch: mockFetch(() => jsonResponse({ status: "error", reason: "empty content" }, 200), []),
  });
  await assert.rejects(
    () => client.memorize({ content: "" }),
    (err: unknown) => {
      assert.ok(err instanceof OmniMemError);
      assert.equal(err.message, "empty content");
      return true;
    },
  );
});

test("network rejection maps to OmniMemServerError", async () => {
  const client = new OmniMemClient({
    fetch: () => Promise.reject(new Error("ECONNREFUSED")),
  });
  await assert.rejects(() => client.recall({ query: "x" }), OmniMemServerError);
});

test("health uses GET /api/health", async () => {
  const sink: Recorded[] = [];
  const client = new OmniMemClient({
    fetch: mockFetch(() => jsonResponse({ status: "healthy", store_count: 3 }), sink),
  });
  const h = await client.health();
  assert.equal(sink[0]?.init.method, "GET");
  assert.equal(sink[0]?.url, "http://127.0.0.1:8765/api/health");
  assert.equal(h.status, "healthy");
});
