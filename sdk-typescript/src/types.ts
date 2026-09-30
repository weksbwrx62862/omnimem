/**
 * Request/response types for the OmniMem REST API.
 *
 * These mirror the server contract in ~/.hermes/plugins/omnimem:
 *   POST /api/{tool}   -> dispatches to OmniMemSDK.<tool>(**body)
 *   GET  /api/health   -> OmniMemSDK.health_check()
 *   GET  /api/tools    -> { tools: string[] }
 *   GET  /metrics      -> Prometheus text exposition
 * Tool set (_TOOLS):
 *   memorize recall reflect govern compact detail export import health
 * Auth: `Authorization` header on every call; `X-Admin-Token` additionally
 * required on the admin-only endpoints export/import.
 *
 * Bodies are intentionally open-ended (`[key: string]: unknown`), because the
 * server forwards unknown keys straight through to the handler as **kwargs.
 * Only the fields the handlers explicitly read are typed here.
 */

/** Known tool endpoints exposed under /api/. */
export type ToolName =
  | "memorize"
  | "recall"
  | "reflect"
  | "govern"
  | "compact"
  | "detail"
  | "export"
  | "import"
  | "health";

/** Privacy tiers (matches config `default_privacy` choices). */
export type Privacy = "public" | "team" | "personal" | "secret";

/** Retrieval modes (matches config `retrieval_mode` choices). */
export type RecallMode = "rag" | "hybrid" | "vector" | "bm25";

/**
 * Memory type. `"fact"` is the server default; other values are opaque to the
 * transport layer and interpreted by the extraction pipeline, so any string is
 * accepted.
 */
export type MemoryType = "fact" | "preference" | "event" | "entity" | "procedure" | (string & {});

/** Injectable fetch implementation (used to unit-test without a network). */
export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

/** Options for constructing an {@link OmniMemClient}. */
export interface ClientOptions {
  /** Base URL of the OmniMem REST server. Defaults to `http://127.0.0.1:8765`. */
  baseUrl?: string;
  /** API key sent as the `Authorization` header. */
  apiKey?: string;
  /** Admin token sent as `X-Admin-Token` on export/import. */
  adminToken?: string;
  /** Client-side timeout per request, in milliseconds. Defaults to 30000. */
  timeoutMs?: number;
  /** Override the fetch implementation (testing/DI). Defaults to global fetch. */
  fetch?: FetchLike;
  /** Extra headers merged into every request. */
  defaultHeaders?: Record<string, string>;
}

export interface MemorizeInput {
  content: string;
  memory_type?: MemoryType;
  confidence?: number;
  privacy?: Privacy;
  scope?: string;
  [key: string]: unknown;
}

export interface RecallInput {
  query: string;
  mode?: RecallMode;
  max_tokens?: number;
  type_filter?: string;
  /** Emit the per-channel scoring breakdown under `_explain`. */
  explain?: boolean;
  [key: string]: unknown;
}

export interface ReflectInput {
  query: string;
  [key: string]: unknown;
}

export interface GovernInput {
  action: string;
  [key: string]: unknown;
}

export interface CompactInput {
  budget?: number;
  [key: string]: unknown;
}

export interface DetailGetInput {
  memory_id: string;
  [key: string]: unknown;
}

export interface DetailListInput {
  [key: string]: unknown;
}

export interface DetailEventsInput {
  from_turn?: number;
  to_turn?: number;
  [key: string]: unknown;
}

export interface ExportInput {
  output_path?: string;
  format?: string;
  [key: string]: unknown;
}

export interface ImportInput {
  input_path: string;
  [key: string]: unknown;
}

/** A single recalled memory item. Shape is server-defined and open-ended. */
export interface RecalledMemory {
  id?: string;
  content?: string;
  score?: number;
  memory_type?: string;
  privacy?: string;
  [key: string]: unknown;
}

/** Response of `recall`. Core fields typed, rest passed through. */
export interface RecallResult {
  status?: string;
  results?: RecalledMemory[];
  memories?: RecalledMemory[];
  count?: number;
  reason?: string;
  _explain?: unknown;
  [key: string]: unknown;
}

/** Response of `health_check`. */
export interface HealthResult {
  status: "healthy" | "degraded" | "unhealthy" | (string & {});
  session_id?: string;
  data_dir?: string;
  store_accessible?: boolean;
  store_count?: number;
  chromadb_accessible?: boolean;
  vector_count?: number;
  circuit_breaker_state?: string;
  [key: string]: unknown;
}

/** Generic JSON result returned by most mutating endpoints. */
export interface ApiResult {
  status?: string;
  error?: string;
  reason?: string;
  [key: string]: unknown;
}
