import type {
  ApiResult,
  ClientOptions,
  CompactInput,
  DetailEventsInput,
  DetailGetInput,
  DetailListInput,
  ExportInput,
  FetchLike,
  GovernInput,
  HealthResult,
  ImportInput,
  MemorizeInput,
  RecallInput,
  RecallResult,
  ReflectInput,
  ToolName,
} from "./types.ts";
import {
  AuthError,
  ForbiddenError,
  NotFoundError,
  OmniMemError,
  OmniMemServerError,
  RateLimitError,
  TimeoutError,
} from "./errors.ts";

const DEFAULT_BASE_URL = "http://127.0.0.1:8765";
const DEFAULT_TIMEOUT_MS = 30_000;

interface RequestOptions {
  method: "GET" | "POST";
  admin?: boolean;
  body?: unknown;
  /** Accept a non-JSON text response (used by /metrics). */
  textResponse?: boolean;
}

/**
 * Thin, dependency-free client over the OmniMem REST API.
 *
 * Every SDK method maps 1:1 to a `POST /api/{tool}` dispatch on the server, so
 * the returned shapes are exactly the dicts OmniMemSDK produces. Requires a
 * global `fetch` (Node 18+).
 */
export class OmniMemClient {
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly adminToken: string | undefined;
  private readonly timeoutMs: number;
  private readonly fetchImpl: FetchLike;
  private readonly defaultHeaders: Record<string, string>;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.adminToken = options.adminToken;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const injected = options.fetch;
    this.fetchImpl =
      injected ??
      ((globalThis.fetch as FetchLike | undefined)?.bind(globalThis) as FetchLike | undefined) ??
      (() => {
        throw new OmniMemError("No fetch implementation available; Node 18+ or pass options.fetch");
      });
    this.defaultHeaders = options.defaultHeaders ?? {};
  }

  /** Store a memory. Maps to SDK.memorize. */
  memorize(input: MemorizeInput): Promise<ApiResult> {
    return this.#call<ApiResult>("memorize", "POST", input);
  }

  /** Retrieve relevant memories. Maps to SDK.recall. */
  recall(input: RecallInput): Promise<RecallResult> {
    return this.#call<RecallResult>("recall", "POST", input);
  }

  /** Deep reflection over a query. Maps to SDK.reflect. */
  reflect(input: ReflectInput): Promise<ApiResult> {
    return this.#call<ApiResult>("reflect", "POST", input);
  }

  /** Governance action (export/reencrypt/reindex/...). Maps to SDK.govern. */
  govern(input: GovernInput): Promise<ApiResult> {
    return this.#call<ApiResult>("govern", "POST", input);
  }

  /** Pre-compaction preparation. Maps to SDK.compact. */
  compact(input: CompactInput = {}): Promise<ApiResult> {
    return this.#call<ApiResult>("compact", "POST", input);
  }

  /** Fetch one memory's detail. Maps to SDK.detail. */
  detail(input: DetailGetInput): Promise<ApiResult> {
    return this.#call<ApiResult>("detail", "POST", { action: "get", ...input });
  }

  /** List memories. Maps to SDK.detail_list. */
  detailList(input: DetailListInput = {}): Promise<ApiResult> {
    return this.#call<ApiResult>("detail", "POST", { action: "list", ...input });
  }

  /** Turn-range event log. Maps to SDK.detail_events. */
  detailEvents(input: DetailEventsInput = {}): Promise<ApiResult> {
    return this.#call<ApiResult>("detail", "POST", { action: "events", ...input });
  }

  /** Admin-only: export memories. Maps to SDK.export_memories. */
  exportMemories(input: ExportInput = {}): Promise<ApiResult> {
    return this.#call<ApiResult>("export", "POST", input, { admin: true });
  }

  /** Admin-only: import memories. Maps to SDK.import_memories. */
  importMemories(input: ImportInput): Promise<ApiResult> {
    return this.#call<ApiResult>("import", "POST", input, { admin: true });
  }

  /** Health probe. Maps to GET /api/health. */
  health(): Promise<HealthResult> {
    return this.#raw<HealthResult>("health", { method: "GET" });
  }

  /** List available tool endpoints. Maps to GET /api/tools. */
  tools(): Promise<{ tools: ToolName[] }> {
    return this.#raw<{ tools: ToolName[] }>("tools", { method: "GET" });
  }

  /** Prometheus metrics text. Maps to GET /metrics. */
  metrics(): Promise<string> {
    return this.#metrics();
  }

  async #call<T>(endpoint: ToolName, method: RequestOptions["method"], body: unknown, opts: { admin?: boolean } = {}): Promise<T> {
    return this.#raw<T>(endpoint, { method, body, admin: opts.admin });
  }

  async #metrics(): Promise<string> {
    const url = `${this.baseUrl}/metrics`;
    const res = await this.#fetch(url, { method: "GET", headers: this.#headers({}) });
    const text = await res.text();
    if (!res.ok) {
      this.#throwHttp(res, text, "/metrics");
    }
    return text;
  }

  async #raw<T>(segment: string, req: RequestOptions): Promise<T> {
    const url = `${this.baseUrl}/api/${segment}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    const headers = this.#headers(req.admin ? { admin: true } : {});
    const init: RequestInit = { method: req.method, headers, signal: controller.signal as AbortSignal };
    if (req.method === "POST" && req.body !== undefined) {
      init.body = JSON.stringify(req.body);
      headers["Content-Type"] = "application/json";
    }
    let res: Response;
    try {
      res = await this.#fetch(url, init);
    } catch (err) {
      clearTimeout(timer);
      if ((err as { name?: string })?.name === "AbortError") {
        throw new TimeoutError(`Request to /api/${segment} timed out after ${this.timeoutMs}ms`, {
          endpoint: segment,
        });
      }
      throw new OmniMemServerError(`Network error calling /api/${segment}: ${String(err)}`, {
        endpoint: segment,
        detail: err,
      });
    }
    clearTimeout(timer);

    const rawText = await res.text();
    let parsed: unknown = undefined;
    if (rawText.length > 0) {
      try {
        parsed = JSON.parse(rawText) as unknown;
      } catch {
        parsed = rawText;
      }
    }

    if (!res.ok) {
      this.#throwHttp(res, parsed, segment);
    }

    // The server also signals soft errors inside a 200 JSON body.
    if (parsed && typeof parsed === "object") {
      const obj = parsed as Record<string, unknown>;
      if (typeof obj["error"] === "string") {
        throw new OmniMemError(obj["error"], { status: res.status, endpoint: segment, detail: obj });
      }
      if (obj["status"] === "error" && typeof obj["reason"] === "string") {
        throw new OmniMemError(obj["reason"], { status: res.status, endpoint: segment, detail: obj });
      }
    }
    return parsed as T;
  }

  #fetch(url: string, init: RequestInit): Promise<Response> {
    // RequestInit from lib.dom carries an AbortSignal typed as AbortSignal;
    // undici's fetch accepts the same shape. Cast through unknown to stay
    // compatible across @types/node versions.
    return this.fetchImpl(url, init as unknown as RequestInit);
  }

  #headers(extra: { admin?: boolean }): Record<string, string> {
    const h: Record<string, string> = { Accept: "application/json", ...this.defaultHeaders };
    if (this.apiKey) {
      h["Authorization"] = this.#authValue(this.apiKey);
    }
    if (extra?.admin && this.adminToken) {
      h["X-Admin-Token"] = this.adminToken;
    }
    return h;
  }

  /** Allow both raw keys and `Bearer <key>` style values. */
  #authValue(key: string): string {
    return /^(Bearer|Basic)\s/i.test(key) ? key : `Bearer ${key}`;
  }

  #throwHttp(res: Response, parsed: unknown, endpoint: string): never {
    const message = extractMessage(parsed) ?? `${res.status} ${res.statusText}`;
    const opts = { endpoint, detail: parsed };
    switch (res.status) {
      case 401:
        throw new AuthError(message, opts);
      case 403:
        throw new ForbiddenError(message, opts);
      case 404:
        throw new NotFoundError(message, opts);
      case 429: {
        const retry = Number(res.headers.get("retry-after"));
        throw new RateLimitError(message, {
          ...opts,
          retryAfterSeconds: Number.isFinite(retry) ? retry : undefined,
        });
      }
      default:
        if (res.status >= 500) {
          throw new OmniMemServerError(message, { status: res.status, ...opts });
        }
        throw new OmniMemError(message, { status: res.status, ...opts });
    }
  }
}

function extractMessage(parsed: unknown): string | undefined {
  if (typeof parsed === "string" && parsed.length > 0) return parsed;
  if (parsed && typeof parsed === "object") {
    const obj = parsed as Record<string, unknown>;
    if (typeof obj["error"] === "string") return obj["error"];
    if (typeof obj["reason"] === "string") return obj["reason"];
    if (typeof obj["message"] === "string") return obj["message"];
  }
  return undefined;
}
