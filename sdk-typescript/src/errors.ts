/**
 * Error types surfaced by OmniMemClient.
 *
 * The OmniMem REST layer returns JSON bodies with an `error` or
 * `{status: "error", reason}` shape on failure, and uses HTTP status codes
 * for auth (401/403), rate limiting (429) and missing endpoints (404).
 * These are mapped onto a small typed hierarchy so callers can branch
 * programmatically instead of parsing strings.
 */

/** Base class for every error thrown by the SDK. */
export class OmniMemError extends Error {
  readonly status: number | undefined;
  readonly endpoint: string | undefined;
  readonly detail: unknown;

  constructor(
    message: string,
    options: { status?: number; endpoint?: string; detail?: unknown } = {},
  ) {
    super(message);
    this.name = "OmniMemError";
    this.status = options.status;
    this.endpoint = options.endpoint;
    this.detail = options.detail;
  }
}

/** Missing/invalid API key (HTTP 401). */
export class AuthError extends OmniMemError {
  constructor(message: string, options: { endpoint?: string; detail?: unknown } = {}) {
    super(message, { status: 401, ...options });
    this.name = "AuthError";
  }
}

/** Admin endpoint called without a valid admin token (HTTP 403). */
export class ForbiddenError extends OmniMemError {
  constructor(message: string, options: { endpoint?: string; detail?: unknown } = {}) {
    super(message, { status: 403, ...options });
    this.name = "ForbiddenError";
  }
}

/** Rate limiter tripped (HTTP 429). */
export class RateLimitError extends OmniMemError {
  readonly retryAfterSeconds: number | undefined;

  constructor(
    message: string,
    options: { endpoint?: string; detail?: unknown; retryAfterSeconds?: number } = {},
  ) {
    super(message, { status: 429, ...options });
    this.name = "RateLimitError";
    this.retryAfterSeconds = options.retryAfterSeconds;
  }
}

/** The requested tool endpoint does not exist (HTTP 404). */
export class NotFoundError extends OmniMemError {
  constructor(message: string, options: { endpoint?: string; detail?: unknown } = {}) {
    super(message, { status: 404, ...options });
    this.name = "NotFoundError";
  }
}

/** Server-side failure (HTTP 5xx) or a transport/network error. */
export class OmniMemServerError extends OmniMemError {
  constructor(message: string, options: { status?: number; endpoint?: string; detail?: unknown } = {}) {
    super(message, { status: options.status ?? 500, endpoint: options.endpoint, detail: options.detail });
    this.name = "OmniMemServerError";
  }
}

/** Raised when the request exceeds the configured client-side timeout. */
export class TimeoutError extends OmniMemError {
  constructor(message: string, options: { endpoint?: string } = {}) {
    super(message, { endpoint: options.endpoint });
    this.name = "TimeoutError";
  }
}
