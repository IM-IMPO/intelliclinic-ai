// API client for the IntelliClinic AI backend.
//
// Every request goes through `request()` so that the bearer token, the base
// URL and the error shape are handled in exactly one place.
//
// Route paths and field names were verified against the live OpenAPI document
// of the running backend. Two things this client deliberately does NOT do:
//
//   1. It does not treat an "unavailable" integration as an empty result.
//      GET /patients returns { results: [], unavailable: true, detail } when
//      OpenMRS is not configured. That is an explicit unavailable state and is
//      passed through so the UI can say so.
//
//   2. It does not guess a health endpoint. The backend exposes liveness at
//      /health and readiness at /api/v1/system/ready. There is no
//      /api/v1/health.

const TOKEN_KEY = "ic_token";
const USER_KEY = "ic_user";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

export function getStoredUser() {
  try {
    return JSON.parse(localStorage.getItem(USER_KEY) || "null");
  } catch {
    return null;
  }
}

export function setSession(token, user) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

// API base is origin + /api/v1. In production the FastAPI app serves the
// built SPA from the same origin; in dev, Vite proxies /api to the backend
// (see vite.config.js).
export const API_ROOT = (
  import.meta.env.VITE_API_BASE_URL || window.location.origin
).replace(/\/api\/v1\/?$/, "");

export const API_BASE = `${API_ROOT}/api/v1`;

async function request(path, { method = "GET", body, signal, timeoutMs } = {}) {
  const headers = {};
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";

  // Algorithms 3 and 5 can be slow on CPU: validation is CPU-only and
  // answering runs a local model, which legitimately takes tens of seconds.
  // A caller may raise the ceiling, but the caller's abort signal always wins.
  const ctrl = new AbortController();
  const timer =
    timeoutMs > 0
      ? setTimeout(() => ctrl.abort(new Error("timeout")), timeoutMs)
      : null;
  const onAbort = () => ctrl.abort(new Error("aborted"));
  signal?.addEventListener("abort", onAbort, { once: true });

  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: ctrl.signal,
    });
  } catch (err) {
    if (err.message === "aborted") throw err;
    if (err.message === "timeout") {
      throw new ApiError(
        "The request timed out. A model may still be loading - try again.",
        0,
        "timeout",
      );
    }
    throw new ApiError(
      "Cannot reach the backend. Is the server running on the configured URL?",
      0,
      String(err),
    );
  } finally {
    if (timer) clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }

  const text = await res.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }

  if (!res.ok) {
    const detail = data?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail) && detail[0]?.msg
          ? detail[0].msg
          : `Request failed with status ${res.status}`;
    throw new ApiError(message, res.status, detail);
  }
  return data;
}

// Algorithm 5 answers from a local model; validation is CPU-only. Generous
// ceiling, still abortable.
const SLOW = { timeoutMs: 300_000 };

export const api = {
  // -- auth ---------------------------------------------------------------
  login: (username, password) =>
    request("/auth/login", { method: "POST", body: { username, password } }),
  logout: () => request("/auth/logout", { method: "POST" }),
  me: () => request("/auth/me"),

  // -- pipeline -----------------------------------------------------------
  // POST /chat { message, session_id?, patient_id?, encounter_id?, dry_run }
  //   -> { request_id, intent, route, stages, extraction, validation,
  //        retrieval, answer, requires_confirmation, saved, created_at }
  chat: (payload, signal) =>
    request("/chat", { method: "POST", body: payload, signal, ...SLOW }),

  // -- individual algorithms ---------------------------------------------
  intent: (message, signal) =>
    request("/intent", { method: "POST", body: { message }, signal }),
  extract: (text, encounterId, signal) =>
    request("/extract", {
      method: "POST",
      body: { text, encounter_id: encounterId },
      signal,
    }),
  validate: (payload, signal) =>
    request("/validate", { method: "POST", body: payload, signal, ...SLOW }),
  retrieve: (query, topK, signal) =>
    request("/retrieve", {
      method: "POST",
      body: { query, top_k: topK },
      signal,
    }),
  answer: (payload, signal) =>
    request("/answer", { method: "POST", body: payload, signal, ...SLOW }),

  // -- human approval gate -----------------------------------------------
  // GET /review-queue -> { awaiting_review: [{ request_id, created_at,
  //   created_by, observation_count, validation_status, openmrs_configured }],
  //   note }
  // GET /review-queue/{id} -> the stored record incl. payload + observations.
  // POST /review-queue/{id}/approve takes
  //   { request_id, approved_by, corrections, payload?, idempotency_key }
  reviewQueue: () => request("/review-queue"),
  reviewItem: (requestId) => request(`/review-queue/${requestId}`),
  approve: (requestId, payload) =>
    request(`/review-queue/${requestId}/approve`, {
      method: "POST",
      body: payload,
    }),

  // -- knowledge base -----------------------------------------------------
  // POST /knowledge-base/documents requires document_id, title,
  // source_version and text; optional effective_date, review_status, topic.
  // GET  /knowledge-base/documents -> { documents, collection,
  //                                    embedding_model, indexing_status }
  documents: () => request("/knowledge-base/documents"),
  indexDocument: (payload) =>
    request("/knowledge-base/documents", { method: "POST", body: payload }),
  setDocumentStatus: (documentId, reviewStatus) =>
    request(`/knowledge-base/documents/${documentId}/status`, {
      method: "POST",
      body: { review_status: reviewStatus },
    }),
  deleteDocument: (documentId) =>
    request(`/knowledge-base/documents/${documentId}`, { method: "DELETE" }),

  // -- integrations -------------------------------------------------------
  // Returns { results: [], unavailable: true, detail } when OpenMRS is not
  // configured. Callers must not read results.length on its own.
  patients: (q = "", limit = 20) =>
    request(`/patients?q=${encodeURIComponent(q)}&limit=${limit}`),
  patient: (uuid) => request(`/patients/${uuid}`),

  systemStatus: () => request("/system/status"),
  // Readiness, not liveness: reports which dependencies are actually usable.
  systemReady: () => request("/system/ready"),
  auditLogs: () => request("/audit-logs"),
  users: () => request("/users"),
  setUserRole: (userId, role) =>
    request(`/users/${userId}/role`, { method: "PATCH", body: { role } }),
  deactivateUser: (userId) =>
    request(`/users/${userId}/deactivate`, { method: "POST" }),
};
