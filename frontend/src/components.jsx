// Shared presentation primitives.
//
// Colour is reserved for clinical meaning. The accent is used for structure
// (links, focus rings, the brand mark); severity, validation status, grounding
// status and backend honesty get their own hues. Nothing here is decorative.

export function Pill({ tone = "neutral", children, title }) {
  return (
    <span className={`pill pill-${tone}`} title={title}>
      {children}
    </span>
  );
}

const SEVERITY_TONE = {
  PARSE_ERROR: "alert",
  URGENT_ALERT: "alert",
  PLAUSIBILITY_WARNING: "warn",
  CONFIG_MISSING: "info",
};

export function severityPill(severity) {
  return (
    <Pill tone={SEVERITY_TONE[severity] || "neutral"} title={`Severity: ${severity}`}>
      {severity}
    </Pill>
  );
}

const STATUS_TONE = {
  VALID: "ok",
  REQUIRES_REVIEW: "warn",
  REJECTED: "alert",
  FAILED: "alert",
  SUBMITTED: "ok",
  DUPLICATE: "info",
  UNCONFIGURED: "info",
  PARTIALLY_SUPPORTED: "warn",
  SUPPORTED: "ok",
  UNVERIFIED: "warn",
  CONFLICTING: "alert",
  PENDING_REVIEW: "warn",
  APPROVED: "ok",
  DISABLED: "neutral",
  AMBIGUOUS: "warn",
  ENCODE_DATA: "info",
  ASK_QUESTION: "info",
};

export function statusPill(status) {
  if (!status) return null;
  const tone = STATUS_TONE[status] || "neutral";
  return (
    <Pill tone={tone} title={`Status: ${status}`}>
      {status}
    </Pill>
  );
}

// A stub backend must never be invisible. This badge is rendered wherever a
// result came from an inference stage, so "not a real model" is always on
// screen next to the answer.
export function BackendBadge({ backend, modelName, title }) {
  if (!backend) return null;
  const stub = String(backend).toLowerCase() !== "real";
  return (
    <span
      className={`backend-badge ${stub ? "backend-stub" : "backend-real"}`}
      title={
        title ||
        `Inference backend: ${backend}. ${
          stub
            ? "This stage is running a deterministic fallback, not a trained model."
            : "This stage ran on the configured real model."
        }`
      }
    >
      {stub ? "STUB backend" : "real backend"}: {backend}
      {modelName ? ` · ${modelName}` : ""}
    </span>
  );
}

export function Empty({ children }) {
  return <div className="empty">{children}</div>;
}

export function Loading({ children = "Loading…" }) {
  return <div className="loading">{children}</div>;
}

export function Notice({ tone = "info", children }) {
  if (!children) return null;
  return <div className={`notice notice-${tone}`}>{children}</div>;
}

// Backend shapes vary: a value may be a scalar, a list, an object, null, or
// missing entirely. Rendering "null" or "[object Object]" as a clinical value
// would be a lie, so everything is normalised to readable text here.
export function fmt(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (Array.isArray(value)) {
    if (value.length === 0) return "none";
    return value.map((v) => fmt(v)).join(", ");
  }
  if (typeof value === "object") {
    const entries = Object.entries(value);
    if (entries.length === 0) return "empty";
    return entries.map(([k, v]) => `${k}: ${fmt(v)}`).join(" · ");
  }
  return String(value);
}

export function kv(label, value, { mono = false } = {}) {
  return (
    <div className="kv">
      <dt>{label}</dt>
      <dd className={mono ? "mono" : undefined}>{fmt(value)}</dd>
    </div>
  );
}

export function Code({ children }) {
  return <code className="code">{children}</code>;
}
