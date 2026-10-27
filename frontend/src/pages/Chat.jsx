import { useState } from "react";
import { api } from "../api.js";
import { useAuth } from "../auth.jsx";
import {
  BackendBadge,
  Empty,
  Notice,
  Pill,
  fmt,
  severityPill,
  statusPill,
} from "../components.jsx";

// Realistic clinical prompts. The impossible ones (210 C) exist deliberately:
// a validator that accepts them is the failure mode this screen must expose.
const EXAMPLES = [
  "Patient Juan has a fever of 38.5 C and BP is 120/80, complains of cough.",
  "Patient Maria temperature 41 C, pulse 210, BP 70/40.",
  "SpO2 85 percent on room air, respiratory rate 32.",
  "Patient has no fever and no cough today.",
  "Diagnosed with community acquired pneumonia, started on amoxicillin 500 mg three times daily.",
  "Temperature is 210 C.",
  "History of hypertension, no chest pain.",
  "What is the recommended dose of amoxicillin for adults?",
];

const ASSERTION_TONE = {
  PRESENT: "ok",
  NEGATED: "info",
  UNCERTAIN: "warn",
  HISTORICAL: "warn",
};

const STAGE_TONE = {
  ok: "ok",
  warning: "warn",
  error: "alert",
  skipped: "neutral",
};

const GROUNDING_TONE = {
  SUPPORTED: "ok",
  PARTIALLY_SUPPORTED: "warn",
  UNVERIFIED: "warn",
  CONFLICTING: "alert",
};

const EVIDENCE_TONE = {
  SUFFICIENT: "ok",
  INSUFFICIENT: "warn",
  CONFLICTING: "alert",
};

function IntentCard({ intent }) {
  if (!intent) return null;
  const alts = Object.entries(intent.alternatives || {});
  return (
    <div className="card">
      <header>
        <div>
          <h2>Intent</h2>
          <p className="sub">Algorithm 1 · intent classification</p>
        </div>
        <BackendBadge backend={intent.backend} modelName={intent.model_name} />
      </header>

      <div className="pill-row" style={{ marginBottom: 10 }}>
        <Pill tone="info" title={`Threshold ${intent.threshold}`}>
          {intent.intent}
        </Pill>
        <Pill tone={intent.confidence >= intent.threshold ? "ok" : "warn"}>
          {(intent.confidence * 100).toFixed(1)}% confidence
        </Pill>
        <Pill tone="neutral">threshold {intent.threshold}</Pill>
        {intent.clarification_prompt && <Pill tone="warn">clarification requested</Pill>}
      </div>

      {intent.clarification_prompt && (
        <Notice tone="warn">{intent.clarification_prompt}</Notice>
      )}

      {alts.length > 0 && (
        <div className="kv-stack">
          <div className="kv">
            <dt>Alternatives</dt>
            <dd>
              {alts.map(([k, v]) => `${k} ${(v * 100).toFixed(1)}%`).join(" · ")}
            </dd>
          </div>
          <div className="kv">
            <dt>Model</dt>
            <dd className="mono">
              {intent.model_name} · {intent.model_version}
            </dd>
          </div>
        </div>
      )}
    </div>
  );
}

function StagesCard({ stages }) {
  if (!stages?.length) return null;
  return (
    <div className="card">
      <header>
        <div>
          <h2>Pipeline stages</h2>
          <p className="sub">Every stage reports whether it ran a real model or a fallback</p>
        </div>
      </header>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>Stage</th>
              <th>Algorithm</th>
              <th>Backend</th>
              <th>Status</th>
              <th>Summary</th>
              <th className="right">ms</th>
            </tr>
          </thead>
          <tbody>
            {stages.map((s, i) => (
              <tr key={`${s.stage}-${i}`}>
                <td className="mono nowrap">{s.stage}</td>
                <td className="mono">{s.algorithm}</td>
                <td>
                  <BackendBadge backend={s.backend} />
                </td>
                <td>
                  <Pill tone={STAGE_TONE[s.status] || "neutral"}>{s.status}</Pill>
                </td>
                <td>{fmt(s.summary)}</td>
                <td className="num right">{s.elapsed_ms}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ExtractionCard({ extraction }) {
  if (!extraction) return null;
  const entities = extraction.entities || [];
  return (
    <div className="card">
      <header>
        <div>
          <h2>Extracted entities</h2>
          <p className="sub">
            Algorithm 2 · {entities.length} entity{entities.length === 1 ? "" : "s"}
          </p>
        </div>
        <BackendBadge
          backend={extraction.backend}
          modelName={extraction.model_name}
        />
      </header>

      {entities.length === 0 ? (
        <Empty>No clinical entities were found in this text.</Empty>
      ) : (
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>Entity</th>
                <th>Type</th>
                <th>Assertion</th>
                <th className="right">Confidence</th>
                <th>Value</th>
                <th>Review</th>
              </tr>
            </thead>
            <tbody>
              {entities.map((e, i) => (
                <tr key={`${e.start}-${i}`}>
                  <td className="mono">“{e.text}”</td>
                    <td className="mono nowrap">{e.type}</td>
                  <td>
                    <Pill tone={ASSERTION_TONE[e.assertion] || "neutral"}>
                      {e.assertion}
                    </Pill>
                  </td>
                  <td className="num right">{(e.confidence * 100).toFixed(1)}%</td>
                  <td>
                    {fmt(e.value_candidate)}
                    {e.unit_candidate ? ` ${e.unit_candidate}` : ""}
                  </td>
                  <td>
                    {e.requires_review ? (
                      <Notice tone="warn">{fmt(e.review_reason)}</Notice>
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function ValidationCard({ validation }) {
  if (!validation) return null;
  const obs = validation.observations || [];
  const issues = validation.issues || [];
  const missing = validation.configuration_missing || [];

  return (
    <div className="card">
      <header>
        <div>
          <h2>Clinical validation</h2>
          <p className="sub">
            Algorithm 3 · {obs.length} observation{obs.length === 1 ? "" : "s"},{" "}
            {issues.length} issue{issues.length === 1 ? "" : "s"}
          </p>
        </div>
        <div className="pill-row">
          {statusPill(validation.status)}
          {validation.rejected?.length > 0 && (
            <Pill tone="alert">{validation.rejected.length} rejected</Pill>
          )}
        </div>
      </header>

      {obs.length === 0 && issues.length === 0 && (
        <Empty>Nothing to validate in this message.</Empty>
      )}

      {obs.length > 0 && (
        <div className="section">
          <h3>Observations</h3>
          <div className="tw">
            <table>
              <thead>
                <tr>
                  <th>Concept</th>
                  <th>Value</th>
                  <th>Unit</th>
                  <th>Status</th>
                  <th>Source text</th>
                  <th>Concept UUID</th>
                </tr>
              </thead>
              <tbody>
                {obs.map((o, i) => (
                  <tr key={`${o.concept_name}-${i}`}>
                    <td>{o.concept_name}</td>
                    <td>{fmt(o.value)}</td>
                    <td className="mono">{fmt(o.unit)}</td>
                    <td>{statusPill(o.status)}</td>
                    <td className="mono">{fmt(o.source_entity_text)}</td>
                    <td className="mono small">
                      {o.concept_uuid ? o.concept_uuid : "not configured"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {issues.length > 0 && (
        <div className="section">
          <h3>Issues</h3>
          <div className="stack">
            {issues.map((iss, i) => (
              <div
                key={`${iss.field}-${i}`}
                className="notice notice-warn"
                style={{ display: "grid", gap: 2 }}
              >
                <div className="flex spread">
                  <strong className="mono">{iss.field}</strong>
                  {severityPill(iss.severity)}
                </div>
                <div>{iss.message}</div>
                {iss.raw_value !== null && iss.raw_value !== undefined && (
                  <div className="mono small muted">raw value: {iss.raw_value}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {missing.length > 0 && (
        <div className="section">
          <Notice tone="warn">
            <strong>Submission is blocked.</strong> These concepts have no OpenMRS
            UUID configured: <span className="mono">{missing.join(", ")}</span>. A
            null UUID means the site must configure it — it is not a data error and
            it does not make the reading itself invalid.
          </Notice>
        </div>
      )}

      <div className="section">
        <div className="kv">
          <dt>can_submit</dt>
          <dd>
            {validation.can_submit ? (
              <Pill tone="ok">true — values and configuration both clear</Pill>
            ) : (
              <Pill tone="alert">false — nothing will be written</Pill>
            )}
          </dd>
          <dt>request_id</dt>
          <dd className="mono small">{validation.request_id}</dd>
        </div>
      </div>
    </div>
  );
}

function RetrievalCard({ retrieval }) {
  if (!retrieval) return null;
  const docs = retrieval.retrieved_documents || [];
  return (
    <div className="card">
      <header>
        <div>
          <h2>Retrieved documents</h2>
          <p className="sub">
            Algorithm 4 · {docs.length} chunk{docs.length === 1 ? "" : "s"} · query: “
            {retrieval.query}”
          </p>
        </div>
        <div className="pill-row">
          <Pill tone={EVIDENCE_TONE[retrieval.evidence_status] || "neutral"}>
            {retrieval.evidence_status}
          </Pill>
          <Pill tone="neutral">{retrieval.embedding_model}</Pill>
        </div>
      </header>

      {docs.length === 0 ? (
        <Empty>No approved documents matched this question.</Empty>
      ) : (
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>Title</th>
                <th>Section</th>
                <th>Version</th>
                <th>Review</th>
                <th className="right">Similarity</th>
              </tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.chunk_id}>
                  <td>{d.title}</td>
                  <td className="mono">{fmt(d.section)}</td>
                  <td className="mono">{d.source_version}</td>
                  <td>
                    <Pill tone={d.approved ? "ok" : "warn"}>{d.review_status}</Pill>
                  </td>
                  <td className="num right">
                    {(d.similarity_score * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {(retrieval.notes || []).length > 0 && (
        <div className="section">
          <ul className="small muted" style={{ margin: 0, paddingLeft: 18 }}>
            {retrieval.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function AnswerCard({ answer }) {
  if (!answer) return null;
  return (
    <div className="card">
      <header>
        <div>
          <h2>Answer</h2>
          <p className="sub">Algorithm 5 · grounded generation</p>
        </div>
        <div className="pill-row">
          <Pill tone={GROUNDING_TONE[answer.grounding_status] || "neutral"}>
            {answer.grounding_status}
          </Pill>
          <BackendBadge backend={answer.backend} modelName={answer.model_name} />
        </div>
      </header>

      <p style={{ fontSize: 15, lineHeight: 1.65, whiteSpace: "pre-wrap" }}>
        {answer.answer}
      </p>

      {answer.sources?.length > 0 && (
        <div className="section">
          <h3>Sources</h3>
          <div className="stack">
            {answer.sources.map((s) => (
              <div key={s.chunk_id} className="kv">
                <dt>Citation</dt>
                <dd>
                  {s.title}
                  {s.section ? ` — ${s.section}` : ""} ({s.source_version})
                  {s.similarity_score != null && (
                    <span className="muted"> · similarity{" "}
                      {(s.similarity_score * 100).toFixed(1)}%</span>
                  )}
                  <br />
                  <span className="mono small muted">
                    {s.document_id} / {s.chunk_id}
                  </span>
                </dd>
              </div>
            ))}
          </div>
        </div>
      )}

      {answer.conflicts?.length > 0 && (
        <div className="section">
          <Notice tone="alert">
            <strong>Conflicting evidence.</strong>{" "}
            {answer.conflicts.join(" ")}
          </Notice>
        </div>
      )}

      {answer.limitations?.length > 0 && (
        <div className="section">
          <h3>Limitations</h3>
          <ul className="small muted" style={{ margin: 0, paddingLeft: 18 }}>
            {answer.limitations.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function Chat() {
  const { user } = useAuth();
  const [message, setMessage] = useState("");
  const [patientId, setPatientId] = useState("");
  const [encounterId, setEncounterId] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function send(e) {
    e?.preventDefault();
    if (busy || !message.trim()) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const res = await api.chat({
        message: message.trim(),
        patient_id: patientId.trim() || null,
        encounter_id: encounterId.trim() || null,
        dry_run: true,
      });
      setResult(res);
    } catch (err) {
      setError(err.message || "Request failed.");
    } finally {
      setBusy(false);
    }
  }

  const route = result?.route;

  return (
    <div className="grid grid-2">
      <form className="card" onSubmit={send}>
        <header>
          <div>
            <h2>Clinical input</h2>
            <p className="sub">
              Signed in as {user?.display_name} ({user?.role})
            </p>
          </div>
        </header>

        <div className="field">
          <label htmlFor="msg">
            Message <span className="opt">— vitals, findings, or a question</span>
          </label>
          <textarea
            id="msg"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Patient Maria temperature 41 C, pulse 210, BP 70/40."
          />
        </div>

        <div className="field-row">
          <div className="field">
            <label htmlFor="pid">
              Patient ID <span className="opt">(optional)</span>
            </label>
            <input
              id="pid"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              placeholder="OpenMRS patient identifier"
            />
          </div>
          <div className="field">
            <label htmlFor="eid">
              Encounter ID <span className="opt">(optional)</span>
            </label>
            <input
              id="eid"
              value={encounterId}
              onChange={(e) => setEncounterId(e.target.value)}
              placeholder="OpenMRS encounter identifier"
            />
          </div>
        </div>

        <div className="btn-row">
          <button className="primary" type="submit" disabled={busy || !message.trim()}>
            {busy ? "Processing…" : "Process message"}
          </button>
          <span className="hint">
            Dry-run by default: nothing is written to OpenMRS without explicit
            approval.
          </span>
        </div>

        {error && (
          <div style={{ marginTop: 12 }}>
            <Notice tone="alert">{error}</Notice>
          </div>
        )}

        <div className="section">
          <h3>Examples</h3>
          <div className="examples">
            {EXAMPLES.map((ex) => (
              <button key={ex} type="button" onClick={() => setMessage(ex)}>
                {ex.length > 62 ? `${ex.slice(0, 62)}…` : ex}
              </button>
            ))}
          </div>
        </div>
      </form>

      <div>
        {!result && !error && (
          <Empty>
            Submit a message to see intent routing, entity extraction, validation
            and any retrieved evidence.
          </Empty>
        )}

        {result && (
          <div className="stack">
            <div className="card">
              <header>
                <div>
                  <h2>Route</h2>
                  <p className="sub">
                    request {result.request_id} · {result.created_at}
                  </p>
                </div>
                <div className="pill-row">
                  <Pill tone={route === "ENCODE_DATA" ? "info" : route === "ASK_QUESTION" ? "info" : "warn"}>
                    {route}
                  </Pill>
                  {result.saved ? (
                    <Pill tone="ok">saved</Pill>
                  ) : (
                    <Pill tone="warn">not saved</Pill>
                  )}
                  {result.requires_confirmation && (
                    <Pill tone="alert">requires confirmation</Pill>
                  )}
                </div>
              </header>

              {result.saved ? (
                <Notice tone="ok">
                  This payload was written to OpenMRS after approval.
                </Notice>
              ) : (
                <Notice tone="info">
                  Dry run — no write was attempted. Approve the payload in the review
                  queue to submit it.
                </Notice>
              )}
            </div>

            <IntentCard intent={result.intent} />
            <StagesCard stages={result.stages} />
            <ExtractionCard extraction={result.extraction} />
            <ValidationCard validation={result.validation} />
            <RetrievalCard retrieval={result.retrieval} />
            <AnswerCard answer={result.answer} />
          </div>
        )}
      </div>
    </div>
  );
}
