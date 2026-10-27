import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useAuth } from "../auth.jsx";
import { Empty, Loading, Notice, Pill, fmt, statusPill } from "../components.jsx";

// AI-extracted observations sit here until a clinician approves them. Nothing in
// this queue has been written to OpenMRS, and approval is the only write path.

function CorrectionEditor({ item, onChange }) {
  // The stored payload is editable: a clinician may fix a mis-extracted value
  // before approving. An empty value means "accept what the pipeline produced".
  const obs = item.observations || item.payload?.observations || [];
  if (obs.length === 0) return null;

  return (
    <div className="section">
      <h3>Corrections</h3>
      <p className="sub small">
        Edit any value the pipeline got wrong, then approve. Blank means keep the
        extracted value.
      </p>
      <div className="field-row">
        {obs.map((o, i) => (
          <div className="field" key={`${o.concept_name}-${i}`}>
            <label htmlFor={`corr-${item.request_id}-${i}`}>{o.concept_name}</label>
            <input
              id={`corr-${item.request_id}-${i}`}
              defaultValue={`${fmt(o.value)}${o.unit ? ` ${o.unit}` : ""}`}
              placeholder="keep extracted value"
              onChange={(e) => onChange(i, e.target.value)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

function Item({ item, onDone }) {
  const { user } = useAuth();
  const canApprove = user?.role === "DOCTOR" || user?.role === "ADMINISTRATOR";
  const [busy, setBusy] = useState(false);
  const [outcome, setOutcome] = useState(null);
  const [error, setError] = useState("");
  const [edits, setEdits] = useState({});
  const [detail, setDetail] = useState(null);

  const obs = item.observations || item.payload?.observations || [];

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const d = await api.reviewItem(item.request_id);
        if (!cancelled) setDetail(d);
      } catch {
        // The summary list is still usable without the full payload.
        if (!cancelled) setDetail(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [item.request_id]);

  function setEdit(i, value) {
    setEdits((prev) => ({ ...prev, [i]: value }));
  }

  async function approve() {
    if (busy) return;
    setBusy(true);
    setError("");
    setOutcome(null);
    const corrections = {};
    obs.forEach((o, i) => {
      const v = edits[i];
      if (v !== undefined && v.trim() !== "") corrections[o.concept_name] = v.trim();
    });
    try {
      const res = await api.approve(item.request_id, {
        request_id: item.request_id,
        approved_by: user?.username || user?.user_id,
        corrections,
        idempotency_key: `ui-${item.request_id}-${Date.now()}`,
      });
      setOutcome(res);
      onDone?.();
    } catch (e) {
      setError(e.message || "Approval failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <header>
        <div>
          <h2 className="mono">{item.request_id}</h2>
          <p className="sub">
            {item.observation_count ?? obs.length} observation
            {(item.observation_count ?? obs.length) === 1 ? "" : "s"} · created{" "}
            {item.created_at} by {item.created_by}
          </p>
        </div>
        <div className="pill-row">
          {statusPill(item.validation_status)}
          <Pill tone={item.openmrs_configured ? "ok" : "warn"}>
            {item.openmrs_configured ? "OpenMRS configured" : "OpenMRS unconfigured"}
          </Pill>
        </div>
      </header>

      {!item.openmrs_configured && (
        <Notice tone="warn">
          OpenMRS is not configured, so approval will be recorded but no encounter
          will be written. This is reported, not silently skipped.
        </Notice>
      )}

      {obs.length > 0 && (
        <div className="section">
          <h3>Observations awaiting approval</h3>
          <div className="tw">
            <table>
              <thead>
                <tr>
                  <th>Concept</th>
                  <th>Value</th>
                  <th>Unit</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {obs.map((o, i) => (
                  <tr key={`${o.concept_name}-${i}`}>
                    <td>{o.concept_name}</td>
                    <td>{fmt(o.value)}</td>
                    <td className="mono">{fmt(o.unit)}</td>
                    <td>{statusPill(o.status)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {detail?.issues?.length > 0 && (
        <div className="section">
          <h3>Validation issues</h3>
          <ul className="small" style={{ margin: 0, paddingLeft: 18 }}>
            {detail.issues.map((iss, i) => (
              <li key={i}>
                <span className="mono">{iss.field}</span> — {iss.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      <CorrectionEditor item={detail || item} onChange={setEdit} />

      {canApprove ? (
        <div className="btn-row">
          <button className="primary" onClick={approve} disabled={busy}>
            {busy ? "Submitting…" : "Approve and submit"}
          </button>
          <span className="hint">
            Requires the DOCTOR or ADMINISTRATOR role. Approval is the only path to
            an OpenMRS write.
          </span>
        </div>
      ) : (
        <Notice tone="info">
          Signed in as {user?.role}. Only DOCTOR and ADMINISTRATOR roles may submit
          to OpenMRS.
        </Notice>
      )}

      {error && (
        <div style={{ marginTop: 12 }}>
          <Notice tone="alert">{error}</Notice>
        </div>
      )}

      {outcome && (
        <div style={{ marginTop: 12 }}>
          {outcome.status === "SUBMITTED" ? (
            <Notice tone="ok">
              Submitted. Encounter {outcome.openmrs_encounter_uuid || "—"} ·{" "}
              {outcome.openmrs_observation_uuids?.length || 0} observation
              {(outcome.openmrs_observation_uuids?.length || 0) === 1 ? "" : "s"}
            </Notice>
          ) : (
            <Notice tone="warn">
              {outcome.status}: {outcome.message}
            </Notice>
          )}
        </div>
      )}
    </div>
  );
}

export default function ReviewQueue() {
  const [queue, setQueue] = useState(null);
  const [error, setError] = useState("");

  async function load() {
    try {
      const res = await api.reviewQueue();
      setQueue(res.awaiting_review || []);
      setError("");
    } catch (e) {
      setError(e.message || "Could not load the review queue.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  if (error) {
    return <Notice tone="alert">{error}</Notice>;
  }
  if (queue === null) {
    return <Loading>Loading review queue…</Loading>;
  }

  return (
    <div className="stack">
      <div className="card">
        <header>
          <div>
            <h2>Review queue</h2>
            <p className="sub">
              {queue.length} payload{queue.length === 1 ? "" : "s"} awaiting human
              approval
            </p>
          </div>
          <button className="ghost" onClick={load}>
            Refresh
          </button>
        </header>
        <p className="small muted">
          Nothing here has been written to OpenMRS. Extraction alone never writes;
          a clinician must approve the payload first.
        </p>
      </div>

      {queue.length === 0 ? (
        <Empty>
          No pending payloads. Process a clinical message with dry-run enabled to
          populate this queue.
        </Empty>
      ) : (
        queue.map((item) => <Item key={item.request_id} item={item} onDone={load} />)
      )}
    </div>
  );
}
