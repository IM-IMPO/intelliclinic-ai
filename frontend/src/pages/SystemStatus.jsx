import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useAuth } from "../auth.jsx";
import { Loading, Notice, Pill, fmt } from "../components.jsx";

// This page exists to make capability claims auditable. Every row below is read
// from the server's own settings report — nothing is inferred from the fact that
// the UI rendered. A capability that is not configured is labelled as such rather
// than hidden behind a disabled button that implies it would work.

const ALGO_LABEL = {
  intent: "Algorithm 1 · Intent classification",
  extraction: "Algorithm 2 · Entity extraction",
  validation: "Algorithm 3 · Validation",
  retrieval: "Algorithm 4 · Retrieval",
  answering: "Algorithm 5 · Grounded answering",
};

function backendTone(backend, configured) {
  if (configured === false) return "warn";
  if (backend === "real" || backend === "sklearn") return "ok";
  if (backend === "stub" || backend === "unavailable") return "alert";
  return "neutral";
}

export default function SystemStatus() {
  const { user } = useAuth();
  const isAdmin = user?.role === "ADMINISTRATOR";

  const [status, setStatus] = useState(null);
  const [ready, setReady] = useState(null);
  const [logs, setLogs] = useState(null);
  const [error, setError] = useState("");

  async function load() {
    try {
      const [s, r] = await Promise.allSettled([api.systemStatus(), api.systemReady()]);
      if (s.status === "fulfilled") setStatus(s.value);
      else setError(s.reason?.message || "System status unavailable.");

      if (r.status === "fulfilled") setReady(r.value);
    } catch (e) {
      setError(e.message || "System status unavailable.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  // Audit log is admin-only; a failed load must not blank the page.
  useEffect(() => {
    if (!isAdmin) return;
    api
      .auditLogs()
      .then((res) => setLogs(res))
      .catch(() => setLogs(null));
  }, [isAdmin]);

  if (error) return <Notice tone="alert">{error}</Notice>;
  if (!status) return <Loading>Reading server capability report…</Loading>;

  const algos = status.algorithms || {};
  const integrations = status.integrations || {};
  const environment = status.environment || {};

  return (
    <div className="stack">
      <div className="card">
        <header>
          <div>
            <h2>System status</h2>
            <p className="sub">
              GET /api/v1/system/status · {environment.app_env} · v
              {environment.version ?? "unknown"}
            </p>
          </div>
          <button className="ghost" onClick={load}>
            Refresh
          </button>
        </header>

        {ready && (
          <div style={{ marginTop: 8 }}>
            <Notice tone={ready.ready ? "ok" : "warn"}>
              Readiness: {ready.ready ? "READY" : "NOT READY"} · documents indexed{" "}
              {fmt(ready.documents_indexed)} · OpenMRS {ready.openmrs ? "reachable" : "unreachable"}
              {ready.openmrs_detail ? ` (${ready.openmrs_detail})` : ""}
            </Notice>
          </div>
        )}
      </div>

      <div className="card">
        <header>
          <div>
            <h2>Algorithms</h2>
            <p className="sub">
              The `backend` field is the server's own claim about which implementation
              produced results. It is not a UI decoration.
            </p>
          </div>
        </header>
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>Algorithm</th>
                <th>Backend</th>
                <th>Model</th>
                <th>Configured</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(algos).map(([key, a]) => (
                <tr key={key}>
                  <td>{ALGO_LABEL[key] || key}</td>
                  <td>
                    <Pill tone={backendTone(a?.backend, a?.configured)}>
                      {a?.backend ?? "unknown"}
                    </Pill>
                  </td>
                  <td className="mono small">{a?.model ?? "—"}</td>
                  <td>{a?.configured ? "yes" : "no"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {Object.values(algos).some((a) => a?.note) && (
          <ul className="small muted notes">
            {Object.entries(algos)
              .filter(([, a]) => a?.note)
              .map(([key, a]) => (
                <li key={key}>
                  <strong>{ALGO_LABEL[key] || key}:</strong> {a.note}
                </li>
              ))}
          </ul>
        )}
      </div>

      <div className="grid grid-2">
        <div className="card">
          <header>
            <div>
              <h2>Integrations</h2>
              <p className="sub">External systems the server can actually reach.</p>
            </div>
          </header>
          <div className="tw">
            <table>
              <tbody>
                <tr>
                  <td>OpenMRS</td>
                  <td className="right">
                    <Pill tone={integrations.openmrs?.configured ? "ok" : "warn"}>
                      {integrations.openmrs?.configured ? "configured" : "not configured"}
                    </Pill>
                  </td>
                </tr>
                <tr>
                  <td>OpenMRS base URL</td>
                  <td className="mono small right nowrap">
                    {integrations.openmrs?.base_url || "—"}
                  </td>
                </tr>
                <tr>
                  <td>OpenMRS reachable</td>
                  <td className="right">
                    <Pill tone={integrations.openmrs?.reachable ? "ok" : "warn"}>
                      {integrations.openmrs?.reachable ? "yes" : "no"}
                    </Pill>
                  </td>
                </tr>
                <tr>
                  <td>Chroma persist dir</td>
                  <td className="mono small right nowrap">
                    {integrations.chroma_persist_dir || "—"}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <header>
            <div>
              <h2>Environment</h2>
              <p className="sub">Non-secret settings. Credentials are never shown here.</p>
            </div>
          </header>
          <div className="tw">
            <table>
              <tbody>
                <tr>
                  <td>app_env</td>
                  <td className="mono small right nowrap">{environment.app_env ?? "—"}</td>
                </tr>
                <tr>
                  <td>version</td>
                  <td className="mono small right nowrap">{environment.version ?? "—"}</td>
                </tr>
                <tr>
                  <td>rate limit</td>
                  <td className="mono small right nowrap">
                    {environment.rate_limit_per_minute ?? "—"}/min
                  </td>
                </tr>
                <tr>
                  <td>OpenMRS detail</td>
                  <td className="small right">{integrations.openmrs?.detail ?? "—"}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="card">
        <header>
          <div>
            <h2>Audit log</h2>
            <p className="sub">
              {isAdmin
                ? "Pipeline run records. Patient-identifiable free text is never stored here."
                : "Administrator role required."}
            </p>
          </div>
          {logs && <Pill tone="neutral">{fmt(logs.count)} events</Pill>}
        </header>

        {!isAdmin ? (
          <Notice tone="info">Sign in as an administrator to read the audit log.</Notice>
        ) : !logs ? (
          <Loading>Loading audit log…</Loading>
        ) : !logs.events || logs.events.length === 0 ? (
          <Notice tone="info">No pipeline runs recorded yet.</Notice>
        ) : (
          <>
            <div className="tw">
              <table>
                <thead>
                  <tr>
                    <th>When</th>
                    <th>Event</th>
                    <th>User</th>
                    <th>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {logs.events.slice(0, 30).map((e, i) => (
                    <tr key={i}>
                      <td className="mono small nowrap">{e.created_at || e.ts || "—"}</td>
                      <td className="mono small">{e.event || e.type || "—"}</td>
                      <td className="small">{e.user || e.username || "—"}</td>
                      <td className="small">{e.detail || e.summary || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="small muted" style={{ marginTop: 10 }}>
              {logs.note || "Run records store routes, entity types, backends and document titles only."}
            </p>
          </>
        )}
      </div>
    </div>
  );
}
