import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useAuth } from "../auth.jsx";
import { Empty, Loading, Notice, Pill, fmt } from "../components.jsx";

// Only APPROVED documents are ever cited. A PENDING_REVIEW or DISABLED document
// still shows up here, because its state is exactly what a reviewer needs to see.

const STATUSES = ["PENDING_REVIEW", "APPROVED", "DISABLED"];
const STATUS_TONE = { APPROVED: "ok", PENDING_REVIEW: "warn", DISABLED: "neutral" };

const TOPICS = [
  "general",
  "antibiotics",
  "cardiology",
  "emergency",
  "paediatrics",
  "guideline",
];

const EMPTY_FORM = {
  document_id: "",
  title: "",
  source_version: "",
  effective_date: "",
  topic: "general",
  text: "",
};

function IndexForm({ onDone }) {
  const [form, setForm] = useState(EMPTY_FORM);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  function set(k, v) {
    setForm((f) => ({ ...f, [k]: v }));
  }

  async function submit(e) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const res = await api.indexDocument({
        document_id: form.document_id.trim(),
        title: form.title.trim(),
        source_version: form.source_version.trim(),
        effective_date: form.effective_date || None,
        topic: form.topic,
        review_status: "PENDING_REVIEW",
        text: form.text,
      });
      setResult(res);
      setForm(EMPTY_FORM);
      onDone?.();
    } catch (err) {
      setError(err.message || "Indexing failed.");
    } finally {
      setBusy(false);
    }
  }

  const ready =
    form.document_id.trim() && form.title.trim() && form.source_version.trim() && form.text.trim();

  return (
    <form className="card" onSubmit={submit}>
      <header>
        <div>
          <h2>Index a document</h2>
          <p className="sub">
            POST /api/v1/knowledge-base/documents · new documents start PENDING_REVIEW
          </p>
        </div>
        <button className="primary" disabled={busy || !ready}>
          {busy ? "Indexing…" : "Index"}
        </button>
      </header>

      <div className="grid grid-2">
        <div className="field">
          <label htmlFor="d-id">Document ID *</label>
          <input
            id="d-id"
            className="mono"
            value={form.document_id}
            onChange={(e) => set("document_id", e.target.value)}
            placeholder="who-pneumonia-2026"
          />
        </div>
        <div className="field">
          <label htmlFor="d-ver">Source version *</label>
          <input
            id="d-ver"
            className="mono"
            value={form.source_version}
            onChange={(e) => set("source_version", e.target.value)}
            placeholder="v1.2"
          />
        </div>
      </div>

      <div className="field">
        <label htmlFor="d-title">Title *</label>
        <input
          id="d-title"
          value={form.title}
          onChange={(e) => set("title", e.target.value)}
          placeholder="WHO guideline: community-acquired pneumonia"
        />
      </div>

      <div className="grid grid-2">
        <div className="field">
          <label htmlFor="d-eff">Effective date</label>
          <input
            id="d-eff"
            type="date"
            value={form.effective_date}
            onChange={(e) => set("effective_date", e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="d-topic">Topic</label>
          <select
            id="d-topic"
            value={form.topic}
            onChange={(e) => set("topic", e.target.value)}
          >
            {TOPICS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="field">
        <label htmlFor="d-text">Document text *</label>
        <textarea
          id="d-text"
          rows={7}
          value={form.text}
          onChange={(e) => set("text", e.target.value)}
          placeholder="Full guideline text. This is what retrieval searches and what citations quote."
        />
      </div>

      {error && (
        <div style={{ marginTop: 12 }}>
          <Notice tone="alert">{error}</Notice>
        </div>
      )}
      {result && (
        <div style={{ marginTop: 12 }}>
          <Notice tone="ok">
            {result.status} · {result.document_id} · {result.chunks_indexed} chunks · sha256{" "}
            <span className="mono">{String(result.sha256).slice(0, 16)}…</span>
          </Notice>
        </div>
      )}
    </form>
  );
}

export default function KnowledgeBase() {
  const { user } = useAuth();
  const canIndex = user?.role === "DOCTOR" || user?.role === "ADMINISTRATOR";
  const canDelete = user?.role === "ADMINISTRATOR";

  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("ALL");

  async function load() {
    try {
      const res = await api.documents();
      setData(res);
      setError("");
    } catch (e) {
      setError(e.message || "Could not load documents.");
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function setStatus(documentId, reviewStatus) {
    try {
      await api.setDocumentStatus(documentId, reviewStatus);
      await load();
    } catch (e) {
      setError(e.message || "Status change failed.");
    }
  }

  async function remove(documentId) {
    if (!window.confirm(`Delete ${documentId}? This removes all indexed chunks.`)) return;
    try {
      await api.deleteDocument(documentId);
      await load();
    } catch (e) {
      setError(e.message || "Delete failed.");
    }
  }

  if (error) return <Notice tone="alert">{error}</Notice>;
  if (!data) return <Loading>Loading documents…</Loading>;

  const docs = data.documents || [];
  const shown = filter === "ALL" ? docs : docs.filter((d) => d.review_status === filter);
  const counts = STATUSES.reduce(
    (acc, s) => ({ ...acc, [s]: docs.filter((d) => d.review_status === s).length }),
    {}
  );

  return (
    <div className="stack">
      <div className="card">
        <header>
          <div>
            <h2>Knowledge base</h2>
            <p className="sub">
              {docs.length} document{docs.length === 1 ? "" : "s"} indexed · collection{" "}
              <span className="mono">{data.collection}</span> · embedding{" "}
              <span className="mono">{data.embedding_model}</span>
            </p>
          </div>
          <div className="pill-row">
            <Pill tone={data.indexing_status === "ready" ? "ok" : "warn"}>
              index {data.indexing_status}
            </Pill>
            <button className="ghost" onClick={load}>
              Refresh
            </button>
          </div>
        </header>
        <p className="small muted">
          Only APPROVED documents are retrieved and cited. Everything else is visible
          here so a reviewer can decide, and DISABLED documents are excluded from
          retrieval without being deleted.
        </p>
      </div>

      <div className="card">
        <header>
          <div>
            <h2>Documents</h2>
            <p className="sub">{shown.length} shown</p>
          </div>
          <div className="pill-row">
            {["ALL", ...STATUSES].map((s) => (
              <button
                key={s}
                className={filter === s ? "chip on" : "chip"}
                onClick={() => setFilter(s)}
              >
                {s === "ALL" ? `all (${docs.length})` : `${s.toLowerCase()} (${counts[s]})`}
              </button>
            ))}
          </div>
        </header>

        {shown.length === 0 ? (
          <Empty>
            {docs.length === 0
              ? "No documents indexed. Add one below — retrieval will report INSUFFICIENT until an APPROVED document matches."
              : "No documents with this status."}
          </Empty>
        ) : (
          <div className="tw">
            <table>
              <thead>
                <tr>
                  <th>Document</th>
                  <th>Topic</th>
                  <th className="right">Chunks</th>
                  <th>Review</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((d) => (
                  <tr key={d.document_id}>
                    <td>
                      <div>{d.title || d.document_id}</div>
                      <div className="small muted mono">
                        {d.document_id} · {d.source_version}
                        {d.effective_date ? ` · eff. ${d.effective_date}` : ""}
                      </div>
                    </td>
                    <td className="mono">{d.topic}</td>
                    <td className="num right">{fmt(d.chunk_count)}</td>
                    <td>
                      <Pill tone={STATUS_TONE[d.review_status] || "neutral"}>
                        {d.review_status}
                      </Pill>
                    </td>
                    <td className="nowrap">
                      <div className="btn-row">
                        {canIndex && d.review_status !== "APPROVED" && (
                          <button
                            className="ghost small"
                            onClick={() => setStatus(d.document_id, "APPROVED")}
                          >
                            Approve
                          </button>
                        )}
                        {canIndex && d.review_status === "APPROVED" && (
                          <button
                            className="ghost small"
                            onClick={() => setStatus(d.document_id, "DISABLED")}
                          >
                            Disable
                          </button>
                        )}
                        {canIndex && d.review_status === "PENDING_REVIEW" && (
                          <button
                            className="ghost small"
                            onClick={() => setStatus(d.document_id, "DISABLED")}
                          >
                            Disable
                          </button>
                        )}
                        {canDelete && (
                          <button
                            className="ghost small danger"
                            onClick={() => remove(d.document_id)}
                          >
                            Delete
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {canIndex ? (
        <IndexForm onDone={load} />
      ) : (
        <Notice tone="info">
          Signed in as {user?.role}. Indexing and review-status changes require the
          DOCTOR or ADMINISTRATOR role. You can still read the document list.
        </Notice>
      )}
    </div>
  );
}
