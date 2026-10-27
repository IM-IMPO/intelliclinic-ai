import { useState } from "react";
import { api } from "../api.js";
import { BackendBadge, Empty, Loading, Notice, Pill, statusPill } from "../components.jsx";

// Each card calls a real endpoint and shows which backend actually produced the
// result. A stub is always labelled as such — this page exists to make that
// visible, not to hide it.

const INTENT_EXAMPLES = [
  "Patient Juan has a fever of 38.5 C and BP is 120/80, complains of cough.",
  "Patient Maria temperature 41 C, pulse 210, BP 70/40.",
  "What is the recommended dose of amoxicillin for adults?",
  "Patient Reyes temperature 38.9 C",
];

const EXTRACT_EXAMPLES = [
  "Patient Juan has a fever of 38.5 C and BP is 120/80, complains of cough.",
  "SpO2 85 percent on room air, respiratory rate 32.",
  "History of hypertension, no chest pain.",
  "Diagnosed with community acquired pneumonia, started on amoxicillin 500 mg three times daily.",
];

const RETRIEVE_EXAMPLES = [
  "amoxicillin dose adults",
  "community acquired pneumonia treatment",
  "normal blood pressure range",
];

function Result({ title, sub, result, error, render }) {
  return (
    <div className="card">
      <header>
        <div>
          <h2>{title}</h2>
          <p className="sub">{sub}</p>
        </div>
      </header>
      {error && <Notice tone="alert">{error}</Notice>}
      {!result && !error && <Loading>Running…</Loading>}
      {result && render(result)}
    </div>
  );
}

export default function Algorithms() {
  const [intentText, setIntentText] = useState(INTENT_EXAMPLES[0]);
  const [extractText, setExtractText] = useState(EXTRACT_EXAMPLES[0]);
  const [query, setQuery] = useState(RETRIEVE_EXAMPLES[0]);
  const [busy, setBusy] = useState({});
  const [errors, setErrors] = useState({});
  const [results, setResults] = useState({});

  async function run(key, fn) {
    setBusy((b) => ({ ...b, [key]: true }));
    setErrors((e) => ({ ...e, [key]: "" }));
    try {
      const res = await fn();
      setResults((r) => ({ ...r, [key]: res }));
    } catch (e) {
      setErrors((er) => ({ ...er, [key]: e.message || "Request failed." }));
    } finally {
      setBusy((b) => ({ ...b, [key]: false }));
    }
  }

  return (
    <div className="grid grid-2">
      <div className="stack">
        <form
          className="card"
          onSubmit={(e) => {
            e.preventDefault();
            run("intent", () => api.intent(intentText));
          }}
        >
          <header>
            <div>
              <h2>Algorithm 1 · Intent classification</h2>
              <p className="sub">POST /api/v1/intent</p>
            </div>
            <button className="primary" disabled={busy.intent || !intentText.trim()}>
              {busy.intent ? "Classifying…" : "Classify"}
            </button>
          </header>
          <div className="field">
            <label htmlFor="it">Message</label>
            <textarea
              id="it"
              value={intentText}
              onChange={(e) => setIntentText(e.target.value)}
            />
          </div>
          <div className="examples">
            {INTENT_EXAMPLES.map((ex) => (
              <button key={ex} type="button" onClick={() => setIntentText(ex)}>
                {ex.length > 58 ? `${ex.slice(0, 58)}…` : ex}
              </button>
            ))}
          </div>
        </form>

        <form
          className="card"
          onSubmit={(e) => {
            e.preventDefault();
            run("extract", () => api.extract(extractText));
          }}
        >
          <header>
            <div>
              <h2>Algorithm 2 · Entity extraction</h2>
              <p className="sub">POST /api/v1/extract</p>
            </div>
            <button className="primary" disabled={busy.extract || !extractText.trim()}>
              {busy.extract ? "Extracting…" : "Extract"}
            </button>
          </header>
          <div className="field">
            <label htmlFor="et">Clinical text</label>
            <textarea
              id="et"
              value={extractText}
              onChange={(e) => setExtractText(e.target.value)}
            />
          </div>
          <div className="examples">
            {EXTRACT_EXAMPLES.map((ex) => (
              <button key={ex} type="button" onClick={() => setExtractText(ex)}>
                {ex.length > 58 ? `${ex.slice(0, 58)}…` : ex}
              </button>
            ))}
          </div>
        </form>

        <form
          className="card"
          onSubmit={(e) => {
            e.preventDefault();
            run("retrieve", () => api.retrieve(query, 5));
          }}
        >
          <header>
            <div>
              <h2>Algorithm 4 · Retrieval</h2>
              <p className="sub">POST /api/v1/retrieve</p>
            </div>
            <button className="primary" disabled={busy.retrieve || !query.trim()}>
              {busy.retrieve ? "Retrieving…" : "Retrieve"}
            </button>
          </header>
          <div className="field">
            <label htmlFor="q">Query</label>
            <input id="q" value={query} onChange={(e) => setQuery(e.target.value)} />
          </div>
          <div className="examples">
            {RETRIEVE_EXAMPLES.map((ex) => (
              <button key={ex} type="button" onClick={() => setQuery(ex)}>
                {ex}
              </button>
            ))}
          </div>
        </form>
      </div>

      <div className="stack">
        <Result
          title="Intent result"
          sub="Algorithm 1"
          result={results.intent}
          error={errors.intent}
          render={(r) => (
            <>
              <div className="pill-row" style={{ marginBottom: 10 }}>
                <Pill tone="info">{r.result?.intent}</Pill>
                <Pill tone="neutral">
                  {(r.result?.confidence * 100).toFixed(1)}%
                </Pill>
                <Pill tone="neutral">threshold {r.result?.threshold}</Pill>
              </div>
              <BackendBadge
                backend={r.result?.backend}
                modelName={r.result?.model_name}
              />
              {r.result?.clarification_prompt && (
                <div style={{ marginTop: 10 }}>
                  <Notice tone="warn">{r.result.clarification_prompt}</Notice>
                </div>
              )}
              <div style={{ marginTop: 10 }}>
                <p className="small muted">{r.route_explanation}</p>
              </div>
              {Object.keys(r.result?.alternatives || {}).length > 0 && (
                <div className="section">
                  <h3>Alternatives</h3>
                  <div className="pill-row">
                    {Object.entries(r.result.alternatives).map(([k, v]) => (
                      <Pill key={k} tone="neutral">
                        {k} {(v * 100).toFixed(0)}%
                      </Pill>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        />

        <Result
          title="Extraction result"
          sub="Algorithm 2"
          result={results.extract}
          error={errors.extract}
          render={(r) => (
            <>
              <div className="pill-row" style={{ marginBottom: 10 }}>
                <Pill tone="neutral">{r.entities?.length || 0} entities</Pill>
              </div>
              <BackendBadge backend={r.backend} modelName={r.model_name} />
              <div className="section">
                {r.entities?.length ? (
                  <div className="tw">
                    <table>
                      <thead>
                        <tr>
                          <th>Entity</th>
                          <th>Type</th>
                          <th>Assertion</th>
                          <th className="right">Conf</th>
                        </tr>
                      </thead>
                      <tbody>
                        {r.entities.map((e, i) => (
                          <tr key={i}>
                            <td className="mono">“{e.text}”</td>
                            <td className="mono">{e.type}</td>
                            <td>{statusPill(e.assertion)}</td>
                            <td className="num right">
                              {(e.confidence * 100).toFixed(1)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <Empty>No entities found.</Empty>
                )}
              </div>
            </>
          )}
        />

        <Result
          title="Retrieval result"
          sub="Algorithm 4"
          result={results.retrieve}
          error={errors.retrieve}
          render={(r) => (
            <>
              <div className="pill-row" style={{ marginBottom: 10 }}>
                <Pill tone="neutral">{r.retrieved_documents?.length || 0} chunks</Pill>
                <Pill tone="neutral">{r.evidence_status}</Pill>
              </div>
              <BackendBadge backend={r.backend} modelName={r.embedding_model} />
              <div className="section">
                {r.retrieved_documents?.length ? (
                  <div className="tw">
                    <table>
                      <thead>
                        <tr>
                          <th>Title</th>
                          <th>Review</th>
                          <th className="right">Similarity</th>
                        </tr>
                      </thead>
                      <tbody>
                        {r.retrieved_documents.map((d) => (
                          <tr key={d.chunk_id}>
                            <td>{d.title}</td>
                            <td>{statusPill(d.review_status)}</td>
                            <td className="num right">
                              {(d.similarity_score * 100).toFixed(1)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <Empty>No approved documents matched.</Empty>
                )}
              </div>
            </>
          )}
        />
      </div>
    </div>
  );
}
