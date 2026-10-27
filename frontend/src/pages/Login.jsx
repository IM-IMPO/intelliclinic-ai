import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth.jsx";
import { Notice } from "../components.jsx";

// Demo accounts. Passwords come from the environment (NURSE_PASSWORD,
// DOCTOR_PASSWORD, ADMIN_PASSWORD); these are the development fallbacks the
// server seeds when they are unset. They are reported by the login screen so
// nobody has to guess, and the server refuses production without overrides.
const EXAMPLES = [
  { label: "doctor", u: "doctor", p: "doctor-dev-password", role: "doctor" },
  { label: "nurse", u: "nurse", p: "nurse-dev-password", role: "nurse" },
  { label: "admin", u: "admin", p: "admin-dev-password", role: "admin" },
];

export default function Login({ bootError }) {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(e) {
    e?.preventDefault();
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      await login(username.trim(), password);
      navigate("/", { replace: true });
    } catch (err) {
      // A wrong password and a down backend must look different: one is the
      // user's to fix, the other is ours to report.
      setError(err.message || "Sign-in failed.");
    } finally {
      setBusy(false);
    }
  }

  // Named `applyExample`, not `useExample`: a `use*` name makes the
  // rules-of-hooks lint rule treat this as a hook, and it is a plain event
  // handler, so it would raise a false "hook called inside a callback" error.
  function applyExample(u, p) {
    setUsername(u);
    setPassword(p);
    setError("");
  }

  return (
    <div className="login-wrap">
      <form className="login-card card" onSubmit={submit}>
        <h1>IntelliClinic AI</h1>
        <p className="sub">Sign in to the clinical assistant</p>

        {bootError && <Notice tone="warn">{bootError}</Notice>}

        <div className="field" style={{ marginTop: 12 }}>
          <label htmlFor="u">Username</label>
          <input
            id="u"
            autoComplete="username"
            autoFocus
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>

        <div className="field">
          <label htmlFor="p">Password</label>
          <input
            id="p"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>

        {error && (
          <div style={{ marginTop: 10 }}>
            <Notice tone="alert">{error}</Notice>
          </div>
        )}

        <div className="btn-row" style={{ marginTop: 14 }}>
          <button className="primary" type="submit" disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </div>

        <p className="hint" style={{ marginTop: 14 }}>
          Sessions are verified against the server on every load — a token left
          over from a previous server run is rejected, not trusted.
        </p>

        <div className="examples">
          <span className="small muted" style={{ width: "100%" }}>
            Development accounts (from the server configuration):
          </span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              type="button"
              onClick={() => applyExample(ex.u, ex.p)}
              title={`Sign in as ${ex.role}`}
            >
              {ex.label}
            </button>
          ))}
        </div>
      </form>
    </div>
  );
}
