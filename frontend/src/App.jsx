import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { api } from "./api.js";
import { useAuth, displayName } from "./auth.jsx";
import Login from "./pages/Login.jsx";
import Chat from "./pages/Chat.jsx";
import ReviewQueue from "./pages/ReviewQueue.jsx";
import Algorithms from "./pages/Algorithms.jsx";
import KnowledgeBase from "./pages/KnowledgeBase.jsx";
import SystemStatus from "./pages/SystemStatus.jsx";

const NAV = [
  { to: "/", label: "Clinical input", end: true },
  { to: "/review", label: "Review queue", badge: true },
  { to: "/algorithms", label: "Algorithms" },
  { to: "/knowledge-base", label: "Knowledge base" },
  { to: "/status", label: "System status" },
];

function Shell() {
  const { user, logout } = useAuth();
  const location = useLocation();
  const [pending, setPending] = useState(0);

  // The review-queue badge is informational only. A failure to load it must
  // not blank the navigation, so it is handled independently of the route.
  // The count is rendered right here, so it stays local to Shell.
  useEffect(() => {
    if (location.pathname !== "/review") {
      api
        .reviewQueue()
        .then((res) => setPending(res?.awaiting_review?.length ?? 0))
        .catch(() => setPending(0));
    } else {
      setPending(0);
    }
  }, [location.pathname]);

  return (
    <div className="shell">
      <header className="topbar">
        <NavLink to="/" className="brand">
          <span className="mark" aria-hidden="true">
            +
          </span>
          <span>
            IntelliClinic AI
            <small>Intelligent Clinical Assistant</small>
          </span>
        </NavLink>

        <nav className="nav" aria-label="Primary">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => (isActive ? "active" : undefined)}
            >
              {item.label}
              {item.badge && pending > 0 && (
                <span className="nav-badge" title={`${pending} awaiting review`}>
                  {pending}
                </span>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="who">
          <span>{displayName(user)}</span>
          <span className="code">{user?.role}</span>
          <button className="ghost small" onClick={logout}>
            Sign out
          </button>
        </div>
      </header>

      <main className="main">
        <Routes>
          <Route path="/" element={<Chat />} />
          <Route path="/review" element={<ReviewQueue />} />
          <Route path="/algorithms" element={<Algorithms />} />
          <Route path="/knowledge-base" element={<KnowledgeBase />} />
          <Route path="/status" element={<SystemStatus />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <footer className="foot">
        IntelliClinic AI · clinical decisions remain the responsibility of the
        treating clinician
      </footer>
    </div>
  );
}

export default function App() {
  const { user, booting, bootError } = useAuth();

  if (booting) {
    return (
      <div className="login-wrap">
        <div className="login-card">
          <h1>IntelliClinic AI</h1>
          <p className="sub">Restoring session…</p>
          <div className="loading">Verifying the stored session with the server</div>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <Routes>
        <Route path="*" element={<Login bootError={bootError} />} />
      </Routes>
    );
  }

  return <Shell />;
}
