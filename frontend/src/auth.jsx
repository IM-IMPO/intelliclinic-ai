import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, clearSession, getStoredUser, getToken, setSession } from "./api.js";

const AuthContext = createContext(null);

// The backend's sessions are held in memory. A token left in localStorage from
// a previous server run is not proof of a live session, so a stored token is
// always re-verified against GET /auth/me before it is trusted. Failing closed
// is the correct default: a stale token must never grant access.
export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [booting, setBooting] = useState(true);
  const [bootError, setBootError] = useState("");

  useEffect(() => {
    let alive = true;
    const token = getToken();
    if (!token) {
      setBooting(false);
      return () => {};
    }
    api
      .me()
      .then((res) => {
        if (alive) setUser(res.user);
      })
      .catch((err) => {
        if (!alive) return;
        // A 401 means the token is dead - clear it. Anything else (backend
        // unreachable, 403) is reported rather than silently signing the user
        // out, because signing out would hide a real problem.
        if (err.status === 401) {
          clearSession();
          setUser(null);
        } else {
          setBootError(err.message || "Could not verify the stored session.");
        }
      })
      .finally(() => {
        if (alive) setBooting(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  const login = useCallback(async (username, password) => {
    const res = await api.login(username, password);
    setSession(res.token, res.user);
    setUser(res.user);
    setBootError("");
    return res.user;
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      // Signing out locally must succeed even if the server call fails.
    } finally {
      clearSession();
      setUser(null);
    }
  }, []);

  const value = useMemo(
    () => ({ user, booting, bootError, login, logout }),
    [user, booting, bootError, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}

// Backend roles: DOCTOR, ADMINISTRATOR, NURSE. The server enforces every
// permission server-side; this helper only decides what the UI offers. Hiding
// a button is a courtesy - it is never the security boundary.
export const ROLES = ["ADMINISTRATOR", "DOCTOR", "NURSE"];

export function roleAtLeast(user, minimum) {
  const order = { NURSE: 1, DOCTOR: 2, ADMINISTRATOR: 3 };
  if (!user?.role) return false;
  return (order[user.role] || 0) >= (order[minimum] || 0);
}

export function can(user, action) {
  if (!user?.role) return false;
  switch (action) {
    // Only DOCTOR and ADMINISTRATOR may submit to OpenMRS (server-enforced).
    case "approve":
    case "index_documents":
    case "set_document_status":
      return roleAtLeast(user, "DOCTOR");
    // Administration is restricted to ADMINISTRATOR.
    case "delete_document":
    case "audit_logs":
    case "manage_users":
      return roleAtLeast(user, "ADMINISTRATOR");
    default:
      return true;
  }
}

export function displayName(user) {
  return user?.display_name || user?.username || "unknown";
}

export { getStoredUser };
