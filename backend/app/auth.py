"""Authentication and role-based access control.

Roles are coarse and deliberately overlap as little as possible:

``NURSE``        enter and review observations, read patients, ask questions.
``DOCTOR``       everything a nurse can do, plus approve and submit to OpenMRS,
                 and manage the knowledge base.
``ADMINISTRATOR`` user/role management, document approval, system settings.

Every protected endpoint calls :func:`require` with the roles it accepts. The
check runs server-side on every request; the frontend's own gating is purely
cosmetic and is never treated as an authorization decision.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

from .config import Settings, get_settings
from .logging_util import get_logger

# PBKDF2 is used rather than plaintext comparison so that a leaked process dump
# does not hand over usable passwords. bcrypt/argon2 are preferable and are
# supported by bumping _hash/_verify.
_PBKDF2_ROUNDS = 240_000


class Role(str, Enum):
    NURSE = "NURSE"
    DOCTOR = "DOCTOR"
    ADMINISTRATOR = "ADMINISTRATOR"


ROLE_HIERARCHY: dict[Role, set[Role]] = {
    Role.ADMINISTRATOR: {Role.ADMINISTRATOR, Role.DOCTOR, Role.NURSE},
    Role.DOCTOR: {Role.DOCTOR, Role.NURSE},
    Role.NURSE: {Role.NURSE},
}


@dataclass
class User:
    user_id: str
    username: str
    display_name: str
    role: Role
    password_hash: str
    active: bool = True
    failed_logins: int = 0
    locked_until: float | None = None
    last_login: str | None = None


@dataclass
class Session:
    token: str
    user_id: str
    role: Role
    created_at: float
    expires_at: float
    ip: str | None = None


class AuthConfigError(RuntimeError):
    """A required auth setting is missing, so the app must not start."""


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


# --------------------------------------------------------------------------
# Password hashing
# --------------------------------------------------------------------------


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ROUNDS
    )
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, rounds, salt, expected = stored.split("$", 3)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(rounds)
        )
        return hmac.compare_digest(digest.hex(), expected)
    except Exception:
        return False


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


class AuthStore:
    """In-memory user/session store with a JSON persistence file.

    Swappable for an OpenMRS-backed or LDAP-backed store without touching the
    route layer: only :meth:`authenticate`, :meth:`session_for`, and
    :meth:`users` are called from request handlers.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self._users: dict[str, User] = {}
        self._sessions: dict[str, Session] = {}
        self._seed()

    def _seed(self) -> None:
        """Create the initial accounts from configuration.

        Passwords come from the environment (``NURSE_PASSWORD`` and friends).
        There is deliberately no committed default password: the ``nurse`` and
        ``doctor`` accounts can approve writes to OpenMRS, so a password that
        ships in the repository is a live credential on the most sensitive path
        of the system.

        When ``app_env`` is not ``production`` and a password is unset, a random
        one is generated and written to a gitignored local file that the
        development scripts read back. In ``production`` an unset password is a
        hard error rather than a silently guessable login.
        """

        import json
        import os
        import secrets
        from pathlib import Path

        log = get_logger("auth")

        dev = self.s.app_env != "production"
        defaults = [
            ("nurse", "Nurse Demo", Role.NURSE, "NURSE_PASSWORD"),
            ("doctor", "Doctor Demo", Role.DOCTOR, "DOCTOR_PASSWORD"),
            ("admin", "Administrator", Role.ADMINISTRATOR, "ADMIN_PASSWORD"),
        ]

        store_path = Path(__file__).resolve().parent / ".dev-credentials.json"
        generated: dict[str, str] = {}
        if dev:
            try:
                generated = json.loads(store_path.read_text("utf-8"))
            except (OSError, ValueError):
                generated = {}

        for uname, display, role, env_key in defaults:
            pw = os.environ.get(env_key)
            if not pw and not dev:
                raise AuthConfigError(
                    f"{env_key} is not set and app_env=production; refusing to "
                    f"create the '{uname}' account with a guessable password."
                )
            if not pw:
                pw = generated.get(uname)
                if not pw:
                    pw = secrets.token_urlsafe(24)
                    generated[uname] = pw
                    log.warning(
                        "%s unset; generated a random development password for "
                        "'%s' (written to %s).",
                        env_key,
                        uname,
                        store_path.name,
                    )
            self._users[uname] = User(
                user_id=uname,
                username=uname,
                display_name=display,
                role=role,
                password_hash=hash_password(pw),
            )

        if generated:
            store_path.write_text(
                json.dumps(generated, indent=2, sort_keys=True) + "\n", "utf-8"
            )

    # -- users ----------------------------------------------------------

    def users(self) -> list[dict[str, Any]]:
        return [
            {
                "user_id": u.user_id,
                "username": u.username,
                "display_name": u.display_name,
                "role": u.role.value,
                "active": u.active,
                "last_login": u.last_login,
                "locked": u.locked_until is not None and u.locked_until > time.time(),
            }
            for u in self._users.values()
        ]

    def set_role(self, user_id: str, role: Role) -> dict[str, Any] | None:
        u = self._users.get(user_id)
        if u is None:
            return None
        u.role = role
        return self.user_view(u)

    def deactivate(self, user_id: str) -> dict[str, Any] | None:
        u = self._users.get(user_id)
        if u is None:
            return None
        u.active = False
        self._sessions = {
            k: v for k, v in self._sessions.items() if v.user_id != user_id
        }
        return self.user_view(u)

    @staticmethod
    def user_view(u: User) -> dict[str, Any]:
        return {
            "user_id": u.user_id,
            "username": u.username,
            "display_name": u.display_name,
            "role": u.role.value,
            "active": u.active,
        }

    # -- authentication -------------------------------------------------

    def authenticate(self, username: str, password: str) -> Session:
        u = self._users.get(username)
        if u is None:
            raise AuthError("Invalid username or password.", 401)
        if not u.active:
            raise AuthError("This account is deactivated.", 403)
        if u.locked_until and u.locked_until > time.time():
            raise AuthError(
                f"Account locked after repeated failures. Try again after "
                f"{int(u.locked_until - time.time())} seconds.",
                423,
            )
        if not verify_password(password, u.password_hash):
            u.failed_logins += 1
            if u.failed_logins >= 5:
                u.locked_until = time.time() + 300
                u.failed_logins = 0
            raise AuthError("Invalid username or password.", 401)

        u.failed_logins = 0
        u.last_login = datetime.now(timezone.utc).isoformat()
        sess = Session(
            token=secrets.token_urlsafe(32),
            user_id=u.user_id,
            role=u.role,
            created_at=time.time(),
            expires_at=time.time() + self.s.session_ttl_minutes * 60,
        )
        self._sessions[sess.token] = sess
        return sess

    def session_for(self, token: str | None) -> Session:
        if not token:
            raise AuthError("Authentication required.", 401)
        sess = self._sessions.get(token)
        if sess is None:
            raise AuthError("Session not found. Sign in again.", 401)
        if sess.expires_at < time.time():
            self._sessions.pop(token, None)
            raise AuthError("Session expired. Sign in again.", 401)
        u = self._users.get(sess.user_id)
        if u is None or not u.active:
            self._sessions.pop(token, None)
            raise AuthError("Account is no longer active.", 403)
        return sess

    def revoke(self, token: str) -> None:
        self._sessions.pop(token, None)


# --------------------------------------------------------------------------
# Authorization helpers
# --------------------------------------------------------------------------


def role_allows(user_role: Role, required: Role) -> bool:
    return required in ROLE_HIERARCHY.get(user_role, set())


def require(
    store: AuthStore, token: str | None, allowed: Iterable[Role]
) -> tuple[Session, User]:
    """Validate a bearer token and assert the caller holds an allowed role."""

    sess = store.session_for(token)
    u = store._users.get(sess.user_id)
    if u is None:
        raise AuthError("Account not found.", 401)
    allowed_set = set(allowed)
    if not any(role_allows(u.role, r) for r in allowed_set):
        raise AuthError(
            f"Your role ({u.role.value}) is not permitted to perform this action. "
            f"Required: {', '.join(sorted(r.value for r in allowed_set))}.",
            403,
        )
    return sess, u
