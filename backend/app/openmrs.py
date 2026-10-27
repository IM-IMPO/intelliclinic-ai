"""OpenMRS REST integration.

All clinical writes go through the OpenMRS REST API. Nothing here touches
OpenMRS's MySQL tables.

Safety properties:

* **Idempotency.** Every write carries an ``Idempotency-Key`` header derived
  from a caller-supplied key, and submitted payloads are fingerprint-stored so
  a retried approval returns the original result instead of creating a second
  encounter. Duplicate submissions are the most common real-world cause of
  duplicated clinical data, so this is not optional.
* **Fail closed.** If OpenMRS is not configured, the caller gets
  ``UNCONFIGURED`` and an explicit message - never a silent success.
* **Least privilege.** Credentials come from the server environment only and
  are never echoed into responses, logs, or error text.
* **No success without confirmation.** The write is only reported as saved once
  OpenMRS returns an encounter UUID.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .logging_util import get_logger
from .state_util import read_json_state
from .schemas import ApprovalResponse


class OpenMRSClient:
    """Thin, typed wrapper over the OpenMRS REST resources we use."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.base = (settings.openmrs_base_url or "").rstrip("/")
        self._client: httpx.Client | None = None

    @property
    def configured(self) -> bool:
        return bool(self.base and self.s.openmrs_username and self.s.openmrs_password)

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base,
                auth=(self.s.openmrs_username or "", self.s.openmrs_password or ""),
                timeout=self.s.openmrs_timeout_s,
                headers={"Accept": "application/json"},
            )
        return self._client

    def health(self) -> tuple[bool, str]:
        if not self.configured:
            return False, "OpenMRS is not configured (set OPENMRS_BASE_URL, OPENMRS_USERNAME, OPENMRS_PASSWORD)."
        try:
            r = self._http().get("/ws/rest/v1/session")
            if r.status_code == 200:
                return True, "Authenticated session confirmed."
            if r.status_code in (401, 403):
                return False, f"Authentication failed (HTTP {r.status_code}). Check OPENMRS credentials."
            return False, f"Unexpected response HTTP {r.status_code}."
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def find_patient(self, identifier: str) -> dict[str, Any] | None:
        """Look up a patient by identifier. Returns None when not found."""

        if not self.configured:
            raise RuntimeError("OpenMRS not configured")
        r = self._http().get(
            "/ws/rest/v1/patient",
            params={"identifier": identifier, "v": "full"},
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0] if results else None

    def resolve_patient_uuid(self, identifier: str) -> str:
        """Resolve a patient identifier to an OpenMRS UUID, refusing to guess.

        A patient identifier typed by a clinician is not a UUID, so it must be
        resolved against OpenMRS before anything can be written. OpenMRS may
        return several patients for one identifier string; picking the first
        would silently attach clinical data to the wrong person, so ambiguity
        is raised rather than resolved.
        """

        if not self.configured:
            raise OpenMRSWriteError(
                "OpenMRS is not configured, so patient identifier "
                f"'{identifier}' cannot be resolved."
            )
        r = self._http().get(
            "/ws/rest/v1/patient",
            params={"identifier": identifier, "v": "full"},
        )
        if r.status_code == 404:
            raise OpenMRSWriteError(
                f"No patient in OpenMRS matches identifier '{identifier}'."
            )
        r.raise_for_status()
        results = r.json().get("results") or []
        if not results:
            raise OpenMRSWriteError(
                f"No patient in OpenMRS matches identifier '{identifier}'."
            )
        if len(results) > 1:
            raise OpenMRSWriteError(
                f"Identifier '{identifier}' matches {len(results)} patients in "
                "OpenMRS. Refusing to choose between them; use a unique identifier."
            )
        uuid = results[0].get("uuid")
        if not uuid:
            raise OpenMRSWriteError(
                f"Patient matched by identifier '{identifier}' returned no uuid."
            )
        return str(uuid)

    def resolve_concept(self, uuid: str) -> bool:
        """Confirm a concept UUID exists on this instance before writing."""

        if not self.configured:
            return False
        r = self._http().get(f"/ws/rest/v1/concept/{uuid}")
        return r.status_code == 200

    def create_encounter(self, body: dict[str, Any], idempotency_key: str) -> str:
        """POST an encounter and return its UUID. Raises on failure."""

        r = self._http().post(
            "/ws/rest/v1/encounter",
            json=body,
            headers={"Idempotency-Key": idempotency_key},
        )
        if r.status_code not in (200, 201):
            raise OpenMRSWriteError(
                f"Encounter creation failed: HTTP {r.status_code} - {r.text[:400]}"
            )
        uuid = r.json().get("uuid")
        if not uuid:
            raise OpenMRSWriteError("OpenMRS returned no encounter uuid.")
        return str(uuid)

    def create_observation(self, body: dict[str, Any], idempotency_key: str) -> str:
        r = self._http().post(
            "/ws/rest/v1/obs",
            json=body,
            headers={"Idempotency-Key": idempotency_key},
        )
        if r.status_code not in (200, 201):
            raise OpenMRSWriteError(
                f"Observation creation failed: HTTP {r.status_code} - {r.text[:400]}"
            )
        uuid = r.json().get("uuid")
        if not uuid:
            raise OpenMRSWriteError("OpenMRS returned no obs uuid.")
        return str(uuid)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class OpenMRSWriteError(RuntimeError):
    pass


class IdempotencyStoreError(RuntimeError):
    """The idempotency store could not be read, so no write may proceed."""


# --------------------------------------------------------------------------
# Idempotency store
# --------------------------------------------------------------------------


def _store_path(settings: Settings) -> Path:
    return settings.data_dir / "idempotency.json"


def _load_store(settings: Settings) -> dict[str, Any]:
    p = _store_path(settings)
    if not p.exists():
        return {}

    def _error(path: Path, detail: str) -> IdempotencyStoreError:
        return IdempotencyStoreError(
            f"The idempotency store {path} is unusable ({detail}). No write has "
            f"been attempted: continuing without it could create duplicate "
            f"encounters in OpenMRS. Restore the file from backup or delete it "
            f"deliberately and re-approve."
        )

    # The shape is checked, not just the syntax: a file that parses to a list or
    # a string is no more a store than a truncated one, and returning it would
    # let a re-submitted approval through as though it were the first.
    return read_json_state(
        p,
        is_valid=lambda v: isinstance(v, dict) and all(isinstance(k, str) for k in v),
        on_error=_error,
        logger_name="openmrs",
        what="a JSON object of idempotency keys",
    )


def _save_store(settings: Settings, data: dict[str, Any]) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    _store_path(settings).write_text(json.dumps(data, indent=2), encoding="utf-8")


def submit_approved_observations(
    *,
    payload: dict[str, Any],
    approved_by: str,
    idempotency_key: str,
    settings: Settings,
) -> ApprovalResponse:
    """Write an approved payload to OpenMRS, idempotently.

    This function is only ever called after an explicit human approval. It is
    never invoked by the extraction or validation stages.
    """

    if not settings.openmrs_configured:
        return ApprovalResponse(
            request_id=str(payload.get("request_id") or ""),
            status="UNCONFIGURED",
            message=(
                "Refusing to write: OpenMRS is not configured. Set "
                "OPENMRS_BASE_URL, OPENMRS_USERNAME and OPENMRS_PASSWORD on the "
                "server. No partial data was sent."
            ),
        )

    # --- idempotency gate ---
    try:
        store = _load_store(settings)
    except IdempotencyStoreError as exc:
        # Converted here rather than at the API boundary so the approval
        # contract stays intact: the caller always receives an
        # ApprovalResponse, never an unhandled exception.
        return ApprovalResponse(
            request_id=str(payload.get("request_id") or ""),
            status="FAILED",
            message=str(exc),
        )
    fingerprint = hashlib.sha256(
        json.dumps(
            {"payload": payload, "approved_by": approved_by}, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()

    existing = store.get(idempotency_key)
    if existing:
        if existing.get("fingerprint") == fingerprint:
            return ApprovalResponse(
                request_id=str(payload.get("request_id") or ""),
                status="DUPLICATE",
                openmrs_encounter_uuid=existing.get("encounter_uuid"),
                openmrs_observation_uuids=existing.get("observation_uuids", []),
                message=(
                    "This exact payload was already submitted with this "
                    "idempotency key. The original result is returned; no "
                    "duplicate encounter was created."
                ),
                audit_event_id=existing.get("audit_event_id"),
            )

    client = OpenMRSClient(settings)
    try:
        ok, detail = client.health()
        if not ok:
            return ApprovalResponse(
                request_id=str(payload.get("request_id") or ""),
                status="FAILED",
                message=f"OpenMRS is not reachable: {detail} Nothing was written.",
            )

        patient_uuid = str(payload.get("patient_uuid") or "")
        if not patient_uuid:
            # A clinician enters an identifier, not a UUID, so it has to be
            # resolved against OpenMRS. If it cannot be resolved the write is
            # refused rather than attached to an unknown patient.
            identifier = str(payload.get("patient_identifier") or "").strip()
            if not identifier:
                return ApprovalResponse(
                    request_id=str(payload.get("request_id") or ""),
                    status="FAILED",
                    message=(
                        "The approved payload has no patient target. Provide "
                        "patient_uuid, or patient_identifier so the patient can "
                        "be resolved against OpenMRS. Nothing was written."
                    ),
                )
            patient_uuid = client.resolve_patient_uuid(identifier)

        encounter_uuid = client.create_encounter(
            body={
                "patient": patient_uuid,
                "encounterType": payload.get("encounter_type_uuid"),
                "visit": payload.get("visit_uuid"),
                "encounterDatetime": payload.get("encounter_datetime")
                or datetime.now(timezone.utc).isoformat(),
                "location": payload.get("location_uuid"),
            },
            idempotency_key=idempotency_key,
        )

        obs_uuids: list[str] = []
        for obs in payload.get("observations", []):
            if not isinstance(obs, dict):
                continue
            concept_uuid = obs.get("concept_uuid")
            if not concept_uuid:
                # Unconfigured concept: there is nothing safe to write, and
                # guessing a UUID is forbidden.
                continue
            if not client.resolve_concept(concept_uuid):
                raise OpenMRSWriteError(
                    f"Concept {concept_uuid} does not exist on this OpenMRS "
                    "instance. The observation was not written."
                )
            obs_uuids.append(
                client.create_observation(
                    body={
                        "concept": concept_uuid,
                        "person": patient_uuid,
                        "encounter": encounter_uuid,
                        "obsDatetime": payload.get("encounter_datetime")
                        or datetime.now(timezone.utc).isoformat(),
                        "value": obs.get("value"),
                        "comment": f"IntelliClinic AI - approved by {approved_by}",
                    },
                    idempotency_key=f"{idempotency_key}:{concept_uuid}",
                )
            )

        store[idempotency_key] = {
            "fingerprint": fingerprint,
            "encounter_uuid": encounter_uuid,
            "observation_uuids": obs_uuids,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "approved_by": approved_by,
        }
        _save_store(settings, store)

        return ApprovalResponse(
            request_id=str(payload.get("request_id") or ""),
            status="SUBMITTED",
            openmrs_encounter_uuid=encounter_uuid,
            openmrs_observation_uuids=obs_uuids,
            message=(
                f"OpenMRS confirmed the write: encounter {encounter_uuid} with "
                f"{len(obs_uuids)} observation(s)."
            ),
            audit_event_id=f"audit-{idempotency_key[:16]}",
        )
    except OpenMRSWriteError as exc:
        return ApprovalResponse(
            request_id=str(payload.get("request_id") or ""),
            status="FAILED",
            message=f"OpenMRS rejected the write: {exc}",
        )
    except Exception as exc:
        return ApprovalResponse(
            request_id=str(payload.get("request_id") or ""),
            status="FAILED",
            message=f"Unexpected error during submission: {type(exc).__name__}: {exc}",
        )
    finally:
        client.close()
