"""IntelliClinic AI - FastAPI application entry point.

Run:
    cd backend
    uvicorn app.main:app --reload --port 8000

Interactive OpenAPI docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .algorithms.answering import GroundedAnswerer
from .algorithms.extraction import EntityExtractor
from .algorithms.intent import IntentClassifier
from .algorithms.retrieval import RegistryUnreadableError, RetrievalEngine
from .algorithms.validation import ClinicalValidator
from .auth import AuthError, AuthStore, Role, require
from .config import get_settings, settings_report
from .logging_util import get_logger
from .openmrs import OpenMRSClient
from .pipeline import Pipeline
from .schemas import (
    AnswerRequest,
    AnswerResponse,
    ApprovalRequest,
    ApprovalResponse,
    ChatRequest,
    ChatResponse,
    ExtractionRequest,
    ExtractionResponse,
    IntentRequest,
    IntentResponse,
    RetrievalRequest,
    RetrievalResponse,
    ValidationRequest,
    ValidationResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="IntelliClinic AI",
    description=(
        "Intelligent clinical assistant. Five specialised algorithms connected by "
        "an orchestration layer: BERT intent classification, Bio_ClinicalBERT "
        "entity extraction, deterministic clinical validation, ChromaDB semantic "
        "retrieval, and grounded Llama question answering."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Singletons (constructed once; each loads its own model lazily)
# --------------------------------------------------------------------------

auth_store = AuthStore()
intent_svc = IntentClassifier()
extraction_svc = EntityExtractor()
validation_svc = ClinicalValidator()
retrieval_svc = RetrievalEngine()
answering_svc = GroundedAnswerer()
pipeline = Pipeline()


@app.exception_handler(RegistryUnreadableError)
async def _registry_unreadable(request, exc):
    """Report an unreadable document registry as 503 with the real cause.

    Without this the failure is a generic 500 and the operator has to read the
    server log to learn that the knowledge base itself is broken rather than
    simply empty.
    """

    return JSONResponse(
        status_code=503,
        content={
            "detail": str(exc),
            "kind": "REGISTRY_UNREADABLE",
        },
    )


@app.exception_handler(AuthError)
async def _auth_error_handler(request: Request, exc: AuthError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


def _token(authorization: str | None = Header(default=None)) -> str | None:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return authorization


def _rate_limit(request: Request) -> None:
    """Minimal per-IP limiter. Replace with Redis in production."""

    client_ip = request.client.host if request.client else "unknown"
    bucket = _RATELIMIT.setdefault(client_ip, [])
    now = time.time()
    bucket[:] = [t for t in bucket if now - t < 60]
    if len(bucket) >= get_settings().rate_limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Slow down and retry.",
        )
    bucket.append(now)


_RATELIMIT: dict[str, list[float]] = {}


@app.middleware("http")
async def _correlation(request: Request, call_next):
    request.state.request_id = f"http-{time.time_ns() % 1_000_000_000}"
    request.state.started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Response-Time-Ms"] = str(
        int((time.perf_counter() - request.state.started) * 1000)
    )
    return response


# --------------------------------------------------------------------------
# Health, readiness, system status
# --------------------------------------------------------------------------


@app.get("/health", tags=["System"])
def health() -> dict[str, Any]:
    """Liveness only - does not check dependencies."""

    return {"status": "ok", "service": "intelliclinic-ai", "version": "0.1.0"}


@app.get("/api/v1/system/status", tags=["System"])
def system_status() -> dict[str, Any]:
    """Capability map. The UI uses this to label real vs simulated features."""

    report = settings_report()
    ok, detail = (True, "not checked")
    client = OpenMRSClient(get_settings())
    try:
        ok, detail = client.health()
    finally:
        client.close()
    report["integrations"]["openmrs"]["reachable"] = ok
    report["integrations"]["openmrs"]["detail"] = detail
    report["integrations"]["openmrs"]["base_url"] = get_settings().openmrs_base_url
    return report


@app.get("/api/v1/system/ready", tags=["System"])
def ready() -> dict[str, Any]:
    """Readiness: reports which dependencies are actually usable."""

    client = OpenMRSClient(get_settings())
    try:
        openmrs_ok, openmrs_detail = client.health()
    finally:
        client.close()

    checks = {
        "intent_classifier": intent_svc.backend,
        "entity_extraction": extraction_svc.backend,
        "validation": "real",
        "retrieval": retrieval_svc.backend,
        "answering": answering_svc.backend,
        "openmrs": openmrs_ok,
    }
    ready_flag = all(
        checks[k] in ("real", "sklearn") for k in ("intent_classifier", "validation")
    )
    return {
        "ready": ready_flag,
        "checks": checks,
        "openmrs_detail": openmrs_detail,
        "documents_indexed": len(retrieval_svc.list_documents()),
    }


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


@app.post("/api/v1/auth/login", tags=["Auth"])
def login(body: dict[str, str]) -> dict[str, Any]:
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    if not username or not password:
        raise HTTPException(status_code=422, detail="username and password are required")
    sess = auth_store.authenticate(username, password)
    user = auth_store._users[sess.user_id]
    return {
        "token": sess.token,
        "expires_at": sess.expires_at,
        "user": auth_store.user_view(user),
    }


@app.post("/api/v1/auth/logout", tags=["Auth"])
def logout(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = _token(authorization)
    if token:
        auth_store.revoke(token)
    return {"status": "signed_out"}


@app.get("/api/v1/auth/me", tags=["Auth"])
def me(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    sess, user = require(auth_store, _token(authorization), [Role.NURSE])
    return {
        "user": auth_store.user_view(user),
        "session_expires_at": sess.expires_at,
        "server_time": time.time(),
    }


# --------------------------------------------------------------------------
# Algorithm 1 - intent
# --------------------------------------------------------------------------


@app.post("/api/v1/intent", response_model=IntentResponse, tags=["Algorithm 1"])
def classify_intent(
    req: IntentRequest, authorization: str | None = Header(default=None)
) -> IntentResponse:
    require(auth_store, _token(authorization), [Role.NURSE])
    return intent_svc.classify(req)


# --------------------------------------------------------------------------
# Algorithm 2 - extraction
# --------------------------------------------------------------------------


@app.post("/api/v1/extract", response_model=ExtractionResponse, tags=["Algorithm 2"])
def extract_entities(
    req: ExtractionRequest, authorization: str | None = Header(default=None)
) -> ExtractionResponse:
    require(auth_store, _token(authorization), [Role.NURSE])
    return extraction_svc.extract(req)


# --------------------------------------------------------------------------
# Algorithm 3 - validation
# --------------------------------------------------------------------------


@app.post("/api/v1/validate", response_model=ValidationResponse, tags=["Algorithm 3"])
def validate_observations(
    req: ValidationRequest, authorization: str | None = Header(default=None)
) -> ValidationResponse:
    require(auth_store, _token(authorization), [Role.NURSE])
    return validation_svc.validate(req)


# --------------------------------------------------------------------------
# Algorithm 4 - retrieval
# --------------------------------------------------------------------------


@app.post("/api/v1/retrieve", response_model=RetrievalResponse, tags=["Algorithm 4"])
def retrieve(
    req: RetrievalRequest, authorization: str | None = Header(default=None)
) -> RetrievalResponse:
    require(auth_store, _token(authorization), [Role.NURSE])
    return retrieval_svc.retrieve(req)


# --------------------------------------------------------------------------
# Algorithm 5 - answering
# --------------------------------------------------------------------------


@app.post("/api/v1/answer", response_model=AnswerResponse, tags=["Algorithm 5"])
def answer(
    req: AnswerRequest, authorization: str | None = Header(default=None)
) -> AnswerResponse:
    require(auth_store, _token(authorization), [Role.NURSE])
    return answering_svc.answer(req)


# --------------------------------------------------------------------------
# Pipeline orchestration
# --------------------------------------------------------------------------


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Pipeline"])
def chat(
    req: ChatRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> ChatResponse:
    _rate_limit(request)
    sess, user = require(auth_store, _token(authorization), [Role.NURSE])
    return pipeline.process(req, actor=user.user_id)


@app.get("/api/v1/review-queue", tags=["Pipeline"])
def review_queue(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    sess, user = require(auth_store, _token(authorization), [Role.NURSE])
    return {
        "awaiting_review": [
            {
                "request_id": rid,
                "created_at": rec["created_at"],
                "created_by": rec["created_by"],
                "observation_count": len(rec.get("observations", [])),
                "validation_status": rec.get("payload", {}).get("validation_status"),
                "openmrs_configured": get_settings().openmrs_configured,
            }
            for rid, rec in pipeline._approvals.items()
        ],
        "note": (
            "AI-extracted observations wait here until a clinician approves them. "
            "Nothing in this queue has been written to OpenMRS."
        ),
    }


@app.get("/api/v1/review-queue/{request_id}", tags=["Pipeline"])
def review_item(
    request_id: str, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    require(auth_store, _token(authorization), [Role.NURSE])
    rec = pipeline.pending_payload(request_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="No pending payload for that request_id")
    return rec


@app.post("/api/v1/review-queue/{request_id}/approve", response_model=ApprovalResponse, tags=["Pipeline"])
def approve(
    request_id: str,
    req: ApprovalRequest,
    authorization: str | None = Header(default=None),
) -> ApprovalResponse:
    """Human approval gate. Only DOCTOR and ADMINISTRATOR may submit to OpenMRS."""

    sess, user = require(auth_store, _token(authorization), [Role.DOCTOR, Role.ADMINISTRATOR])
    if req.request_id != request_id:
        raise HTTPException(
            status_code=422,
            detail="request_id in the body must match the URL path.",
        )
    return pipeline.approve(req, actor=user.user_id)


# --------------------------------------------------------------------------
# Knowledge base (Algorithm 4 administration)
# --------------------------------------------------------------------------


@app.get("/api/v1/knowledge-base/documents", tags=["Knowledge Base"])
def list_documents(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require(auth_store, _token(authorization), [Role.NURSE])
    docs = retrieval_svc.list_documents()
    return {
        "documents": docs,
        "collection": get_settings().chroma_collection,
        "embedding_model": get_settings().embedding_model,
        "indexing_status": "ready" if retrieval_svc.ready else "unavailable",
    }


@app.post("/api/v1/knowledge-base/documents", tags=["Knowledge Base"])
def index_document(
    body: dict[str, Any], authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    require(auth_store, _token(authorization), [Role.DOCTOR, Role.ADMINISTRATOR])
    required = ["document_id", "title", "source_version", "text"]
    missing = [k for k in required if not body.get(k)]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required fields: {', '.join(missing)}",
        )
    try:
        result = retrieval_svc.index_document(
            document_id=str(body["document_id"]),
            title=str(body["title"]),
            source_version=str(body["source_version"]),
            effective_date=body.get("effective_date"),
            review_status=str(body.get("review_status", "PENDING_REVIEW")),
            topic=str(body.get("topic", "general")),
            text=str(body["text"]),
            uploaded_by=(auth_store._users.get(_who(authorization)) or {}).display_name
            if _who(authorization)
            else None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result


@app.delete("/api/v1/knowledge-base/documents/{document_id}", tags=["Knowledge Base"])
def delete_document(
    document_id: str, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    require(auth_store, _token(authorization), [Role.ADMINISTRATOR])
    try:
        return retrieval_svc.remove_document(document_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/v1/knowledge-base/documents/{document_id}/status", tags=["Knowledge Base"])
def set_document_status(
    document_id: str,
    body: dict[str, str],
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    require(auth_store, _token(authorization), [Role.DOCTOR, Role.ADMINISTRATOR])
    status = str(body.get("review_status") or "").upper()
    if status not in ("APPROVED", "PENDING_REVIEW", "DISABLED"):
        raise HTTPException(
            status_code=422,
            detail="review_status must be APPROVED, PENDING_REVIEW or DISABLED",
        )
    try:
        return retrieval_svc.set_review_status(document_id, status)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/v1/knowledge-base/test-retrieval", tags=["Knowledge Base"])
def test_retrieval(
    body: dict[str, Any], authorization: str | None = Header(default=None)
) -> RetrievalResponse:
    require(auth_store, _token(authorization), [Role.DOCTOR, Role.ADMINISTRATOR])
    return retrieval_svc.retrieve(
        RetrievalRequest(
            query=str(body.get("query") or ""),
            top_k=int(body.get("top_k", 5)),
            approved_only=bool(body.get("approved_only", True)),
            topic_filter=body.get("topic_filter"),
            min_score=body.get("min_score"),
        )
    )


def _who(authorization: str | None) -> str | None:
    token = _token(authorization)
    if not token:
        return None
    sess = auth_store._sessions.get(token)
    return sess.user_id if sess else None


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------


@app.get("/api/v1/audit-logs", tags=["Audit"])
def audit_logs(
    limit: int = Query(default=100, ge=1, le=500),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    require(auth_store, _token(authorization), [Role.ADMINISTRATOR])
    path = get_settings().data_dir / "pipeline_runs.jsonl"
    rows: list[dict[str, Any]] = []
    unreadable = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                row = json.loads(line)
            except Exception:
                # A skipped line is a run the operator cannot see. Counted so the
                # gap is visible instead of invisible; the alternative - stopping
                # at the first bad line - would hide every run after it.
                unreadable += 1
                continue
            if not isinstance(row, dict):
                # Valid JSON of the wrong shape is still not a run record. Taking
                # it as one would put a non-record into the audit trail.
                unreadable += 1
                continue
            rows.append(row)
    if unreadable:
        get_logger("main").warning(
            "%d of the last %d pipeline-run record(s) in %s are corrupt and "
            "were skipped; the audit trail is incomplete",
            unreadable,
            limit,
            path,
        )
    return {
        "events": list(reversed(rows)),
        "count": len(rows),
        "unreadable_records": unreadable,
        "patient_identifiable_content_stored": False,
        "note": (
            "Pipeline run records store routes, entity types, model backends, and "
            "document titles only. Free-text patient notes are never written to "
            "this log."
        ),
    }


# --------------------------------------------------------------------------
# Users and roles
# --------------------------------------------------------------------------


@app.get("/api/v1/users", tags=["Administration"])
def list_users(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require(auth_store, _token(authorization), [Role.ADMINISTRATOR])
    return {"users": auth_store.users()}


@app.patch("/api/v1/users/{user_id}/role", tags=["Administration"])
def update_role(
    user_id: str,
    body: dict[str, str],
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    require(auth_store, _token(authorization), [Role.ADMINISTRATOR])
    role_raw = str(body.get("role") or "").upper()
    try:
        role = Role(role_raw)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"role must be one of {', '.join(r.value for r in Role)}",
        )
    updated = auth_store.set_role(user_id, role)
    if updated is None:
        raise HTTPException(status_code=404, detail="user not found")
    return {"user": updated}


@app.post("/api/v1/users/{user_id}/deactivate", tags=["Administration"])
def deactivate_user(
    user_id: str, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    require(auth_store, _token(authorization), [Role.ADMINISTRATOR])
    updated = auth_store.deactivate(user_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="user not found")
    return {"user": updated}


# --------------------------------------------------------------------------
# Patient records (OpenMRS-backed)
# --------------------------------------------------------------------------


@app.get("/api/v1/patients", tags=["Patients"])
def search_patients(
    q: str = Query(default="", min_length=1, max_length=100),
    limit: int = Query(default=20, ge=1, le=100),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    require(auth_store, _token(authorization), [Role.NURSE])
    client = OpenMRSClient(get_settings())
    try:
        if not get_settings().openmrs_configured:
            return {
                "results": [],
                "unavailable": True,
                "detail": (
                    "OpenMRS is not configured on this server, so no live patient "
                    "search is possible. This is an explicit unavailable state, "
                    "not an empty result."
                ),
            }
        r = client._http().get(
            "/ws/rest/v1/patient",
            params={"q": q, "limit": limit, "v": "default"},
        )
        if r.status_code >= 400:
            return {
                "results": [],
                "unavailable": True,
                "detail": f"OpenMRS returned HTTP {r.status_code}.",
            }
        results = r.json().get("results", [])
        return {
            "results": [
                {
                    "uuid": p.get("uuid"),
                    "display": p.get("display"),
                    "identifiers": [
                        {
                            "identifier": i.get("identifier"),
                            "type": (i.get("identifierType") or {}).get("display"),
                        }
                        for i in (p.get("identifiers") or [])
                    ],
                }
                for p in results
            ],
            "unavailable": False,
            "detail": "Live results from OpenMRS.",
        }
    finally:
        client.close()


@app.get("/api/v1/patients/{uuid}", tags=["Patients"])
def patient_detail(
    uuid: str, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    require(auth_store, _token(authorization), [Role.NURSE])
    client = OpenMRSClient(get_settings())
    try:
        if not get_settings().openmrs_configured:
            return {
                "unavailable": True,
                "detail": "OpenMRS is not configured, so patient details are unavailable.",
            }
        r = client._http().get(f"/ws/rest/v1/patient/{uuid}", params={"v": "full"})
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="patient not found in OpenMRS")
        if r.status_code >= 400:
            return {"unavailable": True, "detail": f"OpenMRS returned HTTP {r.status_code}."}
        p = r.json()
        enc = (
            client._http()
            .get(f"/ws/rest/v1/patient/{uuid}/encounter", params={"v": "default"})
            .json()
            .get("results", [])
        )
        return {
            "unavailable": False,
            "patient": {
                "uuid": p.get("uuid"),
                "display": p.get("display"),
                "person": p.get("person"),
                "identifiers": p.get("identifiers"),
            },
            "encounters": [
                {
                    "uuid": e.get("uuid"),
                    "encounterType": (e.get("encounterType") or {}).get("display"),
                    "encounterDatetime": e.get("encounterDatetime"),
                }
                for e in enc
            ][:50],
        }
    finally:
        client.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
