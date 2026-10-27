"""Pydantic request/response contracts shared by all IntelliClinic AI services.

These schemas are the single source of truth for:
  * the FastAPI OpenAPI documentation,
  * the pipeline orchestration layer,
  * the React/TypeScript frontend (``frontend/src/types/api.ts`` is generated
    from ``scripts/generate_frontend_types.py`` reading this module's OpenAPI).

Nothing in this module performs I/O. Every service returns one of these models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------
# Routing primitives
# --------------------------------------------------------------------------


class Intent(str, Enum):
    """The two primary routes Algorithm 1 may select."""

    ENCODE_DATA = "ENCODE_DATA"
    ASK_QUESTION = "ASK_QUESTION"
    AMBIGUOUS = "AMBIGUOUS"


class Confidence(BaseModel):
    """Probability mass over the intent space, plus model provenance."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    confidence: float = Field(..., ge=0.0, le=1.0)
    alternatives: dict[str, float] = Field(default_factory=dict)
    threshold: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    model_version: str
    backend: Literal["real", "stub"] = "real"
    clarification_prompt: str | None = None

    @field_validator("alternatives")
    @classmethod
    def _sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            return v
        total = sum(v.values())
        if total and abs(total - 1.0) > 0.02:
            raise ValueError(f"alternatives must sum to 1.0, got {total}")
        return v


class PipelineStage(str, Enum):
    RECEIVED = "RECEIVED"
    INTENT_CLASSIFIED = "INTENT_CLASSIFIED"
    ENTITIES_EXTRACTED = "ENTITIES_EXTRACTED"
    DOCUMENTS_RETRIEVED = "DOCUMENTS_RETRIEVED"
    VALIDATION_COMPLETED = "VALIDATION_COMPLETED"
    ANSWER_GENERATED = "ANSWER_GENERATED"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    SAVED = "SAVED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


# --------------------------------------------------------------------------
# Algorithm 1 - BERT intent classification
# --------------------------------------------------------------------------


class IntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, max_length=20_000)
    session_id: str | None = None
    """Explicit override for clinical/administrative traffic that must never
    reach the question-answering path (e.g. bulk record repair)."""

    force_route: Literal["ENCODE_DATA", "ASK_QUESTION"] | None = None
    patient_context_id: str | None = None


class IntentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: Confidence
    elapsed_ms: int
    request_id: str
    route_explanation: str


# --------------------------------------------------------------------------
# Algorithm 2 - Bio_ClinicalBERT entity extraction
# --------------------------------------------------------------------------


class EntityType(str, Enum):
    PATIENT_IDENTIFIER = "PATIENT_IDENTIFIER"
    DEMOGRAPHIC = "DEMOGRAPHIC"
    TEMPERATURE = "TEMPERATURE"
    BLOOD_PRESSURE = "BLOOD_PRESSURE"
    PULSE = "PULSE"
    RESPIRATORY_RATE = "RESPIRATORY_RATE"
    OXYGEN_SATURATION = "OXYGEN_SATURATION"
    SYMPTOM = "SYMPTOM"
    DIAGNOSIS = "DIAGNOSIS"
    MEDICATION = "MEDICATION"
    DOSAGE = "DOSAGE"
    ALLERGY = "ALLERGY"
    TEST = "TEST"
    PROCEDURE = "PROCEDURE"
    DATE = "DATE"
    DURATION = "DURATION"


class Assertion(str, Enum):
    """Whether the source text actually asserts this entity as present."""

    PRESENT = "PRESENT"
    NEGATED = "NEGATED"
    UNCERTAIN = "UNCERTAIN"
    HISTORICAL = "HISTORICAL"


class ClinicalEntity(BaseModel):
    """One extracted entity, with offsets for auditability."""

    model_config = ConfigDict(extra="forbid")

    type: EntityType
    text: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    assertion: Assertion = Assertion.PRESENT
    confidence: float = Field(..., ge=0.0, le=1.0)
    value_candidate: float | str | None = None
    unit_candidate: str | None = None
    requires_review: bool = False
    review_reason: str | None = None

    @field_validator("end")
    @classmethod
    def _ordered(cls, v: int, info: Any) -> int:
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("end offset must be >= start offset")
        return v


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=20_000)
    encounter_id: str | None = None


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[ClinicalEntity]
    elapsed_ms: int
    request_id: str
    model_name: str
    model_version: str
    backend: Literal["real", "stub"] = "real"
    raw_bio_tags: list[dict[str, Any]] | None = None


# --------------------------------------------------------------------------
# Algorithm 3 - regex + clinical validation
# --------------------------------------------------------------------------


class Severity(str, Enum):
    PARSE_ERROR = "PARSE_ERROR"
    PLAUSIBILITY_WARNING = "PLAUSIBILITY_WARNING"
    URGENT_ALERT = "URGENT_ALERT"
    # A configuration gap is neither a malformed value nor an abnormal
    # reading. It must not flip the *clinical* verdict to REJECTED - that
    # would conflate "the site has not configured a concept UUID" with
    # "this patient data is wrong" - but it does block submission.
    CONFIG_MISSING = "CONFIG_MISSING"


class ValidationStatus(str, Enum):
    VALID = "VALID"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"
    REJECTED = "REJECTED"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    severity: Severity
    message: str
    raw_value: str | None = None
    """Rejected values are preserved verbatim for authorised review."""


class ObservationField(BaseModel):
    """A normalised observation mapped onto a configured OpenMRS concept."""

    model_config = ConfigDict(extra="forbid")

    concept_uuid: str | None = None
    concept_name: str
    # Scalar observations (vitals) are a single number; symptom, diagnosis,
    # medication and allergy findings are collected as a list of free-text
    # values. A list is a real shape here, not a coercion bug - restricting the
    # union to scalars is what made a symptom list fail validation.
    value: float | int | str | list[str]
    unit: str | None = None
    status: ValidationStatus = ValidationStatus.VALID
    source_entity_text: str | None = None


class ValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[ClinicalEntity]
    patient_age_years: float | None = None
    patient_sex: Literal["M", "F", "U"] = "U"
    encounter_datetime: datetime | None = None


class ValidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[ObservationField]
    issues: list[ValidationIssue]
    # Clinical verdict about the values themselves. Deliberately independent of
    # configuration_missing - an unconfigured concept must not make a correct
    # reading look invalid.
    status: ValidationStatus
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    # Field names whose OpenMRS concept UUID is not configured. Submission is
    # blocked while this is non-empty.
    configuration_missing: list[str] = Field(default_factory=list)
    # True only when the values are clinically acceptable AND every concept is
    # configured. Both gates must pass before a write is attempted.
    can_submit: bool = False
    elapsed_ms: int
    request_id: str


# --------------------------------------------------------------------------
# Algorithm 4 - ChromaDB semantic retrieval
# --------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    title: str
    section: str | None = None
    content: str
    source_version: str
    effective_date: str | None = None
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    approved: bool = True
    review_status: str = "APPROVED"


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=4_000)
    top_k: int = Field(default=5, ge=1, le=50)
    approved_only: bool = True
    topic_filter: str | None = None
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)


class RetrievalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    retrieved_documents: list[RetrievedChunk]
    elapsed_ms: int
    request_id: str
    evidence_status: Literal["SUFFICIENT", "INSUFFICIENT", "CONFLICTING"]
    embedding_model: str
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Algorithm 5 - Llama grounded answering
# --------------------------------------------------------------------------


class GroundingStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNVERIFIED = "UNVERIFIED"
    CONFLICTING = "CONFLICTING"


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    section: str | None = None
    source_version: str
    chunk_id: str
    similarity_score: float | None = None


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=4_000)
    context_chunks: list[RetrievedChunk] = Field(default_factory=list)
    max_tokens: int = Field(default=400, ge=16, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[Citation]
    grounding_status: GroundingStatus
    conflicts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    model_name: str
    model_version: str
    backend: Literal["real", "stub"] = "real"
    elapsed_ms: int
    request_id: str


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


class StageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: PipelineStage
    algorithm: str
    backend: Literal["real", "stub"]
    elapsed_ms: int
    status: Literal["ok", "warning", "error", "skipped"]
    summary: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, max_length=20_000)
    session_id: str | None = None
    patient_id: str | None = None
    encounter_id: str | None = None
    dry_run: bool = True
    """Default True: never writes to OpenMRS without explicit approval."""


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    intent: Confidence
    route: Intent
    stages: list[StageResult]
    extraction: ExtractionResponse | None = None
    validation: ValidationResponse | None = None
    retrieval: RetrievalResponse | None = None
    answer: AnswerResponse | None = None
    requires_confirmation: bool = False
    saved: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ApprovalRequest(BaseModel):
    """Human approval of a pending payload.

    ``payload`` is optional. When omitted, the server uses the payload it stored
    when the message was processed, and applies ``corrections`` on top. That is
    the normal flow: a clinician reviews what the pipeline extracted, edits the
    fields that are wrong, and approves. Supplying a full ``payload`` replaces
    the stored one, which is allowed but must be an explicit choice.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    approved_by: str
    corrections: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] | None = None
    idempotency_key: str = Field(..., min_length=8)


class ApprovalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Literal["SUBMITTED", "DUPLICATE", "FAILED", "UNCONFIGURED"]
    openmrs_encounter_uuid: str | None = None
    openmrs_observation_uuids: list[str] = Field(default_factory=list)
    message: str
    audit_event_id: str | None = None
