"""Pipeline orchestration.

Routes each message down exactly one of the two supported paths and records
every stage. It never runs all five algorithms for one message, and it never
writes to OpenMRS as a side effect of extraction.

DATA ENTRY
    message -> Algorithm 1 -> Algorithm 2 -> Algorithm 3 -> AWAITING_CONFIRMATION
                                                                       |
                                            (explicit human approval) |
                                                                       v
                                                        OpenMRS REST API

QUESTION ANSWERING
    message -> Algorithm 1 -> Algorithm 4 -> Algorithm 5 -> cited answer
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .algorithms.answering import GroundedAnswerer
from .algorithms.extraction import EntityExtractor
from .algorithms.intent import IntentClassifier
from .algorithms.retrieval import RetrievalEngine
from .algorithms.validation import ClinicalValidator
from .config import Settings, get_settings
from .schemas import (
    AnswerRequest as AnswerReq,
    ChatRequest,
    ChatResponse,
    ExtractionRequest as ExtractionReq,
    Intent,
    IntentRequest,
    PipelineStage,
    RetrievalRequest as RetrievalReq,
    StageResult,
    ValidationRequest,
    ApprovalRequest,
    ApprovalResponse,
)


@dataclass
class _Ledger:
    """Append-only record of pipeline runs, stored outside OpenMRS's database."""

    path: Path
    entries: list[dict[str, Any]] = field(default_factory=list)


class Pipeline:
    """Coordinates the five algorithm services into the two routes."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.intent = IntentClassifier(self.s)
        self.extraction = EntityExtractor(self.s)
        self.validation = ClinicalValidator(self.s)
        self.retrieval = RetrievalEngine(self.s)
        self.answering = GroundedAnswerer(self.s)
        self._ledger_path = self.s.data_dir / "pipeline_runs.jsonl"
        self._approvals: dict[str, dict[str, Any]] = {}
        self._load_approvals()

    # ------------------------------------------------------------------

    def process(self, req: ChatRequest, actor: str = "unknown") -> ChatResponse:
        started = time.perf_counter()
        rid = f"req-{time.time_ns() % 1_000_000_000}"
        stages: list[StageResult] = []

        # --- Stage 0: received ---
        stages.append(
            StageResult(
                stage=PipelineStage.RECEIVED,
                algorithm="orchestrator",
                backend="real",
                elapsed_ms=0,
                status="ok",
                summary=(
                    f"Message received ({len(req.message)} chars). "
                    "No patient data has been transmitted to any external service."
                ),
                detail={"dry_run": req.dry_run, "actor": actor},
            )
        )

        # --- Stage 1: Algorithm 1 ---
        intent_resp = self.intent.classify(
            IntentRequest(
                message=req.message,
                session_id=req.session_id,
                patient_context_id=req.patient_id,
            )
        )
        stages.append(
            StageResult(
                stage=PipelineStage.INTENT_CLASSIFIED,
                algorithm="Algorithm 1 - BERT intent classification",
                backend=intent_resp.result.backend,
                elapsed_ms=intent_resp.elapsed_ms,
                status=(
                    "ok"
                    if intent_resp.result.intent is not Intent.AMBIGUOUS
                    else "warning"
                ),
                summary=(
                    f"Route {intent_resp.result.intent.value} "
                    f"(confidence {intent_resp.result.confidence:.2f}, "
                    f"threshold {intent_resp.result.threshold:.2f})."
                ),
                detail={
                    "alternatives": intent_resp.result.alternatives,
                    "model": intent_resp.result.model_name,
                    "explanation": intent_resp.route_explanation,
                    "clarification": intent_resp.result.clarification_prompt,
                },
            )
        )

        if intent_resp.result.intent is Intent.AMBIGUOUS:
            stages.append(
                StageResult(
                    stage=PipelineStage.AWAITING_CONFIRMATION,
                    algorithm="orchestrator",
                    backend="real",
                    elapsed_ms=0,
                    status="warning",
                    summary=(
                        "Intent was ambiguous. Nothing was extracted, retrieved, "
                        "or saved. The message is retained and the user is asked "
                        "to clarify."
                    ),
                    detail={"clarification": intent_resp.result.clarification_prompt},
                )
            )
            return self._finish(rid, intent_resp, stages, started, clarification=True)

        if intent_resp.result.intent is Intent.ENCODE_DATA:
            return self._run_data_entry(req, rid, started, stages, intent_resp, actor)
        return self._run_question(req, rid, started, stages, intent_resp)

    # -- route A: data entry ----------------------------------------------

    def _run_data_entry(
        self,
        req: ChatRequest,
        rid: str,
        started: float,
        stages: list[StageResult],
        intent_resp: Any,
        actor: str,
    ) -> ChatResponse:
        extraction = self.extraction.extract(
            ExtractionReq(text=req.message, encounter_id=req.encounter_id)
        )
        stages.append(
            StageResult(
                stage=PipelineStage.ENTITIES_EXTRACTED,
                algorithm="Algorithm 2 - Bio_ClinicalBERT entity extraction",
                backend=extraction.backend,
                elapsed_ms=extraction.elapsed_ms,
                status="ok" if extraction.entities else "warning",
                summary=(
                    f"Extracted {len(extraction.entities)} entit"
                    f"{'y' if len(extraction.entities) == 1 else 'ies'}"
                    + (
                        f", {sum(1 for e in extraction.entities if e.requires_review)} flagged for review."
                        if extraction.entities
                        else ". No clinical entities were found in the text."
                    )
                ),
                detail={
                    "entity_types": sorted({e.type.value for e in extraction.entities}),
                    "requires_review": [
                        e.text for e in extraction.entities if e.requires_review
                    ],
                    "model": extraction.model_name,
                },
            )
        )

        validation = self.validation.validate(
            ValidationRequest(entities=extraction.entities)
        )
        stages.append(
            StageResult(
                stage=PipelineStage.VALIDATION_COMPLETED,
                algorithm="Algorithm 3 - regex and clinical validation",
                backend="real",
                elapsed_ms=validation.elapsed_ms,
                status=(
                    "error"
                    if validation.status.value == "REJECTED"
                    else ("warning" if validation.status.value == "REQUIRES_REVIEW" else "ok")
                ),
                summary=(
                    f"{len(validation.observations)} observation(s) prepared, "
                    f"{len(validation.issues)} issue(s), status "
                    f"{validation.status.value}. Awaiting human confirmation."
                ),
                detail={
                    "issues": [
                        {"field": i.field, "severity": i.severity.value, "message": i.message}
                        for i in validation.issues
                    ],
                    "rejected": validation.rejected,
                },
            )
        )

        # Stored for the later explicit approval step. Never auto-submitted.
        # The clinician-entered patient identifier is carried through so the
        # approval step can resolve it to an OpenMRS UUID; without a patient
        # target the write has nothing safe to attach to and must be refused.
        approval_payload = dict(validation.payload)
        if req.patient_id:
            approval_payload["patient_identifier"] = req.patient_id

        self._approvals[rid] = {
            "payload": approval_payload,
            "observations": [o.model_dump(mode="json") for o in validation.observations],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": actor,
            "request_id": rid,
        }

        stages.append(
            StageResult(
                stage=PipelineStage.AWAITING_CONFIRMATION,
                algorithm="orchestrator",
                backend="real",
                elapsed_ms=0,
                status="warning",
                summary=(
                    "Awaiting staff confirmation. Nothing has been written to "
                    "OpenMRS. AI-extracted values are never saved automatically."
                ),
                detail={
                    "observation_count": len(validation.observations),
                    "openmrs_configured": self.s.openmrs_configured,
                },
            )
        )

        return self._finish(
            rid, intent_resp, stages, started, extraction=extraction, validation=validation
        )

    # -- route B: question answering --------------------------------------

    def _run_question(
        self,
        req: ChatRequest,
        rid: str,
        started: float,
        stages: list[StageResult],
        intent_resp: Any,
    ) -> ChatResponse:
        retrieval = self.retrieval.retrieve(
            RetrievalReq(query=req.message, top_k=5, approved_only=True)
        )
        stages.append(
            StageResult(
                stage=PipelineStage.DOCUMENTS_RETRIEVED,
                algorithm="Algorithm 4 - ChromaDB semantic retrieval",
                backend=self.retrieval.backend,
                elapsed_ms=retrieval.elapsed_ms,
                status=(
                    "ok"
                    if retrieval.evidence_status == "SUFFICIENT"
                    else "warning"
                ),
                summary=(
                    f"{len(retrieval.retrieved_documents)} passage(s) retrieved "
                    f"from approved documents; evidence status "
                    f"{retrieval.evidence_status}."
                ),
                detail={
                    "evidence_status": retrieval.evidence_status,
                    "notes": retrieval.notes,
                    "documents": sorted(
                        {d.title for d in retrieval.retrieved_documents}
                    ),
                },
            )
        )

        answer = self.answering.answer(
            AnswerReq(
                question=req.message,
                context_chunks=retrieval.retrieved_documents,
            )
        )
        stages.append(
            StageResult(
                stage=PipelineStage.ANSWER_GENERATED,
                algorithm="Algorithm 5 - Llama grounded answering",
                backend=answer.backend,
                elapsed_ms=answer.elapsed_ms,
                status=(
                    "ok"
                    if answer.grounding_status.value == "SUPPORTED"
                    else "warning"
                ),
                summary=(
                    f"{len(answer.sources)} source(s) cited; grounding status "
                    f"{answer.grounding_status.value}."
                ),
                detail={
                    "grounding_status": answer.grounding_status.value,
                    "conflicts": answer.conflicts,
                    "limitations": answer.limitations,
                    "model": answer.model_name,
                },
            )
        )

        return self._finish(
            rid, intent_resp, stages, started, retrieval=retrieval, answer=answer
        )

    # -- completion -------------------------------------------------------

    def _finish(
        self,
        rid: str,
        intent_resp: Any,
        stages: list[StageResult],
        started: float,
        extraction: Any = None,
        validation: Any = None,
        retrieval: Any = None,
        answer: Any = None,
        clarification: bool = False,
    ) -> ChatResponse:
        total = int((time.perf_counter() - started) * 1000)
        resp = ChatResponse(
            request_id=rid,
            intent=intent_resp.result,
            route=intent_resp.result.intent,
            stages=stages,
            extraction=extraction,
            validation=validation,
            retrieval=retrieval,
            answer=answer,
            requires_confirmation=(
                clarification
                or (validation is not None and not clarification)
            ),
            saved=False,
        )
        self._record(resp, total)
        return resp

    def _record(self, resp: ChatResponse, total_ms: int) -> None:
        """Persist a run record. Patient-identifiable text is NOT stored."""

        self.s.data_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "request_id": resp.request_id,
            "at": datetime.now(timezone.utc).isoformat(),
            "route": resp.route.value,
            "confidence": resp.intent.confidence,
            "total_ms": total_ms,
            "saved": resp.saved,
            "backends": {s.algorithm: s.backend for s in resp.stages},
            "entity_types": (
                sorted({e.type.value for e in resp.extraction.entities})
                if resp.extraction
                else []
            ),
            "validation_status": resp.validation.status.value if resp.validation else None,
            "evidence_status": resp.retrieval.evidence_status if resp.retrieval else None,
            "grounding_status": resp.answer.grounding_status.value if resp.answer else None,
            "document_titles": (
                sorted({c.title for c in resp.retrieval.retrieved_documents})
                if resp.retrieval
                else []
            ),
        }
        with self._ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    # -- approval ---------------------------------------------------------

    def pending_payload(self, request_id: str) -> dict[str, Any] | None:
        return self._approvals.get(request_id)

    def approve(self, req: ApprovalRequest, actor: str) -> ApprovalResponse:
        """Submit a human-approved payload. Idempotent via idempotency_key."""

        from .openmrs import submit_approved_observations

        rec = self._approvals.get(req.request_id)
        if rec is None:
            return ApprovalResponse(
                request_id=req.request_id,
                status="FAILED",
                message=(
                    "No pending payload for this request_id. It may have expired, "
                    "already been submitted, or never existed. Nothing was written."
                ),
            )

        if req.payload is not None:
            payload = dict(req.payload)
        else:
            payload = dict(rec["payload"])
        payload.update(req.corrections)
        # The approved observations are stored alongside the payload, not inside
        # it. They have to be attached here, or the write path receives an
        # empty list, creates an encounter with no observations, and reports
        # success while the clinical data is silently dropped.
        payload["observations"] = rec.get("observations", [])

        return submit_approved_observations(
            payload=payload,
            approved_by=req.approved_by or actor,
            idempotency_key=req.idempotency_key,
            settings=self.s,
        )

    def _load_approvals(self) -> None:
        return
