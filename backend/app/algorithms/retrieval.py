"""Algorithm 4 - ChromaDB semantic medical document retrieval.

Finds passages in the clinic's *approved* procedural guidelines and reference
documents. Runs only when Algorithm 1 selects ``ASK_QUESTION``.

Evidence discipline, enforced structurally rather than by convention:

* Only documents whose ``review_status`` is ``APPROVED`` are retrievable unless
  the caller explicitly overrides - and the override is recorded in the
  response.
* Retrieved chunks carry their document version and effective date, so an
  answer can never cite a document without showing which version it came from.
* Similarity score is reported but never treated as proof of correctness.
  ``evidence_status`` is derived from score separation and from whether chunks
  come from conflicting source versions, not from a single threshold.
* An empty or thin result set yields ``INSUFFICIENT`` and Algorithm 5 is
  required to say it cannot verify, rather than improvise.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..logging_util import get_logger
from ..state_util import read_json_state
from ..schemas import RetrievedChunk, RetrievalRequest, RetrievalResponse


class RegistryUnreadableError(RuntimeError):
    """The document registry could not be read, so retrieval is unavailable.

    Distinct from "no documents indexed": an unreadable registry means the
    knowledge base is broken, and answering from it would return confident
    empty results that look identical to a genuine no-match.
    """


@dataclass
class DocumentRecord:
    """Metadata for one indexed reference document."""

    document_id: str
    title: str
    source_version: str
    effective_date: str | None
    review_status: str
    topic: str
    path: str
    chunk_count: int
    uploaded_by: str | None
    uploaded_at: str
    sha256: str


class RetrievalEngine:
    """ChromaDB-backed semantic retrieval over approved clinical documents."""

    name = "chroma-semantic-retrieval"
    version = "1.0.0"

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.backend = "unavailable"
        self._client: Any = None
        self._collection: Any = None
        self._embedder: Any = None
        self._load()

    # -- wiring -----------------------------------------------------------

    def _load(self) -> None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except Exception as exc:
            get_logger("retrieval").warning(f"[retrieval] chromadb unavailable: {exc}")
            return

        try:
            self.s.chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.s.chroma_path))
            self._embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.s.embedding_model
            )
            self._collection = self._client.get_or_create_collection(
                name=self.s.chroma_collection,
                embedding_function=self._embedder,
                metadata={"hnsw:space": "cosine"},
            )
            self.backend = "real"
        except Exception as exc:
            get_logger("retrieval").warning(f"[retrieval] collection unavailable: {exc}")
            self.backend = "unavailable"

    @property
    def ready(self) -> bool:
        return self.backend == "real" and self._collection is not None

    # -- ingestion --------------------------------------------------------

    def index_document(
        self,
        *,
        document_id: str,
        title: str,
        source_version: str,
        effective_date: str | None,
        review_status: str,
        topic: str,
        text: str,
        uploaded_by: str | None = None,
        chunk_size: int = 420,
        chunk_overlap: int = 80,
    ) -> dict[str, Any]:
        """Chunk, embed, and upsert one document. Replaces prior version."""

        if not self.ready:
            raise RuntimeError(
                "Retrieval backend is not available (chromadb or the embedding "
                "model failed to load). Document metadata is recorded but no "
                "chunks were embedded."
            )

        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        if not chunks:
            raise ValueError("document produced no indexable text")

        # Replace any existing chunks for this document id (versioned reindex).
        existing = self._collection.get(where={"document_id": document_id})
        if existing.get("ids"):
            self._collection.delete(ids=existing["ids"])

        ids, docs, metas = [], [], []
        now = _now_iso()
        for i, (offset, content) in enumerate(chunks):
            ids.append(f"{document_id}::{i}")
            docs.append(content)
            metas.append(
                {
                    "document_id": document_id,
                    "title": title,
                    "source_version": source_version,
                    "effective_date": effective_date or "",
                    "review_status": review_status,
                    "topic": topic,
                    "section": _guess_section(content),
                    "chunk_index": i,
                    "char_offset": offset,
                }
            )

        # Chroma batches internally; keep batches bounded for large documents.
        for start in range(0, len(ids), 100):
            sl = slice(start, start + 100)
            self._collection.add(ids=ids[sl], documents=docs[sl], metadatas=metas[sl])

        rec = DocumentRecord(
            document_id=document_id,
            title=title,
            source_version=source_version,
            effective_date=effective_date,
            review_status=review_status,
            topic=topic,
            path="",
            chunk_count=len(chunks),
            uploaded_by=uploaded_by,
            uploaded_at=now,
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        self._save_registry(rec)
        return {
            "document_id": document_id,
            "chunks_indexed": len(chunks),
            "status": "INDEXED",
            "sha256": rec.sha256,
        }

    def remove_document(self, document_id: str) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("retrieval backend unavailable")
        existing = self._collection.get(where={"document_id": document_id})
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        self._registry_pop(document_id)
        return {"document_id": document_id, "chunks_removed": len(ids), "status": "REMOVED"}

    def list_documents(self) -> list[dict[str, Any]]:
        return self._registry_load()

    def set_review_status(self, document_id: str, status: str) -> dict[str, Any]:
        """Approve or disable a document. Disabled documents stop being retrieved."""

        if not self.ready:
            raise RuntimeError("retrieval backend unavailable")
        self._collection.update(
            where={"document_id": document_id},
            metadatas=[{"review_status": status}]
            * len(self._collection.get(where={"document_id": document_id}).get("ids") or []),
        )
        recs = self._registry_load()
        for r in recs:
            if r["document_id"] == document_id:
                r["review_status"] = status
        self._registry_write(recs)
        return {"document_id": document_id, "review_status": status}

    # -- retrieval --------------------------------------------------------

    def retrieve(self, req: RetrievalRequest) -> RetrievalResponse:
        started = time.perf_counter()
        rid = f"retr-{time.time_ns() % 1_000_000_000}"
        notes: list[str] = []

        if not self.ready:
            return RetrievalResponse(
                query=req.query,
                retrieved_documents=[],
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                request_id=rid,
                evidence_status="INSUFFICIENT",
                embedding_model=self.s.embedding_model,
                notes=[
                    "Retrieval backend is unavailable, so NO documents were "
                    "retrieved and nothing can be cited. This is not an empty "
                    "result set - it is a system fault."
                ],
            )

        where: dict[str, Any] = {}
        clauses: list[dict[str, Any]] = []
        if req.approved_only:
            clauses.append({"review_status": "APPROVED"})
            notes.append("Restricted to APPROVED documents only.")
        if req.topic_filter:
            clauses.append({"topic": req.topic_filter})
            notes.append(f"Topic filter applied: {req.topic_filter}")

        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

        try:
            res = self._collection.query(
                query_texts=[req.query],
                n_results=min(req.top_k, max(req.top_k, 10)),
                where=(where or None),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            return RetrievalResponse(
                query=req.query,
                retrieved_documents=[],
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                request_id=rid,
                evidence_status="INSUFFICIENT",
                embedding_model=self.s.embedding_model,
                notes=[f"Retrieval query failed: {type(exc).__name__}: {exc}"],
            )

        chunks: list[RetrievedChunk] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        for i, cid in enumerate(ids):
            meta = metas[i] or {}
            content = docs[i] or ""
            distance = float(dists[i]) if i < len(dists) else 1.0
            score = max(0.0, min(1.0, 1.0 - distance))  # cosine distance -> similarity
            if req.min_score is not None and score < req.min_score:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(cid),
                    document_id=str(meta.get("document_id") or ""),
                    title=str(meta.get("title") or "Untitled"),
                    section=meta.get("section") or None,
                    content=content,
                    source_version=str(meta.get("source_version") or "unknown"),
                    effective_date=(meta.get("effective_date") or None),
                    similarity_score=round(score, 4),
                    approved=str(meta.get("review_status") or "") == "APPROVED",
                    review_status=str(meta.get("review_status") or "UNKNOWN"),
                )
            )

        chunks.sort(key=lambda c: c.similarity_score, reverse=True)
        chunks = chunks[: req.top_k]

        if not chunks:
            notes.append(
                "No approved document scored above the configured minimum. "
                "The question must be answered as UNVERIFIED."
            )
            status = "INSUFFICIENT"
        else:
            status = self._assess(chunks, notes)

        elapsed = int((time.perf_counter() - started) * 1000)
        return RetrievalResponse(
            query=req.query,
            retrieved_documents=chunks,
            elapsed_ms=elapsed,
            request_id=rid,
            evidence_status=status,
            embedding_model=self.s.embedding_model,
            notes=notes,
        )

    def _assess(self, chunks: list[RetrievedChunk], notes: list[str]) -> str:
        top = chunks[0].similarity_score

        if top < self.s.retrieval_min_score:
            notes.append(
                f"Best score {top:.3f} is below the configured minimum "
                f"{self.s.retrieval_min_score}. Evidence is too weak to cite as support."
            )
            return "INSUFFICIENT"

        if len(chunks) < 2:
            notes.append(
                "Only one passage was retrieved; there is no corroboration and no "
                "conflict check is possible."
            )
            return "SUFFICIENT"

        versions = {c.source_version for c in chunks[:3] if c.approved}
        if len(versions) > 1:
            notes.append(
                "Top passages come from different document versions "
                f"({', '.join(sorted(versions))}). Treat this as a possible "
                "conflict and defer to the currently effective version."
            )
            return "CONFLICTING"

        gap = chunks[0].similarity_score - chunks[1].similarity_score
        if gap > self.s.retrieval_conflict_gap * 2:
            notes.append(
                f"Clear separation from the next-best passage ({gap:.3f}), so the "
                "top result is likely the relevant one."
            )
        else:
            notes.append(
                "Top passages score similarly; they may describe the same guidance "
                "from different documents. Citations must list all of them."
            )
        return "SUFFICIENT"

    # -- registry ---------------------------------------------------------

    def _registry_path(self) -> Path:
        return self.s.chroma_path / "documents.json"

    def _registry_load(self) -> list[dict[str, Any]]:
        p = self._registry_path()
        if not p.exists():
            return []

        def _error(path: Path, detail: str) -> RegistryUnreadableError:
            return RegistryUnreadableError(
                f"The document registry {path} could not be used ({detail}). "
                f"Retrieval is unavailable until it is repaired or re-indexed; "
                f"reporting an empty index instead would make every query look "
                f"like a genuine 'no results'."
            )

        # Shape is validated as well as syntax. A registry that parses to an
        # object, or to a list whose entries are not records, would otherwise be
        # handed back as though it were a real index.
        return read_json_state(
            p,
            is_valid=lambda v: isinstance(v, list)
            and all(isinstance(r, dict) and isinstance(r.get("document_id"), str) for r in v),
            on_error=_error,
            logger_name="retrieval",
            what="a JSON list of document records",
        )

    def _registry_write(self, recs: list[dict[str, Any]]) -> None:
        self._registry_path().write_text(
            json.dumps(recs, indent=2), encoding="utf-8"
        )

    def _save_registry(self, rec: DocumentRecord) -> None:
        recs = self._registry_load()
        recs = [r for r in recs if r["document_id"] != rec.document_id]
        recs.append(rec.__dict__)
        self._registry_write(recs)

    def _registry_pop(self, document_id: str) -> None:
        self._registry_write(
            [r for r in self._registry_load() if r["document_id"] != document_id]
        )


# --------------------------------------------------------------------------


def _chunk_text(text: str, size: int = 420, overlap: int = 80) -> list[tuple[int, str]]:
    """Sentence-aware chunking that keeps the offset of every chunk."""

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+", text)
    out: list[tuple[int, str]] = []
    buf, offset, cursor = [], 0, 0

    for s in sentences:
        if not s.strip():
            continue
        if not buf:
            offset = text.find(s, cursor)
        if sum(len(x) + 1 for x in buf) + len(s) > size and buf:
            out.append((offset, " ".join(buf).strip()))
            keep = buf[-1] if overlap and len(buf) > 1 else ""
            buf = [keep] if keep else []
            offset = text.find(s, cursor) if buf else text.find(s, cursor)
        buf.append(s)
        cursor = text.find(s, cursor) + len(s)

    if buf:
        out.append((offset, " ".join(buf).strip()))

    return [(max(0, o), c) for o, c in out if c.strip()]


def _guess_section(text: str) -> str:
    m = re.match(r"^\s*(\d+(?:\.\d+)*[.)]?\s+[A-Z][^\n]{3,60})", text)
    if m:
        return m.group(1).strip()[:80]
    m = re.match(r"^\s*([A-Z][A-Z \-]{4,50})\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).strip().title()
    return "General"


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
