"""Algorithm 2 - Bio_ClinicalBERT medical entity extraction.

Runs only when Algorithm 1 selects ``ENCODE_DATA``.

Two backends:

``real``  A genuine BIO-tagging token classifier (``samrawal/bert-base-uncased_
       clinical-ner``) driven through ``transformers``. Word-piece tokens are
       projected onto character offsets so every entity carries the original
       text and its span, and consecutive sub-words are re-joined. Negation,
       uncertainty, and historical cues are then applied as a *separate*
       assertion pass over the extracted spans - NER finds the span, assertion
       classification decides whether it is actually present. That separation is
       what makes "no fever" come back as ``NEGATED`` rather than as a fever.

``stub``  A deterministic rule extractor using the same schema. It exists so the
       API contract is testable without a GPU, and it is reported as
       ``backend="stub"`` in every response. It is never silently substituted.

Nothing here invents a value: if a span is not present in the text it is not
returned. Uncertain spans are flagged ``requires_review`` rather than dropped.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, get_settings
from ..schemas import (
    Assertion,
    ClinicalEntity,
    EntityType,
    ExtractionRequest,
    ExtractionResponse,
)
from ..logging_util import get_logger

# The clinical NER checkpoint's label vocabulary, mapped onto our entity types.
# Keys are normalised (lower-cased, BIO prefix stripped) so the same map works
# for ``B-problem``, ``B-DISEASE``, ``B-SIGN`` and friends. Anything outside
# this map - including the all-important ``O`` (outside) - resolves to None and
# produces no span. Mapping ``O`` to a real type would emit an entity for every
# non-entity token in the sentence, so it deliberately stays absent.
_LABEL_MAP: dict[str, EntityType] = {
    "disease": EntityType.DIAGNOSIS,
    "diagnosis": EntityType.DIAGNOSIS,
    "condition": EntityType.DIAGNOSIS,
    "sign": EntityType.SYMPTOM,
    "symptom": EntityType.SYMPTOM,
    "complaint": EntityType.SYMPTOM,
    "problem": EntityType.SYMPTOM,  # refined by _refine_problem below
    "treatment": EntityType.MEDICATION,
    "drug": EntityType.MEDICATION,
    "medication": EntityType.MEDICATION,
    "dosage": EntityType.DOSAGE,
    "dose": EntityType.DOSAGE,
    # "test" and "procedure" are NOT symptoms. The checkpoint has a real `test`
    # label (labs / imaging); folding it into SYMPTOM would surface a creatinine
    # result inside the symptom list, which is exactly the misclassification that
    # can hide a finding from the clinician.
    "test": EntityType.TEST,
    "procedure": EntityType.PROCEDURE,
    "vital": EntityType.SYMPTOM,
    "allergy": EntityType.ALLERGY,
}

# Spans the NER model labels as a generic "problem" but which are really
# diagnoses. The checkpoint has one label for both, so this list resolves the
# ambiguity; anything unmatched stays a SYMPTOM, which is the safer default.
_DIAGNOSIS_TERMS = (
    "pneumonia", "hypertension", "diabetes", "asthma", "tuberculosis",
    "malaria", "dengue", "anemia", "anaemia", "hypothermia", "cancer",
    "failure", "syndrome", "disease", "disorder", "infection",
)

_NEGATION_CUES = (
    "no ",
    "not ",
    "denies ",
    "deny ",
    "denied ",
    "without ",
    "w/o ",
    "absent ",
    "negative for ",
    "free of ",
    "never had ",
    "ruled out ",
    "no evidence of ",
    "unremarkable for ",
)

_UNCERTAINTY_CUES = (
    "possible ",
    "possibly ",
    "suspected ",
    "suspect ",
    "may have ",
    "might have ",
    "could be ",
    "probable ",
    "likely ",
    "questionable ",
    "versus ",
    "vs ",
    "rule out ",
)

_HISTORICAL_CUES = (
    "history of ",
    "past medical history ",
    "past history ",
    "hx of ",
    "previous ",
    "previously ",
    "last year ",
    "last month ",
    "years ago ",
    "months ago ",
    "weeks ago ",
    "chronic ",
)

# Cues that look like history cues but assert a *current* finding. They are
# checked first and clear the historical state.
_ACTIVE_CUES = (
    "diagnosed with ",
    "dx with ",
    "known ",
    "presents with ",
    "presenting with ",
    "started on ",
    "prescribed ",
    "reports ",
    "complains of ",
    "complaining of ",
)


# --------------------------------------------------------------------------
# Allergy cues
# --------------------------------------------------------------------------
#
# The checkpoint's vocabulary has NO allergy label - its labels are
# problem/treatment/test only. So a drug the patient is *allergic to* arrives as
# a treatment span (or a problem span). Left uncorrected, "Allergy penicillin"
# is reported to the clinician as a medication the patient is taking, and the
# allergen vanishes from the allergy list entirely - the single mistake in this
# pipeline that can directly harm a patient. These cues retype it.
_ALLERGY_CUES = (
    "allergic to ",
    "allergy to ",
    "allergies to ",
    "allergy: ",
    "allergy ",
    "allergies: ",
    "allergies ",
    "drug allergy ",
    "known allergy ",
)

# Scoped negation, applied to ALLERGY spans only. "No known drug allergies"
# negates the allergy list, not the whole sentence, so it must NOT become a
# general negation cue (it would otherwise mark any later entity as negated).
_NKDA_CUES = (
    "no known drug allergies",
    "no known drug allergy",
    "no known allergies",
    "no known allergy",
    "nkda",
)

# Words that may follow "allergy" without being an allergen. Without this guard
# a bare "allergy noted" is recorded as an allergy to "noted", and "allergies are
# well controlled" as an allergy to "are".
_ALLERGY_STOPWORDS = frozenset(
    {
        "noted", "status", "recorded", "documented", "unknown", "none", "nil", "reviewed",
        "updated", "listed", "checked", "verified", "confirmed", "assessment",
        "denied", "present", "absent", "reported", "today", "now", "given",
        "are", "is", "was", "were", "be", "been", "being", "have", "has", "had",
        "and", "or", "to", "of", "the", "a", "an", "for", "with", "no", "not",
        "on", "at", "by", "in", "from", "this", "that", "these", "those", "she",
        "he", "her", "his", "they", "patient", "well", "controlled",
    }
)

# Words that can never be a clinical entity. The checkpoint sometimes labels
# the discourse frame of a sentence ("Patient", "He", "Today") as a span, and
# emitting that as a finding would put noise into the review queue and, once
# approved, into OpenMRS.
#
# This list is deliberately narrow: closed-class function words and record
# furniture only. Nothing here could ever name a symptom, drug, test or
# allergen, so filtering on it cannot hide a real finding.
_NON_ENTITY_TERMS = frozenset(
    {
        "patient", "patients", "he", "she", "they", "him", "her", "them",
        "his", "hers", "their", "it", "its", "this", "that", "these", "those",
        "today", "now", "here", "reported", "reports", "report", "presents",
        "presenting", "presented", "present", "noted", "reviewed", "checked",
        "given", "seen", "status", "assessment", "plan", "note", "notes",
        "record", "records", "history", "male", "female", "man", "woman",
        "mr", "mrs", "ms", "dr", "md", "do", "none", "nil", "unknown",
    }
)


def _cue_only_vocabulary() -> frozenset[str]:
    """Words that only ever appear as *triggers*, never as findings.

    Derived from the cue lexicons already in this module rather than hand-copied,
    so adding a new cue cannot silently reintroduce the false positive: "allergy"
    is the word that routes a span to ALLERGY and "status" is one of the words that
    may follow it. Neither names a symptom, drug, test or allergen.
    """

    words = set(_NON_ENTITY_TERMS)
    words |= set(_ALLERGY_STOPWORDS)
    for cue in _ALLERGY_CUES:
        words.update(cue.split())
    for cue in _NKDA_CUES:
        words.update(cue.split())
    return frozenset(words)


_CUE_ONLY = _cue_only_vocabulary()


def _is_cue_only(text: str) -> bool:
    """True when a span carries trigger vocabulary and nothing else.

    The model labels the cue itself as an entity: "Allergy status noted." comes
    back as a SYMPTOM whose text is "Allergy status", and "allergies are well
    controlled" as a SYMPTOM "allergies". Those words tell us how to *read* a span,
    they are not the finding. Emitting them would put a phantom finding in the
    review queue and, once a clinician approves it, into OpenMRS.
    """

    stripped = text.strip().lower()
    if not stripped:
        return True
    if stripped in _CUE_ONLY:
        return True
    return all(tok in _CUE_ONLY for tok in stripped.split())

# A span whose own text is the cue plus the allergen: "Allergy penicillin".
# NB: `allergies?` would mean "allergie" + optional "s" - the word is "allergy"
# with a y, so the two spellings are written out explicitly.
_ALLERGY_HEAD_RE = re.compile(
    r"(?i)^\s*(?:allergic\s+to|allerg(?:y|ies))\s*:?\s*([A-Za-z][A-Za-z0-9 \-]{1,40})$"
)


@dataclass
class _Span:
    type: EntityType
    text: str
    start: int
    end: int
    score: float = 1.0
    sub: list[str] = field(default_factory=list)
    # Per-token model probabilities for the tokens that make up this span. The
    # mean becomes the span confidence, so the number the API reports is a real
    # measurement rather than a constant.
    tok_scores: list[float] = field(default_factory=list)

    def add_token_score(self, p: float) -> None:
        self.tok_scores.append(float(p))

    def resolve_score(self) -> None:
        if self.tok_scores:
            self.score = sum(self.tok_scores) / len(self.tok_scores)


class EntityExtractor:
    """Extracts clinical entities with offsets, assertion, and confidence."""

    name = "bio-clinical-bert-ner"
    version = "1.0.0"

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.backend = "stub"
        self._model_name = self.s.ner_model_name
        self._pipe: Any = None
        self._id2label: dict[int, str] = {}
        self._last_tags: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.s.ner_backend != "real":
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.s.ner_model_name)
            model = AutoModelForTokenClassification.from_pretrained(
                self.s.ner_model_name
            )
            self._id2label = {
                int(i): str(l) for i, l in model.config.id2label.items()
            }
            self._tok = tok
            self._model = model
            self._torch = torch
            self.backend = "real"
        except Exception as exc:
            get_logger("extraction").warning(f"[ner] real backend unavailable, using rule fallback: {exc}")
            self.backend = "stub"

    # ------------------------------------------------------------------

    def extract(self, req: ExtractionRequest) -> ExtractionResponse:
        started = time.perf_counter()
        rid = f"ner-{time.time_ns() % 1_000_000_000}"

        if self.backend == "real":
            spans = self._extract_model(req.text)
            source = "Bio_ClinicalBERT token classification"
            # The checkpoint has no vital-sign label, so measurements are
            # anchored deterministically and merged in. See algorithms/vitals.py.
            spans = _merge_vitals(spans, req.text)
        else:
            spans = self._extract_rules(req.text)
            source = "deterministic rule extractor (model not loaded)"

        # The checkpoint has no allergy label, so an allergen arrives typed as a
        # medication or a problem. Retype before assertion runs, otherwise the
        # allergen is reported as a drug the patient takes.
        spans = _refine_allergies(spans, req.text)

        # Drop spans that are pure record furniture ("Patient", "Today"). The
        # The NER checkpoint sometimes labels the frame of a sentence, or the cue
        # that introduces a finding, as an entity in its own right. Emitting
        # "Patient" as a TEST, or "Allergy status" as a SYMPTOM, would put a false
        # finding in the clinician's review queue and, once approved, into OpenMRS.
        spans = [s for s in spans if not _is_cue_only(s.text)]

        entities = [self._finalise(s, req.text) for s in spans]
        entities = _dedupe(entities)

        elapsed = int((time.perf_counter() - started) * 1000)
        return ExtractionResponse(
            entities=entities,
            elapsed_ms=elapsed,
            request_id=rid,
            model_name=self._model_name,
            model_version=self.version,
            backend="stub" if self.backend == "stub" else "real",
            raw_bio_tags=(
                None if self.backend == "stub" else self._last_tags
            ),
        )

    # -- real backend -----------------------------------------------------

    def _extract_model(self, text: str) -> list[_Span]:
        """Token-classify and project sub-word tags back to character spans.

        Minted tokens (``#``-prefixed) never start a new span. That matters for
        this checkpoint: a bare numeric token such as ``38`` followed by the
        minted ``##5`` would otherwise be emitted as a standalone DIAGNOSIS
        because the token pair ("C", "D") happens to be a real disease label in
        the tag vocabulary. Re-joining sub-words first removes that failure.
        """

        torch = self._torch
        enc = self._tok(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"][0].tolist()

        with torch.no_grad():
            logits = self._model(**enc).logits[0]
        preds = logits.argmax(dim=-1).tolist()
        # Real per-token confidence: the softmax probability the model assigned
        # to the label it predicted. Reporting this instead of a constant 1.0 is
        # what lets a misclassification ("NK" read as a symptom) carry a low
        # confidence and therefore be flagged for review.
        tok_probs = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()

        self._last_tags = [
            {
                "token": self._tok.convert_ids_to_tokens(int(i)),
                "label": self._id2label.get(int(p), "O"),
                "start": int(offsets[i][0]),
                "end": int(offsets[i][1]),
            }
            for i, p in enumerate(preds)
        ]

        def is_minted(idx: int) -> bool:
            raw = self._last_tags[idx]["token"] or ""
            return raw.startswith("##")

        spans: list[_Span] = []
        current: _Span | None = None
        for i, p in enumerate(preds):
            label = self._id2label.get(int(p), "O")
            s, e = int(offsets[i][0]), int(offsets[i][1])
            if s == e:  # special token
                continue

            mapped = _map_label(label)
            is_begin = label.upper().startswith("B-")

            # A minted continuation always extends the current span regardless
            # of its own predicted label.
            if not is_begin and is_minted(i) and current is not None:
                current.end = e
                current.sub.append(self._last_tags[i]["token"])
                current.add_token_score(tok_probs[i])
                continue

            if is_begin or mapped is None or (current is not None and current.type != mapped):
                if current is not None:
                    current.resolve_score()
                    spans.append(current)
                current = _Span(mapped, "", s, e) if mapped else None
                if current is not None:
                    current.sub = [self._last_tags[i]["token"]]
                    current.add_token_score(tok_probs[i])
                continue

            if current is not None:
                current.end = e
                current.sub.append(self._last_tags[i]["token"])
                current.add_token_score(tok_probs[i])

        if current is not None:
            current.resolve_score()
            spans.append(current)

        # A BIO tagger sometimes re-emits B- instead of I- when it restarts a
        # span at a word boundary ("hyper" B-problem / "tension" B-problem
        # instead of B- then I-). Two adjacent spans of the same type with no
        # gap between them are one clinical entity, so they are joined here.
        joined: list[_Span] = []
        for sp in spans:
            if (
                joined
                and sp.type is not None
                and joined[-1].type == sp.type
                and joined[-1].end >= sp.start
            ):
                prev = joined[-1]
                prev.end = max(prev.end, sp.end)
                prev.sub.extend(sp.sub)
                prev.tok_scores.extend(sp.tok_scores)
                prev.resolve_score()
            else:
                joined.append(sp)
        spans = joined

        out: list[_Span] = []
        for sp in spans:
            if sp.type is None:
                continue
            raw = text[sp.start : sp.end]
            # trim partial words introduced by sub-word boundaries
            raw, sp.start, sp.end = _trim_word_edges(raw, sp.start, sp.end, text)
            if not raw.strip():
                continue
            sp.text = raw
            if sp.type is EntityType.SYMPTOM:
                sp.type = _refine_problem(raw)
            out.append(sp)
        return out

    # -- rule fallback ----------------------------------------------------

    def _extract_rules(self, text: str) -> list[_Span]:
        """Deterministic extraction. Same schema, no model, honestly labelled.

        The tokenizer matches only alphanumeric runs. A ``/`` is deliberately
        NOT part of a token, so a unit pair like ``38.5 C`` cannot be seen as a
        single token that happens to resemble the ``("C","D")`` disease label.
        """

        spans: list[_Span] = []
        used: list[tuple[int, int]] = []

        def free(s: int, e: int) -> bool:
            return all(e <= a or s >= b for a, b in used)

        def add(t: EntityType, s: int, e: int, score: float = 0.85) -> None:
            if free(s, e):
                used.append((s, e))
                spans.append(_Span(t, text[s:e], s, e, score))

        # --- vital signs (specific first, so "fever" cannot steal "38.5 C") ---
        for m in re.finditer(
            r"(?P<sys>\d{2,3})\s*(?:/|over)\s*(?P<dia>\d{2,3})\s*"
            r"(?:mm\s*hg|mmhg)?",
            text,
            re.IGNORECASE,
        ):
            add(EntityType.BLOOD_PRESSURE, m.start(), m.end(), 0.95)

        for m in re.finditer(
            r"(?<![\d./-])(\d{2}(?:\.\d)?)\s*(?:°?\s*(C|F)\b|degrees?\s*(celsius|fahrenheit)\b)",
            text,
            re.IGNORECASE,
        ):
            add(EntityType.TEMPERATURE, m.start(), m.end(), 0.95)

        for m in re.finditer(
            r"\b(?:pulse|heart rate|hr)\s*(?:is|of|was|at)?\s*:?\s*(\d{2,3})\s*(?:bpm)?",
            text,
            re.IGNORECASE,
        ):
            add(EntityType.PULSE, m.start(), m.end(), 0.9)

        for m in re.finditer(
            r"\b(?:respiratory rate|resp rate|rr|respirations)\s*(?:is|of|was|at)?\s*:?\s*(\d{1,2})\s*(?:/min|bpm)?",
            text,
            re.IGNORECASE,
        ):
            add(EntityType.RESPIRATORY_RATE, m.start(), m.end(), 0.9)

        for m in re.finditer(
            r"\b(?:spo2|o2 sat|oxygen saturation|sat)\s*(?:is|of|was|at|on)?\s*:?\s*(\d{2,3})\s*(?:%|percent)?",
            text,
            re.IGNORECASE,
        ):
            add(EntityType.OXYGEN_SATURATION, m.start(), m.end(), 0.9)

        for m in re.finditer(
            r"(?P<val>\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml|units?)\b"
            r"(?:\s*(?:three times|twice|once|daily|bid|tid|qid|q\d+h|every\s+\d+\s*hours?))?",
            text,
            re.IGNORECASE,
        ):
            add(EntityType.DOSAGE, m.start(), m.end(), 0.8)

        # --- medications (generic + common names) ---
        meds = (
            "paracetamol", "acetaminophen", "ibuprofen", "amoxicillin",
            "amlodipine", "metformin", "salbutamol", "albuterol", "oxytocin",
            "aspirin", "azithromycin", "ceftriaxone", "omeprazole",
        )
        for name in meds:
            for m in re.finditer(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
                add(EntityType.MEDICATION, m.start(), m.end(), 0.9)

        # --- allergies ---
        # The bare form ("Allergy penicillin", "allergies: penicillin") is
        # included, not just "allergic to X": that is how allergy lists are
        # actually written, and requiring "to" or a colon silently dropped the
        # allergen. The stopword guard keeps "allergy noted" and "allergies are
        # well controlled" from being recorded as allergies to "noted" / "are".
        for m in re.finditer(
            r"(?:allergic\s+to|allerg(?:y|ies)\s*(?:to|:)?)\s*"
            r"([A-Za-z][A-Za-z0-9 \-]{1,40})",
            text,
            re.IGNORECASE,
        ):
            s, e = m.start(1), m.end(1)
            e = _clamp_clause(text, s, e)
            first = text[s:e].strip().split()[0].lower().strip(".,;:") if text[s:e].strip() else ""
            if first and first not in _ALLERGY_STOPWORDS:
                add(EntityType.ALLERGY, s, e, 0.9)

        # --- symptoms / diagnoses ---
        lexicon = (
            "fever", "cough", "headache", "dizziness", "nausea", "vomiting",
            "diarrhoea", "diarrhea", "chest pain", "dyspnoea", "dyspnea",
            "shortness of breath", "wheeze", "rash", "itching", "fatigue",
            "chills", "sweating", "abdominal pain", "sore throat", "back pain",
            "joint pain", "weakness", "numbness", "bleeding", "swelling",
            "community acquired pneumonia", "pneumonia", "hypertension",
            "diabetes", "diabetes mellitus", "asthma", "tuberculosis", "malaria",
            "dengue", "anemia", "anaemia", "hypothermia",
        )
        for term in lexicon:
            for m in re.finditer(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
                add(EntityType.SYMPTOM, m.start(), m.end(), 0.88)

        # --- patient identifier / demographics ---
        for m in re.finditer(
            r"\b(?:patient\s+)?([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)", text
        ):
            add(EntityType.PATIENT_IDENTIFIER, m.start(1), m.end(1), 0.8)

        for m in re.finditer(r"\b(\d{1,3})\s*(?:y/o|yo|year[s]? old|years of age)\b", text, re.IGNORECASE):
            add(EntityType.DEMOGRAPHIC, m.start(), m.end(), 0.85)

        spans.sort(key=lambda s: s.start)
        return spans

    # -- assertion pass ---------------------------------------------------

    def _finalise(self, span: _Span, text: str) -> ClinicalEntity:
        """Attach negation/uncertainty/history state and a value candidate.

        Cue precedence is negation > uncertainty > active > historical. A cue
        that merely contains the word "history" does not make a finding past:
        "diagnosed with community acquired pneumonia" is a current diagnosis,
        and marking it HISTORICAL silently drops it from the clinical payload -
        the patient's actual diagnosis disappears. Only cues that genuinely
        locate the finding in the past set Assertion.HISTORICAL.
        """
        from .vitals import implausibility_reason

        window_start = max(0, span.start - 50)
        prefix = text[window_start : span.start].lower()
        lower_text = text.lower()

        assertion = Assertion.PRESENT
        reason: str | None = None

        # Pick the cue closest to the entity, not the first cue in the tuple:
        # a 50-char prefix can contain both "history of" (for one problem) and
        # "chronic" (for another), and the nearest one governs this span.
        def _nearest(cues: tuple[str, ...]) -> tuple[int, str] | None:
            best: tuple[int, str] | None = None
            for cue in cues:
                idx = prefix.rfind(cue)
                if idx != -1 and (best is None or idx > best[0]):
                    best = (idx, cue)
            return best

        neg = _nearest(_NEGATION_CUES)
        unc = _nearest(_UNCERTAINTY_CUES)
        act = _nearest(_ACTIVE_CUES)
        hist = _nearest(_HISTORICAL_CUES)

        # NKDA is scoped to allergies: it negates the allergy list, not the whole
        # sentence, so it must NOT become a general negation cue (it would
        # otherwise mark any later entity - a drug started afterwards - as
        # negated). The tokenizer also splits "No known drug allergies" into
        # fragments ("NK", "known drug allergies"), so the test is on the span
        # text and its type, not the type alone.
        span_lower = span.text.lower()
        nkda = (
            _nearest(_NKDA_CUES)
            if span.type is EntityType.ALLERGY
            or "allerg" in span_lower
            or span_lower in ("nk", "nkda")
            else None
        )

        if nkda is not None:
            assertion = Assertion.NEGATED
            reason = (
                f'No-known-allergies cue "{nkda[1].strip()}" precedes the entity; '
                "the text states there is no allergy to record."
            )
        elif neg is not None:
            assertion = Assertion.NEGATED
            reason = (
                f'Negation cue "{neg[1].strip()}" precedes the entity; the '
                "text does not assert this finding as present."
            )
        elif unc is not None:
            assertion = Assertion.UNCERTAIN
            reason = f'Uncertainty cue "{unc[1].strip()}" precedes the entity.'
        elif act is not None and (hist is None or act[0] >= hist[0]):
            assertion = Assertion.PRESENT
            reason = (
                f'Active cue "{act[1].strip()}" precedes the entity, so this is a '
                "current finding, not a historical one."
            )
        elif hist is not None:
            assertion = Assertion.HISTORICAL
            reason = f'Historical cue "{hist[1].strip()}" within 50 chars before the entity.'

        value, unit = _value_candidate(span.type, span.text, lower_text)
        # A measurement outside the physiologically possible range is almost
        # certainly a transcription or unit error. It is flagged, never dropped:
        # discarding it would silently lose a real value the clinician typed.
        implausible = implausibility_reason(span.type, value, unit)

        # Every condition that forces human review contributes a reason, so the
        # review queue can tell the clinician *why* rather than showing a bare
        # flag. Only the reasons that actually forced review are reported - an
        # ordinary finding with no cue gets no reason string at all.
        review_reasons: list[str] = []
        if assertion is not Assertion.PRESENT and reason:
            review_reasons.append(reason)
        if implausible:
            review_reasons.append(implausible)
        if span.score < self.s.ner_confidence_floor:
            review_reasons.append(
                f"Model confidence {span.score:.2f} is below the configured floor "
                f"of {self.s.ner_confidence_floor}; flagged for human review."
            )
        requires_review = bool(review_reasons)
        reason = " ".join(review_reasons) if review_reasons else None

        return ClinicalEntity(
            type=span.type,
            text=span.text,
            start=span.start,
            end=span.end,
            assertion=assertion,
            confidence=round(float(span.score), 4),
            value_candidate=value,
            unit_candidate=unit,
            requires_review=requires_review,
            review_reason=reason,
        )


def _merge_vitals(spans: list[_Span], text: str) -> list[_Span]:
    """Add deterministically-anchored vitals, dropping duplicates by span.

    A neural span that merely *names* a vital ("BP", "SpO2") without carrying
    the measurement is discarded in favour of the anchored match that includes
    the value and unit. A neural span that overlaps an anchored vital is also
    discarded, so a reading is never reported twice.
    """

    from .vitals import scan_vitals

    matches = scan_vitals(text)
    if not matches:
        return spans

    vital_spans: list[_Span] = []
    for m in matches:
        vital_spans.append(
            _Span(
                type=m.type,
                text=text[m.start : m.end],
                start=m.start,
                end=m.end,
                score=0.97,
            )
        )

    kept: list[_Span] = []
    for sp in spans:
        overlaps = any(sp.start < v.end and sp.end > v.start for v in vital_spans)
        if overlaps:
            continue
        # A bare label with no measurement is not an observation.
        if sp.type in (
            EntityType.SYMPTOM,
            EntityType.DIAGNOSIS,
        ) and re.fullmatch(
            r"(?i)\s*(blood\s+pressure|bp|temperature|temp|pulse|heart\s*rate|hr|"
            r"respiratory\s*rate|resp\s*rate|rr|spo2|o2\s*sat|oxygen\s*saturation)\s*",
            sp.text,
        ):
            continue
        kept.append(sp)

    return sorted(kept + vital_spans, key=lambda s: s.start)


def _map_label(label: str) -> EntityType | None:
    """Resolve a raw BIO tag onto an entity type, or None for ``O``/unknown."""

    up = (label or "").upper()
    if up == "O" or not up:
        return None
    body = up.split("-", 1)[1] if "-" in up else up
    return _LABEL_MAP.get(body.lower())


def _refine_problem(span_text: str) -> EntityType:
    """Promote a generic NER 'problem' span to DIAGNOSIS when it looks like one."""

    low = span_text.lower()
    return (
        EntityType.DIAGNOSIS
        if any(term in low for term in _DIAGNOSIS_TERMS)
        else EntityType.SYMPTOM
    )


def _value_candidate(t: EntityType, text: str, lower_text: str) -> tuple[Any, str | None]:
    """Parse a numeric candidate out of an entity span. Never guesses."""

    if t is EntityType.TEMPERATURE:
        m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(?:°\s*)?([CF])?\b", text)
        if m:
            v = float(m.group(1))
            unit = (m.group(2) or "C").upper()
            if re.search(r"\d\s*°?\s*F\b|degrees?\s*fahrenheit", text, re.IGNORECASE):
                unit = "F"
            return v, unit
    if t is EntityType.BLOOD_PRESSURE:
        return text.strip(), None
    if t is EntityType.OXYGEN_SATURATION:
        # Strip the label ("SpO2", "O2 sat") before parsing, otherwise the "2"
        # in "SpO2" is read as the saturation value.
        stripped = re.sub(
            r"(?i)^\s*(sp\s*o\s*2|o\s*2\s*sat|oxygen\s*saturation|sats?|sat)\b[\s:]*",
            "",
            text,
        )
        m = re.search(r"(\d{2,3})", stripped)
        if m:
            return float(m.group(1)), "%"
    if t in (EntityType.PULSE, EntityType.RESPIRATORY_RATE):
        stripped = re.sub(
            r"(?i)^\s*(pulse|heart\s*rate|hr|respiratory\s*rate|resp\s*rate|rr)\b[\s:]*",
            "",
            text,
        )
        m = re.search(r"(\d{1,3})", stripped)
        if m:
            return float(m.group(1)), None
    if t is EntityType.DOSAGE:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)?", text, re.IGNORECASE)
        if m:
            return float(m.group(1)), (m.group(2) or "").upper() or None
    return None, None


def _refine_allergies(spans: list[_Span], text: str) -> list[_Span]:
    """Retype spans that an allergy cue marks as allergies. Mutates in place.

    Two shapes arrive from the model, and both have to be handled:

    * the cue sits *inside* one span - the checkpoint groups the label with the
      drug, so "Allergy penicillin" is a single treatment span. The allergen is
      recovered from the span's own text and the span is narrowed to it, so the
      cue is not carried into the allergy list as though it were the allergen;
    * the cue sits *before* a span - "allergic to penicillin" is split by the
      tokenizer into a problem span ("allergic") and a treatment span
      ("penicillin"), so the cue is found in the text preceding the span.

    Either way the span becomes an ALLERGY. Nothing is deleted: a span that
    cannot be shown to be an allergy keeps its original type.
    """

    for sp in spans:
        if sp.type is EntityType.ALLERGY:
            continue

        cued = any(
            cue in text[max(0, sp.start - 60) : sp.start].lower()
            for cue in _ALLERGY_CUES
        )

        m = _ALLERGY_HEAD_RE.match(text[sp.start : sp.end])
        if m is not None:
            gs, ge = m.span(1)
            raw = text[sp.start + gs : sp.start + ge]
            lead = len(raw) - len(raw.lstrip())
            tail = raw.strip()
            first = tail.split()[0].lower().strip(".,;:") if tail else ""
            if first and first not in _ALLERGY_STOPWORDS:
                sp.start = sp.start + gs + lead
                sp.end = sp.start + len(tail)
                sp.text = tail
                cued = True

        if cued:
            sp.type = EntityType.ALLERGY

    return spans


def _dedupe(entities: list[ClinicalEntity]) -> list[ClinicalEntity]:
    """Drop an entity fully contained in an overlapping higher-scoring one."""

    out: list[ClinicalEntity] = []
    for e in sorted(entities, key=lambda x: (-x.confidence, x.start)):
        if any(
            o.start <= e.start and o.end >= e.end and o.confidence >= e.confidence
            for o in out
        ):
            continue
        out.append(e)
    return sorted(out, key=lambda x: x.start)


def _trim_word_edges(raw: str, start: int, end: int, text: str) -> tuple[str, int, int]:
    """Snap a sub-word boundary back to a real word boundary."""

    while start < end and raw and not raw[0].isalnum() and raw[0] not in "/°":
        start += 1
        raw = text[start:end]
    while end > start and raw and not raw[-1].isalnum() and raw[-1] not in "/°":
        end -= 1
        raw = text[start:end]
    return raw, start, end


def _clamp_clause(text: str, s: int, e: int) -> int:
    """Do not let an allergy span run to the end of the sentence."""

    for sep in (",", ".", ";", " and ", " but "):
        i = text.find(sep, s)
        if i != -1 and i < e:
            e = i
    return e
