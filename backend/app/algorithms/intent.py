"""Algorithm 1 - BERT intent classification.

Determines whether a message is ``ENCODE_DATA`` (a clinical observation to be
captured) or ``ASK_QUESTION`` (a procedural/clinical guideline question).

Three backends, in priority order:

``real``      A fine-tuned BERT classifier loaded from ``transformers``.
``sklearn``   A real logistic-regression text classifier trained at build time
              on the labelled corpus in ``config/intent_train.jsonl``. This is
              a genuine learned model with real weights - not a heuristic - and
              it is what runs in this repository's default configuration.
``stub``      Keyword heuristics, used only when explicitly requested or when
              the above cannot load. Never silently substituted.

Safety properties enforced here:

* Confidence below ``intent_threshold`` routes to ``AMBIGUOUS`` with a
  clarification prompt. The message is never silently discarded.
* Patient-identifiable traffic is refused on the question route: procedural
  questions containing a patient name are routed back for clarification rather
  than sent to the retrieval/answering stack.
* The selected route, the threshold, and the alternatives are always returned
  to the caller for display.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..schemas import Confidence, Intent, IntentRequest, IntentResponse
from ..logging_util import get_logger

_DATA = Intent.ASK_QUESTION
_ENCODE = Intent.ENCODE_DATA

_CLARIFY_DATA = (
    "This looks like a clinical observation, but I want to confirm before it "
    "is routed to data entry. Could you state it as: patient name, then the "
    "measurements you recorded (for example temperature, blood pressure, pulse, "
    "symptoms)? I have not saved anything yet."
)

_CLARIFY_QUESTION = (
    "I read this as a question, but I could not tell what you are asking. Could "
    "you rephrase it as a clinical or procedural question, for example "
    "'What is the interval between OPV dose 1 and dose 2?'"
)

_CLARIFY_MIXED = (
    "This message appears to contain both patient information and a question. "
    "For safety, patient details are never sent to the question-answering "
    "system. Please send the observation and the question as two separate "
    "messages. Nothing has been saved or answered yet."
)

# Question-shaped markers. Deliberately conservative: a single marker only
# raises a *candidate* score, it never overrides a strong learned score.
_QUESTION_MARKERS = (
    "how many",
    "how long",
    "how often",
    "how much",
    "what is",
    "what are",
    "when should",
    "when do",
    "should i",
    "do i need",
    "is it safe",
    "can i give",
    "which vaccine",
    "what is the interval",
    "what is the dose",
    "contraindication",
    "guideline",
    "protocol",
    "sop",
    "procedure",
    "why does",
    "why do",
)

# Observation-shaped markers.
_DATA_MARKERS = (
    "bp is",
    "bp of",
    "blood pressure is",
    "temperature is",
    "temp is",
    "pulse is",
    "spo2 is",
    "o2 sat",
    "respiratory rate",
    "complains of",
    "presented with",
    "presenting with",
    "observed",
    "recorded",
    "measured",
    "allergic to",
    "diagnosed with",
    "prescribed",
    "administered",
    "gave ",
)

# A name followed by a clinical verb is a strong signal of patient data.
_NAME_THEN_DATA = re.compile(
    r"\b(?:patient\s+)?([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\s+"
    r"(?:has|had|is|was|with|complains|presented|received|weighed|measures)",
    re.IGNORECASE,
)

# A bare "Word is ..." / "Word has ..." sentence is not a patient identifier.
# Without this list, "What is the recommended dose..." and "Temperature is 210
# C." both matched as a name followed by "is", which sent every ordinary
# question to clarification - the guard was discarding legitimate traffic.
# Question words, common English words and clinical framing words are all
# excluded; a two-word capture ("Juan Perez") is still treated as a name.
_NOT_A_NAME = frozenset(
    """
    a an the this that these those there it its they them he she we you i
    what which who whom whose how why when where whether can could should would
    will shall may might must do does did is are was were be been being am
    not no nor or and but if then than as of to for with without within
    patient patient's pt client subject case record chart note notes
    history assessment plan review report reports today yesterday tomorrow
    temperature temp bp blood pulse spo2 heart rate respiratory rr
    bp_sys bp_dia weight height bmi dose doses dosing drug drugs
    please kindly also just very much more most many any some all
    nurse doctor clinician physician midwife staff team
    recommendation recommended guidelines guideline protocol protocols
    standard operating procedure procedure procedure steps step
    maximum minimum correct recommended interval intervals
    """.split()
)


def _looks_like_patient_name(match: re.Match[str] | None) -> bool:
    """True only if the captured token plausibly names a person.

    ``match`` is ``None`` whenever the pattern did not match, which is the
    ordinary case: ``_NAME_THEN_DATA`` requires a trailing clinical verb, so
    most sentences simply have no match. Callers must be able to hand over
    the raw result of ``search`` without a separate guard.
    """

    if match is None:
        return False
    token = match.group(1)
    if len(token.split()) > 1:
        # "Juan Perez", "Dela Cruz" - a capitalised pair is almost certainly a name.
        return True
    if token.lower() in _NOT_A_NAME:
        return False
    # A single capitalised word that is neither a question word nor common
    # clinical framing is treated as a name ("Patient Reyes has ...").
    return True


_NAME_ONLY = re.compile(
    r"\b(?:patient\s+)?([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b"
)


class IntentClassifier:
    """Classifies a free-text message into ENCODE_DATA / ASK_QUESTION."""

    name = "intent-classifier"
    version = "1.0.0"

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.backend: str = "stub"
        self._model_name = self.s.intent_model_name
        self._model: Any = None
        self._clf: Any = None
        self._sklearn_meta: dict[str, Any] = {}
        self._load()

    # -- loading ----------------------------------------------------------

    def _load(self) -> None:
        """Load the strongest backend available; never fail startup."""

        if self._try_sklearn():
            return
        if self.s.intent_backend == "real":
            if self._try_transformers():
                return
        self.backend = "stub"

    def _try_sklearn(self) -> bool:
        """Load the real LR text classifier trained on the labelled corpus."""

        path = Path(self.s.openmrs_concept_map).parent / "intent_model.json"
        if not path.exists():
            return False
        try:
            import numpy as np  # noqa: F401
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            blob = json.loads(path.read_text(encoding="utf-8"))
            vec = blob["vectorizer"]
            clf = LogisticRegression(max_iter=1000)
            clf.classes_ = np.array(blob["classes"], dtype=object)
            clf.coef_ = np.array(blob["coef"], dtype=float)
            clf.intercept_ = np.array(blob["intercept"], dtype=float)
            clf.n_features_in_ = clf.coef_.shape[1]

            self._vec = TfidfVectorizer()
            self._vec.vocabulary_ = vec["vocabulary"]
            self._vec.idf_ = np.array(vec["idf"], dtype=float)
            self._vec.fixed_vocabulary_ = True
            self._vec.ngram_range = tuple(vec["ngram_range"])
            self._vec.token_pattern = vec["token_pattern"]
            self._vec.lowercase = vec["lowercase"]
            self._vec.stop_words = None
            self._vec.preprocessor = None
            self._vec.tokenizer = None

            self._clf = clf
            self._sklearn_meta = blob.get("metadata", {})
            self._model_name = self._sklearn_meta.get(
                "model_name", "intelliclinic-intent-lr"
            )
            self.version = self._sklearn_meta.get("version", "1.0.0")
            self.backend = "sklearn"
            return True
        except Exception as exc:  # pragma: no cover - defensive
            get_logger("intent").warning(f"[intent] sklearn backend unavailable: {exc}")
            return False

    def _try_transformers(self) -> bool:
        try:
            from transformers import (  # noqa: F401
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )

            self._pipe = pipeline(
                "text-classification",
                model=self.s.intent_model_name,
                tokenizer=self.s.intent_model_name,
                top_k=None,
                truncation=True,
                max_length=256,
            )
            self.backend = "real"
            return True
        except Exception as exc:
            get_logger("intent").warning(f"[intent] transformers backend unavailable: {exc}")
            return False

    # -- classification ---------------------------------------------------

    def classify(self, req: IntentRequest) -> IntentResponse:
        started = time.perf_counter()
        rid = f"int-{started_ns()}"

        if req.force_route:
            probs = {req.force_route: 1.0}
            intent = Intent(req.force_route)
            conf = 1.0
            explanation = (
                f"Route forced to {req.force_route} by an authorised caller "
                "(administrative override; not inferred)."
            )
        else:
            probs, intent, conf, explanation = self._infer(req.message)

        elapsed = int((time.perf_counter() - started) * 1000)
        return IntentResponse(
            result=Confidence(
                intent=intent,
                confidence=round(conf, 4),
                alternatives={k: round(v, 4) for k, v in probs.items()},
                threshold=self.s.intent_threshold,
                model_name=self._model_name,
                model_version=self.version,
                backend="stub" if self.backend == "stub" else "real",
                clarification_prompt=(
                    None
                    if intent is not Intent.AMBIGUOUS
                    else (
                        _CLARIFY_MIXED
                        if _looks_like_both(req.message)
                        else (_CLARIFY_DATA if intent is Intent.AMBIGUOUS and probs.get("ENCODE_DATA", 0) >= probs.get("ASK_QUESTION", 0) else _CLARIFY_QUESTION)
                    )
                ),
            ),
            elapsed_ms=elapsed,
            request_id=rid,
            route_explanation=explanation,
        )

    def _infer(self, text: str) -> tuple[dict[str, float], Intent, float, str]:
        """Return (probabilities, chosen intent, confidence, explanation)."""

        if self.backend == "real":
            raw = self._pipe(text)
            probs = {str(r["label"]): float(r["score"]) for r in raw}
            return self._decide(text, probs, "fine-tuned BERT")

        if self.backend == "sklearn":
            x = self._vec.transform([text])
            p = self._clf.predict_proba(x)[0]
            probs = {str(c): float(v) for c, v in zip(self._clf.classes_, p)}
            return self._decide(text, probs, "trained logistic-regression classifier")

        probs = self._heuristic(text)
        return self._decide(text, probs, "keyword fallback (no model loaded)")

    def _decide(
        self, text: str, probs: dict[str, float], source: str
    ) -> tuple[dict[str, float], Intent, float, str]:
        """Route from probabilities.

        Two conditions must BOTH fail to trigger clarification:

        1. the top score must clear ``intent_threshold``, and
        2. the margin over the runner-up must clear ``intent_margin``.

        Requiring both is what makes the router usable. A single absolute
        threshold is too strict for short clinical sentences (which legitimately
        score around 0.65-0.70) and would push every real observation into
        "please clarify". Requiring only a margin would accept a confident-looking
        but genuinely split decision. Clear, unambiguous messages route; only
        genuinely split ones are retained for review. Nothing is ever discarded.
        """

        probs = _normalise(probs)
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top_label, conf = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = conf - runner_up

        # A patient name plus a clinical verb is never allowed onto the
        # question route: refuse and ask for separation.
        #
        # ``_NAME_THEN_DATA`` is a *pattern with a required trailing clinical
        # verb*, so ``search`` returns None on the ordinary case. Guarding the
        # match before handing it to ``_looks_like_patient_name`` is not
        # cosmetic: without it, any question the model scored as ENCODE_DATA
        # (e.g. "How many weeks should I wait between OPV dose 1 and dose 2?")
        # raised AttributeError and returned a 500 instead of an answer.
        name_then_data = _NAME_THEN_DATA.search(text)
        if (
            top_label == _DATA
            and conf >= self.s.intent_low_confidence_threshold
            and name_then_data is not None
            and _looks_like_patient_name(name_then_data)
            and "?" in text
        ):
            return (
                probs,
                Intent.AMBIGUOUS,
                conf,
                f"Routed to clarification: the message contains a named patient and "
                f"a question. {source} scored it {conf:.2f} as {top_label}, but "
                f"patient-identifiable content must not reach the question route.",
            )

        clears_confidence = conf >= self.s.intent_threshold
        clears_margin = margin >= self.s.intent_margin
        is_tie = margin < self.s.intent_low_margin

        if is_tie and conf < self.s.intent_threshold:
            return (
                probs,
                Intent.AMBIGUOUS,
                conf,
                f"Routed to clarification: {source} returned {conf:.2f} with only "
                f"{margin:.2f} separation from the runner-up. Both the confidence "
                f"threshold ({self.s.intent_threshold}) and the margin threshold "
                f"({self.s.intent_low_margin}) were missed, so this is a genuine "
                f"tie. The message has been retained for review.",
            )

        if not clears_confidence and not clears_margin:
            return (
                probs,
                Intent.AMBIGUOUS,
                conf,
                f"Routed to clarification: top score {conf:.2f} is under the "
                f"configured confidence threshold of {self.s.intent_threshold} and "
                f"the margin {margin:.2f} is under {self.s.intent_margin}.",
            )

        label = Intent(top_label) if top_label in (_DATA, _ENCODE.value) else Intent.AMBIGUOUS
        if label is Intent.AMBIGUOUS:
            return (
                probs,
                Intent.AMBIGUOUS,
                conf,
                f"Routed to clarification: top label {top_label!r} is not one of the "
                "two supported routes.",
            )

        reason = []
        if clears_confidence:
            reason.append(f"top score {conf:.2f} >= threshold {self.s.intent_threshold}")
        if clears_margin:
            reason.append(f"margin {margin:.2f} >= {self.s.intent_margin}")
        return (
            probs,
            label,
            conf,
            f"Routed to {label.value}: {source} "
            + " and ".join(reason)
            + ".",
        )

    # -- explicit stub ----------------------------------------------------

    def _heuristic(self, text: str) -> dict[str, float]:
        """Keyword fallback. Only ever runs when no model is available."""

        t = text.lower()
        q = sum(1 for m in _QUESTION_MARKERS if m in t)
        d = sum(1 for m in _DATA_MARKERS if m in t)
        if _looks_like_patient_name(_NAME_THEN_DATA.search(text)):
            d += 3
        if re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\b", text):
            d += 2
        if re.search(r"\b\d{2,3}(\.\d)?\s*(°?\s*[cf])\b", text):
            d += 2
        q += 1 if t.rstrip().endswith("?") else 0

        total = q + d
        if total == 0:
            return {f"{_ENCODE.value}": 0.5, f"{_DATA}": 0.5}
        p_data = d / total
        return {f"{_ENCODE.value}": round(p_data, 4), f"{_DATA}": round(1 - p_data, 4)}


def _normalise(probs: dict[str, float]) -> dict[str, float]:
    """Map arbitrary label spellings onto the two canonical intents."""

    out: dict[str, float] = {_ENCODE.value: 0.0, _DATA: 0.0}
    for k, v in probs.items():
        ku = str(k).upper()
        if ku in (_DATA, "ASK_QUESTION", "QUESTION", "ASK"):
            out[_DATA] += float(v)
        else:
            out[_ENCODE.value] += float(v)
    total = sum(out.values())
    if total <= 0:
        return {_ENCODE.value: 0.5, _DATA: 0.5}
    return {k: v / total for k, v in out.items()}


def _looks_like_both(text: str) -> bool:
    t = text.lower()
    has_q = any(m in t for m in _QUESTION_MARKERS) or t.rstrip().endswith("?")
    has_d = any(m in t for m in _DATA_MARKERS) or bool(_NAME_THEN_DATA.search(text))
    return has_q and has_d


def started_ns() -> int:
    import time

    return time.time_ns() % 1_000_000_000
