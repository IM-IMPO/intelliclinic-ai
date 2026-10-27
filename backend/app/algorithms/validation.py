"""Algorithm 3 - Regex and clinical validation.

Turns the entities from Algorithm 2 into typed observations ready for OpenMRS.
This service is entirely deterministic: same input, same output, same reason
codes. That is deliberate - a clinician must be able to reproduce and audit why
a value was accepted or rejected.

Three separable outcome classes, because conflating them is how abnormal
readings get silently "corrected":

``PARSE_ERROR``          The text is not a usable value at all. Rejected.
``PLAUSIBILITY_WARNING`` The value is syntactically fine but clinically unusual.
                         NEVER deleted and NEVER changed - surfaced for review.
``URGENT_ALERT``         The value is outside a safe range. Surfaced loudly.

Unit normalisation is performed before plausibility checks so that 102 F is
judged as 38.9 C, not as an impossible 102 C.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..schemas import (
    Assertion,
    ClinicalEntity,
    EntityType,
    ObservationField,
    Severity,
    ValidationIssue,
    ValidationRequest,
    ValidationResponse,
    ValidationStatus,
)
from ..logging_util import get_logger

_F_TO_C = lambda f: (f - 32.0) * 5.0 / 9.0  # noqa: E731


@dataclass(frozen=True)
class _Rule:
    key: str
    entity: EntityType
    # Plausible band: outside it the number is not a measurement a living
    # patient can have, so it is reported (never silently corrected).
    lo: float
    hi: float
    # Urgent band: inside the plausible band but clinically critical. Kept as a
    # separate band so a real emergency (41 C, SpO2 85%) is not masked by a
    # generous plausible maximum.
    urgent_lo: float | None
    urgent_hi: float | None
    unit: str


class ClinicalValidator:
    """Normalises, validates, and maps extracted entities onto OpenMRS concepts."""

    name = "clinical-validation"
    version = "1.0.0"

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self._concepts: dict[str, Any] = {}
        self._load_concepts()

    def _load_concepts(self) -> None:
        path = Path(self.s.openmrs_concept_map)
        if not path.is_absolute():
            from ..config import BACKEND_ROOT

            path = BACKEND_ROOT / path
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            self._concepts = blob.get("concepts", {})
        except Exception as exc:
            get_logger("validation").warning(f"[validate] concept map unreadable ({exc}); all concept lookups will fail closed")
            self._concepts = {}

    # -- rule table -------------------------------------------------------

    def _rules(self) -> list[_Rule]:
        """Plausibility bounds, read from configuration - not hardcoded."""
        s = self.s
        return [
            _Rule("temperature_c", EntityType.TEMPERATURE,
                  s.temp_c_min, s.temp_c_max,
                  s.temp_urgent_low, s.temp_urgent_high, "C"),
            _Rule("bp_systolic", EntityType.BLOOD_PRESSURE,
                  s.bp_sys_min, s.bp_sys_max,
                  s.bp_sys_urgent_low, s.bp_sys_urgent_high, "mmHg"),
            _Rule("bp_diastolic", EntityType.BLOOD_PRESSURE,
                  s.bp_dia_min, s.bp_dia_max,
                  s.bp_dia_urgent_low, s.bp_dia_urgent_high, "mmHg"),
            _Rule("pulse", EntityType.PULSE,
                  s.pulse_min, s.pulse_max,
                  s.pulse_urgent_low, s.pulse_urgent_high, "beats/min"),
            _Rule("respiratory_rate", EntityType.RESPIRATORY_RATE,
                  s.rr_min, s.rr_max,
                  s.rr_urgent_low, s.rr_urgent_high, "breaths/min"),
            _Rule("spo2", EntityType.OXYGEN_SATURATION,
                  s.spo2_min, s.spo2_max,
                  s.spo2_urgent_low, None, "%"),
        ]

    # -- entry point ------------------------------------------------------

    def validate(self, req: ValidationRequest) -> ValidationResponse:
        started = time.perf_counter()
        rid = f"val-{time.time_ns() % 1_000_000_000}"

        observations: list[ObservationField] = []
        issues: list[ValidationIssue] = []
        rejected: list[dict[str, Any]] = []
        payload: dict[str, Any] = {}

        # --- vitals ---
        vitals = self._collect_vitals(req.entities, issues, rejected)
        payload.update(vitals)

        for key, rule in self._rules_map().items():
            value = vitals.get(key)
            if value is None:
                continue
            normalised, issue = self._plausibility(key, rule, value)
            if issue:
                issues.append(issue)
            observations.append(
                ObservationField(
                    concept_uuid=self._concept_uuid(key),
                    concept_name=self._concept_display(key),
                    value=normalised,
                    unit=rule.unit,
                    status=(
                        ValidationStatus.VALID
                        if issue is None
                        else (
                            ValidationStatus.REQUIRES_REVIEW
                            if issue.severity is not Severity.PARSE_ERROR
                            else ValidationStatus.REJECTED
                        )
                    ),
                    source_entity_text=self._source_text(req.entities, rule),
                )
            )

        # --- symptom / diagnosis / medication / allergy collections ---
        for key, etype in (
            ("symptoms", EntityType.SYMPTOM),
            ("diagnoses", EntityType.DIAGNOSIS),
            ("medications", EntityType.MEDICATION),
            ("allergies", EntityType.ALLERGY),
        ):
            texts = self._collect_texts(req.entities, etype)
            if texts:
                payload[key] = texts
                observations.append(
                    ObservationField(
                        concept_uuid=self._concept_uuid(key),
                        concept_name=self._concept_display(key),
                        value=texts,
                        unit=None,
                        status=ValidationStatus.VALID,
                    )
                )

        # --- assertion safety gate ---
        negated = [e for e in req.entities if e.assertion is Assertion.NEGATED]
        for e in negated:
            issues.append(
                ValidationIssue(
                    field=e.type.value,
                    severity=Severity.PLAUSIBILITY_WARNING,
                    message=(
                        f'"{e.text}" is negated in the source text and was NOT '
                        "included as a positive finding. It is retained here for "
                        "the reviewing clinician."
                    ),
                    raw_value=e.text,
                )
            )
        for e in req.entities:
            if e.assertion is Assertion.HISTORICAL:
                issues.append(
                    ValidationIssue(
                        field=e.type.value,
                        severity=Severity.PLAUSIBILITY_WARNING,
                        message=(
                            f'"{e.text}" is stated as a past finding and was NOT '
                            "included as a current positive finding. It is retained "
                            "here for the reviewing clinician, who can re-assert it "
                            "if it is still active."
                        ),
                        raw_value=e.text,
                    )
                )
        for e in req.entities:
            if e.assertion is Assertion.UNCERTAIN:
                issues.append(
                    ValidationIssue(
                        field=e.type.value,
                        severity=Severity.PLAUSIBILITY_WARNING,
                        message=(
                            f'"{e.text}" is stated with uncertainty. Confirm before '
                            "treating it as an established finding."
                        ),
                        raw_value=e.text,
                    )
                )

        # --- missing concept mapping is a configuration failure, not a guess ---
        missing = [o.concept_name for o in observations if o.concept_uuid is None]
        for name in dict.fromkeys(missing):
            issues.append(
                ValidationIssue(
                    field=name,
                    severity=Severity.CONFIG_MISSING,
                    message=(
                        f"No OpenMRS concept UUID is configured for '{name}'. "
                        "Refusing to submit: configure this in "
                        "backend/config/concepts.json or set it from the OpenMRS "
                        "instance. No UUID has been invented."
                    ),
                )
            )

        status = self._overall_status(observations, issues, rejected)
        config_missing = [i for i in issues if i.severity is Severity.CONFIG_MISSING]
        payload["validation_status"] = status.value
        payload["issue_count"] = len(issues)
        payload["rejected_count"] = len(rejected)
        payload["requires_human_confirmation"] = True
        # Submission is blocked by an unconfigured concept *or* by a value the
        # clinician has not yet accepted. These are two different reasons, so
        # they are reported separately.
        payload["configuration_missing"] = sorted({i.field for i in config_missing})
        # "can submit" means the values are clinically acceptable *and* every
        # concept is configured. REQUIRES_REVIEW is deliberately excluded: the
        # clinician has not yet accepted those values, so they are not submittable
        # even though nothing about them is wrong.
        payload["can_submit"] = (
            not config_missing and status is ValidationStatus.VALID
        )

        elapsed = int((time.perf_counter() - started) * 1000)
        return ValidationResponse(
            observations=observations,
            issues=issues,
            status=status,
            rejected=rejected,
            payload=payload,
            configuration_missing=sorted({i.field for i in config_missing}),
            can_submit=not config_missing and status is ValidationStatus.VALID,
            elapsed_ms=elapsed,
            request_id=rid,
        )

    # -- helpers ----------------------------------------------------------

    def _rules_map(self) -> dict[str, _Rule]:
        return {r.key: r for r in self._rules()}

    def _collect_vitals(
        self,
        entities: list[ClinicalEntity],
        issues: list[ValidationIssue],
        rejected: list[dict[str, Any]],
    ) -> dict[str, float]:
        out: dict[str, float] = {}

        for e in entities:
            if e.assertion is Assertion.NEGATED:
                continue

            if e.type is EntityType.TEMPERATURE:
                v = self._parse_temperature(e, issues, rejected)
                if v is not None:
                    out["temperature_c"] = v

            elif e.type is EntityType.BLOOD_PRESSURE:
                sys_v, dia_v = self._parse_bp(e, issues, rejected)
                if sys_v is not None:
                    out["bp_systolic"] = sys_v
                if dia_v is not None:
                    out["bp_diastolic"] = dia_v

            elif e.type is EntityType.PULSE:
                v = self._parse_int(e, issues, rejected, "pulse")
                if v is not None:
                    out["pulse"] = v

            elif e.type is EntityType.RESPIRATORY_RATE:
                v = self._parse_int(e, issues, rejected, "respiratory_rate")
                if v is not None:
                    out["respiratory_rate"] = v

            elif e.type is EntityType.OXYGEN_SATURATION:
                v = self._parse_float(e, issues, rejected, "spo2")
                if v is not None:
                    out["spo2"] = v

        return out

    def _parse_temperature(
        self, e: ClinicalEntity, issues: list[ValidationIssue], rejected: list[dict]
    ) -> float | None:
        """Read the temperature, preferring Algorithm 2's parsed value.

        Algorithm 2 already stripped the label word ("temperature") and detected
        the unit; re-parsing the raw span here with a bare number regex is what
        made "temperature 38.5" report the label's own digits. The regex below
        is a fallback for entities built by the rule/stub backends, which have
        no value_candidate.
        """
        raw = e.text
        value: float | None = None
        unit = "C"

        if isinstance(e.value_candidate, (int, float)):
            value = float(e.value_candidate)
            unit = (e.unit_candidate or "C").upper()
        else:
            m = re.search(r"(\d{2,3}(?:\.\d+)?)", raw)
            if not m:
                issues.append(
                    ValidationIssue(
                        field=self._concept_display("temperature_c"),
                        severity=Severity.PARSE_ERROR,
                        message=f'Could not read a numeric temperature from "{raw}".',
                        raw_value=raw,
                    )
                )
                rejected.append(
                    {
                        "field": self._concept_display("temperature_c"),
                        "raw": raw,
                        "reason": "no_numeric_value",
                    }
                )
                return None
            value = float(m.group(1))
            if re.search(r"\d\s*°?\s*F\b|degrees?\s*fahrenheit", raw, re.IGNORECASE):
                unit = "F"

        if unit == "F":
            issues.append(
                ValidationIssue(
                    field=self._concept_display("temperature_c"),
                    severity=Severity.PLAUSIBILITY_WARNING,
                    message=(
                        f"Source unit is Fahrenheit ({value} F); normalised to "
                        f"{_F_TO_C(value):.2f} C for validation and storage."
                    ),
                    raw_value=raw,
                )
            )
            value = round(_F_TO_C(value), 2)
        return value

    def _parse_bp(
        self, e: ClinicalEntity, issues: list[ValidationIssue], rejected: list[dict]
    ) -> tuple[float | None, float | None]:
        # Algorithm 2 returns the un-split "120/80" string as value_candidate for
        # blood pressure; a bare number regex over e.text would read the label
        # digits instead.
        raw = e.value_candidate if isinstance(e.value_candidate, str) else e.text
        m = re.search(r"(\d{2,3})\s*(?:/|over)\s*(\d{2,3})", raw)
        if not m:
            issues.append(
                ValidationIssue(
                    field=self._concept_display("bp_systolic"),
                    severity=Severity.PARSE_ERROR,
                    message=(
                        f'Could not split a systolic/diastolic pair from "{e.text}". '
                        "Expected a form like 120/80. Not guessed."
                    ),
                    raw_value=e.text,
                )
            )
            rejected.append(
                {
                    "field": self._concept_display("bp_systolic"),
                    "raw": e.text,
                    "reason": "unparsable_pair",
                }
            )
            return None, None

        s, d = int(m.group(1)), int(m.group(2))
        if s <= d:
            issues.append(
                ValidationIssue(
                    field=self._concept_display("bp_systolic"),
                    severity=Severity.PLAUSIBILITY_WARNING,
                    message=(
                        f"Systolic ({s}) is not greater than diastolic ({d}). This "
                        "is not a valid blood pressure pairing; both values are "
                        "preserved unchanged for review."
                    ),
                    raw_value=e.text,
                )
            )
        return float(s), float(d)

    def _parse_int(
        self,
        e: ClinicalEntity,
        issues: list[ValidationIssue],
        rejected: list[dict],
        key: str,
    ) -> float | None:
        return self._parse_float(e, issues, rejected, key)

    def _parse_float(
        self,
        e: ClinicalEntity,
        issues: list[ValidationIssue],
        rejected: list[dict],
        key: str,
    ) -> float | None:
        # Prefer the number Algorithm 2 already isolated. Parsing e.text with a
        # bare digit regex reads the "2" out of "SpO2" - a nonsense saturation.
        value = e.value_candidate if isinstance(e.value_candidate, (int, float)) else None
        if value is not None:
            return float(value)

        m = re.search(r"(\d{1,3}(?:\.\d+)?)", e.text)
        if not m:
            issues.append(
                ValidationIssue(
                    field=self._concept_display(key),
                    severity=Severity.PARSE_ERROR,
                    message=f'Could not read a numeric value from "{e.text}".',
                    raw_value=e.text,
                )
            )
            rejected.append(
                {
                    "field": self._concept_display(key),
                    "raw": e.text,
                    "reason": "no_numeric_value",
                }
            )
            return None
        return float(m.group(1))

    def _plausibility(
        self, key: str, rule: _Rule, value: float
    ) -> tuple[float, ValidationIssue | None]:
        """Bounds check. Alters nothing; only reports.

        Order matters. The plausible band is checked first, but the urgent band
        is evaluated *independently* - it must not be nested inside the
        plausible test. A 41 C reading is a genuine measurement and a
        hyperpyrexia emergency; nesting the urgent check inside the plausible
        range let it pass because 41 < 45.
        """

        # 1. Not a value a living patient can have: almost certainly a
        #    transcription or unit error. Reported, never corrected.
        if value < rule.lo or value > rule.hi:
            return value, ValidationIssue(
                field=self._concept_display(key),
                severity=Severity.PLAUSIBILITY_WARNING,
                message=(
                    f"{value:g} {rule.unit} is outside the plausible range for a "
                    f"living patient ({rule.lo:g}-{rule.hi:g} {rule.unit}). This "
                    "usually means a transcription or unit error. The value is "
                    "unchanged and flagged for human confirmation - it has NOT "
                    "been corrected or discarded."
                ),
                raw_value=f"{value:g}",
            )

        # 2. Physically possible but immediately life-threatening.
        if _is_urgent(rule, value):
            lo = "no lower limit" if rule.urgent_lo is None else f"{rule.urgent_lo:g}"
            hi = "no upper limit" if rule.urgent_hi is None else f"{rule.urgent_hi:g}"
            return value, ValidationIssue(
                field=self._concept_display(key),
                severity=Severity.URGENT_ALERT,
                message=(
                    f"{value:g} {rule.unit} is within the physically possible range "
                    f"but outside the safe band ({lo}-{hi} {rule.unit}). Treat as "
                    "clinically urgent. The value is unchanged and must be "
                    "confirmed by a clinician."
                ),
                raw_value=f"{value:g}",
            )

        return value, None

    # Words the NER model tags as "problem" that are not clinical findings.
    # The checkpoint has a single generic `problem` label and frequently tags
    # the framing word "patient" (and the patient's name) as a problem, which
    # then lands in the symptom list. These are dropped here rather than in the
    # model, so the rule and real backends stay comparable.
    _NON_FINDING = frozenset(
        {
            "patient", "patient's", "pt", "client", "subject",
            "he", "she", "they", "him", "her", "them",
            "history", "report", "notes", "note", "today",
            "complains", "complaining", "denies", "denied",
            "presents", "presented", "presents with", "known for",
            "assessment", "plan", "review", "none", "no", "not",
        }
    )

    def _collect_texts(
        self, entities: list[ClinicalEntity], etype: EntityType
    ) -> list[str]:
        seen: dict[str, None] = {}
        for e in entities:
            if e.type is not etype or e.assertion is not Assertion.PRESENT:
                continue
            text = e.text.strip()
            # A bare stopword is a model artefact, not a finding. Keep it out of
            # the payload - inventing a symptom would be worse than dropping one.
            if text.lower() in self._NON_FINDING:
                continue
            # A span that is only a person's name is not a clinical finding.
            if etype is EntityType.SYMPTOM and len(text.split()) == 1 and text[:1].isupper():
                continue
            seen.setdefault(text.lower(), None)
        return list(seen)

    def _overall_status(
        self,
        observations: list[ObservationField],
        issues: list[ValidationIssue],
        rejected: list[dict],
    ) -> ValidationStatus:
        # CONFIG_MISSING never changes the clinical verdict: it says the site
        # has not wired up a concept, not that the value is wrong. Counting it
        # as a PARSE_ERROR here would report a correct reading of 120/80 as
        # REJECTED merely because concepts.json is unconfigured.
        clinical = [i for i in issues if i.severity is not Severity.CONFIG_MISSING]
        if any(i.severity is Severity.PARSE_ERROR for i in clinical):
            return ValidationStatus.REJECTED
        if observations and any(o.status is not ValidationStatus.VALID for o in observations):
            return ValidationStatus.REQUIRES_REVIEW
        if clinical or rejected:
            return ValidationStatus.REQUIRES_REVIEW
        if not observations:
            return ValidationStatus.REJECTED
        return ValidationStatus.VALID

    #: Rule keys are plural where the registry is singular. English cannot
    #: recover "diagnosis" from "diagnoses" mechanically, so the irregulars are
    #: stated explicitly rather than guessed.
    _PLURAL_TO_SINGULAR = {
        "symptoms": "symptom",
        "diagnoses": "diagnosis",
        "medications": "medication",
        "allergies": "allergy",
    }

    def _concept_entry(self, key: str) -> dict[str, Any] | None:
        """Resolve a rule key to its registry entry, or None when unmapped.

        Rule keys and registry keys do not always agree: the rules say
        ``temperature_c`` where the registry says ``temperature``, and the
        collection rules say ``symptoms`` where the registry says ``symptom``.
        The UUID lookup and the display lookup MUST go through this one
        function. If they resolved independently, the UI could name a concept
        one way on its observation row and a different way on its alert, and a
        missing entry would surface as a raw machine key such as ``symptoms``.
        """
        stem = key.replace("_c", "")
        alias = self._PLURAL_TO_SINGULAR.get(key) or self._PLURAL_TO_SINGULAR.get(stem)
        candidates = [key, stem]
        if alias:
            candidates.append(alias)
        if stem.endswith("s") and not stem.endswith("ss"):
            candidates.append(stem[:-1])
        for cand in candidates:
            entry = self._concepts.get(cand)
            if isinstance(entry, dict):
                return entry
        return None

    def _concept_uuid(self, key: str) -> str | None:
        entry = self._concept_entry(key)
        if entry is None:
            return None
        return entry.get("uuid")

    def _concept_display(self, key: str) -> str:
        entry = self._concept_entry(key)
        if entry is None:
            # Nothing in the registry matches. This is reached for real when the
            # concept map cannot be read (see _load_concepts), not only in the
            # hypothetical case the original comment assumed. Returning the raw key
            # would put a machine identifier on screen and into the review queue,
            # so the last resort is a readable form of it - never the key itself.
            return self._humanise(key)
        return str(entry.get("display") or self._humanise(key))

    @staticmethod
    def _humanise(key: str) -> str:
        """A readable label from a concept key, used only when the registry is absent.

        Deliberately not a lookup table: a second source of truth for display
        names would drift from the registry and could disagree with what OpenMRS
        actually stores under.
        """

        text = key.replace("_c", " ").replace("_", " ")
        return " ".join(w for w in text.split() if w).title()

    def _source_text(
        self, entities: list[ClinicalEntity], rule: _Rule
    ) -> str | None:
        for e in entities:
            if e.type is rule.entity:
                return e.text
        return None


def _is_urgent(rule: _Rule, value: float) -> bool:
    if rule.urgent_lo is not None and value < rule.urgent_lo:
        return True
    if rule.urgent_hi is not None and value > rule.urgent_hi:
        return True
    return False
