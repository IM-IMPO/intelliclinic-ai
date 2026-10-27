"""Deterministic vital-sign anchoring.

Why this module exists alongside Algorithm 2 instead of inside it.

The Bio_ClinicalBERT checkpoint we run (``samrawal/bert-base-uncased_clinical-ner``)
has exactly four labels: ``problem``, ``treatment``, ``test`` and ``O``. It has
**no label for vital signs**, so a blood pressure reading is emitted as a
"problem" span (``BP``) and a temperature is missed entirely. That is a property
of the checkpoint, not a bug in the extraction code, and no amount of prompt
tweaking changes it.

Vital signs are therefore anchored by a deterministic scanner that looks for
``label + value + unit`` patterns (``BP is 120/80``, ``temperature 38.5 C``,
``SpO2 91%``). This is the correct division of labour:

* the neural model finds **free-text clinical language** - symptoms,
  diagnoses, medications - where patterns are unbounded and rules would fail;
* the deterministic scanner finds **structured measurements**, where patterns
  are fixed and where an ML model is the less reliable tool.

The two are merged with de-duplication by character span, so a vital sign is
never emitted twice. The scanner only reports what it actually matched: it never
invents a number, and it never widens a match to include a neighbouring token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import Assertion, EntityType


@dataclass(frozen=True)
class VitalMatch:
    type: EntityType
    start: int
    end: int
    value_candidate: float | None
    unit_candidate: str | None


# Ordered most-specific first so that a temperature pattern cannot be swallowed
# by a generic numeric rule.
_PATTERNS: list[tuple[EntityType, re.Pattern[str], str, bool]] = [
    # blood pressure: "BP 120/80", "blood pressure is 120/80 mmHg"
    (
        EntityType.BLOOD_PRESSURE,
        re.compile(
            r"(?:blood\s+pressure|bp)\s*(?:is|of|was|at|:)?\s*"
            r"(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\s*(?:mm\s*hg|mmhg)?",
            re.IGNORECASE,
        ),
        "mmHg",
        False,
    ),
    # temperature with an explicit unit. "fever" is a label keyword as well as
    # "fever of": a bare "fever 38.5 C" is the most common way a nurse writes a
    # temperature, and requiring "fever of" silently dropped the reading. The
    # unit is still mandatory, so the symptom word alone can never become a
    # temperature (we never invent a unit for a bare "fever").
    (
        EntityType.TEMPERATURE,
        re.compile(
            r"(?:temperature|temp|fever(?:\s+of)?)\s*(?:is|of|was|at|:)?\s*"
            r"(\d{2,3}(?:\.\d+)?)\s*(?:°\s*)?([CF])\b",
            re.IGNORECASE,
        ),
        "C",
        True,
    ),
    (
        EntityType.TEMPERATURE,
        re.compile(
            r"(?:temperature|temp|fever(?:\s+of)?)\s*(?:is|of|was|at|:)?\s*"
            r"(\d{2,3}(?:\.\d+)?)\s*(?:degrees?)?\s*(?:celsius|centigrade|fahrenheit)\b",
            re.IGNORECASE,
        ),
        "C",
        True,
    ),
    # pulse
    (
        EntityType.PULSE,
        re.compile(
            r"(?:pulse|heart\s*rate|hr)\s*(?:is|of|was|at|:)?\s*(\d{2,3})\s*(?:bpm|beats?/\s*min)?\b",
            re.IGNORECASE,
        ),
        "beats/min",
        True,
    ),
    # respiratory rate
    (
        EntityType.RESPIRATORY_RATE,
        re.compile(
            r"(?:respiratory\s*rate|resp\s*rate|respirations|rr)\s*(?:is|of|was|at|:)?\s*"
            r"(\d{1,2})\s*(?:/min|breaths?/\s*min)?\b",
            re.IGNORECASE,
        ),
        "breaths/min",
        True,
    ),
    # oxygen saturation
    (
        EntityType.OXYGEN_SATURATION,
        re.compile(
            r"(?:spo2|o2\s*sat|oxygen\s*saturation|sats?)\s*(?:is|of|was|at|on)?\s*:?\s*"
            r"(\d{2,3})\s*(?:%|percent|perc)?\b",
            re.IGNORECASE,
        ),
        "%",
        True,
    ),
]

_BP_RE = re.compile(r"(\d{2,3})\s*(?:/|over)\s*(\d{2,3})")


def scan_vitals(text: str) -> list[VitalMatch]:
    """Find vital-sign measurements in ``text``. Only real matches are returned."""

    out: list[VitalMatch] = []
    claimed: list[tuple[int, int]] = []

    for etype, pattern, unit, numeric in _PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < b and m.end() > a for a, b in claimed):
                continue
            claimed.append((m.start(), m.end()))

            if etype is EntityType.BLOOD_PRESSURE:
                g = _BP_RE.search(m.group(0))
                value = float(g.group(1)) if g else None
            else:
                raw = m.group(1)
                value = float(raw) if raw is not None else None
                if value is not None and re.search(r"\b(f|fahrenheit)\b", m.group(0), re.IGNORECASE):
                    unit = "F"

            out.append(
                VitalMatch(
                    type=etype,
                    start=m.start(),
                    end=m.end(),
                    value_candidate=value,
                    unit_candidate=unit,
                )
            )

    out.sort(key=lambda v: v.start)
    return out


def assertion_for(match: VitalMatch, text: str) -> Assertion:
    """Negation/history check for a vital match, same rules as Algorithm 2."""

    from .extraction import _HISTORICAL_CUES, _NEGATION_CUES, _UNCERTAINTY_CUES

    prefix = text[max(0, match.start - 60) : match.start].lower()
    for cue in _NEGATION_CUES:
        if prefix.rfind(cue) != -1:
            return Assertion.NEGATED
    for cue in _UNCERTAINTY_CUES:
        if prefix.rfind(cue) != -1:
            return Assertion.UNCERTAIN
    for cue in _HISTORICAL_CUES:
        if prefix.rfind(cue) != -1:
            return Assertion.HISTORICAL
    return Assertion.PRESENT


# Physiologically possible ranges, in the unit the scanner reports (deg C,
# beats/min, breaths/min, %). A value outside these bounds cannot be a real
# measurement: it is a transcription error or a unit mistake. Such a value is
# flagged for human review, never dropped - discarding it would silently lose a
# number the clinician actually typed, and the whole point of this pipeline is
# that a finding must never vanish.
_PLAUSIBLE_RANGES: dict[EntityType, tuple[float, float]] = {
    EntityType.TEMPERATURE: (25.0, 45.0),
    EntityType.PULSE: (20.0, 250.0),
    EntityType.RESPIRATORY_RATE: (4.0, 60.0),
    EntityType.OXYGEN_SATURATION: (50.0, 100.0),
    EntityType.BLOOD_PRESSURE: (40.0, 300.0),
}


def implausibility_reason(
    etype: EntityType, value: float | str | None, unit: str | None
) -> str | None:
    """Explain why a measured value is not physiologically possible, else None.

    Fahrenheit temperatures are checked against their own range rather than
    converted, so a 98.6 F reading is not misjudged against the Celsius bounds.
    A blood-pressure span arrives as the string ``"120/80"``, so its systolic
    component is what gets compared. A value that cannot be read as a number
    is left alone rather than guessed at - inventing a number here would be
    worse than declining to judge it.
    """

    if value is None:
        return None
    bounds = _PLAUSIBLE_RANGES.get(etype)
    if bounds is None:
        return None

    numeric = _as_number(value, etype)
    if numeric is None:
        return None

    lo, hi = bounds
    if etype is EntityType.TEMPERATURE and (unit or "").upper().startswith("F"):
        lo, hi = 77.0, 113.0

    if lo <= numeric <= hi:
        return None
    return (
        f"{etype.value.replace('_', ' ').title()} {numeric:g} {unit or ''} is outside "
        f"the physiologically possible range {lo:g}-{hi:g}{unit or ''}; this is "
        "almost certainly a transcription or unit error and must be confirmed by "
        "a clinician before it is written to a record."
    )


def _as_number(value: float | str, etype: EntityType) -> float | None:
    """Read a comparable number out of a parsed value, or None if unreadable.

    ``_value_candidate`` returns a float for most vitals but the raw
    ``"120/80"`` text for blood pressure, so both shapes have to be accepted.
    """

    if isinstance(value, bool):  # bool is an int subclass; never a vital sign
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    if etype is EntityType.BLOOD_PRESSURE:
        m = re.match(r"\s*(\d{1,3})", value)
        return float(m.group(1)) if m else None
    m = re.match(r"\s*(\d{1,3}(?:\.\d+)?)", value)
    return float(m.group(1)) if m else None
