"""Probe the two backend fixes (temperature anchor, allergy retyping)."""
import sys

sys.path.insert(0, ".")
from app.algorithms.extraction import EntityExtractor
from app.schemas import ExtractionRequest

ex = EntityExtractor()
print(f"backend={ex.backend}  model={getattr(ex, '_model_name', None)}")
print("=" * 78)

CASES = [
    ("T1", "Patient Juan has fever 38.5 C, BP 120/80, pulse 80."),
    ("T2", "fever of 41 C"),
    ("T3", "temperature is 36.6 C"),
    ("T4", "Patient is febrile."),
    ("A1", "Allergy penicillin."),
    ("A2", "Patient is allergic to penicillin."),
    ("A3", "allergies: sulfa drugs"),
    ("A7", "allergy to sulfa"),
    ("A8", "allergies: penicillin"),
    ("T5", "Creatinine is 2.1 and chest xray shows infiltrate."),
    ("A4", "NKDA. No known drug allergies."),
    ("A5", "Allergy status noted."),
    ("A6", "allergies are well controlled"),
    ("R1", "Patient Juan has fever 38.5 C, BP 120/80, pulse 80, respiratory "
           "rate 18, SpO2 98 percent, complains of cough. Diagnosed with "
           "community acquired pneumonia, started on amoxicillin 500 mg. "
           "Allergy penicillin."),
]

for tag, text in CASES:
    ents = ex.extract(ExtractionRequest(text=text))
    print(f"\n[{tag}] {text}")
    if not ents.entities:
        print("    (no entities)")
    for e in ents.entities:
        val = f"{e.value_candidate}{e.unit_candidate or ''}" if e.value_candidate else "-"
        print(
            f"    {e.type.name:20s} conf={e.confidence:.2f} "
            f"assert={e.assertion.value:10s} text={e.text!r:34s} val={val} "
            f"review={e.requires_review}"
        )
        if e.review_reason:
            print(f"        reason: {e.review_reason}")
