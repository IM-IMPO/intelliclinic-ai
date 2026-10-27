"""Verify the two extraction fixes: real confidence and physiological range check.

Run from backend/:  python _verify_fixes.py
"""

from app.algorithms.extraction import EntityExtractor
from app.schemas import ExtractionRequest

CASES = [
    ("T1", "Patient Juan has fever 38.5 C, BP 120/80, pulse 80."),
    ("T2", "Patient Maria temperature 41 C, pulse 210, BP 70/40."),
    ("T3", "Temperature is 210 C."),
    ("T4", "Patient is febrile."),
    ("T5", "Creatinine is 2.1 and chest xray shows infiltrate."),
]

svc = EntityExtractor()
print("=== extraction: real confidence + plausibility ===")
for tag, text in CASES:
    resp = svc.extract(ExtractionRequest(text=text))
    print(f"\n{tag}: {text!r}")
    for e in resp.entities:
        print(
            f"  {e.type.value:<18} {e.text!r:<24} conf={e.confidence:.3f} "
            f"assert={e.assertion.value:<11} review={e.requires_review}"
        )
        if e.review_reason:
            print(f"      reason: {e.review_reason}")

print("\n=== implausibility_reason unit table ===")
from app.algorithms.vitals import implausibility_reason
from app.schemas import EntityType as ET

checks = [
    (ET.TEMPERATURE, 210.0, "C"),
    (ET.TEMPERATURE, 38.5, "C"),
    (ET.TEMPERATURE, 98.6, "F"),
    (ET.TEMPERATURE, 212.0, "F"),
    (ET.PULSE, 210.0, None),
    (ET.PULSE, 80.0, None),
    (ET.OXYGEN_SATURATION, 91.0, "%"),
    (ET.OXYGEN_SATURATION, 3.0, "%"),
    (ET.BLOOD_PRESSURE, 70.0, "mmHg"),
    (ET.BLOOD_PRESSURE, 120.0, "mmHg"),
    (ET.MEDICATION, 500.0, "mg"),
    (ET.TEMPERATURE, None, "C"),
]
for etype, val, unit in checks:
    r = implausibility_reason(etype, val, unit)
    print(
        f"  {etype.value:<20} {str(val):<7} {unit or '':<5} -> {'FLAGGED' if r else 'ok'}"
    )
