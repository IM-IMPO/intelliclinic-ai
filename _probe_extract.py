import sys

sys.path.insert(0, ".")
from app.algorithms.extraction import EntityExtractor
from app.schemas import ExtractionRequest

ex = EntityExtractor()
print("backend:", ex.backend, "| model:", ex._model_name)
print("=" * 74)

CASES = [
    "Patient Juan has fever 38.5 C, BP 120/80, pulse 80, complains of cough.",
    "Allergy penicillin.",
    "Patient is allergic to penicillin.",
    "Patient Maria temperature 41 C, pulse 210, BP 70/40.",
    "Temperature is 210 C.",
    "SpO2 85 percent on room air, respiratory rate 32.",
]

for c in CASES:
    r = ex.extract(ExtractionRequest(text=c))
    print("TEXT:", c)
    for e in r.entities:
        print(
            f"   {e.type.value:22s} conf={e.confidence:.2f} "
            f"assert={e.assertion.value:11s} text={e.text!r} "
            f"val={e.value_candidate!r} unit={e.unit_candidate!r} "
            f"review={e.requires_review}"
        )
    if not r.entities:
        print("   (none)")
    print("-" * 74)
