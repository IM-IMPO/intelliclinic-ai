import sys

sys.path.insert(0, ".")
from app.algorithms.intent import IntentClassifier
from app.schemas import IntentRequest

c = IntentClassifier()
msgs = [
    "Temperature is 210 C.",
    "Temperature is 41 C and heart rate is 210.",
    "What is the recommended dose of amoxicillin for adults?",
    "SpO2 85 percent on room air, respiratory rate 32.",
    "History of hypertension, no chest pain.",
    "Diagnosed with community acquired pneumonia, started on amoxicillin 500 mg three times daily.",
]
for m in msgs:
    r = c.classify(IntentRequest(message=m))
    print(f"{m[:58]:60s} -> {r.result.intent.value:14s} conf={r.result.confidence:.3f} {r.result.backend}")
    if getattr(r.result, "scores", None):
        for k, v in sorted(r.scores.items(), key=lambda kv: -kv[1])[:4]:
            print(f"      {k:22s} {v:.4f}")
    print()
