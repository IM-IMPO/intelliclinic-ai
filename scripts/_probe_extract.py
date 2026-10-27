"""In-process end-to-end probe of both pipeline routes."""
import sys, json
sys.path.insert(0, "../backend")
from app.pipeline import Pipeline
from app.schemas import ChatRequest

CASES = [
    "normal data", "Patient Juan has a fever of 38.5 C and BP is 120/80, complains of cough.",
    "abnormal", "Patient Maria temperature 41 C, pulse 210, BP 70/40.",
    "hypoxia", "SpO2 85 percent on room air, respiratory rate 32.",
    "negated", "Patient has no fever and no cough today.",
    "diagnosis+med", "Diagnosed with community acquired pneumonia, started on amoxicillin 500 mg three times daily.",
    "impossible", "Temperature is 210 C.",
    "history", "History of hypertension, no chest pain.",
    "question", "What is the recommended dose of amoxicillin for adults?",
]

p = Pipeline()
for i in range(0, len(CASES), 2):
    label, msg = CASES[i], CASES[i + 1]
    r = p.process(ChatRequest(message=msg))
    print("=" * 70)
    print(f"[{label}] {msg}")
    print(f"   route={r.route.value} conf={r.intent.confidence} backend={r.intent.backend}")
    v = r.validation
    if v is not None:
        print(f"   status={v.status.value} can_submit={v.can_submit} cfg_missing={v.configuration_missing}")
        for o in v.observations:
            print(f"     OBS {o.concept_name} = {o.value!r} [{o.status.value}] uuid={o.concept_uuid}")
        for s in v.issues:
            print(f"     ISSUE {s.severity.value}: {s.message[:96]}")
    else:
        a = r.answer
        print("   NO VALIDATION STAGE | answer:", (a.answer or "")[:110] if a else None)
        if a:
            print("     grounding:", a.grounding_status, "citations:", len(a.sources or []))
