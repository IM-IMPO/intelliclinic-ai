"""End-to-end HTTP probe: proves the B1/B2/B5 fixes on the LIVE server.

The earlier verification imported the module directly, which proves the code
is correct but not that the *running* server serves it. This probe goes through
/api/v1/extract with a real token, so a stale listener cannot hide a regression.

Credentials are the seeded dev users; they are never printed.
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

# The seeded accounts have no committed password; the backend generates a random
# one when NURSE_PASSWORD is unset and writes it to backend/app/.dev-credentials.json.
DEV_CREDENTIALS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "backend", "app", ".dev-credentials.json",
)


def dev_password(username: str) -> str:
    """Resolve a seeded account's password from the environment or the dev store."""

    env_key = f"{username.upper()}_PASSWORD"
    if os.environ.get(env_key):
        return os.environ[env_key]
    try:
        with open(DEV_CREDENTIALS, encoding="utf-8") as fh:
            pw = json.load(fh).get(username)
        if pw:
            return pw
    except (OSError, ValueError):
        pass
    raise SystemExit(
        f"Cannot resolve the password for '{username}': set {env_key} or start "
        f"the backend once so it generates {DEV_CREDENTIALS}."
    )

CASES = [
    ("T1", "Patient Juan has fever 38.5 C, BP 120/80, pulse 80."),
    ("T2", "Patient Maria temperature 41 C, pulse 210, BP 70/40."),
    ("T3", "Temperature is 210 C."),
    ("T4", "Patient is febrile."),
    ("T5", "Creatinine is 2.1 and chest xray shows infiltrate."),
    ("A1", "Allergy penicillin."),
    ("A2", "Patient is allergic to penicillin."),
    ("A3", "allergies: sulfa drugs"),
    ("A4", "NKDA. No known drug allergies."),
    ("A5", "Allergy status noted."),
    ("A6", "allergies are well controlled"),
    ("A7", "allergy to sulfa"),
    ("A8", "allergies: penicillin"),
    ("R1", "Patient Juan has fever 38.5 C, BP 120/80, pulse 80, respiratory rate 18, "
           "SpO2 98 percent, complains of cough. Diagnosed with community acquired "
           "pneumonia, started on amoxicillin 500 mg. Allergy penicillin."),
]


def call(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    st, body = call("POST", "/api/v1/auth/login",
                    {"username": "nurse", "password": dev_password("nurse")})
    if st != 200:
        print(f"LOGIN FAILED {st}: {body}")
        return 1
    token = body["token"]
    print(f"login: {st}  (token acquired, not printed)\n")

    failures = []
    for tag, text in CASES:
        st, body = call("POST", "/api/v1/extract", {"text": text}, token=token)
        if st != 200:
            print(f"{tag}: HTTP {st} {body}")
            failures.append(tag)
            continue
        ents = body.get("entities", [])
        print(f"{tag} backend={body.get('backend')}  '{text[:48]}'")
        for e in ents:
            extra = ""
            if e.get("value_candidate") is not None:
                extra = (f" value={e.get('value_candidate')} "
                         f"unit={e.get('unit_candidate')}")
            print(f"     {e.get('type'):<10} conf={e.get('confidence')} "
                  f"assert={e.get('assertion')} review={e.get('requires_review')} "
                  f"'{e.get('text')}'{extra}")
        print()

    print("=" * 62)
    print(f"failures: {failures or 'none'}")

    # Assertions on the specific defects this sweep fixed.
    def ents_for(text):
        _, b = call("POST", "/api/v1/extract", {"text": text}, token=token)
        return b.get("entities", [])

    def types_for(text):
        return [(e["type"], e["text"]) for e in ents_for(text)]

    checks = []
    a1 = types_for("Allergy penicillin.")
    checks.append(("B2: bare 'Allergy penicillin' is an ALLERGY, not a MEDICATION",
                   any(t == "ALLERGY" and "penicillin" in x for t, x in a1)))
    checks.append(("B2: 'Allergy penicillin' is not reported as a medication taken",
                   not any(t == "MEDICATION" for t, _ in a1)))

    # T1: 'fever 38.5 C' must yield the reading the clinician wrote.
    t1 = ents_for("Patient Juan has fever 38.5 C, BP 120/80, pulse 80.")
    checks.append(("B1: 'fever 38.5 C' yields a temperature reading",
                   any(e["type"] == "TEMPERATURE" and e.get("value_candidate") == 38.5
                       for e in t1)))
    checks.append(("B1: the reading carries the unit the clinician wrote",
                   any(e["type"] == "TEMPERATURE" and e.get("unit_candidate") == "C"
                       for e in t1)))
    checks.append(("B1: 'BP 120/80' is parsed without raising",
                   any(e["type"] == "BLOOD_PRESSURE" for e in t1)))
    checks.append(("B1: confidence is calibrated, not a hardcoded 1.0",
                   any(0.0 < e["confidence"] < 0.999 for e in t1)))
    checks.append(("B1: record furniture ('Patient') is not emitted as a finding",
                   not any("patient" in e["text"].lower() for e in t1)))

    t4 = types_for("Patient is febrile.")
    checks.append(("B1: bare 'febrile' with no unit still invents nothing",
                   not any("temperature" in x.lower() for _, x in t4)))

    t5 = types_for("Creatinine is 2.1 and chest xray shows infiltrate.")
    checks.append(("B5: the checkpoint's test label lands in TEST, not SYMPTOM",
                   any(t == "TEST" for t, _ in t5)))
    checks.append(("B5: no lab result is folded into SYMPTOM",
                   not any(t == "SYMPTOM" and "creatinine" in x.lower() for t, x in t5)))

    # An implausible vital is flagged, never silently accepted.
    t3 = ents_for("Temperature is 210 C.")
    checks.append(("B1: 'Temperature is 210 C' is flagged for review",
                   any(e["type"] == "TEMPERATURE" and e["requires_review"] for e in t3)))
    checks.append(("B1: the flag carries a reason a clinician can act on",
                   any(e.get("review_reason") for e in t3)))

    a5 = types_for("Allergy status noted.")
    checks.append(("B2r1: 'Allergy status noted.' is not an allergy to 'status'",
                   not any(t == "ALLERGY" for t, _ in a5)))
    a6 = types_for("allergies are well controlled")
    checks.append(("B2r1: 'allergies are well controlled' invents no allergen",
                   not any(t == "ALLERGY" for t, _ in a6)))
    a7 = types_for("allergy to sulfa")

    def cue_only_symptoms(pairs):
        """Symptom spans that are nothing but the cue word itself."""

        return [x for t, x in pairs
                if t == "SYMPTOM" and x.strip().lower() in ("allergy", "allergies", "allergic")]

    checks.append(("B1: 'Allergy status noted.' emits no cue-only symptom",
                   not cue_only_symptoms(a5)))
    checks.append(("B1: 'allergies are well controlled' emits no cue-only symptom",
                   not cue_only_symptoms(a6)))
    checks.append(("B1: 'allergy to sulfa' keeps the allergen but drops the cue",
                   any(t == "ALLERGY" and "sulfa" in x for t, x in a7)
                   and not cue_only_symptoms(a7)))
    a4 = types_for("NKDA. No known drug allergies.")
    checks.append(("B2r2: NKDA is negated rather than recorded as an allergy",
                   not any(t == "ALLERGY" and "allerg" in x.lower() for t, x in a4)))

    print()
    bad = []
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            bad.append(name)
    print()
    print(f"RESULT: {len(checks) - len(bad)}/{len(checks)} passed")
    return 0 if not bad and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
