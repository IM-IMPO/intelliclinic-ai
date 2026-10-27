"""End-to-end acceptance probe against a running server.

Exercises the two pipeline routes, the review/approval gate, RBAC enforcement,
and the knowledge-base lifecycle. Prints a PASS/FAIL line per assertion.

    py -3.12 scripts/smoke_test.py            # default http://127.0.0.1:8011
    py -3.12 scripts/smoke_test.py --base http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8011"
PASS, FAIL = [], []

# The seeded accounts have no committed password. The backend generates a random
# one per account when the matching *_PASSWORD variable is unset and writes it
# to backend/app/.dev-credentials.json. These scripts read that file back (or
# the environment, which wins) instead of hardcoding a credential.
DEV_CREDENTIALS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "backend", "app", ".dev-credentials.json",
)


def dev_password(username: str) -> str:
    """Resolve a seeded account's password from the environment or the dev store.

    Raises rather than guessing: a script that cannot authenticate should fail
    loudly, never fall back to a hardcoded password.
    """

    env_key = f"{username.upper()}_PASSWORD"
    if os.environ.get(env_key):
        return os.environ[env_key]
    try:
        with open(DEV_CREDENTIALS, encoding="utf-8") as fh:
            stored = json.load(fh)
        pw = stored.get(username)
        if pw:
            return pw
    except (OSError, ValueError):
        pass
    raise SystemExit(
        f"Cannot resolve the password for '{username}': set {env_key} or start "
        f"the backend once so it generates {DEV_CREDENTIALS}."
    )


def call(method: str, path: str, body: dict | None = None, token: str | None = None):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw[:400]}


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))


def login(username: str, password: str) -> str | None:
    st, body = call("POST", "/api/v1/auth/login", {"username": username, "password": password})
    if st != 200:
        print(f"login failed for {username}: {st} {body}")
        return None
    return body["token"]


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()
    BASE = args.base.rstrip("/")

    print(f"\n== IntelliClinic AI smoke test against {BASE} ==\n")

    # ---- health / readiness ----
    st, h = call("GET", "/health")
    check("health endpoint", st == 200 and h.get("status") == "ok", f"{st} {h}")

    st, r = call("GET", "/api/v1/system/ready")
    checks = r.get("checks", {})
    check("readiness reports intent classifier", checks.get("intent_classifier") in ("real", "sklearn"), str(checks))
    check("readiness reports validation as real", checks.get("validation") == "real", str(checks))
    check("readiness reports retrieval", checks.get("retrieval") in ("real", "unavailable"), str(checks))
    check("readiness reports answering", checks.get("answering") in ("real", "stub"), str(checks))
    check("openmrs absence is explicit", r.get("openmrs_detail") != "", str(r.get("openmrs_detail")))

    # ---- auth ----
    nurse = login("nurse", dev_password("nurse"))
    doctor = login("doctor", dev_password("doctor"))
    admin = login("admin", dev_password("admin"))
    check("nurse login", bool(nurse))
    check("doctor login", bool(doctor))
    check("admin login", bool(admin))

    st, bad = call("POST", "/api/v1/auth/login", {"username": "nurse", "password": "wrong"})
    check("wrong password rejected", st == 401, f"{st}")

    st, noauth = call("POST", "/api/v1/chat", {"message": "hello"})
    check("unauthenticated chat rejected", st == 401, f"{st}")

    # ---- RBAC ----
    st, forbidden = call("GET", "/api/v1/users", token=nurse)
    check("nurse cannot list users", st == 403, f"{st}")

    st, users = call("GET", "/api/v1/users", token=admin)
    check("admin can list users", st == 200 and len(users.get("users", [])) == 3, f"{st}")

    st, forbidden = call("POST", "/api/v1/knowledge-base/documents",
                         {"document_id": "x", "title": "x", "source_version": "1", "text": "x"},
                         token=nurse)
    check("nurse cannot index documents", st == 403, f"{st}")

    # ---- Algorithm 1 ----
    print("\n-- Algorithm 1: intent --")
    st, intent_data = call("POST", "/api/v1/intent",
                           {"message": "Patient Juan has a fever of 38.5 C and BP is 120/80, complains of cough."},
                           token=nurse)
    check("intent endpoint 200", st == 200, f"{st}")
    res = intent_data.get("result", {})
    check("acceptance patient-data route", res.get("intent") == "ENCODE_DATA",
          f"{res.get('intent')} conf={res.get('confidence')}")
    check("intent reports a real backend", res.get("backend") == "real", str(res.get("backend")))
    check("intent returns model version", bool(res.get("model_version")), str(res.get("model_version")))

    st, q = call("POST", "/api/v1/intent",
                 {"message": "How many weeks should I wait between OPV dose 1 and dose 2?"}, token=nurse)
    check("acceptance question route", q.get("result", {}).get("intent") == "ASK_QUESTION",
          str(q.get("result", {}).get("intent")))

    st, amb = call("POST", "/api/v1/intent", {"message": "hello"}, token=nurse)
    check("ambiguous input is not silently dropped",
          amb.get("result", {}).get("intent") == "AMBIGUOUS"
          and amb.get("result", {}).get("clarification_prompt"),
          f"{amb.get('result', {}).get('intent')}")

    # ---- Algorithm 2 ----
    print("\n-- Algorithm 2: entity extraction --")
    st, ext = call("POST", "/api/v1/extract",
                   {"text": "Patient Juan has a fever of 38.5 C and BP is 120/80, complains of cough."},
                   token=nurse)
    check("extraction endpoint 200", st == 200, f"{st}")
    ents = ext.get("entities", [])
    types = {e["type"] for e in ents}
    check("temperature extracted", "TEMPERATURE" in types, str(sorted(types)))
    check("blood pressure extracted", "BLOOD_PRESSURE" in types, str(sorted(types)))
    check("symptoms extracted", "SYMPTOM" in types, str(sorted(types)))
    check("entities carry offsets", all("start" in e and "end" in e for e in ents))
    check("entities carry confidence", all(e.get("confidence", 0) > 0 for e in ents))
    check("extraction reports backend", ext.get("backend") in ("real", "stub"), str(ext.get("backend")))
    print(f"        backend={ext.get('backend')} model={ext.get('model_name')} entities={len(ents)}")

    st, neg = call("POST", "/api/v1/extract",
                   {"text": "Patient has no fever and no cough today."}, token=nurse)
    neg_ents = [e for e in neg.get("entities", []) if e["type"] == "SYMPTOM"]
    check("negated symptom flagged NEGATED",
          all(e.get("assertion") == "NEGATED" for e in neg_ents) and len(neg_ents) >= 1,
          str([(e["text"], e.get("assertion")) for e in neg_ents]))

    st, missing = call("POST", "/api/v1/extract", {"text": "Patient is doing well today."}, token=nurse)
    check("no invented values on empty input",
          not any(e["type"] in ("TEMPERATURE", "BLOOD_PRESSURE", "PULSE") for e in missing.get("entities", [])),
          str([e["type"] for e in missing.get("entities", [])]))

    # ---- Algorithm 3 ----
    print("\n-- Algorithm 3: validation --")
    st, val = call("POST", "/api/v1/validate",
                   {"entities": ents, "patient_age_years": 44, "patient_sex": "M"}, token=nurse)
    check("validation endpoint 200", st == 200, f"{st}")
    obs = val.get("observations", [])
    temps = [o for o in obs if o.get("concept_name") == "Temperature (C)"]
    check("temperature normalised to celsius",
          any(abs(float(o["value"]) - 38.5) < 0.01 for o in temps), str([o.get("value") for o in temps]))
    bp_sys = [o for o in obs if o.get("concept_name") == "Systolic blood pressure"]
    bp_dia = [o for o in obs if o.get("concept_name") == "Diastolic blood pressure"]
    check("blood pressure split into systolic/diastolic",
          any(float(o["value"]) == 120 for o in bp_sys) and any(float(o["value"]) == 80 for o in bp_dia),
          f"sys={[o['value'] for o in bp_sys]} dia={[o['value'] for o in bp_dia]}")
    check("payload carries validation_status", val.get("payload", {}).get("validation_status") in
          ("VALID", "REQUIRES_REVIEW", "REJECTED"), str(val.get("status")))
    check("human confirmation is required", val.get("payload", {}).get("requires_human_confirmation") is True)
    check("missing concept UUID is reported, not invented",
          any("concept UUID" in i.get("message", "") for i in val.get("issues", []))
          or all(o.get("concept_uuid") for o in obs),
          f"issues={[i.get('field') for i in val.get('issues', [])]}")
    print(f"        status={val.get('status')} observations={len(obs)} issues={len(val.get('issues', []))}")

    st, f = call("POST", "/api/v1/validate",
                 {"entities": [{"type": "TEMPERATURE", "text": "abc", "start": 0, "end": 3,
                                "assertion": "PRESENT", "confidence": 0.9}]}, token=nurse)
    check("malformed temperature rejected with reason",
          f.get("status") == "REJECTED" and f.get("rejected"),
          f"{f.get('status')} {f.get('rejected')}")

    st, impl = call("POST", "/api/v1/validate",
                    {"entities": [{"type": "TEMPERATURE", "text": "42.5 C", "start": 0, "end": 7,
                                   "assertion": "PRESENT", "confidence": 0.95}]}, token=nurse)
    # A value outside the physically possible range (34-42 C) is far more likely
    # a transcription/unit error than a real reading, so the rule issues a
    # PLAUSIBILITY_WARNING and preserves the raw value. URGENT_ALERT is reserved
    # for a value that IS physically possible but clinically dangerous.
    impl_warn = [i for i in impl.get("issues", []) if i.get("severity") == "PLAUSIBILITY_WARNING"]
    check("implausible value warns and is preserved",
          len(impl_warn) >= 1 and any(float(o["value"]) == 42.5 for o in impl.get("observations", [])),
          str(impl_warn))

    st, real_urg = call("POST", "/api/v1/validate",
                        {"entities": [{"type": "TEMPERATURE", "text": "41.0 C", "start": 0, "end": 7,
                                       "assertion": "PRESENT", "confidence": 0.95}]}, token=nurse)
    urg = [i for i in real_urg.get("issues", []) if i.get("severity") == "URGENT_ALERT"]
    check("dangerous-but-possible value raises urgent alert",
          len(urg) >= 1, str(urg))

    st, fahr = call("POST", "/api/v1/validate",
                    {"entities": [{"type": "TEMPERATURE", "text": "102.2 F", "start": 0, "end": 8,
                                   "assertion": "PRESENT", "confidence": 0.95}]}, token=nurse)
    fv = [o["value"] for o in fahr.get("observations", []) if o.get("concept_name") == "Temperature (C)"]
    check("fahrenheit converted to celsius", fv and abs(float(fv[0]) - 39.0) < 0.1, str(fv))

    st, bp_bad = call("POST", "/api/v1/validate",
                      {"entities": [{"type": "BLOOD_PRESSURE", "text": "120/80", "start": 0, "end": 7,
                                     "assertion": "PRESENT", "confidence": 0.95}]}, token=nurse)
    check("blood pressure accepted", bp_bad.get("status") in ("VALID", "REQUIRES_REVIEW"), bp_bad.get("status"))

    # ---- Knowledge base + Algorithm 4 ----
    print("\n-- Algorithm 4: retrieval --")
    doc_text = (
        "Approved Immunization Guidelines, version 2024.1, Oral Polio Vaccine Schedule. "
        "The minimum interval between OPV dose 1 and OPV dose 2 is 4 weeks (28 days). "
        "For catch-up schedules, the interval may be reduced to a minimum of 4 weeks "
        "between any two doses. A child who has received 3 doses is considered fully "
        "immunized against polio. OPV is contraindicated in children with known "
        "severe immunodeficiency. Any adverse event following immunization must be "
        "reported within 48 hours to the national surveillance unit.\n\n"
        "Cold Chain and Storage. All vaccines must be stored between +2 degrees "
        "Celsius and +8 degrees Celsius. The refrigerator temperature must be "
        "recorded twice daily. Vaccines exposed to temperatures outside this range "
        "for more than 60 minutes must be quarantined and reported.\n\n"
        "Measles case management. A measles case is considered infectious from "
        "4 days before to 4 days after the onset of the rash. Isolation is "
        "recommended for 4 days after rash onset.\n\n"
        "Needlestick injury procedure. Wash the site with soap and running water. "
        "Report the incident to the supervisor within 1 hour. Complete the incident "
        "form. Assess hepatitis B and HIV status of both the source patient and the "
        "affected health worker within 24 hours."
    )
    st, indexed = call("POST", "/api/v1/knowledge-base/documents", {
        "document_id": "immunization-guidelines-2024",
        "title": "Approved Immunization Guidelines",
        "source_version": "2024.1",
        "effective_date": "2024-01-01",
        "review_status": "APPROVED",
        "topic": "immunization",
        "text": doc_text,
    }, token=doctor)
    check("document indexed", st == 200 and indexed.get("status") == "INDEXED", f"{st} {indexed}")
    print(f"        chunks indexed = {indexed.get('chunks_indexed')}")

    st, retr = call("POST", "/api/v1/retrieve",
                    {"query": "How many weeks should I wait between OPV dose 1 and dose 2?",
                     "top_k": 3, "approved_only": True}, token=nurse)
    check("retrieval returns passages", len(retr.get("retrieved_documents", [])) > 0,
          str(len(retr.get("retrieved_documents", []))))
    check("retrieval reports evidence status", retr.get("evidence_status") in
          ("SUFFICIENT", "INSUFFICIENT", "CONFLICTING"), str(retr.get("evidence_status")))
    check("retrieved chunks carry a version", all(d.get("source_version") for d in retr.get("retrieved_documents", [])),
          str([d.get("source_version") for d in retr.get("retrieved_documents", [])]))
    check("retrieval is backend-real", retr.get("embedding_model") != "", str(retr.get("embedding_model")))
    print(f"        status={retr.get('evidence_status')} notes={len(retr.get('notes', []))}")
    for d in retr.get("retrieved_documents", [])[:2]:
        print(f"          [{d['similarity_score']}] {d['title']} / {d.get('section')} :: {d['content'][:90]}...")

    st, retr_pending = call("POST", "/api/v1/retrieve", {"query": "OPV interval", "approved_only": True}, token=nurse)
    check("approved_only excludes unapproved", retr_pending.get("evidence_status") == "INSUFFICIENT"
          or all(d.get("approved") for d in retr_pending.get("retrieved_documents", [])),
          str(retr_pending.get("evidence_status")))

    st, empty = call("POST", "/api/v1/retrieve", {"query": "what is the best pizza topping"}, token=nurse)
    check("unrelated query yields INSUFFICIENT",
          empty.get("evidence_status") == "INSUFFICIENT" or not empty.get("retrieved_documents"),
          str(empty.get("evidence_status")))

    # ---- Algorithm 5 ----
    print("\n-- Algorithm 5: grounded answering --")
    chunks = retr.get("retrieved_documents", [])
    st, ans = call("POST", "/api/v1/answer",
                   {"question": "How many weeks should I wait between OPV dose 1 and dose 2?",
                    "context_chunks": chunks}, token=nurse)
    check("answer endpoint 200", st == 200, f"{st}")
    check("answer cites sources", len(ans.get("sources", [])) > 0, str(len(ans.get("sources", []))))
    check("answer has a grounding status", ans.get("grounding_status") in
          ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNVERIFIED", "CONFLICTING"), str(ans.get("grounding_status")))
    check("answer names the real document",
          all(s.get("document_id") == "immunization-guidelines-2024" for s in ans.get("sources", [])),
          str([s.get("document_id") for s in ans.get("sources", [])]))
    check("citations carry version", all(s.get("source_version") for s in ans.get("sources", [])))
    check("answer states it is not a substitute", any("judgement" in l.lower() or "judgment" in l.lower()
                                                      for l in ans.get("limitations", [])),
          str(ans.get("limitations")))
    print(f"        backend={ans.get('backend')} status={ans.get('grounding_status')}")
    print(f"        answer: {ans.get('answer', '')[:300]}")

    st, no_ev = call("POST", "/api/v1/answer",
                     {"question": "How many weeks between OPV dose 1 and dose 2?", "context_chunks": []}, token=nurse)
    check("no evidence -> UNVERIFIED, not fabricated",
          no_ev.get("grounding_status") == "UNVERIFIED" and no_ev.get("sources") == [],
          f"{no_ev.get('grounding_status')} sources={len(no_ev.get('sources', []))}")

    # ---- Full pipeline ----
    print("\n-- Pipeline: data entry route --")
    st, chat = call("POST", "/api/v1/chat",
                    {"message": "Patient Juan has a fever of 38.5 C and BP is 120/80, complains of cough.",
                     "patient_id": "MRN-1001"},
                    token=nurse)
    check("chat endpoint 200", st == 200, f"{st}")
    check("data route selected", chat.get("route") == "ENCODE_DATA", str(chat.get("route")))
    stages = [s["stage"] for s in chat.get("stages", [])]
    check("pipeline stages recorded",
          all(s in stages for s in ("RECEIVED", "INTENT_CLASSIFIED", "ENTITIES_EXTRACTED",
                                     "VALIDATION_COMPLETED", "AWAITING_CONFIRMATION")), str(stages))
    check("nothing auto-saved", chat.get("saved") is False, str(chat.get("saved")))
    check("confirmation required", chat.get("requires_confirmation") is True)
    check("extraction stage present", chat.get("extraction") is not None)
    check("validation stage present", chat.get("validation") is not None)
    check("no retrieval on data route", chat.get("retrieval") is None)
    check("no answer on data route", chat.get("answer") is None)
    print(f"        stages={[s['stage'] for s in chat.get('stages', [])]}")

    rid = chat.get("request_id", "")
    st, item = call("GET", f"/api/v1/review-queue/{rid}", token=nurse)
    check("payload appears in review queue", st == 200 and item.get("observations"), f"{st}")

    st, q_list = call("GET", "/api/v1/review-queue", token=nurse)
    check("review queue lists pending item", any(x.get("request_id") == rid for x in q_list.get("awaiting_review", [])))

    st, nurse_approve = call("POST", f"/api/v1/review-queue/{rid}/approve",
                             {"request_id": rid, "approved_by": "nurse",
                              "idempotency_key": "smoke-nurse-1"}, token=nurse)
    check("nurse cannot approve/submit to OpenMRS", st == 403, f"{st}")

    st, doc_approve = call("POST", f"/api/v1/review-queue/{rid}/approve",
                           {"request_id": rid, "approved_by": "doctor",
                            "idempotency_key": "smoke-doctor-1"}, token=doctor)
    check("doctor approval does not claim success without OpenMRS",
          doc_approve.get("status") in ("UNCONFIGURED", "FAILED", "DUPLICATE", "SUBMITTED"),
          f"{st} {doc_approve}")
    check("unconfigured OpenMRS is stated plainly",
          doc_approve.get("status") == "UNCONFIGURED"
          or "OpenMRS" in doc_approve.get("message", ""),
          doc_approve.get("message", ""))
    print(f"        approve status = {doc_approve.get('status')} :: {doc_approve.get('message', '')[:110]}")

    print("\n-- Pipeline: question route --")
    st, chat2 = call("POST", "/api/v1/chat",
                     {"message": "How many weeks should I wait between OPV dose 1 and dose 2?"}, token=nurse)
    check("question route selected", chat2.get("route") == "ASK_QUESTION", str(chat2.get("route")))
    stages2 = [s["stage"] for s in chat2.get("stages", [])]
    check("question route skips extraction",
          "ENTITIES_EXTRACTED" not in stages2 and "VALIDATION_COMPLETED" not in stages2, str(stages2))
    check("question route retrieves documents", chat2.get("retrieval") is not None)
    check("question route produces an answer", chat2.get("answer") is not None)
    check("no extraction object on question route", chat2.get("extraction") is None)
    check("no validation object on question route", chat2.get("validation") is None)
    print(f"        stages={[s['stage'] for s in chat2.get('stages', [])]}")
    if chat2.get("answer"):
        print(f"        grounding={chat2['answer'].get('grounding_status')} :: {chat2['answer'].get('answer', '')[:220]}")

    print("\n-- Audit --")
    st, audit = call("GET", "/api/v1/audit-logs?limit=10", token=admin)
    check("audit log accessible to admin", st == 200 and audit.get("count", 0) > 0, f"{st}")
    check("audit log contains no free-text patient notes",
          audit.get("patient_identifiable_content_stored") is False)
    st, audit_nurse = call("GET", "/api/v1/audit-logs", token=nurse)
    check("nurse cannot read audit log", st == 403, f"{st}")

    print("\n-- Patients (unavailable state) --")
    st, pats = call("GET", "/api/v1/patients?q=Juan", token=nurse)
    check("patient search reports unavailable honestly",
          st == 200 and pats.get("unavailable") is True and pats.get("results") == [],
          f"{st} {str(pats)[:150]}")

    print("\n-- Concept naming (no machine keys on screen) --")
    # The UI renders concept_name and ValidationIssue.field straight from the
    # API, so a raw registry key reaching either field is a visible defect. A
    # regex cannot catch this - "symptoms" is a perfectly ordinary-looking word
    # - so assert against the source of truth instead: every name the API emits
    # for a concept must be one of the display labels configured in
    # config/concepts.json.
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "backend", "config", "concepts.json",
    )
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    displays = {
        v["display"]
        for v in cfg.get("concepts", {}).values()
        if isinstance(v, dict) and v.get("display")
    }
    st, named = call("POST", "/api/v1/validate", {
        "entities": [
            {"type": "TEMPERATURE", "text": "38.5 C", "start": 0, "end": 6,
             "assertion": "PRESENT", "confidence": 0.95},
            {"type": "BLOOD_PRESSURE", "text": "120/80", "start": 0, "end": 6,
             "assertion": "PRESENT", "confidence": 0.95},
            {"type": "PULSE", "text": "80", "start": 0, "end": 2,
             "assertion": "PRESENT", "confidence": 0.95},
            {"type": "RESPIRATORY_RATE", "text": "18", "start": 0, "end": 2,
             "assertion": "PRESENT", "confidence": 0.95},
            {"type": "OXYGEN_SATURATION", "text": "98 percent", "start": 0, "end": 10,
             "assertion": "PRESENT", "confidence": 0.95},
            {"type": "SYMPTOM", "text": "cough", "start": 0, "end": 5,
             "assertion": "PRESENT", "confidence": 0.9},
            {"type": "DIAGNOSIS", "text": "community acquired pneumonia", "start": 0, "end": 28,
             "assertion": "PRESENT", "confidence": 0.9},
            {"type": "MEDICATION", "text": "amoxicillin", "start": 0, "end": 11,
             "assertion": "PRESENT", "confidence": 0.9},
            {"type": "ALLERGY", "text": "penicillin", "start": 0, "end": 10,
             "assertion": "PRESENT", "confidence": 0.9},
        ],
        "patient_age_years": 44,
        "patient_sex": "M",
    }, token=nurse)
    emitted = [o.get("concept_name") for o in named.get("observations", [])]
    emitted += list(named.get("configuration_missing", []))
    emitted += [i.get("field") for i in named.get("issues", [])
                if i.get("severity") == "CONFIG_MISSING"]
    emitted += [r.get("field") for r in named.get("rejected", [])]
    leaked = sorted({n for n in emitted if n and n not in displays})
    check("every concept is named by a configured display label",
          not leaked, f"not in registry: {leaked}")
    print(f"        registry labels = {len(displays)} :: emitted names = "
          f"{sorted({n for n in emitted if n})}")

    print("\n" + "=" * 62)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  FAILED:")
        for f in FAIL:
            print(f"    - {f}")
    print("=" * 62 + "\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
