import re
import sys

sys.path.insert(0, ".")
from app.algorithms import extraction as E

text = "Allergy penicillin."
spans = E.EntityExtractor()._extract_model(text)
print("raw spans from _extract_model:")
for s in spans:
    print(
        f"  type={s.type.name:12s} start={s.start:3d} end={s.end:3d} "
        f"text={s.text!r} slice={text[s.start:s.end]!r}"
    )

print()
print("head regex on span slice:")
for s in spans:
    m = E._ALLERGY_HEAD_RE.match(text[s.start : s.end])
    print(f"  {text[s.start:s.end]!r} -> match={m is not None} groups={m.groups() if m else None}")

print()
print("after _refine_allergies:")
for s in E._refine_allergies(list(spans), text):
    print(f"  type={s.type.name:12s} start={s.start:3d} end={s.end:3d} text={s.text!r}")
