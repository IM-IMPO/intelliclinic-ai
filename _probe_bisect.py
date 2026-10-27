import re
import sys

sys.path.insert(0, ".")
from app.algorithms import extraction as E

BS = chr(92)
p = E._ALLERGY_HEAD_RE.pattern
print("pattern with <BS> markers:", p.replace(BS, "<BS>"))
print("len:", len(p), " backslashes:", p.count(BS))
print()

s = "Allergy penicillin"
steps = [
    "(?i)^",
    "(?i)^" + BS + "s*",
    "(?i)^" + BS + "s*allergies?",
    "(?i)^" + BS + "s*(?:allergic" + BS + "s+to|allergies?)",
    "(?i)^" + BS + "s*(?:allergic" + BS + "s+to|allergies?)" + BS + "s*:?",
    "(?i)^" + BS + "s*(?:allergic" + BS + "s+to|allergies?)" + BS + "s*:?" + BS + "s*",
    "(?i)^" + BS + "s*(?:allergic" + BS + "s+to|allergies?)" + BS + "s*:?" + BS + "s*([A-Za-z][A-Za-z0-9 -]{1,40})$",
]
for t in steps:
    print(f"{str(bool(re.match(t, s))):5s}  {t.replace(BS, '<BS>')}")

print()
print("compiled HEAD_RE.search:", E._ALLERGY_HEAD_RE.search(s))
print("compiled HEAD_RE.match :", E._ALLERGY_HEAD_RE.match(s))
