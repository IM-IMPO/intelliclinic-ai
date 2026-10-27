import sys

sys.path.insert(0, ".")
import torch
from app.algorithms.extraction import EntityExtractor

ex = EntityExtractor()
text = "Patient Maria temperature 41 C, pulse 210, BP 70/40."

enc = ex._tok(
    text,
    return_tensors="pt",
    return_offsets_mapping=True,
    truncation=True,
    max_length=512,
)
offsets = enc.pop("offset_mapping")[0].tolist()
ids = enc["input_ids"][0].tolist()
with torch.no_grad():
    logits = ex._model(**enc).logits[0]
preds = logits.argmax(dim=-1).tolist()
labels = ex._id2label

print("LABEL SET:", {k: v for k, v in sorted(labels.items())})
print("=" * 74)
for i, p in enumerate(preds):
    tok = ex._tok.convert_ids_to_tokens(int(ids[i]))
    s, e = offsets[i]
    print(
        f"  [{i:2d}] {tok!r:18s} off=({s:2d},{e:2d}) "
        f"{text[s:e]!r:22s} -> {labels.get(p, p)}"
    )
