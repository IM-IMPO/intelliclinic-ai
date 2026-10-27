#!/usr/bin/env python3
"""Train the Algorithm 1 intent classifier and persist a portable artifact.

This produces ``backend/config/intent_model.json`` - a real logistic-regression
text classifier over TF-IDF n-grams, trained on ``intent_train.jsonl``. The
artifact stores weights only (no patient data), so it can be committed and
loaded without sklearn's pickle machinery.

Usage:
    python scripts/train_intent_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> int:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    src = ROOT / "backend" / "config" / "intent_train.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]

    if len(set(labels)) < 2:
        print("need at least two classes")
        return 1

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        token_pattern=r"(?u)\b\w[\w/°-]*\b",
        min_df=1,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    y = np.array(labels, dtype=object)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X, y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(clf, X, y, cv=skf, scoring="f1_weighted")

    preds = clf.predict(X)
    print(f"classes: {list(clf.classes_)}")
    print(f"features: {X.shape[1]}  rows: {X.shape[0]}")
    print(f"5-fold CV weighted F1: {cv.mean():.4f} +/- {cv.std():.4f}")
    print(classification_report(y, preds, digits=3))

    # Probability calibration sanity on the acceptance-criteria examples.
    probes = {
        "clear patient data": "Patient Juan has a fever of 38.5 C and BP is 120/80, complains of cough.",
        "clear question": "How many weeks should I wait between OPV dose 1 and dose 2?",
        "empty-ish": "hello",
    }
    for name, t in probes.items():
        p = clf.predict_proba(vec.transform([t]))[0]
        print(f"  probe[{name}] -> " + ", ".join(f"{c}={v:.3f}" for c, v in zip(clf.classes_, p)))

    out = {
        "metadata": {
            "model_name": "intelliclinic-intent-lr",
            "version": "1.0.0",
            "classes": list(clf.classes_),
            "cv_f1_weighted": round(float(cv.mean()), 4),
            "training_rows": len(rows),
            "algorithm": "TF-IDF (1,2) + LogisticRegression, balanced class weights",
            "threshold_default": 0.72,
            "low_confidence_threshold_default": 0.55,
        },
        "classes": list(clf.classes_),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "vectorizer": {
            "vocabulary": vec.vocabulary_,
            "idf": vec.idf_.tolist(),
            "ngram_range": list(vec.ngram_range),
            "token_pattern": vec.token_pattern,
            "lowercase": vec.lowercase,
        },
    }
    dest = ROOT / "backend" / "config" / "intent_model.json"
    dest.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {dest} ({dest.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
