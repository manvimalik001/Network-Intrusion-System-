"""predict.py - load saved artifacts and classify one NSL-KDD connection record."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

model = joblib.load(MODELS_DIR / "model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
encoders = joblib.load(MODELS_DIR / "encoders.pkl")
FEATURES = joblib.load(MODELS_DIR / "features.pkl")
LABELS = [str(x) for x in joblib.load(MODELS_DIR / "labels.pkl")]

CAT_COLS = ["protocol_type", "service", "flag"]


def predict(record: dict) -> dict:
    """record: dict with the 41 feature names as keys (raw values, e.g. 'tcp')."""
    missing = [f for f in FEATURES if f not in record]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    row = pd.DataFrame([{f: record[f] for f in FEATURES}])

    for col in CAT_COLS:
        value = str(row.at[0, col])
        known = set(encoders[col].classes_)
        if value not in known:
            raise ValueError(f"Unknown value '{value}' for {col}. Known: {sorted(known)[:10]}...")
        row[col] = encoders[col].transform([value])

    X = scaler.transform(row[FEATURES])
    proba = model.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    return {
        "label": LABELS[idx],
        "confidence": round(float(proba[idx]), 4),
        "probabilities": {LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)},
    }


def record_from_csv_line(line: str) -> dict:
    """Turn one raw KDDTest+/KDDTrain+ line into a feature dict (extra columns ignored)."""
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < len(FEATURES):
        raise ValueError(f"Expected at least {len(FEATURES)} values, got {len(parts)}")
    rec = {}
    for name, val in zip(FEATURES, parts):
        rec[name] = val if name in CAT_COLS else float(val)
    return rec


if __name__ == "__main__":
    test_path = ROOT / "KDDTest+.txt"
    with open(test_path) as f:
        for line in [next(f) for _ in range(3)]:
            print(line.split(",")[41], "->", predict(record_from_csv_line(line)))