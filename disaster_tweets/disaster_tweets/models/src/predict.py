"""
src/predict.py
Prediction logic, kept separate from the Streamlit user interface.
Loads the saved pipeline (preprocessing + model) - nothing is retrained here.
"""
import json
from pathlib import Path

import joblib
import numpy as np

from src.preprocess import EXTRA_FEATURE_NAMES, ROOT

MODEL_PATH = ROOT / "models" / "disaster_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

MIN_WORDS = 2
MAX_CHARS = 500


class ModelNotFoundError(FileNotFoundError):
    pass


def load_pipeline(path: Path = MODEL_PATH):
    if not Path(path).exists():
        raise ModelNotFoundError(
            f"Model file not found: {path}. Run  python -m src.train  first.")
    return joblib.load(path)


def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


def validate_text(text) -> str:
    """Return a cleaned input string or raise ValueError with a helpful message."""
    if text is None or not str(text).strip():
        raise ValueError("Please enter some tweet text.")
    text = str(text).strip()
    if len(text.split()) < MIN_WORDS:
        raise ValueError(f"Please enter at least {MIN_WORDS} words - a single word has too little context.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Text is too long ({len(text)} characters). Please keep it under {MAX_CHARS}.")
    return text


def _feature_names(pipe):
    if "tfidf" in pipe.named_steps:
        return list(pipe.named_steps["tfidf"].get_feature_names_out())
    union = pipe.named_steps["features"]
    return list(union.transformer_list[0][1].get_feature_names_out()) + EXTRA_FEATURE_NAMES


def explain(pipe, text, top_n=6):
    """For linear models: words in THIS tweet that pushed the score up / down.
    Returns (pushing_to_disaster, pushing_to_not) as lists of (word, contribution)."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return [], []
    X = pipe[:-1].transform([text])
    X = X.tocsr() if hasattr(X, "tocsr") else X
    names = _feature_names(pipe)
    coef = clf.coef_.ravel()
    idx = X.nonzero()[1]
    contrib = [(names[i], float(X[0, i] * coef[i])) for i in idx if i < len(names)
               and names[i] not in EXTRA_FEATURE_NAMES]
    pos = sorted([c for c in contrib if c[1] > 0], key=lambda c: -c[1])[:top_n]
    neg = sorted([c for c in contrib if c[1] < 0], key=lambda c: c[1])[:top_n]
    return pos, neg


def predict_text(pipe, text):
    """Predict a single tweet. Returns a dict with label, disaster probability (if the
    model provides one) and per-word explanation."""
    text = validate_text(text)
    pred = int(pipe.predict([text])[0])
    out = {"text": text, "label": pred, "label_name": "Disaster" if pred else "Not a disaster",
           "probability": None, "pos_words": [], "neg_words": []}
    if hasattr(pipe, "predict_proba"):
        out["probability"] = float(pipe.predict_proba([text])[0, 1])
    out["pos_words"], out["neg_words"] = explain(pipe, text)
    return out


def predict_batch(pipe, texts):
    """Predict many tweets at once (list/Series of strings). Returns (labels, probs|None)."""
    texts = [str(t) if t is not None else "" for t in texts]
    labels = pipe.predict(texts)
    probs = pipe.predict_proba(texts)[:, 1] if hasattr(pipe, "predict_proba") else None
    return labels, probs
