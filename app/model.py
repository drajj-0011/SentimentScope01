"""Core machine-learning logic for SentimentScope.

The model is intentionally small and fully offline: a TF-IDF vectoriser feeding a
logistic-regression classifier, trained on the bundled dataset in ``data/``.
No API keys, no downloads, no GPU.

Author: Debraj
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "sentiment.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACT_DIR / "model.joblib"

LABELS: tuple[str, str, str] = ("negative", "neutral", "positive")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def load_dataset(path: Path | str = DATA_PATH) -> tuple[list[str], list[str]]:
    """Read the bundled CSV and return parallel lists of texts and labels."""
    texts: list[str] = []
    labels: list[str] = []

    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if text and label in LABELS:
                texts.append(text)
                labels.append(label)

    if not texts:
        raise ValueError(f"No usable rows found in {path}")

    return texts, labels


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def build_pipeline() -> Pipeline:
    """Fresh, unfitted scikit-learn pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=8.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(
    texts: Iterable[str] | None = None,
    labels: Iterable[str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[Pipeline, dict]:
    """Fit the pipeline and return it together with a metrics dictionary."""
    if texts is None or labels is None:
        texts, labels = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        list(texts),
        list(labels),
        test_size=test_size,
        random_state=seed,
        stratify=list(labels),
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_fbeta": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "test_size": len(y_test),
        "train_size": len(y_train),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=list(LABELS)).tolist(),
        "labels": list(LABELS),
    }
    return pipeline, metrics


def save_model(pipeline: Pipeline, metrics: dict, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
    """Persist the fitted pipeline + metrics as a single joblib artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "metrics": metrics,
            "version": "1.0.0",
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        path,
    )
    return path


# --------------------------------------------------------------------------- #
# Inference wrapper
# --------------------------------------------------------------------------- #
@dataclass
class SentimentModel:
    """Thin, thread-safe wrapper around the fitted pipeline."""

    pipeline: Pipeline
    metrics: dict = field(default_factory=dict)
    version: str = "1.0.0"
    trained_at: str = ""
    source: str = "memory"

    # -- construction ------------------------------------------------------ #
    @classmethod
    def train(cls, save: bool = True) -> "SentimentModel":
        pipeline, metrics = train_model()
        instance = cls(
            pipeline=pipeline,
            metrics=metrics,
            trained_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            source="trained-now",
        )
        if save:
            save_model(pipeline, metrics)
        return instance

    @classmethod
    def load_or_train(cls, path: Path | str = DEFAULT_MODEL_PATH) -> "SentimentModel":
        path = Path(path)
        if path.exists():
            try:
                blob = joblib.load(path)
                if isinstance(blob, dict) and "pipeline" in blob:
                    return cls(
                        pipeline=blob["pipeline"],
                        metrics=blob.get("metrics", {}),
                        version=blob.get("version", "1.0.0"),
                        trained_at=blob.get("trained_at", ""),
                        source="artifact",
                    )
            except Exception:  # corrupt artifact -> retrain below
                pass
        return cls.train()

    # -- inference --------------------------------------------------------- #
    def predict(self, text: str, top_k: int = 6) -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("text must not be empty")

        started = time.perf_counter()
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        label = classes[int(probabilities.argmax())]

        scores = {
            name: round(float(prob), 4)
            for name, prob in sorted(
                zip(classes, probabilities), key=lambda kv: kv[1], reverse=True
            )
        }

        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        return {
            "label": label,
            "confidence": round(float(probabilities.max()), 4),
            "scores": scores,
            "explanation": self._explain(text, label, top_k=top_k),
            "latency_ms": latency_ms,
            "model_version": self.version,
        }

    def _explain(self, text: str, label: str, top_k: int = 6) -> list[dict]:
        """Per-token contribution of the predicted class (tf-idf * coefficient)."""
        vectoriser: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        classifier: LogisticRegression = self.pipeline.named_steps["classifier"]

        row = vectoriser.transform([text]).tocoo()
        if row.nnz == 0:
            return []

        names = vectoriser.get_feature_names_out()
        weights = classifier.coef_[list(classifier.classes_).index(label)]

        contributions = [
            {"token": str(names[col]), "weight": round(float(value * weights[col]), 4)}
            for col, value in zip(row.col, row.data)
        ]
        contributions.sort(key=lambda item: abs(item["weight"]), reverse=True)
        return contributions[:top_k]

    # -- metadata ---------------------------------------------------------- #
    def info(self) -> dict:
        vectoriser: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        return {
            "name": "SentimentScope",
            "version": self.version,
            "author": "Debraj",
            "algorithm": "TF-IDF (1-2 grams) + Logistic Regression",
            "labels": list(self.pipeline.classes_),
            "vocabulary_size": len(vectoriser.vocabulary_),
            "trained_at": self.trained_at,
            "source": self.source,
            "metrics": self.metrics,
        }


# --------------------------------------------------------------------------- #
# Process-wide singleton
# --------------------------------------------------------------------------- #
_MODEL: SentimentModel | None = None
_LOCK = threading.Lock()


def get_model() -> SentimentModel:
    """Load the artifact once, train on first run if it is missing."""
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                _MODEL = SentimentModel.load_or_train()
    return _MODEL
