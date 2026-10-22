"""Train the SentimentScope model and write the joblib artifact.

Usage:
    python -m scripts.train

Author: Debraj
"""

from __future__ import annotations

import json

from app.model import DATA_PATH, DEFAULT_MODEL_PATH, load_dataset, save_model, train_model


def main() -> None:
    texts, labels = load_dataset(DATA_PATH)
    print(f"dataset      : {DATA_PATH.relative_to(DATA_PATH.parents[2])}")
    print(f"rows         : {len(texts)}")

    for label in ("negative", "neutral", "positive"):
        print(f"  {label:<9}: {labels.count(label)}")

    pipeline, metrics = train_model(texts, labels)
    path = save_model(pipeline, metrics)

    print(f"\naccuracy     : {metrics['accuracy']:.4f}")
    print(f"macro F-beta : {metrics['macro_fbeta']:.4f}")
    print(f"train/test   : {metrics['train_size']}/{metrics['test_size']}")
    print(f"artifact     : {path}")

    print("\nconfusion matrix (rows = truth, cols = predicted)")
    header = "            " + "".join(f"{name:>10}" for name in metrics["labels"])
    print(header)
    for name, row in zip(metrics["labels"], metrics["confusion_matrix"]):
        print(f"  {name:<10}" + "".join(f"{value:>10}" for value in row))

    print("\nsample inference")
    for sentence in (
        "this update fixed everything and feels smoother",
        "it stopped charging after a month",
        "the box contains a charger and a cable",
    ):
        result = pipeline.predict([sentence])[0]
        confidence = max(pipeline.predict_proba([sentence])[0])
        print(f"  {result:<9} ({confidence:.2f})  {sentence}")

    print("\nmetrics json :")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
