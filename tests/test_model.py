"""Unit tests for the model layer.

Run:  pytest -q
Author: Debraj
"""

from __future__ import annotations

import pytest

from app.model import LABELS, SentimentModel, load_dataset, train_model


@pytest.fixture(scope="module")
def model() -> SentimentModel:
    return SentimentModel.train(save=False)


def test_dataset_has_all_three_labels():
    texts, labels = load_dataset()
    assert len(texts) == len(labels)
    assert len(texts) >= 120
    assert set(labels) == set(LABELS)


def test_dataset_rows_are_unique_and_non_empty():
    texts, _ = load_dataset()
    assert all(t.strip() for t in texts)
    assert len(set(texts)) == len(texts), "duplicate sentences weaken the evaluation"


def test_training_reaches_reasonable_accuracy():
    _, metrics = train_model()
    assert metrics["accuracy"] >= 0.80, metrics
    assert metrics["macro_fbeta"] >= 0.75, metrics
    assert len(metrics["confusion_matrix"]) == 3


def test_predict_returns_valid_shape(model: SentimentModel):
    out = model.predict("this is absolutely wonderful and it works perfectly")
    assert out["label"] in LABELS
    assert 0.0 <= out["confidence"] <= 1.0
    assert set(out["scores"]) == set(LABELS)
    assert abs(sum(out["scores"].values()) - 1.0) < 0.01
    assert out["latency_ms"] >= 0
    assert out["model_version"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("i absolutely love this product", "positive"),
        ("complete waste of money and it broke", "negative"),
    ],
)
def test_clear_cases_are_classified_correctly(model: SentimentModel, text, expected):
    assert model.predict(text)["label"] == expected


def test_explanation_tokens_have_signed_weights(model: SentimentModel):
    explanation = model.predict("the battery life is amazing but support was rude")["explanation"]
    assert explanation, "expected at least one contributing token"
    assert all({"token", "weight"} <= set(item) for item in explanation)
    assert any(item["weight"] > 0 for item in explanation)


def test_empty_text_is_rejected(model: SentimentModel):
    with pytest.raises(ValueError):
        model.predict("   ")


def test_unknown_vocabulary_still_returns_a_label(model: SentimentModel):
    out = model.predict("zzzqqq flibbertigibbet wobblewump")
    assert out["label"] in LABELS
    assert out["explanation"] == []


def test_model_info_is_serialisable(model: SentimentModel):
    info = model.info()
    assert info["author"] == "Debraj"
    assert info["vocabulary_size"] > 50
    assert info["algorithm"].startswith("TF-IDF")
