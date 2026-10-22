"""API contract tests using FastAPI's TestClient.

Run:  pytest -q
Author: Debraj
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["author"] == "Debraj"


def test_index_page_credits_author(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "SentimentScope" in res.text
    assert "Debraj" in res.text


def test_model_card(client):
    body = client.get("/api/model").json()
    assert body["author"] == "Debraj"
    assert len(body["labels"]) == 3
    assert 0.0 <= body["metrics"]["accuracy"] <= 1.0


def test_predict_positive(client):
    res = client.post("/api/predict", json={"text": "fantastic quality and it arrived early"})
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == "positive"
    assert body["confidence"] > 0.4
    assert body["explanation"]
    assert "weight" in body["explanation"][0]


def test_predict_without_explanation(client):
    res = client.post(
        "/api/predict", json={"text": "it crashed twice today", "explain": False}
    )
    assert res.status_code == 200
    assert res.json()["explanation"] == []


def test_predict_rejects_blank_text(client):
    res = client.post("/api/predict", json={"text": ""})
    assert res.status_code == 422


def test_predict_rejects_missing_field(client):
    assert client.post("/api/predict", json={}).status_code == 422


def test_batch_predict(client):
    payload = {
        "texts": [
            "i love it and would buy again",
            "the package arrived on tuesday",
            "worst experience i have had in years",
        ]
    }
    res = client.post("/api/predict/batch", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 3
    labels = [item["label"] for item in body["results"]]
    assert labels == ["positive", "neutral", "negative"]


def test_batch_predict_rejects_empty_list(client):
    assert client.post("/api/predict/batch", json={"texts": []}).status_code == 422


def test_examples_endpoint(client):
    body = client.get("/api/examples").json()
    assert len(body["examples"]) >= 3
    assert all({"text", "hint"} <= set(item) for item in body["examples"])


def test_openapi_schema_builds(client):
    schema = client.get("/openapi.json").json()
    assert "/api/predict" in schema["paths"]
    assert schema["info"]["contact"]["name"] == "Debraj"
