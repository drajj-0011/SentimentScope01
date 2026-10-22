"""Pydantic request/response models — the API contract.

Author: Debraj
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Sentence or short review to classify.",
        examples=["the battery life is amazing and setup took two minutes"],
    )
    explain: bool = Field(
        default=True,
        description="Return the tokens that pushed the prediction.",
    )


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Up to 50 sentences in one call.",
    )


class TokenContribution(BaseModel):
    token: str
    weight: float


class Prediction(BaseModel):
    label: str = Field(..., description="negative | neutral | positive")
    confidence: float = Field(..., ge=0, le=1)
    scores: dict[str, float]
    explanation: list[TokenContribution] = []
    latency_ms: float
    model_version: str


class BatchPrediction(BaseModel):
    results: list[Prediction]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    author: str


class ExamplesResponse(BaseModel):
    examples: list[dict]
