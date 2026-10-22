"""FastAPI application for SentimentScope.

Run locally:
    uvicorn app.main:app --reload

Author: Debraj
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model import get_model
from .schemas import (
    BatchPrediction,
    BatchPredictRequest,
    ExamplesResponse,
    HealthResponse,
    Prediction,
    PredictRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
logger = logging.getLogger("sentimentscope")

STATIC_DIR = Path(__file__).parent / "static"

EXAMPLES = [
    {"text": "the battery life is amazing and setup took two minutes", "hint": "clearly positive"},
    {"text": "it broke after two days and support never replied", "hint": "clearly negative"},
    {"text": "it comes in three colors and ships from a warehouse", "hint": "factual / neutral"},
    {"text": "honestly a joy to use every day", "hint": "short & enthusiastic"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model at boot so the first request is not slow."""
    model = get_model()
    logger.info("model ready (source=%s, labels=%s)", model.source, model.pipeline.classes_)
    app.state.model = model
    yield


app = FastAPI(
    title="SentimentScope API",
    description=(
        "A small, fully offline sentiment classifier built with scikit-learn "
        "and served with FastAPI. No API keys, no model downloads.\n\n"
        "**Author:** Debraj"
    ),
    version="1.0.0",
    contact={"name": "Debraj"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page playground."""
    return FileResponse(STATIC_DIR / "index.html")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    model = getattr(app.state, "model", None)
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        author="Debraj",
    )


@app.get("/api/model", tags=["meta"])
def model_info() -> dict:
    """Metadata about the trained estimator: algorithm, vocabulary, metrics."""
    return get_model().info()


@app.get("/api/examples", response_model=ExamplesResponse, tags=["meta"])
def examples() -> ExamplesResponse:
    return ExamplesResponse(examples=EXAMPLES)


@app.post("/api/predict", response_model=Prediction, tags=["predict"])
def predict(payload: PredictRequest) -> Prediction:
    """Classify a single piece of text."""
    try:
        result = get_model().predict(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not payload.explain:
        result["explanation"] = []
    return Prediction(**result)


@app.post("/api/predict/batch", response_model=BatchPrediction, tags=["predict"])
def predict_batch(payload: BatchPredictRequest) -> BatchPrediction:
    """Classify up to 50 sentences in one round trip."""
    model = get_model()
    results: list[Prediction] = []

    for text in payload.texts:
        try:
            results.append(Prediction(**model.predict(text, top_k=4)))
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"invalid entry {text!r}: {exc}"
            ) from exc

    return BatchPrediction(results=results, count=len(results))
