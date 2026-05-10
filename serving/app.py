"""FastAPI app that serves the tennis-prediction model.

At startup it loads the latest `Staging` model from MLflow, connects to the
Feast online store, and builds label encoders from ``data/tennis.csv`` so
raw categorical requests match the trained model.

Inference paths:

- ``POST /predict`` — entity lookup by ``day_id`` (Feast online store).
- ``POST /predict/features`` — raw outlook/temperature/humidity/wind strings,
  no Feast lookup required.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.sklearn
from feast import FeatureStore
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from training.helpers import (
    Outlook,
    Temperature,
    Humidity,
    Wind,
    build_feature_encoders_from_csv,
    encode_raw_features,
)
from paths import resolved_feast_repo_path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_CSV = REPO_ROOT / "data" / "tennis.csv"

state: dict = {"model": None, "store": None, "encoders": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    state["model"] = mlflow.sklearn.load_model("models:/tennis_model/Staging")
    state["store"] = FeatureStore(repo_path=resolved_feast_repo_path())
    ref_csv = os.environ.get("TENNIS_REFERENCE_CSV", str(DEFAULT_REFERENCE_CSV))
    state["encoders"] = build_feature_encoders_from_csv(ref_csv)
    yield


app = FastAPI(title="Tennis Predictor", lifespan=lifespan)


class PredictRequest(BaseModel):
    day_id: int


class PredictFeaturesRequest(BaseModel):
    outlook: Outlook
    temperature: Temperature
    humidity: Humidity
    wind: Wind


class PredictResponse(BaseModel):
    day_id: int
    prediction: int
    label: str


class PredictFeaturesResponse(BaseModel):
    prediction: int
    label: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    feature_vector = state["store"].get_online_features(
        features=[
            "tennis_features:outlook",
            "tennis_features:temperature",
            "tennis_features:humidity",
            "tennis_features:wind",
        ],
        entity_rows=[{"day_id": req.day_id}],
    ).to_df()[["outlook", "temperature", "humidity", "wind"]]

    pred = int(state["model"].predict(feature_vector)[0])
    return PredictResponse(
        day_id=req.day_id,
        prediction=pred,
        label="Yes" if pred else "No",
    )


@app.post("/predict/features", response_model=PredictFeaturesResponse)
def predict_features(req: PredictFeaturesRequest) -> PredictFeaturesResponse:
    payload = req.model_dump()
    try:
        X = encode_raw_features(payload, state["encoders"])
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown or invalid category for one or more features: {exc}",
        ) from exc

    pred = int(state["model"].predict(X)[0])
    return PredictFeaturesResponse(
        prediction=pred,
        label="Yes" if pred else "No",
    )
