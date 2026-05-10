"""Minimal path: load CSV, train a small decision tree, save + log to MLflow.

Writes a joblib file under ``models/`` (override with ``MODEL_OUTPUT``) and logs
the sklearn model to the active MLflow run.

No Feast or model registry required. Start a tracking server first, e.g.:

    mlflow server --host 0.0.0.0 --port 5000

Then:

    MLFLOW_TRACKING_URI=http://localhost:5000 python training/train_mlflow_simple.py

Optional: register the artifact by setting MLFLOW_REGISTERED_MODEL=tennis_model
(same server must have registry support).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO))

from helpers import encode_dataframe, train  # noqa: E402

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "tennis-prediction-simple")
CSV_PATH = Path(os.environ.get("TENNIS_CSV", _REPO / "data" / "tennis.csv"))
MODEL_OUTPUT = Path(
    os.environ.get("MODEL_OUTPUT", _REPO / "models" / "tennis_tree.joblib")
)


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = pd.read_csv(CSV_PATH)
    df = encode_dataframe(df)

    run_name = f"simple-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        model, metrics = train(df)
        mlflow.log_params({**default_mlflow_params(), "source": "csv"})
        mlflow.log_metrics(metrics)

        reg_name = os.environ.get("MLFLOW_REGISTERED_MODEL")
        if reg_name:
            mlflow.sklearn.log_model(
                model, "model", registered_model_name=reg_name
            )
        else:
            mlflow.sklearn.log_model(model, "model")

        MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_OUTPUT)
        mlflow.log_artifact(str(MODEL_OUTPUT))

        print(
            f"Logged run {run_name} — metrics: {metrics}; "
            f"saved model to {MODEL_OUTPUT}"
        )


if __name__ == "__main__":
    main()
