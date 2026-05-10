"""Training entry point: fetch features → train → register in MLflow."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from feast import FeatureStore

_REPO = Path(__file__).resolve().parents[1]
# Allow `python training/train.py` to import sibling modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO))

from helpers import encode_dataframe, should_deploy, train  # noqa: E402
from paths import resolved_feast_repo_path  # noqa: E402
from training.model import default_mlflow_params  # noqa: E402
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.80"))


def fetch_features() -> pd.DataFrame:
    """Pull the historical training set from Feast (offline store)."""
    store = FeatureStore(repo_path=resolved_feast_repo_path())
    entity_df = pd.DataFrame(
        {
            "day_id": list(range(1, 15)),
            "event_timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)] * 14,
        }
    )
    return store.get_historical_features(
        entity_df=entity_df,
        features=[
            "tennis_features:outlook",
            "tennis_features:temperature",
            "tennis_features:humidity",
            "tennis_features:wind",
            "tennis_features:play",
        ],
    ).to_df()


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("tennis-prediction")

    print("Fetching features from Feast...")
    df = fetch_features()
    df = encode_dataframe(df)

    print("Training model...")
    run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        model, metrics = train(df)
        mlflow.log_params(default_mlflow_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="tennis_model",
        )
        print(f"Metrics: {metrics}")

        if should_deploy(metrics, THRESHOLD):
            client = mlflow.MlflowClient()
            latest = client.get_latest_versions(
                "tennis_model", stages=["None"]
            )[0]
            client.transition_model_version_stage(
                "tennis_model", latest.version, "Staging"
            )
            print(f"Model v{latest.version} promoted to Staging")
        else:
            print(
                f"Accuracy {metrics['accuracy']} below {THRESHOLD}. Not promoted."
            )


if __name__ == "__main__":
    main()
