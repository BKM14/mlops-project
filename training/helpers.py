"""Pure functions for the tennis-prediction model.

These helpers intentionally avoid Feast and MLflow imports so they can be
unit-tested in isolation without any infrastructure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from training.model import build_estimator

FEATURE_COLS = ["outlook", "temperature", "humidity", "wind"]
LABEL_COL = "play"

Outlook = Literal["Sunny", "Overcast", "Rain"]
Temperature = Literal["Hot", "Mild", "Cool"]
Humidity = Literal["High", "Normal"]
Wind = Literal["Weak", "Strong"]


def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all categorical columns. Returns an encoded copy."""
    df = df.copy()
    for col in FEATURE_COLS + [LABEL_COL]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def build_feature_encoders(reference: pd.DataFrame) -> dict[str, LabelEncoder]:
    """Fit one LabelEncoder per feature column on the reference dataset.

    Must match the encoding used when building ``data/tennis_feast.parquet``
    (same vocabulary as ``data/tennis.csv``) so raw inputs align with the
    trained model and Feast-stored integers.
    """
    encoders: dict[str, LabelEncoder] = {}
    for col in FEATURE_COLS:
        le = LabelEncoder()
        le.fit(reference[col].astype(str))
        encoders[col] = le
    return encoders


def build_feature_encoders_from_csv(csv_path: str | Path) -> dict[str, LabelEncoder]:
    """Convenience wrapper: load CSV and build encoders."""
    return build_feature_encoders(pd.read_csv(csv_path))


def encode_raw_features(
    features: dict[str, str], encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:
    """Turn raw categorical strings into a single-row encoded feature matrix."""
    row: dict[str, list[int]] = {}
    for col in FEATURE_COLS:
        row[col] = [int(encoders[col].transform([str(features[col])])[0])]
    return pd.DataFrame(row)


def train(df: pd.DataFrame) -> tuple[DecisionTreeClassifier, dict]:
    """Fit a decision-tree classifier on the full dataframe.

    Returns the trained model and a metrics dict.
    """
    X = df[FEATURE_COLS]
    y = df[LABEL_COL]
    model = build_estimator()
    model.fit(X, y)
    accuracy = accuracy_score(y, model.predict(X))
    return model, {"accuracy": round(accuracy, 4), "n_samples": len(df)}


def should_deploy(metrics: dict, threshold: float = 0.80) -> bool:
    """Promotion gate based on training accuracy."""
    return metrics["accuracy"] >= threshold
