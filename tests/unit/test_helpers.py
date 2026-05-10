import pandas as pd
from sklearn.preprocessing import LabelEncoder

from training.helpers import (
    build_feature_encoders_from_csv,
    encode_raw_features,
    encode_dataframe,
    should_deploy,
    train,
)

SAMPLE = pd.DataFrame(
    {
        "outlook": ["Sunny", "Overcast", "Rain", "Sunny", "Rain"],
        "temperature": ["Hot", "Hot", "Mild", "Cool", "Mild"],
        "humidity": ["High", "High", "High", "Normal", "Normal"],
        "wind": ["Weak", "Weak", "Weak", "Weak", "Strong"],
        "play": ["No", "Yes", "Yes", "Yes", "No"],
    }
)


def test_encode_returns_integers():
    df = encode_dataframe(SAMPLE)
    assert df["play"].dtype in ["int32", "int64", "int8"]
    assert set(df["play"].unique()).issubset({0, 1})


def test_train_returns_model_and_metrics():
    df = encode_dataframe(SAMPLE)
    model, metrics = train(df)
    assert hasattr(model, "predict")
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n_samples"] == 5


def test_should_deploy_above_threshold():
    assert should_deploy({"accuracy": 0.90}) is True


def test_should_deploy_below_threshold():
    assert should_deploy({"accuracy": 0.70}) is False


def test_raw_features_match_reference_csv_encoding():
    """Same per-column LabelEncoder fit as parquet build → identical ints."""
    ref = pd.read_csv("data/tennis.csv")
    encoders = build_feature_encoders_from_csv("data/tennis.csv")
    for i in range(len(ref)):
        row = {
            "outlook": ref.loc[i, "outlook"],
            "temperature": ref.loc[i, "temperature"],
            "humidity": ref.loc[i, "humidity"],
            "wind": ref.loc[i, "wind"],
        }
        X = encode_raw_features(row, encoders)
        for col in ["outlook", "temperature", "humidity", "wind"]:
            le = LabelEncoder()
            expected_series = le.fit_transform(ref[col].astype(str))
            assert int(X[col].iloc[0]) == int(expected_series[i])
