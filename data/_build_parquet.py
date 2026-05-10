"""One-off helper to build data/tennis_feast.parquet from data/tennis.csv.

The Feast FeatureView declares every feature column as Int64, and both the
training pipeline (which calls `encode_dataframe`) and the FastAPI serving
pipeline (which feeds online-store values straight into the trained model)
expect integer-encoded features. So we label-encode the categorical columns
here before writing parquet.

Run from the repo root:
    python data/_build_parquet.py
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "tennis.csv"
PARQUET_PATH = ROOT / "tennis_feast.parquet"

CATEGORICAL_COLS = ["outlook", "temperature", "humidity", "wind", "play"]


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    # Label-encode categorical columns so the parquet matches the Int64
    # schema declared in feature_store/feature_repo/features.py.
    for col in CATEGORICAL_COLS:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str)).astype("int64")

    df["day_id"] = df["day"].astype("int64")
    df["event_timestamp"] = pd.Series(
        [pd.Timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc))] * len(df),
        dtype="datetime64[ns, UTC]",
    )

    # Build an Arrow schema with explicit ns precision so the parquet round-trip
    # preserves datetime64[ns, UTC] (Feast's expected timestamp resolution).
    schema = pa.Schema.from_pandas(df, preserve_index=False)
    ts_idx = schema.get_field_index("event_timestamp")
    schema = schema.set(
        ts_idx,
        pa.field("event_timestamp", pa.timestamp("ns", tz="UTC")),
    )

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, PARQUET_PATH, version="2.6")
    print(f"Wrote {PARQUET_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
