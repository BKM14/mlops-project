from pathlib import Path

from feast import FileSource

# Resolve the parquet path absolutely so `feast apply`, `materialize.py`, and
# `train.py` all locate the same file regardless of which directory they are
# invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARQUET_PATH = str(_REPO_ROOT / "data" / "tennis_feast.parquet")

tennis_source = FileSource(
    path=_PARQUET_PATH,
    event_timestamp_column="event_timestamp",
)
