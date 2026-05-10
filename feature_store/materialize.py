"""Push the tennis features from the offline store into the Redis online store.

We use an explicit `materialize()` window starting at 2024-01-01 because the
synthetic dataset stamps every row with that single event timestamp.
`materialize_incremental` would otherwise default the window start to
`now - ttl`, which excludes those rows entirely.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from feast import FeatureStore

from paths import resolved_feast_repo_path

START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def main() -> None:
    store = FeatureStore(repo_path=resolved_feast_repo_path())
    store.materialize(start_date=START_DATE, end_date=datetime.now(tz=timezone.utc))
    print("Materialization complete.")


if __name__ == "__main__":
    main()
