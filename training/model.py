"""Tennis play / no-play classifier: estimator definition and default hyperparameters.

Training code fits this model; serving loads a serialized instance. Change the
architecture or hyperparameters here only.
"""
from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

MAX_DEPTH = 7
RANDOM_STATE = 42


def build_estimator(
    *,
    max_depth: int = MAX_DEPTH,
    random_state: int | None = RANDOM_STATE,
) -> DecisionTreeClassifier:
    """Return an unfitted decision tree (production training spec)."""
    return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)


def default_mlflow_params() -> dict[str, int]:
    """Hyperparameters to log to MLflow (names match the sklearn constructor)."""
    return {"max_depth": MAX_DEPTH, "random_state": RANDOM_STATE}
