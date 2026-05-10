"""Repository-root paths so CLI entry points work no matter the shell cwd."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# This file lives at the repo root.
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FEAST_REPO = REPO_ROOT / "feature_store" / "feature_repo"
FEAST_CONFIG_FILENAME = "feature_store.yaml"


def _is_valid_feast_repo(path: Path) -> bool:
    return (path / FEAST_CONFIG_FILENAME).is_file()


def resolved_feast_repo_path() -> str:
    """Resolve Feast repo: env ``FEAST_REPO_PATH``, else ``<repo>/feature_store/feature_repo``.

    Behaviour:

    - If ``FEAST_REPO_PATH`` is **unset/empty**, use the repo default.
    - If set and **absolute**, use it as-is.
    - If set and **relative**, resolve it from the project root (not cwd).
    - If the resulting path is missing ``feature_store.yaml``, log a warning to
      stderr and fall back to the repo default. This rescues stale exports
      such as ``FEAST_REPO_PATH=feature_store/feature_repo`` set under
      ``$HOME``, which would otherwise look up ``$HOME/feature_store/...``.
    """
    raw = os.environ.get("FEAST_REPO_PATH", "").strip()
    if not raw:
        return str(DEFAULT_FEAST_REPO.resolve())

    p = Path(raw).expanduser()
    candidate = p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()
    if _is_valid_feast_repo(candidate):
        return str(candidate)

    print(
        f"[paths] FEAST_REPO_PATH={raw!r} → {candidate} has no "
        f"{FEAST_CONFIG_FILENAME}; falling back to {DEFAULT_FEAST_REPO}",
        file=sys.stderr,
    )
    return str(DEFAULT_FEAST_REPO.resolve())
