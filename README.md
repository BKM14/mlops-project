# Tennis Play Prediction — MLOps

End-to-end MLOps pipeline that predicts whether weather conditions are
suitable to **play tennis**, using the classic 14-row Play Tennis dataset
(`Outlook, Temperature, Humidity, Wind → Play`).

**Stack:**

- **Python 3.10+** (CI uses 3.11)
- **scikit-learn** — `DecisionTreeClassifier`
- **[Feast](https://feast.dev/)** — file offline store + Redis online store
- **[MLflow](https://mlflow.org/)** — local tracking server + model registry
- **[FastAPI](https://fastapi.tiangolo.com/)** — REST inference endpoints
- **[Pandera](https://pandera.readthedocs.io/)** — dataset schema validation
- **GitHub Actions** — unit tests → data validation → smoke test → train & register

> No Kubernetes. No Kubeflow. Everything runs on plain Python + a couple of local
> infrastructure pieces (Redis, MLflow).

---

## Architecture

```
                        ┌──────────────────────────────────────────────────────────┐
                        │                          Data layer                      │
                        │                                                          │
                        │   data/tennis.csv  ─►  data/_build_parquet.py            │
                        │                       (label-encode + add day_id /       │
                        │                        event_timestamp)                  │
                        │                                  │                       │
                        │                                  ▼                       │
                        │   data/tennis_feast.parquet  (Int64 features, ns UTC)    │
                        └─────────────────────────┬────────────────────────────────┘
                                                  │
                                                  ▼
   ┌──────────────────────────────┐    ┌──────────────────────────────────────────┐
   │ Feature definitions          │    │ Feast offline store (file)               │
   │ feature_store/feature_repo/  │    │   reads tennis_feast.parquet             │
   │   ├─ feature_store.yaml      │    │ Feast online store (Redis @ :6379)       │
   │   ├─ data_sources.py         │◄───│   filled by `feast materialize`          │
   │   └─ features.py             │    └─────────────────┬────────────────────────┘
   └──────────────┬───────────────┘                      │
                  │ feast apply                          │ get_online_features
                  ▼                                      ▼
         registry.db ────► training/train.py        serving/app.py (FastAPI)
                                  │                       ▲
                                  │ get_historical_        │ load Staging model
                                  │   features            │
                                  ▼                       │
                          MLflow Tracking @ :5000  ───────┘
                          + Model Registry (tennis_model → Staging)
```

### What each piece does

1. **CSV → Parquet** (`data/_build_parquet.py`)
   `tennis.csv` is the source of truth. The builder label-encodes the four
   categorical features and the label, adds `day_id` (Int64 entity key) and
   `event_timestamp` (`datetime64[ns, UTC]`, all rows = `2024-01-01`),
   and writes `tennis_feast.parquet`. The Feast `FeatureView` declares all
   features as Int64, so encoding happens **before** parquet — that's why the
   serving path also keeps a CSV-fitted `LabelEncoder` for raw inputs.

2. **Feast** (`feature_store/feature_repo/`)
   - `feature_store.yaml` — project name, file offline store, Redis online store.
   - `data_sources.py` — `FileSource` pointing at the parquet (path resolved
     absolutely, so `feast apply` works from any cwd).
   - `features.py` — `Entity(name="day", join_keys=["day_id"])` + the
     `tennis_features` FeatureView with five Int64 fields and a 365-day TTL.
   - `feast apply` writes `data/registry.db`. `feature_store/materialize.py`
     pushes points-in-time features into Redis using an explicit
     `materialize(start=2024-01-01, end=now)` window.

3. **Training** (`training/`)
   - `helpers.py` — pure functions: `encode_dataframe`, `train`, `should_deploy`,
     `build_feature_encoders_from_csv`, `encode_raw_features`. No Feast or
     MLflow imports here so unit tests don't need infra.
   - `train.py` — entry point: pulls historical features from Feast for
     `day_id ∈ [1, 14]` at `2024-01-01 UTC`, runs the helpers, logs metrics
     and parameters to MLflow, registers the sklearn artifact as
     **`tennis_model`**, and promotes the new version to **`Staging`** when
     accuracy ≥ `ACCURACY_THRESHOLD` (default `0.80`).

4. **Serving** (`serving/app.py`)
   On startup:
   - Connects to MLflow (`MLFLOW_TRACKING_URI`) and loads
     `models:/tennis_model/Staging` (override with `MODEL_URI`).
   - Opens a `FeatureStore` against the local Feast registry (so
     `/predict` by `day_id` can read from Redis).
   - Builds **`LabelEncoder`s** from `data/tennis.csv` so raw string inputs
     align with the integers the model was trained on.

   Two prediction paths:

   | Path | Body | Behavior |
   |------|------|----------|
   | `POST /predict` | `{"day_id": int}` | Online lookup in Redis, then `model.predict` |
   | `POST /predict/features` | raw outlook/temperature/humidity/wind strings | Encoded in-process; **no Redis lookup**. Still requires MLflow at startup. |
   | `GET /health` | — | Liveness probe |

5. **Tests** (`tests/`)
   - `tests/unit/` — encoders, train, deploy gate, and round-trip equivalence
     between the CSV-fit encoders and the parquet ints. No infra needed.
   - `tests/data_validation/` — Pandera schema check on `tennis.csv`.
   - `tests/smoke/` — end-to-end fit + predict on the dataset.

6. **CI/CD** (`.github/workflows/ci_cd.yml`)
   PRs and pushes run the three test jobs in series. The `train-and-register`
   job runs only on `main`, brings up an ephemeral Redis service, an MLflow
   server, applies Feast definitions, materializes, and trains.

---

## Project layout

```
mlops/
├── conftest.py                      # repo-root sys.path for pytest
├── data/
│   ├── tennis.csv                   # raw dataset (source of truth)
│   ├── tennis_feast.parquet         # generated, do not hand-edit
│   └── _build_parquet.py            # CSV → parquet builder
├── feature_store/
│   ├── feature_repo/
│   │   ├── feature_store.yaml       # local + Redis @ localhost:6379
│   │   ├── data_sources.py          # absolute parquet path
│   │   └── features.py              # Entity + FeatureView
│   └── materialize.py               # offline → Redis, [2024-01-01, now]
├── training/
│   ├── helpers.py                   # pure: encode/train/deploy gate
│   └── train.py                     # Feast → train → MLflow → Staging
├── serving/
│   └── app.py                       # FastAPI + lifespan model loader
├── tests/
│   ├── unit/test_helpers.py
│   ├── data_validation/test_schema.py
│   └── smoke/test_model_smoke.py
├── paths.py                         # resilient FEAST_REPO_PATH resolver
├── Makefile
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Requirement | Why |
|-------------|-----|
| **Python 3.10+** with `python3` on `PATH` | Makefile defaults to `python3` (override with `make … PYTHON=python`) |
| **Redis** on `localhost:6379` | Feast online store |
| **MLflow tracking server** on `localhost:5000` | Used by `train.py` and the FastAPI app |

Optional: Docker for Redis (single-line bring-up).

Install Python deps:

```bash
python3 -m pip install -r requirements.txt
```

---

## Setup commands (local)

Run all of these from the **repository root** (`/home/balaji/mlops`).

### 1. Bring up Redis

```bash
docker run -d --name redis-tennis -p 6379:6379 redis
# or, if Redis is installed natively:
redis-server &
```

### 2. Bring up MLflow

```bash
mlflow server --host 0.0.0.0 --port 5000 &
```

### 3. Install + bootstrap

```bash
make setup
# = pip install -r requirements.txt
# + python3 data/_build_parquet.py
# + cd feature_store/feature_repo && feast apply
```

### 4. Materialize features into Redis

Required so `POST /predict` (online lookup) works.

```bash
make materialize
```

### 5. Run tests

No Redis or MLflow required for tests.

```bash
make test
```

### 6. Train + register the model

```bash
make train
# expected tail:
# Metrics: {'accuracy': 0.8571, 'n_samples': 14}
# Model v1 promoted to Staging
```

### 7. Start the FastAPI server

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 make serve
# Uvicorn running on http://127.0.0.1:8000
```

---

## Makefile reference

| Command | What it does |
|---------|---------------|
| `make setup` | Install requirements, build parquet, run `feast apply` |
| `make parquet` | Rebuild `data/tennis_feast.parquet` from `data/tennis.csv` |
| `make materialize` | Push offline features into Redis (window `[2024-01-01, now]`) |
| `make test` | `pytest tests/ -v` |
| `make train` | `python3 training/train.py` |
| `make serve` | `python3 -m uvicorn serving.app:app --reload` |

Override the interpreter: `make train PYTHON=/usr/bin/python3.11`.

### Equivalent raw commands

```bash
python3 -m pip install -r requirements.txt
python3 data/_build_parquet.py
cd feature_store/feature_repo && feast apply && cd -
python3 feature_store/materialize.py
python3 -m pytest tests/ -v
python3 training/train.py
python3 -m uvicorn serving.app:app --reload --host 0.0.0.0 --port 8000
```

---

## Environment variables

### Training (`training/train.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking + registry URI |
| `ACCURACY_THRESHOLD` | `0.80` | Min accuracy required to promote to Staging |
| `FEAST_REPO_PATH` | *(unset → `<repo>/feature_store/feature_repo`)* | Absolute, or relative to the **project root** |

### Serving (`serving/app.py`)

| Variable | Required | Description |
|----------|----------|-------------|
| `MLFLOW_TRACKING_URI` | **Yes** | Same URI used at training time |
| `MODEL_URI` | No | Defaults to `models:/tennis_model/Staging`; override to pin a version (`models:/tennis_model/3`) |
| `FEAST_REPO_PATH` | No | If unset, resolves to `<repo>/feature_store/feature_repo` (resilient to a stale shell `export`) |
| `TENNIS_REFERENCE_CSV` | No | Defaults to `<repo>/data/tennis.csv`; encoders for `/predict/features` |

---

## API & cURL

Base URL in examples: `http://localhost:8000`. Replace with your deploy URL
when needed.

### `GET /health`

Liveness probe.

```bash
curl -sS http://localhost:8000/health
# {"status":"ok"}
```

### `POST /predict` — predict by `day_id`

Reads encoded features from the Feast online store (Redis). Only works for
`day_id ∈ [1, 14]` after `make materialize`.

```bash
curl -sS -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"day_id": 1}'
# {"day_id":1,"prediction":0,"label":"No"}

curl -sS -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"day_id": 3}'
# {"day_id":3,"prediction":1,"label":"Yes"}

curl -sS -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"day_id": 14}'
# {"day_id":14,"prediction":0,"label":"No"}
```

### `POST /predict/features` — predict from raw weather strings

Does **not** query Redis. Encodes inputs in-process using the CSV-fit
`LabelEncoder`s. Useful when there is no `day_id` (e.g. predicting today).

Allowed values (must match `data/tennis.csv` exactly):

| Field | Values |
|-------|--------|
| `outlook` | `Sunny`, `Overcast`, `Rain` |
| `temperature` | `Hot`, `Mild`, `Cool` |
| `humidity` | `High`, `Normal` |
| `wind` | `Weak`, `Strong` |

```bash
curl -sS -X POST http://localhost:8000/predict/features \
  -H 'Content-Type: application/json' \
  -d '{
    "outlook": "Sunny",
    "temperature": "Hot",
    "humidity": "High",
    "wind": "Weak"
  }'
# {"prediction":0,"label":"No"}

curl -sS -X POST http://localhost:8000/predict/features \
  -H 'Content-Type: application/json' \
  -d '{
    "outlook": "Overcast",
    "temperature": "Cool",
    "humidity": "Normal",
    "wind": "Strong"
  }'
# {"prediction":1,"label":"Yes"}
```

Invalid categories return **HTTP 422** with a detail message (FastAPI/Pydantic
rejects them at the body level when they're outside the `Literal` set).

### Response schema

| Endpoint | Field | Type | Notes |
|----------|-------|------|-------|
| `/predict` & `/predict/features` | `prediction` | `int` | `0` = No, `1` = Yes |
| | `label` | `str` | `"No"` or `"Yes"` |
| `/predict` only | `day_id` | `int` | Echoed back |

---

## CI/CD

`.github/workflows/ci_cd.yml` runs four jobs:

| Job | Trigger | Purpose |
|-----|---------|---------|
| `unit-tests` | every push / PR | `pytest tests/unit/` |
| `data-validation` | after unit | Pandera schema on `tennis.csv` |
| `smoke-test` | after data-validation | end-to-end fit + predict |
| `train-and-register` | after smoke, **`main` only** | spins up Redis, MLflow, runs `feast apply` + materialize + train |

PRs run the three test jobs without infra. `main` runs the full pipeline.

---

## Verification checklist

After `make setup → make materialize → make train → make serve` you should
hit all of these:

- `make test` is green (7 tests).
- The MLflow UI at `http://localhost:5000` shows experiment `tennis-prediction`
  with a registered model **`tennis_model`** in **Staging**.
- `curl http://localhost:8000/health` returns `{"status":"ok"}`.
- `POST /predict {"day_id":1}` returns `{"label":"No"}`.
- `POST /predict/features` with Sunny/Hot/High/Weak returns `{"label":"No"}`.

---

## Troubleshooting

| Symptom | What to do |
|---------|------------|
| `make: python: No such file` | This Makefile uses `python3`. If you forked an older one, set `make … PYTHON=python3`. |
| `RESOURCE_DOES_NOT_EXIST: tennis_model not found` at startup | `make train` didn't succeed against the same `MLFLOW_TRACKING_URI`. Train first, then start the API. |
| `FeatureViewNotFoundException: Feature view tennis_features does not exist` | Run `cd feature_store/feature_repo && feast apply` (or `make setup`). |
| `Materializing 0 feature views` | The materialize window doesn't cover `event_timestamp` in the parquet. The included script uses `start=2024-01-01` explicitly — re-run after `feast apply`. |
| `FileNotFoundError: …/feature_store/feature_repo/feature_store.yaml` under `$HOME` | Stale `FEAST_REPO_PATH` in your shell. `unset FEAST_REPO_PATH` and retry; `paths.py` also auto-falls back if the env-var path is missing `feature_store.yaml`. |
| `ValueError: invalid literal for int() with base 10: 'Sunny'` during materialize | Parquet wasn't rebuilt after a CSV change. `make parquet` then `make materialize`. |
| Pip resolver conflict involving `pyarrow` | This repo pins `pyarrow==15.0.2` to satisfy MLflow 2.13.x's `pyarrow<16` constraint. Stay on the pinned versions. |
| `/predict` returns null/error for some `day_id` | Run `make materialize` after the latest `feast apply`. |
| `502/connection refused` from `/predict` | Redis isn't reachable on `localhost:6379`. Start it (see Setup §1). |

---

## Quick recap (copy-paste)

```bash
# infra
docker run -d -p 6379:6379 redis
mlflow server --host 0.0.0.0 --port 5000 &

# bootstrap
unset FEAST_REPO_PATH
make setup
make materialize

# verify
make test

# train + serve
make train
MLFLOW_TRACKING_URI=http://localhost:5000 make serve

# call
curl -sS http://localhost:8000/health
curl -sS -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"day_id": 1}'
curl -sS -X POST http://localhost:8000/predict/features \
  -H 'Content-Type: application/json' \
  -d '{"outlook":"Sunny","temperature":"Hot","humidity":"High","wind":"Weak"}'
```
