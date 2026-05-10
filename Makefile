# Override on the command line if needed, e.g. `make train PYTHON=python`.
PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

.PHONY: setup parquet materialize test train train-simple serve

setup:
	$(PIP) install -r requirements.txt
	$(PYTHON) data/_build_parquet.py
	cd feature_store/feature_repo && feast apply

parquet:
	$(PYTHON) data/_build_parquet.py

materialize:
	$(PYTHON) feature_store/materialize.py

test:
	$(PYTHON) -m pytest tests/ -v

train:
	$(PYTHON) training/train.py

# CSV + sklearn + MLflow only (no Feast). Needs MLflow server on MLFLOW_TRACKING_URI.
train-simple:
	$(PYTHON) training/train_mlflow_simple.py

serve:
	$(PYTHON) -m uvicorn serving.app:app --reload
