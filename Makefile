# Override on the command line if needed, e.g. `make train PYTHON=python`.
PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

.PHONY: setup parquet materialize test train serve

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

serve:
	$(PYTHON) -m uvicorn serving.app:app --reload
