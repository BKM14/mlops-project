mlflow server --host 0.0.0.0 --port 5000 &

unset FEAST_REPO_PATH
make setup
make materialize

make test

make train

MLFLOW_TRACKING_URI=http://localhost:5000 make serve

curl -sS http://localhost:8000/health
curl -sS -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"day_id": 1}'
curl -sS -X POST http://localhost:8000/predict/features -H 'Content-Type: application/json' \
  -d '{"outlook":"Sunny","temperature":"Hot","humidity":"High","wind":"Weak"}'