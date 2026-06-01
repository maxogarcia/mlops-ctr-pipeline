# MLOps CTR Prediction Pipeline

A production-grade MLOps pipeline for Ad Click-Through Rate (CTR) Prediction trained on the Criteo dataset. Built to demonstrate end-to-end ML engineering from data versioning to automated retraining to Kubernetes deployment.

## Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.767 |
| F1 Score | 0.538 |
| Precision | 0.441 |
| Recall | 0.690 |
| Prediction Latency | ~23ms |

Trained on 800,000 Criteo ad impressions (1M row sample, 80/20 split)

## Tech Stack

| Layer | Tool |
|-------|------|
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Model | XGBoost |
| Serving | FastAPI + Uvicorn |
| Monitoring | Evidently AI |
| Orchestration | Prefect |
| Containerization | Docker |
| Deployment | Kubernetes |
| CI/CD | GitHub Actions |
| Dataset | Criteo CTR (1M rows) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /predict | POST | Returns click probability and binary prediction |
| /health | GET | Health check with model name and version |
| /metrics | GET | Prometheus metrics |

## Quick Start

git clone https://github.com/maxogarcia/mlops-ctr-pipeline.git
cd mlops-ctr-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mlflow server --host 0.0.0.0 --port 5000
uvicorn src.serve.main:app --host 0.0.0.0 --port 8000

## Automated Retraining Pipeline

1. Evidently AI runs daily drift detection
2. If drift exceeds threshold, Prefect DAG triggers automatically
3. Data reprocessed, model retrained, new version registered in MLflow
4. Serving layer picks up new model version on next deployment

## Docker

docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 maxogarcia/ctr-pipeline:latest

## Kubernetes

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

## Tests

pytest tests/ -v

Built as part of a FAANG-prep ML engineering portfolio.
