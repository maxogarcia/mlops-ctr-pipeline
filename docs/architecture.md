# Architecture

Criteo Dataset -> preprocess.py -> train.py -> MLflow Registry -> FastAPI Server
FastAPI Server -> Prometheus Metrics -> Grafana Dashboard
New Data -> Evidently AI drift detection -> Prefect Retraining DAG -> MLflow Registry
GitHub Push -> GitHub Actions -> DockerHub -> Kubernetes Cluster
