import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, log_loss
)
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "ctr_xgboost")

def load_processed_data():
    print("Loading processed data...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test

def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ctr_prediction")

    X_train, X_test, y_train, y_test = load_processed_data()

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
        "random_state": 42,
    }

    print("Starting MLflow run...")
    with mlflow.start_run():
        model = xgb.XGBClassifier(**params)
        print("Training model...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "log_loss": log_loss(y_test, y_prob),
        }

        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME
        )

        print(f"\nModel registered as '{MODEL_NAME}' in MLflow")

if __name__ == "__main__":
    train()
