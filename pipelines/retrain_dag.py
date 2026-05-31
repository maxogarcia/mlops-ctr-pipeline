import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prefect import flow, task, get_run_logger
from src.monitor.drift import check_drift
from src.data.preprocess import load_data, preprocess
from src.train.train import train
import subprocess

@task
def check_for_drift() -> bool:
    logger = get_run_logger()
    logger.info("Checking for data drift...")
    drift_detected = check_drift()
    if drift_detected:
        logger.warning("Drift detected - triggering retraining pipeline")
    else:
        logger.info("No drift detected - skipping retraining")
    return drift_detected

@task
def reprocess_data():
    logger = get_run_logger()
    logger.info("Reprocessing data...")
    df = load_data("data/raw/dac/train_sample.txt")
    X_train, X_test, y_train, y_test = preprocess(df)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    logger.info("Data reprocessed successfully")

@task
def retrain_model():
    logger = get_run_logger()
    logger.info("Retraining model...")
    train()
    logger.info("Model retrained and registered in MLflow")

@task
def notify(message: str):
    logger = get_run_logger()
    logger.info(f"NOTIFICATION: {message}")
    # In production this would send a Slack/email alert
    print(f"\n{'='*50}")
    print(f"PIPELINE NOTIFICATION: {message}")
    print(f"{'='*50}\n")

@flow(name="ctr-retraining-pipeline")
def retraining_pipeline():
    logger = get_run_logger()
    logger.info("Starting CTR retraining pipeline")

    drift_found = check_for_drift()

    if drift_found:
        notify("Drift detected - starting retraining")
        reprocess_data()
        retrain_model()
        notify("Retraining complete - new model registered in MLflow")
    else:
        notify("No drift detected - pipeline complete, no retraining needed")

if __name__ == "__main__":
    retraining_pipeline()
