import pandas as pd
import json
import os
from datetime import datetime
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

DRIFT_THRESHOLD = 0.5
DRIFT_REPORT_DIR = "reports/drift"
DRIFT_LOG = "reports/drift_log.json"

INT_COLS = [f'int_{i}' for i in range(1, 14)]
CAT_COLS = [f'cat_{i}' for i in range(1, 27)]

def load_reference_data(path: str = "data/processed/X_train.csv", n: int = 10000) -> pd.DataFrame:
    print("Loading reference data...")
    return pd.read_csv(path, nrows=n)

def load_current_data(path: str = "data/processed/X_test.csv", n: int = 10000) -> pd.DataFrame:
    print("Loading current data...")
    return pd.read_csv(path, nrows=n)

def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    print("Running drift analysis...")
    os.makedirs(DRIFT_REPORT_DIR, exist_ok=True)

    report = Report([DataDriftPreset()])
    snapshot = report.run(
        Dataset.from_pandas(reference),
        Dataset.from_pandas(current)
    )

    # Save HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{DRIFT_REPORT_DIR}/drift_report_{timestamp}.html"
    snapshot.save_html(report_path)
    print(f"Drift report saved to {report_path}")

    # Extract drift share from snapshot
    result_dict = snapshot.dict()
    drift_share = 0.0

    try:
        for metric in result_dict.get("metrics", []):
            value = metric.get("value", {})
            if isinstance(value, dict) and "share_of_drifted_columns" in value:
                drift_share = float(value["share_of_drifted_columns"])
                break
    except Exception as e:
        print(f"Could not extract drift share: {e}")

    drift_detected = drift_share > DRIFT_THRESHOLD

    summary = {
        "timestamp": timestamp,
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "threshold": DRIFT_THRESHOLD,
        "report_path": report_path
    }

    # Log drift result
    os.makedirs("reports", exist_ok=True)
    logs = []
    if os.path.exists(DRIFT_LOG):
        with open(DRIFT_LOG) as f:
            logs = json.load(f)
    logs.append(summary)
    with open(DRIFT_LOG, "w") as f:
        json.dump(logs, f, indent=2)

    return summary

def check_drift() -> bool:
    reference = load_reference_data()
    current = load_current_data()
    summary = run_drift_report(reference, current)

    print(f"\nDrift Summary:")
    print(f"  Drift detected: {summary['drift_detected']}")
    print(f"  Share of drifted columns: {summary['drift_share']:.2%}")
    print(f"  Threshold: {summary['threshold']}")

    if summary['drift_share'] > DRIFT_THRESHOLD:
        print("WARNING: Significant drift detected - retraining recommended")
        return True
    else:
        print("OK: No significant drift detected")
        return False

if __name__ == "__main__":
    check_drift()
