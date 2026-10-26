from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "jenkins_builds.csv"
MODEL_PATH = ROOT / "models" / "build_failure_model.pkl"
REPORT_PATH = ROOT / "reports" / "validation_metrics.json"

FEATURES = [
    "code_changes",
    "hour",
    "prev_failure",
    "failure_last_5",
]
TARGET = "failure"
THRESHOLD = 0.30


def main():
    df = pd.read_csv(DATA_PATH).sort_values("build_number").reset_index(drop=True)

    train_end = int(len(df) * 0.60)
    val_end = int(len(df) * 0.80)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": "prebuild_logistic_regression",
        "features": FEATURES,
        "threshold": THRESHOLD,
        "train_builds": int(len(train_df)),
        "validation_builds": int(len(val_df)),
        "validation_failure_rate": float(y_val.mean()),
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
        "average_precision": float(
            average_precision_score(y_val, probabilities)
        ),
        "brier_score": float(brier_score_loss(y_val, probabilities)),
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(
            precision_score(y_val, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_val, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_alarms": int(fp),
        "missed_failures": int(fn),
        "true_positives": int(tp),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2))

    print("Model saved:", MODEL_PATH)
    print("Validation report saved:", REPORT_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
