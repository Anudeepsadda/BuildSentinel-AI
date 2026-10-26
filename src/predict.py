from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "build_failure_model.pkl"

FEATURE_NAMES = [
    "code_changes",
    "hour",
    "prev_failure",
    "failure_last_5",
]

THRESHOLD = 0.30


def predict_failure(features):
    """
    Return the predicted probability of build failure.

    `features` must be a dictionary containing the four pre-build
    features, or a list in FEATURE_NAMES order.
    """
    if isinstance(features, dict):
        missing = [name for name in FEATURE_NAMES if name not in features]
        extra = [name for name in features if name not in FEATURE_NAMES]

        if missing or extra:
            raise ValueError(
                f"Missing features: {missing}; unexpected features: {extra}"
            )

        values = [features[name] for name in FEATURE_NAMES]

    elif isinstance(features, (list, tuple)):
        if len(features) != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} features, got {len(features)}."
            )
        values = list(features)

    else:
        raise TypeError("features must be a dictionary, list, or tuple.")

    try:
        values = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("All feature values must be numeric.") from exc

    if not all(pd.notna(value) for value in values):
        raise ValueError("Feature values must not be missing or NaN.")

    code_changes, hour, prev_failure, failure_last_5 = values

    if code_changes < 1:
        raise ValueError("code_changes must be at least 1.")
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23.")
    if prev_failure not in (0, 1):
        raise ValueError("prev_failure must be 0 or 1.")
    if not 0 <= failure_last_5 <= 5:
        raise ValueError("failure_last_5 must be between 0 and 5.")

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([values], columns=FEATURE_NAMES)

    return float(model.predict_proba(row)[0, 1])
