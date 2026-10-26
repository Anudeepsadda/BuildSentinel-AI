import os
import numpy as np
import pandas as pd

def generate_dataset(n_builds=800, seed=42):
    rng = np.random.default_rng(seed)

    duration = rng.integers(60, 1200, n_builds)
    code_changes = rng.integers(1, 1200, n_builds)
    tests_failed = rng.poisson(2, n_builds)
    hour = rng.integers(0, 24, n_builds)

    failures = []
    previous_failure = 0
    previous_five = []

    for i in range(n_builds):
        # Risk signals known before the current build finishes
        recent_failure_count = sum(previous_five[-5:])

        # Base risk plus interpretable risk contributions
        logit = (
            -2.0
            + 0.65 * (tests_failed[i] > 2)
            + 0.75 * (code_changes[i] > 500)
            + 0.45 * (duration[i] > 800)
            + 0.40 * (hour[i] <= 5)
            + 0.65 * previous_failure
            + 0.20 * recent_failure_count
        )

        probability = 1 / (1 + np.exp(-logit))
        failure = int(rng.random() < probability)

        failures.append(failure)
        previous_five.append(failure)
        previous_failure = failure

    df = pd.DataFrame({
        "build_number": range(1, n_builds + 1),
        "duration": duration,
        "code_changes": code_changes,
        "tests_failed": tests_failed,
        "hour": hour,
        "failure": failures
    })

    # Historical features use only earlier builds
    df["prev_failure"] = df["failure"].shift(1).fillna(0)
    df["failure_last_5"] = (
        df["failure"].shift(1)
        .rolling(window=5, min_periods=1)
        .sum()
        .fillna(0)
    )

    df["prev_failure"] = df["prev_failure"].astype(int)
    df["failure_last_5"] = df["failure_last_5"].astype(int)

    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = generate_dataset()
    df.to_csv("data/jenkins_builds.csv", index=False)

    print("Synthetic dataset generated successfully.")
    print("Shape:", df.shape)
    print("Failure rate:", f"{df['failure'].mean():.2%}")
    print(df.head())
