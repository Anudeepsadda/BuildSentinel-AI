def compute_adjusted_risk(
    ml_risk,
    tests_failed,
    code_changes,
    prev_failure,
    failure_last_5,
    alpha=0.45
):
    tf = min(tests_failed / 10, 1)
    cc = min(code_changes / 1000, 1)
    pf = prev_failure
    hist = min(failure_last_5 / 5, 1)

    quality_score = 1 - (
        0.35 * tf +
        0.35 * cc +
        0.15 * pf +
        0.15 * hist
    )

    quality_score = max(0, min(1, quality_score))
    adjusted_risk = ml_risk * (1 - quality_score * alpha)

    return adjusted_risk, quality_score
