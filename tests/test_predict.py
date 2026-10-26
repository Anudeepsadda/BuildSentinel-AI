import pytest

from src.predict import predict_failure, FEATURE_NAMES


@pytest.fixture
def valid_input():
    return {
        "code_changes": 300,
        "hour": 14,
        "prev_failure": 0,
        "failure_last_5": 1,
    }


def test_valid_input_returns_probability(valid_input):
    probability = predict_failure(valid_input)

    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


def test_list_input_is_supported():
    probability = predict_failure([300, 14, 0, 1])

    assert 0.0 <= probability <= 1.0


def test_missing_feature_is_rejected():
    with pytest.raises(ValueError):
        predict_failure({"code_changes": 300})


def test_unexpected_feature_is_rejected(valid_input):
    valid_input["duration"] = 600

    with pytest.raises(ValueError):
        predict_failure(valid_input)


@pytest.mark.parametrize("hour", [-1, 24, 25])
def test_invalid_hour_is_rejected(valid_input, hour):
    valid_input["hour"] = hour

    with pytest.raises(ValueError):
        predict_failure(valid_input)


@pytest.mark.parametrize("flag", [-1, 2, 0.5])
def test_invalid_previous_failure_is_rejected(valid_input, flag):
    valid_input["prev_failure"] = flag

    with pytest.raises(ValueError):
        predict_failure(valid_input)


@pytest.mark.parametrize("count", [-1, 6])
def test_invalid_failure_history_is_rejected(valid_input, count):
    valid_input["failure_last_5"] = count

    with pytest.raises(ValueError):
        predict_failure(valid_input)


def test_feature_contract_has_four_features():
    assert FEATURE_NAMES == [
        "code_changes",
        "hour",
        "prev_failure",
        "failure_last_5",
    ]
