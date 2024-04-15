import numpy as np
import pandas as pd
import pytest

from pyaldata.utils import determine_ref_field

# default number of timepoints
T = 50
# number of trials
N = 13


def _generate_mock_data(
    correct_spikes: bool = False,
    correct_rates: bool = False,
    incorrect_on_all_trials_rates: bool = False,
    incorrect_on_one_trial_rates: bool = False,
    correct_pos: bool = False,
    incorrect_on_all_trials_pos: bool = False,
    incorrect_on_one_trial_pos: bool = False,
):
    data = {}

    rng = np.random.default_rng()

    data["pmd_spikes"] = [rng.standard_normal((T, 100)) for _ in range(N)]
    data["some_string_field"] = ["asdf" for _ in range(N)]
    data["some_int_field"] = [rng.integers(0, 10) for _ in range(N)]

    if correct_spikes:
        data["correct_spikes"] = [rng.standard_normal((T, 10)) for _ in range(N)]

    if correct_rates:
        data["correct_rates"] = [rng.standard_normal((T, 10)) for _ in range(N)]

    if incorrect_on_all_trials_rates:
        data["incorrect_on_all_trials_rates"] = [
            rng.standard_normal((T - 10, 10)) for _ in range(N)
        ]

    if incorrect_on_one_trial_rates:
        arrays = [rng.standard_normal((T, 10)) for _ in range(N)]
        arrays[11] = arrays[11][: T - 10, :]
        data["incorrect_on_one_trial_rates"] = arrays

    if correct_pos:
        data["correct_pos"] = [rng.standard_normal((T, 2)) for _ in range(N)]

    if incorrect_on_all_trials_pos:
        data["incorrect_on_all_trials_pos"] = [
            rng.standard_normal((T - 10, 2)) for _ in range(N)
        ]

    if incorrect_on_one_trial_pos:
        arrays = [rng.standard_normal((T, 2)) for _ in range(N)]
        arrays[11] = arrays[11][: T - 10, :]
        data["incorrect_on_one_trial_pos"] = arrays

    return pd.DataFrame(data)


def test_one_field():
    df = _generate_mock_data()

    assert determine_ref_field(df) == "pmd_spikes"


def test_correct_spikes():
    df = _generate_mock_data(correct_spikes=True)

    assert determine_ref_field(df) in ["pmd_spikes", "correct_spikes"]


def test_correct_rates():
    df = _generate_mock_data(correct_rates=True)

    assert determine_ref_field(df) in ["pmd_spikes", "correct_rates"]


def test_incorrect_on_all_trials_rates():
    df = _generate_mock_data(incorrect_on_all_trials_rates=True)

    with pytest.raises(ValueError):
        determine_ref_field(df)


def test_incorrect_on_one_trial_rates():
    df = _generate_mock_data(incorrect_on_one_trial_rates=True)

    with pytest.raises(ValueError):
        determine_ref_field(df)
