import numpy as np
import pandas as pd
import pytest

from pyaldata.utils import get_trial_length, get_trial_lengths

from .test_determine_ref_field import N, T, _generate_mock_data


def test_single_field():
    df = _generate_mock_data()

    assert get_trial_length(df.iloc[0]) == T


def test_correct_multiple():
    df = _generate_mock_data(correct_rates=True, correct_spikes=True)

    assert get_trial_length(df.iloc[0]) == T


def test_inconsistent_lengths():
    df = _generate_mock_data(incorrect_on_all_trials_rates=True)

    with pytest.raises(ValueError):
        get_trial_length(df.iloc[0])


def test_get_trial_lengths_all_the_same():
    df = _generate_mock_data()

    expected = T * np.ones(df.shape[0])

    assert np.all(get_trial_lengths(df) == expected)


def test_get_trial_lengths_random_lengths():
    trial_lengths = np.random.randint(0, T, size=N)

    data = {}
    data["pmd_spikes"] = [np.random.normal(size=(length, 100)) for length in trial_lengths]
    df = pd.DataFrame(data)

    assert np.all(get_trial_lengths(df) == trial_lengths)


def test_get_trial_length_df_happy():
    df = _generate_mock_data(correct_rates=True, correct_spikes=True)

    assert get_trial_length(df) == T


def test_get_trial_length_df_different_lengths():
    trial_lengths = np.random.randint(0, T, size=N)

    data = {}
    data["pmd_spikes"] = [np.random.normal(size=(length, 100)) for length in trial_lengths]
    df = pd.DataFrame(data)

    with pytest.raises(ValueError, match="All trials must have the same length."):
        get_trial_length(df)


def test_get_trial_length_df_wrong_type():
    with pytest.raises(TypeError):
        get_trial_length([1, 2, 3])
