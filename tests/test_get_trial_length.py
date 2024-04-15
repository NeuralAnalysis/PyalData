import pytest

from pyaldata.utils import get_trial_length

from .test_determine_ref_field import T, _generate_mock_data


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
