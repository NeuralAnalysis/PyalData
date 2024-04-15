import warnings

import pytest

from pyaldata.utils import get_time_varying_fields

from .test_determine_ref_field import _generate_mock_data


def test_strict_correct_single_field():
    # only time-varying field is pmd_spikes
    df = _generate_mock_data()

    assert get_time_varying_fields(df) == ["pmd_spikes"]


def test_strict_correct_spikes_rates():
    # two other correct spikes/rates fields should also be included
    df = _generate_mock_data(correct_spikes=True, correct_rates=True)

    assert set(get_time_varying_fields(df)) == {
        "pmd_spikes",
        "correct_spikes",
        "correct_rates",
    }


def test_strict_correct_pos():
    # a position field that has the correct time length should be included
    df = _generate_mock_data(correct_spikes=True, correct_pos=True)

    assert set(get_time_varying_fields(df)) == {
        "pmd_spikes",
        "correct_spikes",
        "correct_pos",
    }


def test_strict_incorrect_rates():
    # a rates field that has the wrong length on every trial should raise an error
    df = _generate_mock_data(incorrect_on_one_trial_rates=True)

    with pytest.raises(ValueError):
        get_time_varying_fields(df)


def test_strict_incorrect_pos_not_there():
    # a position field that has incorrect length on each trial should not be included
    df = _generate_mock_data(incorrect_on_all_trials_pos=True)

    assert get_time_varying_fields(df) == ["pmd_spikes"]


def test_strict_warn_if_suspicios():
    # if there is a position field that is the correct length on most of the trials
    # and warn_if_suspicious is True
    # then make sure a warning is given
    df = _generate_mock_data(incorrect_on_one_trial_pos=True)

    with pytest.warns():
        # get_time_varying_fields(df, warn_if_suspicious=True)
        get_time_varying_fields(df)


def test_strict_no_warning_if_not_wanted():
    # if warn_if_suspicious is False, there should be no warning
    df = _generate_mock_data(incorrect_on_one_trial_pos=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_time_varying_fields(df, warn_if_suspicious=False)


def test_strict_wrong_ref_field_given():
    # raise an error if the ref_field given is not in the dataframe
    df = _generate_mock_data()

    with pytest.raises(KeyError):
        get_time_varying_fields(df, ref_field="m1_spikes")


def test_strict_other_ref_field():
    # these two fields will have the same number of timepoints on all trials
    # but pmd_spikes is not the same,
    # so an error should be raised because all spikes/rates should match each other
    df = _generate_mock_data(
        incorrect_on_all_trials_rates=True, incorrect_on_all_trials_pos=True
    )

    assert set(get_time_varying_fields(df, ref_field="incorrect_on_all_trials_rates")) == {
        "incorrect_on_all_trials_rates",
        "incorrect_on_all_trials_pos",
    }


def test_old_correct():
    # from the old behavior only test the happy path
    df = _generate_mock_data(correct_spikes=True, correct_rates=True, correct_pos=True)

    assert set(get_time_varying_fields(df, strict_criterion=False)) == {
        "pmd_spikes",
        "correct_spikes",
        "correct_rates",
        "correct_pos",
    }
