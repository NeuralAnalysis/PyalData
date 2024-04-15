import numpy as np
import pandas as pd
import pytest

from pyaldata.tools import trial_average

n_trials = 64
rng = np.random.default_rng()


def _generate_data():
    data = {}

    data["constant_string_field"] = ["Bob" for _ in range(n_trials)]
    data["changing_string_field"] = [rng.choice(["Bob", "Alice"]) for _ in range(n_trials)]
    data["int_field"] = rng.integers(low=0, high=50, size=n_trials)
    data["float_field"] = rng.random(size=n_trials)
    data["good_rates"] = [i * np.ones((10, 5)) for i in range(n_trials)]

    return pd.DataFrame(data)


def test_changing_string_field_is_dropped():
    df = _generate_data()

    av_df = trial_average(df, None)

    assert av_df.shape[0] == 1
    assert set(av_df.columns) == {
        "constant_string_field",
        "int_field",
        "float_field",
        "good_rates",
    }


def test_array_field():
    df = _generate_data()

    av_df = trial_average(df, None)

    assert set(av_df.columns) == {
        "constant_string_field",
        "int_field",
        "float_field",
        "good_rates",
    }

    expected_av_array = np.mean(range(n_trials)) * np.ones((10, 5))

    assert av_df.shape[0] == 1
    assert np.allclose(av_df["good_rates"].iloc[0], expected_av_array)


def test_error_on_different_lengths():
    df = _generate_data()
    df["diff_length_rates"] = [
        i * np.ones((rng.integers(8, 10), 5)) for i in range(n_trials)
    ]

    with pytest.raises(ValueError):
        trial_average(df, None)


def test_condition_single_field():
    df = _generate_data()
    av_df = trial_average(df, "changing_string_field")
    assert set(av_df.index.values) == {"Alice", "Bob"}


def test_condition_two_fields():
    df = _generate_data()
    df["ab"] = ["a" if i < n_trials // 2 else "b" for i in range(n_trials)]
    df["cd"] = ["c" if i % 2 == 0 else "d" for i in range(n_trials)]

    av_df = trial_average(df, ["ab", "cd"])
    print(av_df.index)

    # we have 4 possible combinations
    assert av_df.shape[0] == 4


def test_condition_array():
    df = _generate_data()

    cond_per_row = [True if i < df.shape[0] // 2 else False for i in range(df.shape[0])]
    av_df = trial_average(df, cond_per_row)

    assert set(av_df.index.values) == {True, False}


def test_condition_callable():
    df = _generate_data()

    av_df = trial_average(
        df, lambda row: "x" if row.changing_string_field == "Alice" else "y"
    )

    assert set(av_df.index.values) == {"x", "y"}
