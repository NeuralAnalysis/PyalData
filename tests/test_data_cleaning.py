import numpy as np

from pyaldata.data_cleaning import clean_0d_array_fields

from .test_determine_ref_field import _generate_mock_data


def test_clean_0d_array_fields_int_only():
    # if the field is already just scalars, it should not be changed
    df = _generate_mock_data()
    idx_vals = [5 for _ in range(df.shape[0])]
    df["idx_field"] = idx_vals

    assert clean_0d_array_fields(df)["idx_field"].tolist() == idx_vals


def test_clean_0d_array_fields_0d_only():
    # if the field is only 0d arrays, they should be converted to scalars
    df = _generate_mock_data()
    idx_vals_scalar = [5 for _ in range(df.shape[0])]
    df["idx_field"] = [np.array(val) for val in idx_vals_scalar]

    assert clean_0d_array_fields(df)["idx_field"].tolist() == idx_vals_scalar


def test_clean_0d_array_fields_1d_only():
    # if the field is only 1d arrays, they should be left alone
    df = _generate_mock_data()
    idx_vals_scalar = [5 for _ in range(df.shape[0])]
    idx_vals_1d_array = [np.array([val]) for val in idx_vals_scalar]
    df["idx_field"] = idx_vals_1d_array

    assert clean_0d_array_fields(df)["idx_field"].tolist() == idx_vals_1d_array


def test_clean_0d_array_fields_first_element_0d():
    df = _generate_mock_data()
    idx_vals_scalar = [5 for _ in range(df.shape[0])]

    idx_vals_mixed = [val for val in idx_vals_scalar]
    idx_vals_mixed[0] = np.array(idx_vals_mixed[0])  # first element is 0d

    df["idx_field"] = idx_vals_mixed

    assert clean_0d_array_fields(df)["idx_field"].tolist() == idx_vals_mixed


def test_clean_0d_array_fields_first_mixed():
    df = _generate_mock_data()
    idx_vals_scalar = [5 for _ in range(df.shape[0])]

    idx_vals_mixed = [val for val in idx_vals_scalar]

    idx_vals_mixed[2] = np.array([idx_vals_mixed[2], idx_vals_mixed[2] * 2])
    idx_vals_mixed[1] = np.array(idx_vals_mixed[1])

    df["idx_field"] = idx_vals_mixed

    assert clean_0d_array_fields(df)["idx_field"].tolist() == idx_vals_mixed
