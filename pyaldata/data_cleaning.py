from . import utils

import numpy as np


@utils.copy_td
def backshift_idx_fields(trial_data):
    """
    Adjust index fields from 1-based to 0-based indexing

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    Returns
    -------
    trial_data with the 'idx_' fields adjusted
    """
    idx_fields = [col for col in trial_data.columns.values if col.startswith("idx")]

    for col in idx_fields:
        # using a list comprehension to still work if the idx field itself is an array
        trial_data[col] = [idx - 1 for idx in trial_data[col]]

    return trial_data


@utils.copy_td
def clean_0d_array_fields(df):
    """
    Loading v7.3 MAT files, sometimes scalers are stored as 0-dimensional arrays for some reason.
    This converts those back to scalars.

    Parameters
    ----------
    df : pd.DataFrame
        data in trial_data format

    Returns
    -------
    a copy of df with the relevant fields changed
    """
    for c in df.columns:
        if isinstance(df[c].values[0], np.ndarray):
            if all([arr.ndim == 0 for arr in df[c]]):
                df[c] = [arr.item() for arr in df[c]]

    return df


@utils.copy_td
def clean_integer_fields(df):
    """
    Modify fields that store integers as floats to store them as integers instead.

    Parameters
    ----------
    df : pd.DataFrame
        data in trial_data format

    Returns
    -------
    a copy of df with the relevant fields changed
    """
    for field in df.columns:
        if isinstance(df[field].values[0], np.ndarray):
            try:
                int_arrays = [np.int32(arr) for arr in df[field]]
            except:
                print(f"array field {field} could not be converted to int.")
            else:
                if all(
                    [
                        np.allclose(int_arr, arr)
                        for (int_arr, arr) in zip(int_arrays, df[field])
                    ]
                ):
                    df[field] = int_arrays
        else:
            if not isinstance(df[field].values[0], str):
                try:
                    int_version = np.int32(df[field])
                except:
                    print(f"field {field} could not be converted to int.")
                else:
                    if np.allclose(int_version, df[field]):
                        df[field] = int_version

    return df
