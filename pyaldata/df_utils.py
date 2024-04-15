import pandas as pd

from . import utils


def concat_TDs(frames, re_index=True):
    """
    Concatenate trial_data structures.
    Supports if structs don't have the same fields, missing values are filled with nan.

    Parameters
    ----------
    frames: sequence of trial_data structs
        ex: frames=[td1, td2, td3]
    re_index: bool, optional, default True
        Sets the index of the struct from 0 to n-1 (n is total number of trials).
        If False, the index from each original frame is maintained (careful: might lead to repeated indices).

    Returns
    -------
    Returns the concatenated dataframe.
        trial_data_total = df1 + df2 +...
    """
    if re_index:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.concat(frames)


@utils.copy_td
def rename_fields(trial_data, fields):
    """
    Rename field inside trial data

    Parameters
    ----------
    trial_data: pd.DataFrame
        trial_data dataframe
    fields: dict
        dictionary where keys are fields to change and the keys are the new names
        ex: fields = {'old_name1':'new_name1', 'old_name2':'new_name2'}

    Returns
    ----------
    trial_data: pd.DataFrame
        data with fields renamed
    """

    for f in fields.keys():
        if f not in trial_data:
            raise ValueError(f"{f} field does not exist in trial data")

    return trial_data.rename(columns=fields)


@utils.copy_td
def copy_fields(trial_data, fields):
    """
    Copy and rename inside trial data

    Parameters
    ----------
    trial_data: pd.DataFrame
        trial_data dataframe
    fields: dict
        dictionary where keys are fields to copy and the keys are the new names
        ex: fields = {'old_name1':'new_name1', 'old_name2':'new_name2'}

    Returns
    ----------
    trial_data: pd.DataFrame
        data with the copied fields with the new name
    """
    # Check if all fields exist
    for f in fields.keys():
        if f not in trial_data:
            raise ValueError(f"{f} field does not exist in trial data")

    for f in fields.keys():
        trial_data[fields[f]] = trial_data[f]

    return trial_data
