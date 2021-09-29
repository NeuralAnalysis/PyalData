import numpy as np
import pandas as pd

import functools
import warnings


def copy_td(func):
    """
    Call copy on the first argument of the function and work on the copied value
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # if there are no positional arguments
        if len(args) == 0:
            df = kwargs["trial_data"]

            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"first argument of {func.__name__} has to be a pandas DataFrame")

            kwargs["trial_data"] = df.copy()
            return func(**kwargs)
        else:
            # dataframe is the first positional argument
            if not isinstance(args[0], pd.DataFrame):
                raise ValueError(f"first argument of {func.__name__} has to be a pandas DataFrame")

            return func(args[0].copy(), *args[1:], **kwargs)

    return wrapper


def remove_suffix(text, suffix):
    """
    Remove suffix from the end of text

    Parameters
    ----------
    text : str
        text from which to remove the suffix
    suffix : str
        suffix to remove from the end of text

    Returns
    -------
    text : str
        text with suffix removed if text ends with suffix
        text untouched if text doesn't end with suffix
    """
    if text.endswith(suffix):
        return text[:-len(suffix)]
    else:
        warnings.warn(f"{text} doesn't end with {suffix}. Didn't remove anything.")
    return text


def all_integer(arr):
    """
    Check if all the values in arr are approximately integers
    """
    return np.all(np.isclose(arr, np.array(arr, dtype=int)))

  
def get_time_varying_fields(trial_data, ref_field=None):
    """
    Identify time-varying fields in the dataset


    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used

    Returns
    -------
    time_fields : list of str
        list of fieldnames that store time-varying signals
    """
    if ref_field is None:
        # look for a spikes field
        ref_field = [col for col in trial_data.columns.values
                     if col.endswith("spikes") or col.endswith("rates")][0]

    # identify candidates based on the first trial
    first_trial = trial_data.iloc[0]
    T = first_trial[ref_field].shape[0]
    time_fields = []
    for col in first_trial.index:
        try:
            if first_trial[col].shape[0] == T:
                time_fields.append(col)
        except:
            pass

    # but check the rest of the trials, too
    ref_lengths = np.array([arr.shape[0] for arr in trial_data[ref_field]])
    for col in time_fields:
        col_lengths = np.array([arr.shape[0] for arr in trial_data[col]])
        assert np.all(col_lengths == ref_lengths), f"not all lengths in {col} match the reference {ref_field}"

    return time_fields


def get_array_fields(trial_data):
    """
    Get the names of columns that contain numpy arrays

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    Returns
    -------
    columns that have array values : list of str
    """
    return [col for col in trial_data.columns if all([isinstance(el, np.ndarray) for el in trial_data[col]])]


def get_string_fields(trial_data):
    """
    Get the names of columns that contain string data

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    Returns
    -------
    columns that have string values : list of str
    """
    return [col for col in trial_data.columns if all([isinstance(el, str) for el in trial_data[col]])]


def get_trial_length(trial, ref_field=None):
    """
    Get the number of time points in the trial

    Parameters
    ----------
    trial : pd.Series
        trial to check
    ref_field : str, optional
        time-varying field to use for identifying the length
        if not given, the first field that ends with "spikes" is used

    Returns
    -------
    length : int
    """
    if ref_field is None:
        ref_field = [col for col in trial.index.values if col.endswith("spikes") or col.endswith("rates")][0]

    return np.size(trial[ref_field], axis=0)
