import functools
import warnings
from typing import Union

import numpy as np
import pandas as pd

__all__ = [
    "copy_td",
    "determine_ref_field",
    "get_array_fields",
    "get_string_fields",
    "get_time_varying_fields",
    "get_trial_length",
    "get_trial_lengths",
    "remove_suffix",
]


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
                raise ValueError(
                    f"first argument of {func.__name__} has to be a pandas DataFrame"
                )

            kwargs["trial_data"] = df.copy()
            return func(**kwargs)
        else:
            # dataframe is the first positional argument
            if not isinstance(args[0], pd.DataFrame):
                raise ValueError(
                    f"first argument of {func.__name__} has to be a pandas DataFrame"
                )

            return func(args[0].copy(), *args[1:], **kwargs)

    return wrapper


def remove_suffix(text: str, suffix: str) -> str:
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
        return text[: -len(suffix)]
    else:
        warnings.warn(f"{text} doesn't end with {suffix}. Didn't remove anything.")
    return text


def determine_ref_field(trial_data: pd.DataFrame) -> str:
    """
    Find a dataframe column that ends with 'spikes' or 'rates', so it is likely to have a time dimension and can be used as a reference.

    Assuming spike and rate fields have time as their first dimension,
    make sure they all have the same number of timepoints,
    then return one of them.
    """
    # look for a spikes field
    spike_rate_fields = [
        col
        for col in trial_data.columns.values
        if col.endswith("spikes") or col.endswith("rates")
    ]

    # make sure they all have the same number of timepoints on all trials
    for i, trial in trial_data.iterrows():
        if len({trial[field].shape[0] for field in spike_rate_fields}) != 1:
            n_tp = {field: trial[field].shape[0] for field in spike_rate_fields}
            raise ValueError(
                f"Number of timepoints in spike/rate fields doesn't match. Found these fields: {n_tp}"
            )

    # if they all match, just return the first one
    return spike_rate_fields[0]


def get_time_varying_fields(
    trial_data: pd.DataFrame,
    ref_field: str = None,
    strict_criterion: bool = True,
    warn_if_suspicious: bool = True,
) -> list[str]:
    """
    Identify time-varying fields in the dataset


    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used

    strict_criterion: bool, default True
        in order for a column to qualify as time-varying,
        it has to contain numpy arrays and the first dimension of the arrays has to match
        the reference field on all trials

    warn_if_suspicious: bool, default True
        only used if strict_criterion is True.
        if a field contains numpy arrays and has the same length as the reference field
        on 80% of the trials, it might be a time-varying field with some errors.
        set this to true to check and warn if a field like that is found
        if strict_criterion is False, such fields are likely to cause an error


    Returns
    -------
    time_fields : list of str
        list of fieldnames that store time-varying signals
    """
    if ref_field is None:
        ref_field = determine_ref_field(trial_data)

    time_fields = []

    if strict_criterion:
        required_match_ratio = 0.8

        ref_lengths = np.array([arr.shape[0] for arr in trial_data[ref_field]])

        for col in get_array_fields(trial_data):
            col_lengths = np.array([arr.shape[0] for arr in trial_data[col]])
            if np.all(col_lengths == ref_lengths):
                time_fields.append(col)
            elif warn_if_suspicious:
                match_ratio = np.mean(col_lengths == ref_lengths)
                if match_ratio >= required_match_ratio:
                    warnings.warn(
                        f"{col} might be a time-varying field. It matches the length of {ref_field} on {match_ratio*100}% of trials"
                    )
    else:
        # use the old method

        # identify candidates based on the first trial
        first_trial = trial_data.iloc[0]
        T = first_trial[ref_field].shape[0]
        for col in first_trial.index:
            try:
                if first_trial[col].shape[0] == T:
                    time_fields.append(col)
            except Exception:
                pass

        # but check the rest of the trials, too
        ref_lengths = np.array([arr.shape[0] for arr in trial_data[ref_field]])
        for col in time_fields:
            col_lengths = np.array([arr.shape[0] for arr in trial_data[col]])
            assert np.all(
                col_lengths == ref_lengths
            ), f"not all lengths in {col} match the reference {ref_field}"

    return time_fields


def get_array_fields(trial_data: pd.DataFrame) -> list[str]:
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
    return [
        col
        for col in trial_data.columns
        if all([isinstance(el, np.ndarray) for el in trial_data[col]])
    ]


def get_string_fields(trial_data: pd.DataFrame) -> list[str]:
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
    return [
        col
        for col in trial_data.columns
        if all([isinstance(el, str) for el in trial_data[col]])
    ]


def _get_trial_length_trial(trial: pd.Series, ref_field: str = None) -> int:
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
        spike_rate_fields = [
            col
            for col in trial.index.values
            if col.endswith("spikes") or col.endswith("rates")
        ]

        if len({trial[field].shape[0] for field in spike_rate_fields}) != 1:
            n_tp = {field: trial[field].shape[0] for field in spike_rate_fields}
            raise ValueError(
                f"Number of timepoints in spike/rate fields doesn't match. Found these fields: {n_tp}"
            )

        ref_field = spike_rate_fields[0]

    return np.size(trial[ref_field], axis=0)


def get_trial_length(
    trial_or_df: Union[pd.Series, pd.DataFrame], ref_field: str = None
) -> int:
    """
    Get the number of time points in a trial, or the number of timepoints in a dataframe
    where all trials are the same length.

    Parameters
    ----------
    trial_or_df :
        Trial or dataframe to check
    ref_field : str, optional
        time-varying field to use for identifying the length
        if not given, the first field that ends with "spikes" is used

    Returns
    -------
    length : int

    Raises
    ------
    ValueError
        If not all the trials in the dataframe have the same length.
    TypeError
        If `trial_or_df` is not a pandas Series or DataFrame.
    """
    if isinstance(trial_or_df, pd.Series):
        return _get_trial_length_trial(trial_or_df, ref_field)
    elif isinstance(trial_or_df, pd.DataFrame):
        unique_trial_lengths = np.unique(get_trial_lengths(trial_or_df, ref_field))

        if len(unique_trial_lengths) != 1:
            raise ValueError("All trials must have the same length.")

        return unique_trial_lengths[0]
    else:
        raise TypeError("trial_or_df must be a pandas Series or DataFrame.")


def get_trial_lengths(trial_data: pd.DataFrame, ref_field: str = None) -> np.ndarray:
    """
    Get the number of time points in all trials.

    Parameters
    ----------
    trial_data : pd.DataFrame
        DataFrame to check.
    ref_field : str, optional
        time-varying field to use for identifying the length
        if not given, the first field that ends with "spikes" is used

    Returns
    -------
    numpy array with the length of each trial
    """
    return trial_data.apply(lambda trial: get_trial_length(trial, ref_field), axis=1).values
