import numpy as np
import pandas as pd

import scipy.signal as scs
from scipy.ndimage import convolve1d

import functools
import warnings


def norm_gauss_window(bin_length, std):
    """
    Gaussian window with its mass normalized to 1

    Parameters
    ----------
    bin_length : float
        binning length of the array we want to smooth in ms
    std : float
        standard deviation of the window
        use hw_to_std to calculate std based from half-width

    Returns
    -------
    win : 1D np.array
        Gaussian kernel with
            length: 10*std/bin_length
            mass normalized to 1
    """
    win = scs.gaussian(int(10*std/bin_length), std/bin_length)
    return win / np.sum(win)


def hw_to_std(hw):
    """
    Convert half-width to standard deviation for a Gaussian window.
    """
    return hw / (2 * np.sqrt(2 * np.log(2)))


def smooth_data(mat, dt=None, std=None, hw=None, win=None):
    """
    Smooth a 1D array or every column of a 2D array

    Parameters
    ----------
    mat : 1D or 2D np.array
        vector or matrix whose columns to smooth
        e.g. recorded spikes in a time x neuron array
    dt : float
        length of the timesteps in seconds
    std : float (optional)
        standard deviation of the smoothing window
    hw : float (optional)
        half-width of the smoothing window
    win : 1D array-like (optional)
        smoothing window to convolve with

    Returns
    -------
    np.array of the same size as mat
    """
    assert only_one_is_not_None((win, hw, std))

    if win is None:
        assert dt is not None, "specify dt if not supplying window"

        if std is None:
            std = hw_to_std(hw)

        win = norm_gauss_window(dt, std)

    if mat.ndim == 1 or mat.ndim == 2:
        return convolve1d(mat, win, axis=0, output=np.float32, mode='reflect')
    else:
        raise ValueError("mat has to be a 1D or 2D array")


def z_score(arr):
    """
    z-score function by removing the mean and dividing by the standard deviation (across time)

    Parameters
    ----------
    arr : np.array
        array to z-score
        time on the first axis

    Returns
    -------
    z-scored array with the same shape as arr
    """
    return (arr - arr.mean(axis=0)) / arr.std(axis=0)


def center(arr):
    """
    Center array by removing the mean across time

    Parameters
    ----------
    arr : np.array
        array to center
        time on the first axis

    Returns
    -------
    centered array with the same shape as arr
    """
    return arr - arr.mean(axis=0)


def only_one_is_not_None(args):
    return sum([arg is not None for arg in args]) == 1


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

  
def get_range(arr, axis=None):
    """
    Difference between the highest and the lowest value

    Parameters
    ----------
    arr : np.array
        typically a time-varying signal
    axis : int, optional
        if None, difference between the largest and smallest value in the array
        if int, calculate the range along the given axis

    Returns
    -------
    if axis=None, a single integer
    if axis is not None, an np.array containing the ranges along the given axis
    """
    return np.max(arr, axis=axis) - np.min(arr, axis=axis)


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
