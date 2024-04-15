import numpy as np

from . import utils
from . import extract_signals


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


@utils.copy_td
def center_signal(trial_data, signals, train_trials=None):
    """
    Center signal by removing the mean across time


    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signals : str or list of str
        column to center
    train_trials : list of int
        indices of the trials to consider when calculating the mean

    Returns
    -------
    trial_data : pd.DataFrame
        data with the given field centered
    """
    if isinstance(signals, str):
        signals = [signals]

    for signal in signals:
        whole_signal = extract_signals.concat_trials(trial_data, signal, train_trials)
        col_mean = np.mean(whole_signal, axis=0)

        trial_data[signal] = [s - col_mean for s in trial_data[signal]]

    return trial_data


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


@utils.copy_td
def z_score_signal(trial_data, signals, train_trials=None):
    """
    z-score signal by removing the mean across time
    and dividing by the standard deviation


    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signals : str or list of str
        column to z-score
    train_trials : list of int
        indices of the trials to consider when calculating the mean and std

    Returns
    -------
    trial_data : pd.DataFrame
        data with the given field z-scored
    """
    if isinstance(signals, str):
        signals = [signals]

    for signal in signals:
        whole_signal = extract_signals.concat_trials(trial_data, signal, train_trials)
        col_mean = np.mean(whole_signal, axis=0)
        col_std = np.std(whole_signal, axis=0)

        trial_data[signal] = [(s - col_mean) / col_std for s in trial_data[signal]]

    return trial_data


@utils.copy_td
def sqrt_transform_signal(trial_data, signals, train_trials=None):
    """
    square-root transform signal

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signals : str
        column to transform
    train_trials : list of int
        warning: not used, only here for consistency with other functions
        indices of the trials to consider when calculating the mean and std

    Returns
    -------
    trial_data : pd.DataFrame
        data with the given field transformed
    """
    if train_trials is not None:
        utils.warnings.warn("train_trials is not used in sqrt_transform")

    if isinstance(signals, str):
        signals = [signals]

    for signal in signals:
        for s in trial_data[signal]:
            if (s < 0).any():
                raise ValueError(
                    "signal cannot contain negative values when square-root transforming"
                )

        trial_data[signal] = [np.sqrt(s) for s in trial_data[signal]]

    return trial_data


@utils.copy_td
def zero_normalize_signal(trial_data, signals, train_trials=None):
    """
    Zero-normalize signal to 0-1 by removing the minimum, then dividing by the range


    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signals : str
        column to normalize
    train_trials : list of int
        indices of the trials to consider when calculating the minimum and range

    Returns
    -------
    trial_data : pd.DataFrame
        data with the given field normalized
    """
    if isinstance(signals, str):
        signals = [signals]

    for signal in signals:
        whole_signal = extract_signals.concat_trials(trial_data, signal, train_trials)
        col_min = np.min(whole_signal, axis=0)
        col_range = get_range(whole_signal, axis=0)

        trial_data[signal] = [(s - col_min) / col_range for s in trial_data[signal]]

    return trial_data


@utils.copy_td
def center_normalize_signal(trial_data, signals, train_trials=None):
    """
    Center-normalize signal by removing the mean, then dividing by the range


    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signals : str
        column to normalize
    train_trials : list of int
        indices of the trials to consider when calculating the mean and range

    Returns
    -------
    trial_data : pd.DataFrame
        data with the given field normalized
    """
    if isinstance(signals, str):
        signals = [signals]

    for signal in signals:
        whole_signal = extract_signals.concat_trials(trial_data, signal, train_trials)
        col_mean = np.mean(whole_signal, axis=0)
        col_range = get_range(whole_signal, axis=0)

        trial_data[signal] = [(s - col_mean) / col_range for s in trial_data[signal]]

    return trial_data


@utils.copy_td
def soft_normalize_signal(trial_data, signals, train_trials=None, alpha=5):
    """
    Soft normalize signal a la Churchland papers

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signals : str
        column to normalize
    train_trials : list of int
        indices of the trials to consider when calculating the range
    alpha : float, default 5
        normalization factor = firing rate range + alpha

    Returns
    -------
    trial_data : pd.DataFrame
        data with the given field soft-normalized
    """
    if isinstance(signals, str):
        signals = [signals]

    for signal in signals:
        whole_signal = extract_signals.concat_trials(trial_data, signal, train_trials)

        norm_factor = get_range(whole_signal) + alpha

        trial_data[signal] = [s / norm_factor for s in trial_data[signal]]

    return trial_data


@utils.copy_td
def transform_signal(trial_data, signals, transformations, train_trials=None, **kwargs):
    """
    Apply transformation(s) to signal


    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signals : str
        column to normalize
    transformations : str or list of str
        transformations to apply
        if it's a list of strings, the corresponding transformations are applied in the given order
        Currently implemented:  'center',
                                'center_normalize',
                                'zero_normalize',
                                'sqrt' or 'sqrt_transform',
                                'z-score' or 'z_score',
                                'zero_normalize',
                                'soft_normalize'
    train_trials : list of int
        indices of the trials to consider for setting up the transformations
    kwargs
        keyword arguments to pass to the transformation functions


    Returns
    -------
    trial_data : pd.DataFrame
        data with the given field transformed
    """
    method_dict = {
        "center": center_signal,
        "center_normalize": center_normalize_signal,
        "zero_normalize": zero_normalize_signal,
        "sqrt_transform": sqrt_transform_signal,
        "sqrt": sqrt_transform_signal,
        "z_score": z_score_signal,
        "z-score": z_score_signal,
        "zero_normalize": zero_normalize_signal,
        "soft_normalize": soft_normalize_signal,
    }

    if isinstance(transformations, str):
        transformations = [transformations]

    if isinstance(signals, str):
        signals = [signals]

    for signal in signals:
        for trans in transformations:
            trial_data = method_dict[trans](trial_data, signal, train_trials, **kwargs)

    return trial_data
