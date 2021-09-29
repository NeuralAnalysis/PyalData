import numpy as np


def concat_trials(trial_data, signal, trial_indices=None):
    """
    Concatenate signal from different trials in time

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signal : str
        name of the field to concatenate
    trial_indices : array-like of ints
        indices of the trials we want to get the signal from

    Returns
    -------
    np.array of the signal in the selected trials
    stacked on top of each other
    """
    if trial_indices is None:
        return np.concatenate(trial_data[signal].values, axis=0)
    else:
        return np.concatenate(trial_data.loc[trial_indices, signal].values, axis=0)


def get_signals(trial_data, signals, trial_indices=None):
    """
    Extract multiple signals

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signals : str or list of str
        name of the fields to concatenate
    trial_indices : array-like of ints
        indices of the trials we want to get the signals from

    Returns
    -------
    np.array of the signals in the selected trials
    merged and stacked on top of each other
    """
    if isinstance(signals, str):
        signals = [signals]

    return np.column_stack([concat_trials(trial_data, s, trial_indices) for s in signals])


def get_sig_by_trial(trial_data, signals, trial_indices=None):
    """
    Extract multiple signals and stack trials along a trial dimension
    resulting in a T x N x n_trials tensor.
    Note that each trial and signal has to have the same length (T).

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signals : str or list of str
        name of the fields to extract
        if multiple, signals are merged
    trial_indices : array-like of ints
        indices of the trials we want to get the signals from

    Returns
    -------
    np.array of the signals in the selected trials
    merged and stacked along a new trial dimension
    shape T x N x n_trials
    """
    if isinstance(signals, str):
        signals = [signals]

    if trial_indices is None:
        trial_indices = trial_data.index

    return np.stack([np.column_stack(row) for row in trial_data.loc[trial_indices, signals].values],
                    axis=-1)


def stack_time_average(trial_data, signal):
    """
    Average signal in time in each trial, then stack them into an
    n_trials x n_features array

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signal : str
        signal to work on

    Returns
    -------
    X : 2D np.array
        n_trials x n_features array in which every row is
        the time-averaged signal in a trial
    """
    return np.stack([np.mean(arr, axis=0) for arr in trial_data[signal]])
