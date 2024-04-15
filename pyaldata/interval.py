import numpy as np

from . import utils
from . import tools

import warnings

warnings.simplefilter("always", UserWarning)


@utils.copy_td
def restrict_to_interval(
    trial_data,
    start_point_name=None,
    end_point_name=None,
    rel_start=0,
    rel_end=0,
    before=None,
    after=None,
    epoch_fun=None,
    warn_per_trial=False,
    reset_index=True,
    ref_field=None,
):
    """
    Restrict time-varying fields to an interval around a time point or between two time points (inclusive on both ends)

    trial_data : pd.DataFrame
        data in trial_data format
    start_point_name : str, optional
        name of the time point around which the interval starts
    end_point_name : str, optional
        name of the time point around which the interval ends
        if None, the interval is created around start_point_name
    rel_start : int, default 0
        when to start extracting relative to the starting time point
        replaces the 'before' option
    rel_end : int, default 0
        when to stop extracting relative to the ending time point
        replaces the 'after' option
    before (deprecated) : int, optional, default None
        number of time points to extract before the starting time point
        Please use rel_start instead.
    after (deprecated): int, optional, default None
        number of time points to extract after the ending time point
        Please use rel_end instead.
    epoch_fun : function, optional
        function that takes a trial and returns the epoch to extract
    warn_per_trial : bool, optional, default False
        give more detailed warnings about indexing in each problematic trial
    reset_index : bool, optional, default True
        whether to reset the dataframe index to [0,1,2,...]
        or keep the original indices of the kept trials
    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used

    Returns
    -------
    data in trial_data format
    """
    if before is not None:
        warnings.warn("'before' is deprecated. Use 'rel_start' instead.")
        rel_start = -before
    if after is not None:
        warnings.warn("'after' is deprecated. Use 'rel_end' instead.")
        rel_end = after

    assert (start_point_name is None) ^ (
        epoch_fun is None
    ), "Give either start_point_name or epoch_fun."

    idx_fields = [col for col in trial_data.columns.values if col.startswith("idx")]
    time_fields = utils.get_time_varying_fields(trial_data, ref_field)

    # generate epoch_fun if the interval is given with time points
    if start_point_name is not None:
        epoch_fun = generate_epoch_fun(start_point_name, end_point_name, rel_start, rel_end)

    # check in which trials the indexing works properly
    kept_trials_mask = np.array(
        [
            _slice_in_trial(trial, epoch_fun(trial), warn_per_trial)
            for (i, trial) in trial_data.iterrows()
        ]
    ).astype(bool)
    # warn about dropping the problematic trials
    if np.any(~kept_trials_mask):
        warnings.warn(
            f"""Dropping the trials with the following IDs because of invalid time indexing. For more information, try warn_per_trial=True

        {trial_data.trial_id.values[~kept_trials_mask]}""",
            stacklevel=3,
        )

    # only keep trials in which the indexing works properly
    trial_data = tools.select_trials(trial_data, kept_trials_mask, reset_index)

    # cut time varying signals
    trim_temp = {
        col: extract_interval_from_signal(trial_data, col, epoch_fun) for col in time_fields
    }
    trial_data = trial_data.assign(**trim_temp)

    # adjust idx fields
    def _adjust_field(val, new_T):
        if isinstance(val, (np.ndarray, list)):
            return np.array([np.nan if (idx < 0 or idx > new_T) else idx for idx in val])
        elif (val < 0) or (val > new_T):
            return np.nan
        else:
            return val

    new_time_lengths = [arr.shape[0] for arr in trial_data[time_fields[0]]]
    zero_points = [epoch_fun(trial).start for (i, trial) in trial_data.iterrows()]

    for col in idx_fields:
        trial_data[col] = [
            idx - zero_point for (idx, zero_point) in zip(trial_data[col], zero_points)
        ]

        # set indices that are now invalid (i.e. not in the restricted interval) to nan
        trial_data[col] = [
            _adjust_field(idx, new_T)
            for (idx, new_T) in zip(trial_data[col], new_time_lengths)
        ]

    return trial_data


def slice_around_index(idx, before, after):
    """
    Return a slice around an index
    Length will be before + after + 1

    Parameters
    ----------
    idx : int
        index around which to create the interval
    before : int
        number of time points before time point
    after : int
        number of time points after time point

    Returns
    -------
    slice(idx-before, idx+after+1)
    """
    start = idx - before
    end = idx + after + 1

    if np.isfinite(start):
        start = int(start)
    if np.isfinite(end):
        end = int(end)

    return slice(start, end)


def slice_around_point(trial, point_name, before, after):
    """
    Return a slice around a time point in the trial
    Length will be before + after + 1

    Parameters
    ----------
    trial : pd.Series
        a row from a trial_data dataframe
        representing a trial
    point_name : str
        name of the time point around which to create the interval
    before : int
        number of time points before time point
    after : int
        number of time points after time point

    Returns
    -------
    slice object
    """
    start = trial[point_name] - before
    end = trial[point_name] + after + 1

    if np.isfinite(start):
        start = int(start)
    if np.isfinite(end):
        end = int(end)

    return slice(start, end)


def slice_between_points(trial, start_point_name, end_point_name, before, after):
    """
    Return a slice that starts before start_point_name and ends after end_point_name

    Parameters
    ----------
    trial : pd.Series
        a row from a trial_data dataframe
        representing a trial
    start_point_name : str
        name of the time point around which the interval starts
    end_point_name : str
        name of the time point around which the interval ends
    before : int
        number of time points before time point
    after : int
        number of time points after time point

    Returns
    -------
    slice object
    """
    start = trial[start_point_name] - before
    end = trial[end_point_name] + after + 1

    if np.isfinite(start):
        start = int(start)
    if np.isfinite(end):
        end = int(end)

    return slice(start, end)


def generate_epoch_fun(start_point_name, end_point_name=None, rel_start=0, rel_end=0):
    """
    Return a function that slices a trial around/between time points

    Parameters
    ----------
    start_point_name : str
        name of the time point around which the interval starts
    end_point_name : str, optional
        name of the time point around which the interval ends
        if None, the interval is created around start_point_name
    rel_start : int, default 0
        when to start extracting relative to the starting time point
        replaces the 'before' option
    rel_end : int, default 0
        when to stop extracting relative to the ending time point
        replaces the 'after' option

    Returns
    -------
    epoch_fun : function
        function that can be used to extract the interval from a trial
    """
    if end_point_name is None:
        epoch_fun = lambda trial: slice_around_point(
            trial, start_point_name, -rel_start, rel_end
        )
    else:
        epoch_fun = lambda trial: slice_between_points(
            trial, start_point_name, end_point_name, -rel_start, rel_end
        )

    return epoch_fun


def extract_interval_from_signal(trial_data, signal, epoch_fun):
    """
    Extract an interval from a time-varying signal in the dataset

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signal : str
        name of the signal to extract
    epoch_fun : function
        function that takes a trial and returns the epoch to extract

    Returns
    -------
    list of the extracted np.arrays
    """
    return [trial[signal][epoch_fun(trial), ...] for (i, trial) in trial_data.iterrows()]


def _slice_in_trial(trial, sl, warn=False):
    """
    Check if the slice is within the trial's time indices

    Parameters
    ----------
    trial : pd.Series
        trial to check
    sl : slice
        slice to check
    warn : bool, optional, default False
        whether to warn if the slice is outside
        the trial's time index

    Returns
    -------
    is_inside : bool
    """
    T = utils.get_trial_length(trial)

    is_inside = True

    if sl.start < 0:
        is_inside = False
        if warn:
            warnings.warn(
                f"Invalid time index on trial with ID {trial.trial_id}. Trying to access index {sl.start} < 0"
            )
    if sl.stop > T:
        is_inside = False
        if warn:
            warnings.warn(
                f"Invalid time index on trial with ID {trial.trial_id}. Trying to access index {sl.stop-1} >= {T}"
            )

    if not np.isfinite(sl.start):
        is_inside = False
        if warn:
            warnings.warn(
                f"Invalid time index on trial with ID {trial.trial_id}. Starting point is {sl.start}"
            )
    if not np.isfinite(sl.stop):
        is_inside = False
        if warn:
            warnings.warn(
                f"Invalid time index on trial with ID {trial.trial_id}. End point is {sl.stop}"
            )

    return is_inside
