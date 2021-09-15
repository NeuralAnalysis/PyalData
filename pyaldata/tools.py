import numpy as np
import pandas as pd

from . import utils
import warnings
warnings.simplefilter("always", UserWarning)


@utils.copy_td
def add_gradient(trial_data, signal, outfield=None, normalize=False):
    """
    Compute the gradient of signal in time

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signal : str
        name of the field whose gradient we want to compute
    outfield : str (optional)
        if given, the name of the field in which to store the gradient
        if not given, 'd' is prepended to the signal
    normalize : bool, default False
        normalize gradient by bin size
        for example put the dt in v = ds/dt :)

    Returns
    -------
    trial_data : pd.DataFrame
        copy of trial_data with the gradient field added
    """
    if outfield is None:
        outfield = 'd' + signal

    trial_data[outfield] = [np.gradient(s, axis=0) for s in trial_data[signal]]

    if normalize:
        bin_size = trial_data.bin_size.values[0]
        assert all(trial_data.bin_size.values == bin_size)

        trial_data[outfield] = trial_data[outfield] / bin_size

    return trial_data


@utils.copy_td
def combine_time_bins(trial_data, n_bins, extra_time_fields=None, ref_field=None):
    """
    Re-bin data by combining n_bins timesteps

    Fields that are adjusted by default are:
        - bin_size
        - spikes
        - rates
        - idx
        - fields found by utils.get_time_varying_fields
    If you want to include others, specify extra_time_fields
    
    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    n_bins : int
        number of bins to combine into one
    extra_time_fields : list of str (optional)
        extra time-varying signals to adjust
    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used

    Returns
    -------
    adjusted trial_data copy
    """
    spike_fields = [col for col in trial_data.columns if col.endswith("spikes")]
    rate_fields = [col for col in trial_data.columns if col.endswith("rates")]
    idx_fields = [col for col in trial_data.columns if col.startswith("idx")]

    # check if there are any time-varying fields left
    other_time_fields = [col for col in utils.get_time_varying_fields(trial_data, ref_field)
                         if (col not in spike_fields) and (col not in rate_fields)]

    if len(trial_data.bin_size.unique()) != 1:
        raise NotImplementedError("implementation assumes that every trial has the same bin_size")

    trial_data["bin_size"] = n_bins * trial_data["bin_size"]

    # adjust indices
    for col in idx_fields:
        trial_data[col] = [idx // n_bins for idx in trial_data[col]]


    # rebin time-varying fields
    def rebin_array(arr, red_fun):
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        T, N = arr.shape
        T = (T // n_bins) * n_bins # throw away last bins

        arr = arr[:T, :]
        arr = arr.reshape(int(T / n_bins), n_bins, N)

        return red_fun(arr, axis=1).squeeze()

    # start with spike fields
    for col in spike_fields:
        # if we think the column still holds spikes
        if np.all([utils.all_integer(arr) for arr in trial_data[col]]):
            f = np.sum
        # if they are not integers anymore, e.g. because they've been smoothed
        else:
            f = np.mean

        trial_data[col] = [rebin_array(arr, f) for arr in trial_data[col]]

    # rebin rate fields
    for col in rate_fields:
        trial_data[col] = [rebin_array(arr, np.mean) for arr in trial_data[col]]



    # hopefully all time-varying fields were caught but the user can also provide some
    if extra_time_fields is not None:
        if isinstance(extra_time_fields, str):
            extra_time_fields = [extra_time_fields]

        # remove duplicate field names
        other_time_fields = list(set(other_time_fields + extra_time_fields))

    # rebin the time-varying fields left
    for col in other_time_fields:
        trial_data[col] = [rebin_array(arr, np.mean) for arr in trial_data[col]]


    return trial_data


@utils.copy_td
def merge_signals(trial_data, signals, out_fieldname):
    """
    Merge two signals under a new name
    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signals : list of str
        name of the fields we want to merge
    out_fieldname : str
        name of the field in which to store the output
        
    Returns
    -------
    trial_data : pd.DataFrame
        copy of trial_data with out_fieldname added
    """
    if isinstance(signals, str):
        raise ValueError("signals should be a list of fields")
    if len(signals) == 1:
        raise ValueError(f"This function is for merging multiple signals. Only got {signals[0]}")

    trial_data[out_fieldname] = [np.column_stack(row) for row in trial_data[signals].values]
    
    return trial_data


@utils.copy_td
def add_norm(trial_data, signal):
    """
    Add the norm of the signal to the dataframe

    Parameters
    ----------
    trial_data : pd.DataFrame
        trial_data dataframe
    signal : str
        field to take the norm of

    Returns
    -------
    td : pd.DataFrame
        trial_data with '_norm' fields added
    """
    norm_field_name = signal + "_norm"

    trial_data[norm_field_name] = [np.linalg.norm(s, axis=1) for s in trial_data[signal]]
    
    return trial_data
    

def trial_average(trial_data, condition, ref_field=None):
    """
    Trial-average signals, optionally after grouping trials by some conditions

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    condition : str, array-like trial_data.index, function, or None
        if None, there's no grouping
        if str, group trials by this field
        if array-like, condition is a value that is assigned to each trial (e.g. df.target_id < 4),
        and trials are grouped based on these values
        if function, it should take a trial and return a value. the trials will be grouped based on these values
    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used

    Returns
    -------
    pd.DataFrame with the fields averaged and the trial_id column dropped
    """
    time_fields = utils.get_time_varying_fields(trial_data, ref_field)
    for col in time_fields:
        assert len(set([arr.shape for arr in trial_data[col]])) == 1, f"Trials should have the same time coordinates."

    if condition is None:
        av_df = trial_data.mean()

        # calculate the mean of array fields one by one
        for col in utils.get_array_fields(trial_data):
            av_df[col] = trial_data[col].mean()

        # keep string fields if they are the same on every trial
        for col in utils.get_string_fields(trial_data):
            if trial_data[col].unique().size == 1:
                av_df[col] = trial_data[col].iloc[0]

        return av_df

    if callable(condition):
        groups = [condition(trial) for (i, trial) in trial_data.iterrows()]
    else:
        groups = condition

    # group by the condition and call trial_average without a condition on the sub-dataframes
    return (pd.DataFrame.from_dict({a : trial_average(b, None) for (a, b) in trial_data.groupby(groups)},
                                   orient="index")
                        .drop("trial_id", axis="columns"))


@utils.copy_td
def subtract_cross_condition_mean(trial_data, cond_idx=None, ref_field=None):
    """
    Find mean across all trials for each time point and subtract it from each trial.
    
    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    cond_idx : array-like
        indices of trials to use for mean
    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used

    Returns
    -------
    trial_data with mean subtracted
    """
    if cond_idx is None:
        cond_idx = trial_data.index

    time_fields = utils.get_time_varying_fields(trial_data, ref_field)
    for col in time_fields:
        assert len(set([arr.shape for arr in trial_data[col]])) == 1, f"Trials should have the same time coordinates in order to average."

    for col in time_fields:
        mean_act = np.mean(trial_data.loc[cond_idx, col], axis=0)
        trial_data[col] = [arr - mean_act for arr in trial_data[col]]
    return trial_data
        

@utils.copy_td
def select_trials(trial_data, query, reset_index=True):
    """
    Select trials based on some criteria

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    query : str, function, array-like
        if array-like, the dataframe is indexed with this
            can be either a list of indices or a mask
        if str, it should express a condition
            it is passed to trial_data.query
        if function/callable, it should take a trial as argument
            and return True for trials you want to keep
    reset_index : bool, optional, default True
        whether to reset the dataframe index to [0,1,2,...]
        or keep the original indices of the kept trials

    Returns
    -------
    trial_data with only the selected trials

    Examples
    --------
    succ_td = select_trials(td, "result == 'R'")

    succ_mask = (td.result == 'R')
    succ_td = select_trials(td, succ_mask)

    train_idx = np.arange(10)
    train_trials = select_trials(td, train_idx)

    right_trials = select_trials(td, lambda trial: np.cos(trial.target_direction) > np.finfo(float).eps)
    """
    if isinstance(query, str):
        trials_to_keep = trial_data.query(query).index
    elif callable(query):
        trials_to_keep = [query(trial) for (i, trial) in trial_data.iterrows()]
    else:
        trials_to_keep = query

    if reset_index:
        return trial_data.loc[trials_to_keep, :].reset_index(drop=True)
    else:
        return trial_data.loc[trials_to_keep, :]


def keep_common_trials(df_a, df_b, join_field='trial_id'):
    """
    Keep only trials with ID that are found in both data sets
    
    Parameters
    ----------s
    df_a : pd.DataFrame
        first data set in trial data format
    df_b : pd.DataFrame
        second data set in trial data format
    join_field : str, optional, default trial_id
        field based on which trials are matched to each other
        
    Returns
    -------
    (subset_a, subset_b) : tuple of dataframes
    """
    common_ids = np.intersect1d(df_a[join_field].values, df_b[join_field].values)
    subset_a = select_trials(df_a, lambda trial: trial[join_field] in common_ids)
    subset_b = select_trials(df_b, lambda trial: trial[join_field] in common_ids)
    
    return subset_a, subset_b

