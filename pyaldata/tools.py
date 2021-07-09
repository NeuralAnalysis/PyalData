import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis

from . import utils

import warnings
warnings.simplefilter("always", UserWarning)

@utils.copy_td
def smooth_signals(trial_data, signals, std=None, hw=None):
    """
    Smooth signal(s)
    
    Parameters
    ----------
    trial_data: pd.DataFrame
        trial data
    signals: list of strings
        signals to be smoothed
    std : float (optional)
        standard deviation of the smoothing window
        default 0.05 seconds
    hw : float (optional)
        half-width of the smoothing window
    
    Returns
    -------
    trial_data: DataFrame with the appropriate fields smoothed
    """
    bin_size = trial_data.iloc[0]['bin_size']

    if hw is None:
        if std is None:
            std = 0.05
    else:
        assert (std is None), "only give hw or std"

        std = utils.hw_to_std(hw)

    win = utils.norm_gauss_window(bin_size, std)

    if isinstance(signals, str):
        signals = [signals]

    for (i, trial) in trial_data.iterrows():
        for sig in signals:
            trial_data.at[i, sig] = utils.smooth_data(trial[sig], win=win)

    return trial_data


@utils.copy_td
def add_firing_rates(trial_data, method, std=None, hw=None, win=None):
    """
    Add firing rate fields calculated from spikes fields

    Parameters
    ----------
    trial_data : pd.DataFrame
        trial_data dataframe
    method : str
        'bin' or 'smooth'
    std : float (optional)
        standard deviation of the Gaussian window to smooth with
        default 0.05 seconds
    hw : float (optional)
        half-width of the of the Gaussian window to smooth with
    win : 1D array
        smoothing window

    Returns
    -------
    td : pd.DataFrame
        trial_data with '_rates' fields added
    """
    spike_fields = [name for name in trial_data.columns.values if name.endswith("_spikes")]
    rate_suffix = "_rates"
    rate_fields = [utils.remove_suffix(name, "_spikes") + rate_suffix for name in spike_fields]

    bin_size = trial_data.iloc[0]['bin_size']

    if method == "smooth":
        if win is None:
            if hw is None:
                if std is None:
                    std = 0.05
            else:
                assert (std is None), "only give hw or std"

                std = utils.hw_to_std(hw)

            win = utils.norm_gauss_window(bin_size, std)

        def get_rate(spikes):
            return utils.smooth_data(spikes, win=win) / bin_size

    elif method == "bin":
        assert all([x is None for x in [std, hw, win]]), "If binning is used, then std, hw, and win have no effect, so don't provide them."

        def get_rate(spikes):
            return spikes / bin_size

    # calculate rates for every spike field
    for (spike_field, rate_field) in zip(spike_fields, rate_fields):
        trial_data[rate_field] = [get_rate(spikes) for spikes in trial_data[spike_field]]

    return trial_data


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
        trial_data[outfield] = trial_data[outfield] / trial_data.bin_size

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
        whole_signal = utils.concat_trials(trial_data, signal, train_trials)
        col_mean = np.mean(whole_signal, axis=0)

        trial_data[signal] = [s - col_mean for s in trial_data[signal]]

    return trial_data


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
        whole_signal = utils.concat_trials(trial_data, signal, train_trials)
        col_mean = np.mean(whole_signal, axis=0)
        col_std = np.std(whole_signal, axis=0)

        col_std[col_std==0] = 1  # to avoid dividing by zero

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
                raise ValueError("signal cannot contain negative values when square-root transforming")

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
        whole_signal = utils.concat_trials(trial_data, signal, train_trials)
        col_min = np.min(whole_signal, axis=0)
        col_range = utils.get_range(whole_signal, axis=0)

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
        whole_signal = utils.concat_trials(trial_data, signal, train_trials)
        col_mean = np.mean(whole_signal, axis=0)
        col_range = utils.get_range(whole_signal, axis=0)

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
        whole_signal = utils.concat_trials(trial_data, signal, train_trials)

        norm_factor = utils.get_range(whole_signal) + alpha

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
    method_dict = {"center" : center_signal,
                   "center_normalize" : center_normalize_signal,
                   "zero_normalize" : zero_normalize_signal,
                   "sqrt_transform" : sqrt_transform_signal,
                   "sqrt" : sqrt_transform_signal,
                   "z_score" : z_score_signal,
                   "z-score" : z_score_signal,
                   "zero_normalize" : zero_normalize_signal,
                   "soft_normalize" : soft_normalize_signal}

    if isinstance(transformations, str):
        transformations = [transformations]

    if isinstance(signals, str):
        signals = [signals]

    for signal in signals:
        for trans in transformations:
            trial_data = method_dict[trans](trial_data, signal, train_trials, **kwargs)

    return trial_data


@utils.copy_td
def restrict_to_interval(trial_data, start_point_name=None, end_point_name=None, rel_start=0, rel_end=0, before=None, after=None, epoch_fun=None, warn_per_trial=False, reset_index=True, ref_field=None):
    """
    Restrict time-varying fields to an interval around a time point or between two time points

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

    assert (start_point_name is None) ^ (epoch_fun is None), "Give either start_point_name or epoch_fun."

    idx_fields = [col for col in trial_data.columns.values if col.startswith("idx")]
    time_fields = utils.get_time_varying_fields(trial_data, ref_field)

    # generate epoch_fun if the interval is given with time points
    if start_point_name is not None:
        epoch_fun = utils.generate_epoch_fun(start_point_name, end_point_name, rel_start, rel_end)

    # check in which trials the indexing works properly
    kept_trials_mask = np.array([utils._slice_in_trial(trial, epoch_fun(trial), warn_per_trial)
                                 for (i, trial) in trial_data.iterrows()]).astype(bool)
    # warn about dropping the problematic trials
    if np.any(~kept_trials_mask):
        warnings.warn(f"""Dropping the trials with the following IDs because of invalid time indexing. For more information, try warn_per_trial=True

        {trial_data.trial_id.values[~kept_trials_mask]}""", stacklevel=3)

    # only keep trials in which the indexing works properly
    trial_data = select_trials(trial_data, kept_trials_mask, reset_index)

    # cut time varying signals
    for col in time_fields:
        trial_data[col] = utils.extract_interval_from_signal(trial_data, col, epoch_fun)

    # adjust idx fields
    def _adjust_field(val, new_T):
        if isinstance(val, (np.ndarray, list)):
            return np.array([np.nan
                             if (idx < 0 or idx > new_T)
                             else idx
                             for idx in val])
        elif ((val < 0) or (val > new_T)):
            return np.nan
        else:
            return val

    new_time_lengths = [arr.shape[0] for arr in trial_data[time_fields[0]]]
    zero_points = [epoch_fun(trial).start for (i, trial) in trial_data.iterrows()]

    for col in idx_fields:
        trial_data[col] = [idx - zero_point for (idx, zero_point) in zip(trial_data[col], zero_points)]

        # set indices that are now invalid (i.e. not in the restricted interval) to nan
        trial_data[col] = [_adjust_field(idx, new_T)
                           for (idx, new_T) in zip(trial_data[col], new_time_lengths)]

    return trial_data


def fit_dim_reduce_model(trial_data, model, signal, train_trials=None, fit_kwargs=None):
     """
     Fit a dimensionality reduction model to train_trials

     Parameters
     ----------
     trial_data : pd.DataFrame
         data in trial_data format
     model : dimensionality reduction model
         model to fit
         has to implement a .fit (and .transform) method
     signal : str
         signal to fit to
     train_trials : list of ints (optional)
         trials to fit the model to
     fit_kwargs : dict (optional)
         parameters to pass to model.fit

     Returns
     -------
     fitted model

     Example
     -------
         from sklearn.decomposition import PCA
         pca_dims = -5
         pca = fit_dim_reduce_model(trial_data, PCA(pca_dims), 'M1_rates')
     """
     if fit_kwargs is None:
         fit_kwargs = {}

     model.fit(utils.concat_trials(trial_data, signal, train_trials),
               **fit_kwargs)

     return model


@utils.copy_td
def apply_dim_reduce_model(trial_data, model, signal, out_fieldname):
   """
   Apply a fitted dimensionality reduction model to all trials

   Parameters
   ----------
   trial_data : pd.DataFrame
       data in trial_data format
   model : dimensionality reduction model
       fitted model
       has to implement a .transform method
   signal : str
       signal to apply to
   out_fieldname : str
       name of the field in which to store the transformed values

   Returns
   -------
   trial_data with out_fieldname added
   """
   trial_data[out_fieldname] = [model.transform(s) for s in trial_data[signal]]

   return trial_data


@utils.copy_td
def dim_reduce(trial_data, model, signal, out_fieldname, train_trials=None, fit_kwargs=None, return_model=False):
   """
   Fit dimensionality reduction model and apply it to all trials

   Parameters
   ----------
   trial_data : pd.DataFrame
       data in trial_data format
   model : dimensionality reduction model
       model to fit
       has to implement a .fit and .transform method
   signal : str
       signal to fit and transform
   out_fieldname : str
       name of the field in which to store the transformed values
   train_trials : list of ints (optional)
       trials to fit the model to
   fit_kwargs : dict (optional)
       parameters to pass to model.fit
   return_model : bool (optional, default False)
       return the fitted model along with the data

   Returns
   -------
   if return_model is False
       trial_data with the projections added in out_fieldname
   if return_model is True
       (trial_data, model)
   """
   model = fit_dim_reduce_model(trial_data, model, signal, train_trials, fit_kwargs)

   if return_model:
       return (apply_dim_reduce_model(trial_data, model, signal, out_fieldname), model)
   else:
       return apply_dim_reduce_model(trial_data, model, signal, out_fieldname)


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
        if (f not in trial_data): 
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
    #Check if all fields exist
    for f in fields.keys():
        if (f not in trial_data): 
            raise ValueError(f"{f} field does not exist in trial data")
            
    for f in fields.keys():
        trial_data[fields[f]] = trial_data[f]
    
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
        

def get_average_firing_rates(trial_data, signal, divide_by_bin_size=None):
    """
    Calculate average firing rates of neurons across all trials

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signal : str
        signal from which to calculate the average firing rates
        ideally spikes or rates
    divide_by_bin_size : bool, optional
        whether to divide by the bin size when calculating the firing rates

    Returns
    -------
    np.array with the average firing rates
    shape (N, ) where N is the number of neurons in signal
    """
    assert len(set(trial_data.bin_size)) == 1, "Function assumes that every trial has the same bin size."

    if signal.endswith("spikes"):
        if divide_by_bin_size is None:
            utils.warnings.warn("Assuming spikes are actually spikes and dividing by bin size.")
            divide_by_bin_size = True
    elif signal.endswith("rates"):
        if divide_by_bin_size is None:
            utils.warnings.warn("Assuming rates are already in Hz and don't have to divide by bin size.")
            divide_by_bin_size = False
    else:
        if divide_by_bin_size is None:
            raise ValueError(f"Please specify divide_by_bin_size. Could not determine it automatically.")

    if divide_by_bin_size:
        return np.mean(utils.concat_trials(trial_data, signal), axis=0) / trial_data.bin_size[0]
    else:
        return np.mean(utils.concat_trials(trial_data, signal), axis=0)


@utils.copy_td
def remove_low_firing_neurons(trial_data, signal, threshold, divide_by_bin_size=None, verbose=False):
    """
    Remove neurons from signal whose average firing rate
    across all trials is lower than a threshold


    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signal : str
        signal from which to calculate the average firing rates
        ideally spikes or rates
    threshold : float
        threshold in Hz
    divide_by_bin_size : bool, optional
        whether to divide by the bin size when calculating the firing rates
    verbose : bool, optional, default False
        print a message about how many neurons were removed

    Returns
    -------
    trial_data with the low-firing neurons removed from the
    signal and the corresponding unit_guide
    """
    av_rates = get_average_firing_rates(trial_data, signal, divide_by_bin_size)
    mask = av_rates >= threshold

    trial_data[signal] = [arr[:, mask] for arr in trial_data[signal]]

    if signal.endswith("_spikes"):
        suffix = "_spikes"
    elif signal.endswith("_rates"):
        suffix = "_rates"
    else:
        utils.warnings.warn("Could not determine which unit_guide to modify.")

    area_name = utils.remove_suffix(signal, suffix)
    unit_guide = area_name + "_unit_guide"

    if unit_guide in trial_data.columns:
        trial_data[unit_guide] = [arr[mask, :] for arr in trial_data[unit_guide]]

    if verbose:
        print(f"Removed {np.sum(~mask)} neurons from {signal}.")

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
