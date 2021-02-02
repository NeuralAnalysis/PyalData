import math
import numpy as np
import pandas as pd

from scipy.stats import norm
import scipy.io
import scipy.signal as scs

from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis

import functools
import warnings


def mat2dataframe(path, shift_idx_fields, td_name=None):
    """
    Load a trial_data .mat file and turn it into a pandas DataFrame

    Parameters
    ----------
    path : str
        path to the .mat file to load
        "Can also pass open file-like object."
    td_name : str, optional
        name of the variable under which the data was saved
    shift_idx_fields : bool
        whether to shift the idx fields
        set to True if the data was exported from matlab
        using its 1-based indexig

    Returns
    -------
    df : pd.DataFrame
        pandas dataframe replicating the trial_data format
        each row is a trial
    """
    try:
        mat = scipy.io.loadmat(path, simplify_cells=True)
    except NotImplementedError:
        try:
            import mat73
        except ImportError:
            raise ImportError("Must have mat73 installed to load mat73 files.")
        else:
            mat = mat73.loadmat(path)

    real_keys = [k for k in mat.keys() if not (k.startswith("__") and k.endswith("__"))]

    if td_name is None:
        if len(real_keys) == 0:
            raise ValueError("Could not find dataset name. Please specify td_name.")
        elif len(real_keys) > 1:
            raise ValueError("More than one datasets found. Please specify td_name.")

        assert len(real_keys) == 1

        td_name = real_keys[0]

    df = pd.DataFrame(mat[td_name])

    if shift_idx_fields:
        df = backshift_idx_fields(df)

    return df 


def getSig(trial_data, trial, signals):
    '''
    Get a matrix containing the requested signal for all time points. 
    
    Input:
    trial_data: DataFrame object with the data
    trial: Index of the trial we are interested in
    signals: String or index of the column we want to select
    
    Output: 
    data: matrix with the value held in column of the specified trial
    
    '''
    data = trial_data.loc[trial,signals]
    
    return data 


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
            length: 5*std/bin_length
            mass normalized to 1
    """
    win = scs.gaussian(int(5*std/bin_length), std/bin_length)
    return win / np.sum(win)


def hw_to_std(hw):
    """
    Convert half-width to standard deviation for a Gaussian window.
    """
    return hw / (2 * np.sqrt(2 * np.log(2)))


def _smooth_1d(arr, win):
    """
    Smooth a 1D array by convolving it with a window

    Parameters
    ----------
    arr : 1D array-like
        time-series to smooth
    win : 1D array-like
        smoothing window to convolve with

    Returns
    -------
    1D np.array of the same length as arr
    """
    return scs.convolve(arr, win, mode = "same")


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

    if mat.ndim == 1:
        return _smooth_1d(mat, win)
    elif mat.ndim == 2:
        return np.column_stack([_smooth_1d(mat[:, i], win) for i in range(mat.shape[1])])
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

def dimReduce(data, params):
    """
    Function to compute the dimensionality reduction. For now handles PCA, PPCA and FA.
    
    Input:
    data: Data to be projected
    params: struct containing parameters
        params['algorithm'] : (string) which algorith, e.g. 'pca', 'fa', 'ppca'
        params['num_dims']: how many dimensions (for FA). Default is dimensionality of input
    
    Output:
    info_out: struct of information
    .w : wieght matrix for projections
    .scores: scores for the components
    .eigen : eigenvalues for PC ranking
    
    """
    alg=params['algorithm'].lower()
    
    if alg =='pca':
        pca = PCA(n_components=data.shape[1])
        pca.fit(data)
        w = np.transpose(pca.components_) # Transpose to have one column per pc
        scores = pca.fit_transform(data) 
        eigen = pca.explained_variance_ 
        mu = pca.mean_
        
        
    
    elif alg == 'fa':
        fa = FactorAnalyzer(n_factors=params['num_dims'],rotation=None, method='ml')
        fa.fit(data)
        w = fa.loadings_
        scores = fa.transform(data)
        eigen, f = fa.get_eigenvalues()
        mu = np.mean(data, axis=0)
        
    out_info = dict()
    out_info['w'] = w
    out_info['scores'] = scores
    out_info['eigen'] = eigen
    out_info['mu'] = mu
    
    return out_info


def all_integer(arr):
    """
    Check if all the values in arr are approximately integers
    """
    return np.all(np.isclose(arr, np.array(arr, dtype=int)))

  
@copy_td
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
        if not given, the first field that ends with "spikes" is used

    Returns
    -------
    time_fields : list of str
        list of fieldnames that store time-varying signals
    """
    if ref_field is None:
        # look for a spikes field
        ref_field = [col for col in trial_data.columns.values if col.endswith("spikes")][0]

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
    return slice(idx-before, idx+after+1)


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
    return slice_around_index(trial[point_name], before, after)


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

    return slice(trial[start_point_name]-before, trial[end_point_name]+after+1)


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
