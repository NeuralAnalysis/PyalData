import numpy as np
import pandas as pd

from typing import Optional

from . import utils


def signal_dimensionality(trial_data: pd.DataFrame, signal: str) -> int:
    """
    Determine signal dimensionality
    
    Parameters
    ----------
    trial_data : pd.DataFrame
        data in TrialData format
    signal : str
        signal whose dimensionality we want to know
    
    Returns
    -------
    d : int
        dimensionality of the signal
    """
    first_element = trial_data[signal].values[0]
    
    if first_element.ndim == 1:
        d = 1
        
        for arr in trial_data[signal].values:
            assert arr.ndim == 1
    else:
        d = first_element.shape[1]
        
        for arr in trial_data[signal].values:
            assert arr.shape[1] == d
        
    return d


@utils.copy_td
def add_gradient(trial_data: pd.DataFrame, signal: str, outfield: Optional[str] = None, normalize: bool = False) -> pd.DataFrame:
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
def add_norm(trial_data: pd.DataFrame, signal: str) -> pd.DataFrame:
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

    if signal_dimensionality(trial_data, signal) == 1:
        # 1D signal's norm is its absolute value
        trial_data[norm_field_name] = [np.abs(s) for s in trial_data[signal]]
    else:
        trial_data[norm_field_name] = [np.linalg.norm(s, axis=1) for s in trial_data[signal]]

    
    return trial_data


