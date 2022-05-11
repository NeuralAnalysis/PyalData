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
