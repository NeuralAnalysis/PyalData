import numpy as np
from . import utils


def trials_are_same_length(trial_data, ref_field=None):
    """
    Check if all trials of a dataset have the same length

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used
    """
    trial_lengths = [
        utils.get_trial_length(trial, ref_field=ref_field)
        for (_, trial) in trial_data.iterrows()
    ]
    return len(set(trial_lengths)) == 1


def all_integer(arr):
    """
    Check if all the values in arr are approximately integers
    """
    return np.all(np.isclose(arr, arr.astype(int)))
