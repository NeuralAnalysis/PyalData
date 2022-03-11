import numpy as np
from . import utils

def trials_are_same_length(trial_data,ref_field=None):
    trial_lengths = [utils.get_trial_length(trial,ref_field=ref_field) for _,trial in trial_data.iterrows()]
    return len(set(trial_lengths)) == 1

def all_integer(arr):
    """
    Check if all the values in arr are approximately integers
    """
    return np.all(np.isclose(arr, np.array(arr, dtype=int)))
