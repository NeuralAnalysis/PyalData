import numpy as np


def split_array(X, lengths):
    """
    Split an array into subarrays of given lengths along its first dimension.
    """
    assert np.sum(lengths) == X.shape[0]

    border_indices = np.insert(np.cumsum(lengths), 0, 0)
    return [
        X[start:end, ...] for (start, end) in zip(border_indices[:-1], border_indices[1:])
    ]
