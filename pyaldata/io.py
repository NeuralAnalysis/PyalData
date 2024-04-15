from . import data_cleaning

import pandas as pd
import scipy.io


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

    df = data_cleaning.clean_0d_array_fields(df)
    df = data_cleaning.clean_integer_fields(df)

    if shift_idx_fields:
        df = data_cleaning.backshift_idx_fields(df)

    return df
