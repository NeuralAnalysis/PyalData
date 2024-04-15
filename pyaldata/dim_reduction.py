from . import utils
from . import extract_signals


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
        pca_dims = 5
        pca = fit_dim_reduce_model(trial_data, PCA(pca_dims), 'M1_rates')
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    model.fit(extract_signals.concat_trials(trial_data, signal, train_trials), **fit_kwargs)

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
def dim_reduce(
    trial_data,
    model,
    signal,
    out_fieldname,
    train_trials=None,
    fit_kwargs=None,
    return_model=False,
):
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
