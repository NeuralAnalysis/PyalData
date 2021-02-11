from . import utils


def fit_regressor_model(trial_data, model, input_signal, output_signal, train_indices=None):
    """
    Fit a regression model that predicts one signal from another

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    model : regressor model
        has to implement a .fit(X, y) method
    input_signal : str
        signal to predict from
    output_signal : str
        signal to predict
    train_indices : array-like of ints, optional
        indices of the trials to train on

    Returns
    -------
    fitted model
    """
    X = utils.concat_trials(trial_data, input_signal, train_indices)
    Y = utils.concat_trials(trial_data, output_signal, train_indices)

    model.fit(X, Y)

    return model


@utils.copy_td
def apply_regressor_model(trial_data, model, signal, out_fieldname):
    """
    Apply a fitted regression model to all trials

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
    trial_data[out_fieldname] = [model.predict(s) for s in trial_data[signal]]

    return trial_data


@utils.copy_td
def regress(trial_data, model, input_signal, output_signal, out_fieldname, train_indices=None, return_model=False):
    """
    Fit and apply a regression model to predict one signal from another

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    model : regressor model
        has to implement a .fit(X, y) method
    input_signal : str
        signal to predict from
    output_signal : str
        signal to predict
    out_fieldname : str
        name of the field in which to store the transformed values
    train_indices : array-like of ints, optional
        indices of the trials to train on
    return_model : bool, optional, default False
        return the fitted model along with the dataframe

    Returns
    -------
    trial_data with out_fieldname added
    """
    model = fit_regressor_model(trial_data, model, input_signal, output_signal, train_indices)
    trial_data = apply_regressor_model(trial_data, model, input_signal, out_fieldname)

    if return_model:
        return trial_data, model
    else:
        return trial_data
