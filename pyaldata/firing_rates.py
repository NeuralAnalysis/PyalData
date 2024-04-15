import numpy as np

from . import smoothing
from . import utils
from . import extract_signals


@utils.copy_td
def add_firing_rates(trial_data, method, std=None, hw=None, win=None, backend="convolve1d"):
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
    backend: str, either 'convolve1d' or 'convolve'
        'convolve1d' (default) uses scipy.ndimage.convolve1d, which is faster in some cases
        'convolve'  uses scipy.signal.convolve, which may scale better for large arrays

    Returns
    -------
    td : pd.DataFrame
        trial_data with '_rates' fields added
    """
    spike_fields = [name for name in trial_data.columns.values if name.endswith("_spikes")]
    rate_suffix = "_rates"
    rate_fields = [
        utils.remove_suffix(name, "_spikes") + rate_suffix for name in spike_fields
    ]

    bin_size = trial_data["bin_size"].values[0]

    if method == "smooth":
        if win is None:
            if hw is None:
                if std is None:
                    std = 0.05
            else:
                assert std is None, "only give hw or std"

                std = smoothing.hw_to_std(hw)

            win = smoothing.norm_gauss_window(bin_size, std)

        def get_rate(spikes):
            return smoothing.smooth_data(spikes, win=win, backend=backend) / bin_size

    elif method == "bin":
        assert all(
            [x is None for x in [std, hw, win]]
        ), "If binning is used, then std, hw, and win have no effect, so don't provide them."

        def get_rate(spikes):
            return spikes / bin_size

    # calculate rates for every spike field
    for spike_field, rate_field in zip(spike_fields, rate_fields):
        trial_data[rate_field] = [get_rate(spikes) for spikes in trial_data[spike_field]]

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
    assert (
        len(set(trial_data.bin_size)) == 1
    ), "Function assumes that every trial has the same bin size."

    if signal.endswith("spikes"):
        if divide_by_bin_size is None:
            utils.warnings.warn(
                "Assuming spikes are actually spikes and dividing by bin size."
            )
            divide_by_bin_size = True
    elif signal.endswith("rates"):
        if divide_by_bin_size is None:
            utils.warnings.warn(
                "Assuming rates are already in Hz and don't have to divide by bin size."
            )
            divide_by_bin_size = False
    else:
        if divide_by_bin_size is None:
            raise ValueError(
                f"Please specify divide_by_bin_size. Could not determine it automatically."
            )

    if divide_by_bin_size:
        return (
            np.mean(extract_signals.concat_trials(trial_data, signal), axis=0)
            / trial_data["bin_size"].values[0]
        )
    else:
        return np.mean(extract_signals.concat_trials(trial_data, signal), axis=0)


@utils.copy_td
def remove_low_firing_neurons(
    trial_data, signal, threshold, divide_by_bin_size=None, verbose=False
):
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
