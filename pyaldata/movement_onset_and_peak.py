import warnings
import numpy as np

from scipy.signal import find_peaks

from . import utils
from . import tools


def get_onset_idx(s, min_ds=1.9, s_thresh=10, peak_divisor=2, method="peaks", debug=False):
    """
    Get the index at the beginning of the uprise of the signal

    Parameters
    ----------
    s : 1D array
        signal
        mostly used for calculating movement onset
    min_ds : float, default 1.9
        minimum acceleration
    s_thresh : float, default 10
        if using thresholding, returns the first index where the signal reaches this threshold
    peak_divisor : float, default 2
        what fraction of the peak acceleration to reach before peak acceleration 
    method : string, default 'peaks'
        possible values: "peaks", "Matt", "threshold"
            peaks: find peaks of the acceleration using scipy.signal.find_peaks
            Matt: find peaks using the second derivative like in TrialData
            threshold: simple thresholding
    debug : bool, default False
        print if falling back to thresholding

    Returns
    -------
    on_idx : int or np.nan
        index of the onset or NaN if none is found
    """
    assert method.lower() in ['peaks', 'scipy', 'matt', 'threshold'], "method has to be one of 'peaks', 'scipy', 'matt', 'threshold']"

    # I'm not sure what this is
    abs_acc_thresh = np.nan

    # initialize to nan
    on_idx = np.nan

    if method.lower() in ['peaks', 'matt']:
        # find peaks of the acceleration
        ds = np.insert(np.diff(s), 0, 0)

        if method.lower() == 'peaks':
            peaks = find_peaks(ds)[0]
        else:
            dds = np.insert(np.diff(ds), 0, 0)
            peaks = np.append((dds[:-1]>0) & (dds[1:]<0), False)
            peaks = np.nonzero(peaks)[0]
        
        # keep only those above a threshold
        mvt_peak = peaks[ds[peaks] > min_ds]
        
        # if there are peaks
        if not len(mvt_peak) == 0:
            # take the first one
            mvt_peak = mvt_peak[0]
            if np.isnan(abs_acc_thresh):
                # Threshold is max of acceleration peak divided by divisor
                thresh = ds[mvt_peak] / peak_divisor
            else:
                thresh = abs_acc_thresh
                
            # initiation is the last time point where ds is below threshold before the peak
            on_idx = [i for i in range(mvt_peak) if ds[i] < thresh][-1]

    # if thresholding is chosen or peak finding didn't work, do thresholding
    if np.isnan(on_idx):
        if debug:
            print('using thresholding')

        if len(np.nonzero((s > s_thresh))[0]) != 0:
            on_idx = np.nonzero((s > s_thresh))[0][0]

        if np.isnan(on_idx): # usually means it never crosses threshold
            warnings.warn("Could not identify movement onset")

    return on_idx


def get_movement_onset(trial, start="idx_go_cue", min_ds=1.9, s_thresh=10, peak_divisor=2, method="peaks", debug=False):
    """
    Get index of movement onset in the trial

    Parameters
    ----------
    trial : pd.Series
        trial to get the movement onset in
        has to have a vel_norm field
    start : int or string, default "idx_go_cue"
        if integer: index after which to consider the velocities
        if string: field containing the index after which to consider the velocities
    min_ds, s_thresh, peak_divisor, debug
        see get_onset_idx function
    s_thresh : float, default 10
        if using thresholding, returns the first index where the signal reaches this threshold
    peak_divisor : float, default 2
        what fraction of the peak acceleration to reach before peak acceleration 
    method : string, default 'peaks'
        possible values: "peaks", "Matt", "threshold"
            peaks: find peaks of the acceleration using scipy.signal.find_peaks
            Matt: find peaks using the second derivative like in TrialData
            threshold: simple thresholding
    debug : bool, default False
        print if falling back to thresholding

    Returns
    -------
    on_idx : int or np.nan
        index of the onset or NaN if none is found
    """
    if isinstance(start, str):
        start_idx = int(trial[start])
    else:
        start_idx = int(start)

    return start_idx + get_onset_idx(trial.vel_norm[start_idx:], min_ds, s_thresh, peak_divisor, method, debug)


@utils.copy_td
def add_movement_onset(trial_data, **kwargs):
    """
    Get index of movement onset in every trial

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
        has to have a vel_norm field
    kwargs
        for a list of keyword arguments, see PyalData.movement_onset.get_movement_onset

    Returns
    -------
    copy of trial_data with "idx_movement_on" field added
    """
    if 'vel' not in trial_data.columns:
        trial_data = tools.add_gradient(trial_data, 'pos', 'vel', True)
    if 'vel_norm' not in trial_data.columns:
        trial_data = tools.add_norm(trial_data, 'vel')

    trial_data["idx_movement_on"] = trial_data.apply(lambda trial: get_movement_onset(trial, **kwargs),
                                                     axis=1)

    return trial_data


def get_peak_speed_idx(trial, start="idx_go_cue"):
    """
    Get the index of peak velocity in the trial

    Parameters
    ----------
    trial : pd.Series
        trial to get the movement onset in
        has to have a vel_norm field
    start : int or string, default "idx_go_cue"
        if integer: index after which to consider the velocities
        if string: field containing the index after which to consider the velocities

    Returns
    -------
    idx : int
        index of maximum velocity
    """
    if isinstance(start, str):
        start_idx = int(trial[start])
    else:
        start_idx = int(start)

    return start_idx + np.argmax(trial.vel_norm[start_idx:])


@utils.copy_td
def add_peak_speed_idx(trial_data, start="idx_go_cue"):
    """
    Get the index of peak velocity in every trial and save it
    in the field "idx_peak_speed"

    Parameters
    ----------
    trial : pd.Series
        trial to get the movement onset in
        has to have a vel_norm field
    start : int or string, default "idx_go_cue"
        if integer: index after which to consider the velocities
        if string: field containing the index after which to consider the velocities

    Returns
    -------
    copy of trial_data with "idx_peak_speed" added
    """
    trial_data["idx_peak_speed"] = trial_data.apply(lambda trial: get_peak_speed_idx(trial),
                                                     axis=1)

    return trial_data


def get_peak_speed(trial, start="idx_go_cue"):
    """
    Get peak speed in the trial during movement.

    Parameters
    ----------
    trial : pd.Series
        trial to get the movement onset in
        has to have a vel_norm field
    start : int or string, default "idx_go_cue"
        if integer: index after which to consider the velocities
        if string: field containing the index after which to consider the velocities

    Returns
    -------
    maximum of the velocity's norm during movement
    """
    if isinstance(start, str):
        start_idx = int(trial[start])
    else:
        start_idx = int(start)

    return np.max(trial.vel_norm[start_idx:])


@utils.copy_td
def add_peak_speed(trial_data, start="idx_go_cue"):
    """
    Get the peak velocity in every trial and save it
    in the field "peak_speed"

    Parameters
    ----------
    trial : pd.Series
        trial to get the movement onset in
        has to have a vel_norm field
    start : int or string, default "idx_go_cue"
        if integer: index after which to consider the velocities
        if string: field containing the index after which to consider the velocities

    Returns
    -------
    copy of trial_data with "peak_speed" added
    """
    trial_data["peak_speed"] = trial_data.apply(lambda trial: get_peak_speed(trial),
                                                axis=1)

    return trial_data
