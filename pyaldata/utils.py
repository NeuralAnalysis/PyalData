import math
import numpy as np
import pandas as pd

from scipy.stats import norm
import scipy.io
import scipy.signal as scs

from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis

def mat2dataframe(path):
    mat = scipy.io.loadmat(path, simplify_cells=True)
    df = pd.DataFrame(mat['trial_data'])

    return df 


def getSig(trial_data, trial, signals):
    '''
    Get a matrix containing the requested signal for all time points. 
    
    Input:
    trial_data: DataFrame object with the data
    trial: Index of the trial we are interested in
    signals: String or index of the column we want to select
    
    Output: 
    data: matrix with the value held in column of the specified trial
    
    '''
    data = trial_data.loc[trial,signals]
    
    return data 


def norm_gauss_window(bin_length, std):
    """
    Gaussian window with its mass normalized to 1

    Parameters
    ----------
    bin_length : float
        binning length of the array we want to smooth in ms
    std : float
        standard deviation of the window
        use hw_to_std to calculate std based from half-width

    Returns
    -------
    win : 1D np.array
        Gaussian kernel with
            length: 5*std/bin_length
            mass normalized to 1
    """
    win = scs.gaussian(int(5*std/bin_length), std/bin_length)
    return win / np.sum(win)


def hw_to_std(hw):
    """
    Convert half-width to standard deviation for a Gaussian window.
    """
    return hw / (2 * np.sqrt(2 * np.log(2)))


def _smooth_1d(arr, win):
    """
    Smooth a 1D array by convolving it with a window

    Parameters
    ----------
    arr : 1D array-like
        time-series to smooth
    win : 1D array-like
        smoothing window to convolve with

    Returns
    -------
    1D np.array of the same length as arr
    """
    return scs.convolve(arr, win, mode = "same")


def smooth_data(mat, dt=None, std=None, hw=None, win=None):
    """
    Smooth a 1D array or every column of a 2D array

    Parameters
    ----------
    mat : 1D or 2D np.array
        vector or matrix whose columns to smooth
        e.g. recorded spikes in a time x neuron array
    dt : float
        length of the timesteps in seconds
    std : float (optional)
        standard deviation of the smoothing window
    hw : float (optional)
        half-width of the smoothing window
    win : 1D array-like (optional)
        smoothing window to convolve with

    Returns
    -------
    np.array of the same size as mat
    """
    assert only_one_is_not_None((win, hw, std))

    if win is None:
        assert dt is not None, "specify dt if not supplying window"

        if std is None:
            std = hw_to_std(hw)

        win = norm_gauss_window(dt, std)

    if mat.ndim == 1:
        return _smooth_1d(mat, win)
    elif mat.ndim == 2:
        return np.column_stack([_smooth_1d(mat[:, i], win) for i in range(mat.shape[1])])
    else:
        raise ValueError("mat has to be a 1D or 2D array")


def only_one_is_not_None(args):
    return sum([arg is not None for arg in args]) == 1


def concatTrials(trial_data, signal, indx_list):
    data=trial_data.loc[indx_list[0],signal]
    for i in indx_list[1:]:
        data=np.concatenate((data,trial_data.loc[i,signal]))

    return data

def dimReduce(data, params):
    """
    Function to compute the dimensionality reduction. For now handles PCA, PPCA and FA.
    
    Input:
    data: Data to be projected
    params: struct containing parameters
        params['algorithm'] : (string) which algorith, e.g. 'pca', 'fa', 'ppca'
        params['num_dims']: how many dimensions (for FA). Default is dimensionality of input
    
    Output:
    info_out: struct of information
    .w : wieght matrix for projections
    .scores: scores for the components
    .eigen : eigenvalues for PC ranking
    
    """
    alg=params['algorithm'].lower()
    
    if alg =='pca':
        pca = PCA(n_components=data.shape[1])
        pca.fit(data)
        w = np.transpose(pca.components_) # Transpose to have one column per pc
        scores = pca.fit_transform(data) 
        eigen = pca.explained_variance_ 
        mu = pca.mean_
        
        
    
    elif alg == 'fa':
        fa = FactorAnalyzer(n_factors=params['num_dims'],rotation=None, method='ml')
        fa.fit(data)
        w = fa.loadings_
        scores = fa.transform(data)
        eigen, f = fa.get_eigenvalues()
        mu = np.mean(data, axis=0)
        
    out_info = dict()
    out_info['w'] = w
    out_info['scores'] = scores
    out_info['eigen'] = eigen
    out_info['mu'] = mu
    
    return out_info

