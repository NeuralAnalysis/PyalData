import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from . import utils


@utils.copy_td
def smooth_signals(trial_data, signals, std=None, hw=None):
    """
    Smooth signal(s)
    
    Parameters
    ----------
    trial_data: pd.DataFrame
        trial data
    signals: list of strings
        signals to be smoothed
    std : float (optional)
        standard deviation of the smoothing window
        default 0.05 seconds
    hw : float (optional)
        half-width of the smoothing window
    
    Returns
    -------
    trial_data: DataFrame with the appropriate fields smoothed
    """
    assert utils.only_one_is_not_None((std, hw))

    bin_size = trial_data.iloc[0]['bin_size']

    if std is None:
        if hw is not None:
            std = utils.hw_to_std(hw)
        else:
            std = 0.05

    win = utils.norm_gauss_window(bin_size, std)

    if isinstance(signals, str):
        signals = [signals]

    for (i, trial) in trial_data.iterrows():
        for sig in signals:
            trial_data.at[i, sig] = utils.smooth_data(trial[sig], win=win)

    return trial_data


@utils.copy_td
def add_firing_rates(trial_data, method, std=None, hw=None, win=None):
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

    Returns
    -------
    td : pd.DataFrame
        trial_data with '_rates' fields added
    """
    spike_fields = [name for name in trial_data.columns.values if name.endswith("_spikes")]
    rate_suffix = "_rates"
    rate_fields = [utils.remove_suffix(name, "_spikes") + rate_suffix for name in spike_fields]

    bin_size = trial_data.iloc[0]['bin_size']

    if method == "smooth":
        # NOTE creating the smoothing window here might be faster
        def get_rate(spikes):
            return utils.smooth_data(spikes, bin_size, std, hw, win) / bin_size

    elif method == "bin":
        assert all([x is None for x in [std, hw, win]]), "If binning is used, then std, hw, and win have no effect, so don't provide them."

        def get_rate(spikes):
            return spikes / bin_size

    # calculate rates for every spike field
    for (spike_field, rate_field) in zip(spike_fields, rate_fields):
        trial_data[rate_field] = [get_rate(spikes) for spikes in trial_data[spike_field]]

    return trial_data


def getTDidx(trial_data, col, v):
    '''
    Return tral_data with only the rows where selected column col holds the specific value v
    
    Input:
    trial_data: DataFrame object with data
    col: column index or string 
    v: value of interest to select data
    
    Output:
    trial_data: DataFrame object with data
    
    '''
    
    return  trial_data.loc[trial_data[col] == v]

def truncTD(trial_data,idx_start, idx_end,signals):
    '''
    Function that will truncate the dataframe based on the values of columns idx_start and idx_end
    
    Input:
    trial_data: DataFrames structure
    idx_start: column index with values of start time
    idx_end: column index with value of start time
    signals: list of signals we want to truncate
    
    Output:
    trial_data: DataFrames structure with truncated data
    
    '''
    trial_data_exit=trial_data.copy()
    for trial in trial_data.index:
        for iSig in signals:
            
            data=np.array(trial_data.loc[trial,iSig])
            idx_s=trial_data.loc[trial,idx_start['target']]+idx_start['shift']
            
            idx_e=trial_data.loc[trial,idx_end['target']]+idx_end['shift']
            
            truncate=data[np.int(idx_s)-1:np.int(idx_e),:]
            trial_data_exit.at[trial,iSig]=truncate
    return trial_data_exit

def projLow (trial_data, params, out_info):
    """
    Function to project data to low dimensional space and store scores in trial data
    
    Input:
    trial_data: DataFrames structure with data
    params: struct containing parameters
        params['algorithm'] : (string) which algorith, e.g. 'pca', 'fa', 'ppca'
        params['signals']: (list) which signals
        params['num_dims']: how many dimensions (for FA). Default is dimensionality of input
    out_info: structure of information obtained from dimReduce.
        out_info['w']: weight matrix for projections
        out_info['scores']: scores for the components
        out_info['eigen']: eigenvalues for PC ranking
    
    Output:
    trial_data: DataFrames structure with data and additional field with 
    
    TODO:
    
    - SUBTRACT MEAN FROM DATA
    
    """
    signals=params['signals']
    for iSig in signals:
        series= trial_data[iSig].copy()
        for trial in trial_data.index:
            data = trial_data.loc[trial,'M1_spikes']
            latent = data.dot(out_info['w'])
            series.at[trial]=latent
        trial_data[iSig+'_'+params['algorithm']]=series
        
    return trial_data

def binTD (trial_data, num_bins, isSpikes):

    fn_spikes = [col for col in trial_data.columns if 'spikes' in col]
    fn_time = ['vel', 'pos','acc']
    fn_idx =  [col for col in trial_data.columns if 'idx' in col]

    do_avg = False
    for trial in trial_data.index:
        t = range(0,np.shape(trial_data.loc[trial,fn_time[0]])[0])
        if do_avg:
            t_bin = [1 , t[-1]+1]
            num_bins = t[-1]
        else:
            t_bin = range(0,t[-1],num_bins)
        
        # update entry to new bin size
        trial_data.at[trial,'bin_size'] = num_bins * trial_data.loc[trial, 'bin_size']

        # process spike fields
        # for now I am going to assume that we already did the smoothing, so we will just do the average
        if isSpikes:
            for iArray in range(len(fn_spikes)):
                temp = np.array(trial_data.loc[trial, fn_spikes[iArray]])
                # fr is size bins * neurons 
                fr = np.zeros((len(t_bin)-1,np.shape(temp)[1]))
                for iBin in range(len(t_bin)-1):
                    fr[iBin,:] = np.sum(temp[t_bin[iBin]:t_bin[iBin+1],:],axis=0)

                trial_data.at[trial, fn_spikes[iArray]] = fr

        else:
            for iArray in range(len(fn_spikes)):
                temp = np.array(trial_data.loc[trial, fn_spikes[iArray]])
                # fr is size bins * neurons 
                fr = np.zeros((len(t_bin)-1,np.shape(temp)[1]))
                for iBin in range(len(t_bin)-1):
                    fr[iBin,:] = np.mean(temp[t_bin[iBin]:t_bin[iBin+1],:])

                trial_data.at[trial, fn_spikes[iArray]] = fr

        # process other time fields
        for iSig in range(len(fn_time)):
            temp = np.array(trial_data.loc[trial, fn_time[iSig]])
            
            kin = np.zeros((len(t_bin)-1,np.shape(temp)[1]))
            for iBin in range(len(t_bin)-1):
                kin[iBin,:] = np.mean(temp[t_bin[iBin]:t_bin[iBin+1],:],axis=0)
            
            trial_data.at[trial, fn_time[iSig]] = kin

        # process index fields
        for iIdx in range(len(fn_idx)):
            temp = trial_data.loc[trial,fn_idx[iIdx]]
            
            if temp > len(t):
                temp = len(t)
            
            if temp <= 0:
                temp = np.nan
            
            if math.isnan(temp):
                temp = np.nan

            if not math.isnan(temp):
                temp = np.int(temp)
                temp_adjust = 0
                temp=t[temp]
                matches = list(i for i,e in enumerate(t_bin) if e <= temp)

                temp_adjust = matches[-1]
            
            trial_data.at[trial, fn_idx[iIdx]] = temp_adjust
    
    return trial_data

