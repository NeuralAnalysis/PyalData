import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from . import utils


def SmoothSignal(trial_data, signals):
    '''
    Smooth signal
    
    Input: 
    trial_data: DataFrame object with the data
    signals: list of strings or index of the signals to be smoothed
    
    Output:
    trial_data: DataFrame objecti with the signals fields smoothed
    '''
    trial_data_exit = trial_data.copy()
    bin_size = trial_data.loc[trial_data.index[0]]['bin_size']
    width = 0.05
    
    for trial in trial_data.index:
        for sig in signals:
            
            data = smooth_data(getSig(trial_data, trial, sig),bin_size,width)
            trial_data_exit.at[trial,sig]=np.array(data)
    return trial_data_exit

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

