import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.io
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


def smooth_data(data, dt, kernel_sd):
    '''
    Convolves spikes counts or firing rates with a Gaussian Kernel to smooth the responses. 
    
    Input: 
    data: array of data( rows: time, columns: signals)
    dt: size of the time steps in data in seconds 
    kernel_sd: gaussian kernel standard deviation
    '''
    
    data=np.asarray(data)
    # Get number of channels and number of samples
    (nbr_samples, nbr_chs) = np.shape(data)
    #Preallocate return matrix
    data_smooth = np.zeros((nbr_samples, nbr_chs))
    # kernel half length is 3 times the SD 
    kernel_hl = math.ceil(3*kernel_sd/(dt))
    # create the kernel -- it will have length 2*kernel_hl+1
    y = [x * dt for x in list(range(-kernel_hl, kernel_hl+1,1))]
    kernel = norm.pdf(y, 0, kernel_sd)
    # compute normalization factor
    nm = np.transpose(np.convolve(kernel, np.ones( nbr_samples)))
    
    
    for i in range(nbr_chs):
        aux_smoothed_FR = np.convolve(np.transpose(kernel), data[:,i])/nm
        data_smooth[:,i] = aux_smoothed_FR[kernel_hl:-(kernel_hl)]
    
    return data_smooth

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

