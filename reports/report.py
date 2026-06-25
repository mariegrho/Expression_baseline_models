import numpy as np
import pandas as pd
from sklearn import metrics

def _local_variance(residuals: np.ndarray):
    """This uses aggregated data residuals[id,time].mean(id).
    This uses the average variance of all 3-value pairs of direct neighbors.
    
    """
    local_variance = [np.var(r) for r in zip(
        np.roll(residuals, shift=1)[1:-1],
        np.roll(residuals, shift=0)[1:-1],
        np.roll(residuals, shift=-1)[1:-1]
        )]
    global_variance = np.var(residuals)

    lgv = np.mean(local_variance) / global_variance
    return float(lgv)

def _autocorrelation(residuals, lag=1):
    """This uses unaggregated data residuals[id,time]"""
    return pd.Series(residuals).autocorr(lag=lag)

def spearman_correlation(obs, pred):
    '''pattern accuracy'''
    from scipy.stats import spearmanr
    corr = spearmanr(obs, pred)[0]
    return float(corr)

def calc_nrmse(obs, pred):
    '''Normalized Root Mean Square Error'''
    rmse = np.sqrt(np.mean((obs - pred)**2))

    # by range
    nrmse_range = rmse / (obs.max() - obs.min())
    # by mean
    nrmse_mean = rmse / obs.mean()
    # by std
    nrmse_std = rmse / obs.std()

    return [nrmse_range, nrmse_mean, nrmse_std]

def calc_mase(obs, pred):
    '''Mean Absolute Scaled Error
        Scaled by mean observed expression'''
    mae_model = np.sum(np.abs(obs - pred))

    mean_obs = np.mean(obs)
    mae_naive = np.sum(np.abs(obs - mean_obs))
    mase = mae_model / mae_naive

    return mase

def calc_bic(idata):
    '''
    k = free parameters
    n = data points
    LL = max. Log-Likelihood (sum)
    '''

    n = len(idata.posterior_model_fits.time)
    k = len(idata.posterior.data_vars)
    LL = idata.log_likelihood.y.mean(dim=("chain", "draw")).sum().item()
    bic =  k * np.log(n) - 2 * LL
    return float(bic)

