import numpy as np
import pandas as pd
import arviz as az
from scipy.stats import spearmanr, pearsonr
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import click

# ========== Metric ===================

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
    corr = spearmanr(obs, pred)[0]
    return float(corr)

def pearson_correlation(obs, pred):
    '''pattern accuracy'''
    corr = pearsonr(obs, pred)[0]
    return float(corr)

def calc_nrmse(obs, pred, norm="mean"):
    '''Normalized Root Mean Square Error'''
    rmse = np.sqrt(((obs - pred)**2).mean(dim="time"))
    if norm == "range":
        return float(rmse / (obs.max(dim="time") - obs.min(dim="time")))
    elif norm == "mean":
        return float(rmse / obs.mean(dim="time"))
    elif norm == "std":
        return float(rmse / obs.std(dim="time"))
    else: 
        raise ValueError(f"Unknown normalization method: {norm}")


def calc_log_rmse(obs, pred):
    '''Root Mean Square Error of log-transformed data'''

    obs = np.log2(obs +1)
    pred = np.log2(pred +1)
    rmse = np.sqrt(((obs - pred)**2).mean(dim="time"))

    return rmse.item()

def calc_mase(obs, pred):
    '''Mean Absolute Scaled Error
        Scaled by mean observed expression'''
    mae_model = np.sum(np.abs(obs - pred))

    mean_obs = np.mean(obs)
    mae_naive = np.sum(np.abs(obs - mean_obs))
    mase = mae_model / mae_naive

    return float(mase)

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

def calc_AIC(idata):
    '''
    k = free parameters
    LL = max. Log-Likelihood (sum)
    '''
    k = len(idata.posterior.data_vars)
    LL = idata.log_likelihood.y.mean(dim=("chain", "draw")).sum().item()
    aic = - 2 * LL + 2*k
    return float(aic)


# grep -l ERROR logs/gof_worker_logs/*.log
# grep -c WARNING logs/gof_worker_logs/*.log