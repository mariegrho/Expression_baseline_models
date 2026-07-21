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

def spearman_correlation(idata):
    '''pattern accuracy'''
    from scipy.stats import spearmanr
    obs = idata.observed_data.y
    pred = idata.posterior_model_fits.y.mean(dim=("chain", "draw"))
    corr = spearmanr(obs, pred)[0]
    return float(corr)

def calc_nrmse(y_true, y_pred):
    '''Normalized Root Mean Square Error'''
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))

    # by range
    nrmse_range = rmse / (y_true.max() - y_true.min())
    # by mean
    nrmse_mean = rmse / y_true.mean()
    # by std
    nrmse_std = rmse / y_true.std()

    return [nrmse_range, nrmse_mean, nrmse_std]

def calc_mase(obs, pred):
    '''Mean Absolute Scaled Error
        Scaled by mean observed expression'''
    mae_model = np.sum(np.abs(obs - pred))

    mean_obs = np.mean(obs)
    mae_naive = np.sum(np.abs(obs - mean_obs))
    mase = mae_model / mae_naive

    return mase

def LOO(idata):
    '''Leave One Out'''
    obs = idata.observed_data.y
    loglik = idata.log_likelihood.mean(dim=("chain", "draw"))

    loo = []
    return loo

def calc_BIC(idata):
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


def WAIC(idata):
    '''Widely Applicable Information Criterion'''
    pp = idata.posterior_model_fits.y.mean(dim=("chain", "draw"))
    loglik = idata.log_likelihood.y.mean(dim=("chain", "draw"))
    waic = -2 * (loglik - sum(np.var(loglik)))
    return waic


def fpe(idata, residual_var="posterior_residuals", var_name="y"):
    "Final Prediction Error --> smaller FPE = better model "
    "n: number of observations"
    "p: number of free parameters"
    "sigma2_hat  = residual variance estimate"
    ""

    resid_group = getattr(idata, residual_var)
    resid = resid_group[var_name]

    resid_mean = resid.mean(dim=("chain", "draw", "source"))  
    resid_vals = resid_mean[~np.isnan(resid_mean)].values

    n = resid_vals.size
    p = len(idata.posterior.data_vars)

    sigma2_hat = np.sum(resid_vals**2) / n
    fpe = sigma2_hat  * (n+p) / (n-p)

    return fpe
