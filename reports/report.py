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

import numpy as np
import arviz as az
import pandas as pd
from typing import List, Dict, Optional, Callable
from scipy.stats import spearmanr


def spearmanr_from_idata(
    idata: az.InferenceData,
    data_vars: Optional[List[str]] = None,
    use_predictions: bool = False,
    obs_transform_funcs: Dict[str, Callable] = {},
):
    """
    Calculate Spearman's rank correlation coefficient (rho) from InferenceData.
    
    Parameters
    ----------
    idata : az.InferenceData
        ArviZ InferenceData object containing posterior samples and observed data
    data_vars : Optional[List[str]], optional
        List of data variables to compute correlation for. If None, uses all variables
        in observed_data.
    use_predictions : bool, optional
        If True, uses posterior_predictive samples. If False, uses posterior_model_fits.
        Default is True.
    obs_transform_funcs : Dict[str, Callable], optional
        Dictionary of transformation functions to apply to observed data for each variable.
        Default is empty dict (no transformations).
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing Spearman's rho values for each data variable.
        Each row represents a data variable, with columns for the correlation statistic.
    """
    if data_vars is None:
        data_vars = list(idata.observed_data.data_vars.keys())

    spearmanr_data_vars = {}
    for dv in data_vars:
        if use_predictions:
            x_0 = idata.posterior_predictive[dv]
            x = idata.observed_data[dv]
        else:
            x_0 = idata.posterior_model_fits[dv]
            obs_transform_func = obs_transform_funcs.get(dv, lambda x: x)
            x = obs_transform_func(idata.observed_data[dv])

        # Calculate Spearman's rho for each chain/draw combination
        # Flatten dimensions to get pairwise correlations
        spearmanr_values = []
        spearmanr_pvalues = []

        # Reshape for correlation calculation
        x_0_flat = x_0.values
        x_flat = x.values

        # Spearman's rho calculation - handle different array shapes
        # For each chain and draw, compute correlation between predicted and observed
        n_chains = x_0.coords.get('chain', {}).size if 'chain' in x_0.coords else 1
        n_draws = x_0.coords.get('draw', {}).size if 'draw' in x_0.coords else 1

        for chain in range(n_chains):
            for draw in range(n_draws):
                if n_chains > 1 and n_draws > 1:
                    pred = x_0_flat[chain, draw]
                elif n_chains > 1:
                    pred = x_0_flat[chain]
                elif n_draws > 1:
                    pred = x_0_flat[draw]
                else:
                    pred = x_0_flat

                # Flatten both arrays for correlation
                pred_flat = pred.flatten()
                obs_flat = x_flat.flatten()

                # Calculate Spearman's rho
                if len(pred_flat) > 0 and len(obs_flat) > 0:
                    rho, pvalue = spearmanr(pred_flat, obs_flat, nan_policy="omit")
                    spearmanr_values.append(rho)
                    spearmanr_pvalues.append(pvalue)

        spearmanr_array = np.array(spearmanr_values)
        spearmanr_pvalues_array = np.array(spearmanr_pvalues)

    return spearmanr_array.mean()
"""
        spearmanr_data_vars.update({dv: {
            "Spearman's rho": spearmanr_array.mean(),
            "Spearman's rho (std)": spearmanr_array.std(),
            "P-value (mean)": spearmanr_pvalues_array.mean(),
        }})

        if len(spearmanr_values) > 1:
            spearmanr_data_vars[dv].update({
                "Spearman's rho (min)": spearmanr_array.min(),
                "Spearman's rho (max)": spearmanr_array.max(),
            })

    spearmanr_data_vars["model"] = np.nan

    return pd.DataFrame(spearmanr_data_vars).T
    #return spearmanr_array
"""
