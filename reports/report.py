import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


def calc_log_rmse(obs, pred):
    '''Root Mean Square Error of log-transformed data'''

    obs = np.log2(obs +1)
    pred = np.log2(pred +1)
    rmse = np.sqrt(np.mean((obs - pred)**2))

    return rmse.item()

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


import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def _compute_rho(model, gid):
    """Runs in a worker process: load one gene's dataset and compute rho."""
    path = os.path.join(PROJECT_ROOT, "results", "120_hpf", model, "all", gid, "numpyro_posterior.nc")
    try:
        obs_ds = xr.open_dataset(path, group="observed_data")
        pred_ds = xr.open_dataset(path, group="posterior_model_fits")

        obs = obs_ds["y"].mean("source")
        pred = pred_ds["y"].mean(dim=("draw", "chain", "source"))
        rho = spearmanr(obs.values, pred.values).correlation

        obs_ds.close()
        pred_ds.close()
    except Exception as e:
        print(f"⚠️ Failed on {gid}: {e}")
        rho = np.nan
    return gid, rho

def calc_rho_full_ds(model, max_workers=None, limit=None):

    csv_path = os.path.join(PROJECT_ROOT, "results_summary", model, "goodness_of_fit_summary.csv")
    df = pd.read_csv(csv_path)
    df = df.set_index("gene_id", drop=False)

    genes = df.index.tolist()
    if limit:
        genes = genes[:limit]

    if max_workers is None:
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    from functools import partial
    worker = partial(_compute_rho, model)

    results = {}
    chunksize = max(1, len(genes) // (max_workers * 4))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for gid, rho in ex.map(worker, genes, chunksize=chunksize):
            results[gid] = rho

    df["rho"] = df.index.map(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved gof file under {csv_path}")


if __name__ == "__main__":
    import sys
    func_name = sys.argv[1]
    model = sys.argv[2]
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None

    globals()[func_name](model, max_workers=max_workers)