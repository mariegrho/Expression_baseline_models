import numpy as np
import pandas as pd
import arviz as az
from scipy.stats import spearmanr, pearsonr
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

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

    # by range
    if norm == "range":
        return rmse / (obs.max(dim="time") - obs.min(dim="time"))
    # by mean
    elif norm == "mean":
        return rmse / obs.mean(dim="time")
    elif norm == "std":
        return rmse / obs.std(dim="time")
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

def calc_AIC(idata):
    '''
    k = free parameters
    LL = max. Log-Likelihood (sum)
    '''
    k = len(idata.posterior.data_vars)
    LL = idata.log_likelihood.y.mean(dim=("chain", "draw")).sum().item()
    aic = - 2 * LL + 2*k
    return float(aic)



# ======== Post fit metric calculation ===================
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def _init_worker_logger(model):
    """Runs once per worker process when the pool starts it up."""
    log_dir = os.path.join(PROJECT_ROOT, "logs", "gof_worker_logs", model)
    os.makedirs(log_dir, exist_ok=True)
    pid = os.getpid()
    log_path = os.path.join(log_dir, f"worker_{pid}.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(process)d] %(levelname)s %(message)s",
    )
    

def _compute_stats(model, gid):
    """Runs in a worker process: load one gene's dataset and compute rho."""
    path = os.path.join(PROJECT_ROOT, "results", "120_hpf", model, "all", gid, "numpyro_posterior.nc")

    try:
        idata = az.from_netcdf(path)
        obs = idata.observed_data.y.mean("source")
        pred = idata.posterior_model_fits.y.mean(dim=("draw", "chain", "source"))

        ll = idata.log_likelihood["y"]  # dims: (chain, draw, time, source)
        grouped_ll = ll.sum(dim="source")   # -> dims: (chain, draw, time)

        # Wrap into a fresh InferenceData for WAIC/LOO
        idata_grouped = az.InferenceData(
            posterior=idata.posterior,          # reuse original posterior
            log_likelihood=xr.Dataset({"y": grouped_ll}),
        )
        
    except Exception as e:
        logging.error(f"Failed to load {gid}: {e}")
        return gid, np.nan, np.nan, np.nan, np.nan

    try:
        nrmse = calc_nrmse(obs.values, pred.values, norm="mean").mean()
    except Exception as e:
        logging.warning(f"NRMSE failed on {gid}: {e}")
        nrmse = np.nan

    try:
        # Watanabe–Akaike information criterion
        waic = az.waic(idata_grouped, pointwise=True).elpd_waic
    except Exception as e:
        logging.warning(f"WAIC failed on {gid}: {e}")
        waic = np.nan

    try:
        loo = az.loo(idata_grouped, pointwise=True).elpd_loo
    except Exception as e:
        logging.warning(f"LOO failed on {gid}: {e}")
        loo = np.nan
    finally:
        idata.close()

    try:
        rho = spearmanr(obs.values, pred.values).correlation
    except Exception as e:
        logging.warning(f"spearmanr failed on {gid}: {e}")
        rho = np.nan
    
    try:
        pearson = pearsonr(obs.values, pred.values).correlation
    except Exception as e:
        logging.warning(f"pearsonr failed on {gid}: {e}")
        pearson = np.nan

    return gid, rho, pearson, waic, loo, nrmse

def calc_rho_full_ds(model, max_workers=None, limit=None):

    csv_path = os.path.join(PROJECT_ROOT, "results/results_summary", model, "goodness_of_fit_summary.csv")
    df = pd.read_csv(csv_path)
    df = df.set_index("gene_id", drop=False)

    assert df.index.is_unique, "Duplicate gene_id values in CSV"

    genes = df.index.tolist()
    if limit:
        genes = genes[:limit]

    if max_workers is None:
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    from functools import partial
    worker = partial(_compute_stats, model)

    results_r = {}
    results_p = {}
    results_w = {}
    results_l = {}
    results_n = {}
    chunksize = max(1, len(genes) // (max_workers * 4))
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_logger, initargs=(model,),) as ex:
        for gid, rho, pearson, waic, loo, nrmse in ex.map(worker, genes, chunksize=chunksize):
            results_r[gid] = rho
            results_p[gid] = pearson
            results_w[gid] = waic
            results_l[gid] = loo
            results_n[gid] = nrmse

    df["rho"] = df.index.map(results_r)
    df["pearsonr"] = df.index.map(results_p)
    df["WAIC"] = df.index.map(results_w)
    df["LOO"] = df.index.map(results_l)
    df["NRMSE"] = df.index.map(results_n)

    print(f"NaN counts — rho: {sum(pd.isna(v) for v in results_r.values())}, "
      f"pearson: {sum(pd.isna(v) for v in results_p.values())}, "
      f"WAIC: {sum(pd.isna(v) for v in results_w.values())}, "
      f"LOO: {sum(pd.isna(v) for v in results_l.values())}, "
      f"NRMSE: {sum(pd.isna(v) for v in results_n.values())}")

    df.to_csv(csv_path, index=False)
    print(f"Saved gof file under {csv_path}")


if __name__ == "__main__":
    import sys
    func_name = sys.argv[1]
    model = sys.argv[2]
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None

    globals()[func_name](model, max_workers=max_workers)


# grep -l ERROR logs/gof_worker_logs/*.log
# grep -c WARNING logs/gof_worker_logs/*.log