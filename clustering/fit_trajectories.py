"""
fit_trajectories.py

Step 1.
Fit GAM trajectories for every gene
and evaluate them on the common time grid.

Input
-----
xarray.Dataset

Dimensions:
    ensembl_gene_id
    time
    source

Output
------
trajectory matrix

(n_genes × n_timepoints)

"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from pygam import LinearGAM, s, f, te
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# ----------------------------------------------------------
# configuration
# ----------------------------------------------------------

N_SPLINES = 15
N_JOBS = -1

DATA_WEIGHT = {
    'BK': 3, 
    'JN': 2, 
    'Pauli': 1, 
    'White': 3,
}

# ----------------------------------------------------------
# fit one gene
# ----------------------------------------------------------
def fit_gene(gene, ds, prediction_grid):

    g = ds.sel(ensembl_gene_id=gene)

    t_all = []
    y_all = []
    source_all = []
    weight_all = []          # <-- per-observation weights, not per-study

    sources = np.atleast_1d(ds.source.values)
    for source_index, source in enumerate(sources):

        y = g.tpm.sel(source=source).values
        y = np.atleast_1d(y)    
        t = ds.time.values

        mask = np.isfinite(y)

        t_all.append(t[mask])
        y_all.append(y[mask])
        source_all.extend([source_index] * mask.sum())

        # repeat the study's weight once per retained observation
        weight_all.extend([DATA_WEIGHT[source]] * mask.sum())

    if len(t_all) == 0:
        return None

    t = np.concatenate(t_all)
    y = np.concatenate(y_all)
    source_all = np.asarray(source_all)
    weights = np.asarray(weight_all, dtype=float)
    X = np.column_stack([t, source_all])

    n_obs = len(y)
    n_splnes = min(N_SPLINES, max(4, n_obs // 3))

    try:
        if len(sources) > 1:
            gam = LinearGAM(s(0, n_splines=n_splnes) + f(1))
            gam.gridsearch(X, y, weights=weights, lam=np.logspace(-3, 3, 8), progress=False) 
        else: 
            gam = LinearGAM(s(0, n_splines=n_splnes))
            gam.gridsearch(X, y, lam=np.logspace(-3, 3, 8), progress=False)
    except Exception as e:
        print(f"[fit_gene] {gene}: {e}")
        return None

    # predict on common time grid
    pred = np.zeros((len(sources), len(prediction_grid)))
    for source_index in range(len(sources)):
        Xpred = np.column_stack([prediction_grid, np.repeat(source_index, len(prediction_grid))])
        pred[source_index] = gam.predict(Xpred)

    mean_curve = pred.mean(axis=0)
    return mean_curve

# ----------------------------------------------------------
# fit every gene
# ----------------------------------------------------------

def fit_all_genes(ds):

    prediction_grid = ds.time.values
    genes = ds.ensembl_gene_id.values
    results = Parallel(n_jobs=N_JOBS)(
        delayed(fit_gene)(gene,ds,prediction_grid)
        for gene in tqdm(genes)
    )
    curves = []
    kept = []

    for gene, curve in zip(genes, results):
        if curve is None:
            continue
        curves.append(curve)
        kept.append(gene)

    curves = np.asarray(curves)
    trajectories = xr.DataArray(
        curves,
        dims=["ensembl_gene_id", "time"],
        coords=dict(
            ensembl_gene_id=kept,
            time=prediction_grid
        ),
        name="trajectory"
    )
    return trajectories

def gof_trajectories(ds, trajectories, t_end):

    from scipy.stats import spearmanr, pearsonr

    """
    Compute goodness-of-fit metrics for every gene
    """

    genes = trajectories.ensembl_gene_id.values
    sources = ds.source.to_series().unique()
    rows = []

    for src in sources:
        gof = {"nrmse": [], "pearson":[], "spearman":[]}
        accept = []
        data = ds.sel(source=src, drop=True)

        for gene in tqdm(genes):

            try:
                y_true = data.sel(ensembl_gene_id=gene, drop=True).tpm
                y_pred = trajectories.sel(ensembl_gene_id=gene, drop=True).values
            except Exception:
                gof.append(np.nan)
                accept.append(False)
                continue

            mask = np.isfinite(y_true) & np.isfinite(y_pred)

            if mask.sum() < 2:
                gof.append(np.nan)
                accept.append(False)
                continue

            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            rmse = np.sqrt(np.mean((y_true.values - y_pred)**2))
            pearson = pearsonr(y_true.values, y_pred)[0]
            spearman = spearmanr(y_true.values, y_pred)[0]

            n_range = y_true.max("time").item() - y_true.min("time").item()  # range
            nrmse = rmse / n_range if n_range > 0 else np.nan

            gof["nrmse"].append(nrmse)
            gof["pearson"].append(pearson)
            gof["spearman"].append(spearman)

            accept.append(bool((nrmse < 0.3) and (pearson > 0.5) and (spearman > 0.5)) if np.isfinite(nrmse) else False)

        rows.append(pd.DataFrame({"ensembl_gene_id": genes, "source": src, 
                                  "nrmse": gof["nrmse"], "pearson": gof["pearson"], "spearman": gof["spearman"],
                                  "accepted": accept, }))

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(f"results/gof_trajectories_{t_end}.csv", index=False)

    return df

# ----------------------------------------------------------
# main
# ----------------------------------------------------------

if __name__ == "__main__":

    DATA = ["all", 'White', "Pauli", "BK", "JN"]
    ds = xr.load_dataset("../data/genes_tpms_white_pauli_JN_BK_mean.nc")
    ds_clean = ds.dropna(dim="time", how="all", subset=["tpm"])

    # Remove low expressed genes -> too noisy, no effective pattern
    mask = (ds_clean.tpm.max(dim="time", skipna=True) >= 1).all(dim="source") 
    ds_clean = ds_clean.sel(ensembl_gene_id=mask)

    # Reduce variance by log scaling
    ds_clean["tpm"] = np.log2(ds_clean.tpm + 1) 

    # z-score
    mean = ds_clean.tpm.mean(dim=("time", "source"))
    std = ds_clean.tpm.std(dim=("time", "source"))
    ds_clean["tpm"] = (ds_clean.tpm - mean) / std

    print(len(ds_clean.ensembl_gene_id))
    for T_END in [120]:
        print(f"fitting over t={T_END} hpf")

        ds_filtered = ds_clean.sel(time=slice(0, T_END))
        trajectories = fit_all_genes(ds_filtered)
        trajectories.to_netcdf(f"results/{DATA[0]}_gene_trajectories_{T_END}.nc")

        print(f"Calculating goodness of fit...")
        #trajectories = xr.load_dataset(f"results/{DATA[0]}_gene_trajectories_{T_END}.nc").trajectory
        gof_trajectories(ds_filtered, trajectories, T_END)
