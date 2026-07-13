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

N_SPLINES = 20
N_JOBS = -1
GOF_THRESHOLD = 0.2

DATA_WEIGHT = {
    'BK': 2, 
    'JN': 1, 
    'Pauli': 1, 
    'White': 3,
}


# ----------------------------------------------------------
# fit one gene
# ----------------------------------------------------------

def sigmoid_weights(t, midpoint, scale=15, floor=0.5, ceiling=1.0):
    """
    Fallende S-Kurve: hohes Plateau für frühe t, sanfter Übergang
    um `midpoint`, niedriges Plateau (floor) für späte t.
    """
    s = 1.0 / (1.0 + np.exp((t - midpoint) / scale))
    return floor + (ceiling - floor) * s


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

    if len(y) < N_SPLINES:   # fewer points than spline basis functions requested
        print(f"[fit_gene] {gene}: only {len(y)} obs, skipping (need >= {N_SPLINES})")
        return None


    n_obs = len(y)
    n_splnes = min(N_SPLINES, max(4, n_obs // 2))

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

    """
    Compute goodness-of-fit metrics for every gene
    """

    genes = trajectories.ensembl_gene_id.values
    sources = ds.source.to_series().unique()
    rows = []

    for src in sources:
        gof = []
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
            n_range = y_true.max("time").item() - y_true.min("time").item()  # range
            nrmse = rmse / n_range if n_range > 0 else np.nan

            gof.append(nrmse)
            accept.append(bool(nrmse < GOF_THRESHOLD) if np.isfinite(nrmse) else False)

        rows.append(pd.DataFrame({"ensembl_gene_id": genes, "source": src, "nrmse": gof, "accepted": accept, }))

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(f"results/gof_trajectories_{t_end}.csv", index=False)

    return df


# ----------------------------------------------------------
# main
# ----------------------------------------------------------

if __name__ == "__main__":

    DATA = ["all", 'White', "Pauli", "BK", "JN"]

    ds = xr.load_dataset("../data/genes_tpms_white_pauli_JN_BK_mean.nc")
    ds["tpm"] = np.log2(ds.tpm + 1)
    #ds = ds.sel(source = ['White'])
    #ds = ds.sel(ensembl_gene_id=ds.ensembl_gene_id.values[0:5])

    ds_clean = ds.dropna(dim="time", how="all", subset=["tpm"])
    # Keep only relevantly expressed genes
    mask = (ds_clean.tpm.max(dim="time", skipna=True) >= 1).any(dim="source")

    for T_END in [24]:
        print(f"fitting over t={T_END} hpf")

        ds_filtered = ds.sel(ensembl_gene_id=mask).sel(time=slice(0, T_END))
        trajectories = fit_all_genes(ds_filtered)
        trajectories.to_netcdf(f"results/{DATA[0]}_gene_trajectories_{T_END}.nc")

        print(f"Calculating goodness of fit...")
        gof_trajectories(ds_filtered, trajectories, T_END)
