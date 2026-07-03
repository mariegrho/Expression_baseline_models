"""
fit_trajectories.py

Fit study-aware GAM trajectories for every gene
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
GOF_THRESHOLD = 0.15

DATA_WEIGHT = {
    'BK': 2, 
    'JN': 1, 
    'Pauli et al.': 1, 
    'White et al.': 5,
}

T_END = 24

# ----------------------------------------------------------
# fit one gene
# ----------------------------------------------------------

def sigmoid_weights(t, midpoint=T_END/2, scale=15, floor=0.5, ceiling=1.0):
    """
    Fallende S-Kurve: hohes Plateau für frühe t, sanfter Übergang
    um `midpoint`, niedriges Plateau (floor) für späte t.
    """
    s = 1.0 / (1.0 + np.exp((t - midpoint) / scale))
    return floor + (ceiling - floor) * s


def fit_gene1(gene, ds, prediction_grid):

    g = ds.sel(ensembl_gene_id=gene)

    t_all = []
    y_all = []
    source_all = []
    weight_all = []          # <-- per-observation weights, not per-study

    sources = ds.source.values

    for source_index, source in enumerate(sources):

        y = g.tpm.sel(source=source).values
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

    y = np.log2(y + 1)

    X = np.column_stack([t, source_all])

    try:
        gam = LinearGAM(s(0, n_splines=N_SPLINES) + f(1))
        gam.gridsearch(X, y, weights=weights, lam=np.logspace(-3, 3, 8), progress=False)
    except Exception as e:
        print(f"[fit_gene] {gene}: {e}")
        return None

    # ---------------------------------------------
    # predict on common time grid
    # ---------------------------------------------
    pred = np.zeros((len(sources), len(prediction_grid)))
    for source_index in range(len(sources)):
        Xpred = np.column_stack([prediction_grid, np.repeat(source_index, len(prediction_grid))])
        pred[source_index] = gam.predict(Xpred)

    # study weighting is now already reflected in the fit itself —
    # average across sources unweighted (see note below)
    mean_curve = pred.mean(axis=0)
    return mean_curve

def fit_gene(gene, ds, prediction_grid):

    g = ds.sel(ensembl_gene_id=gene)

    t_all = []
    y_all = []
    source_all = []
    weights_d = []

    sources = ds.source.values

    # ---------------------------------------------
    # collect observations from every study
    # ---------------------------------------------

    for source_index, source in enumerate(sources):

        y = g.tpm.sel(source=source).values
        t = ds.time.values

        mask = np.isfinite(y)

        #if mask.sum() < 5:
        #    continue

        t_all.append(t[mask])
        y_all.append(y[mask])
        source_all.extend([source_index] * mask.sum())

        weights_d.append(DATA_WEIGHT[source])

    if len(t_all) == 0:
        return None

    t = np.concatenate(t_all)
    y = np.concatenate(y_all)

    source_all = np.asarray(source_all)

    # ---------------------------------------------
    # log transform
    # ---------------------------------------------

    y = np.log2(y + 1)

    # ---------------------------------------------
    # design matrix
    # column 0 = time
    # column 1 = study
    # ---------------------------------------------

    X = np.column_stack([t, source_all])


    tau = 40
    floor = 0.15
    #time_weights = floor + (1 - floor) * np.exp(-t / tau)
    #time_weights = sigmoid_weights(t)

    #plt.plot(t, time_weights, 'o')
    #plt.show()

    try:
        
        #gam = LinearGAM(te(0, 1, n_splines=[N_SPLINES, len(sources)]))
        gam = LinearGAM(s(0,n_splines=N_SPLINES)+f(1))
        #gam.gridsearch(X, y, weights=time_weights, lam=np.logspace(-5, 3, 10), progress=False)
        gam.gridsearch(X,y, lam=np.logspace(-3,3,8), progress=False)
        #print(gam.lam)

    except Exception as e:
        print(f"[fit_gene] {gene}: {e}")
        return None

    # ---------------------------------------------
    # predict on common time grid
    # ---------------------------------------------

    pred = np.zeros((len(sources), len(prediction_grid)))

    for source_index in range(len(sources)):
        Xpred = np.column_stack([prediction_grid, np.repeat(source_index, len(prediction_grid))])
        pred[source_index] = gam.predict(Xpred)

    # ---------------------------------------------
    # average over studies
    # ---------------------------------------------
    #mean_curve = np.average(pred, axis=0, weights=weights_d)
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


def gof_trajectories(ds, trajectories):

    """
    Compute goodness-of-fit metrics for every gene
    """
    ds = np.log2(ds + 1)

    genes = trajectories.ensembl_gene_id.values
    sources = ds.source.to_series().unique()
    rows = []

    for src in sources:
        gof = []
        accept = []
        data = ds.sel(source=src, drop=True)

        for gene in tqdm(genes):

            y_true = data.sel(ensembl_gene_id=gene, drop=True).tpm
            y_pred = trajectories.sel(ensembl_gene_id=gene, drop=True).values

            norm = y_true.max("time").item() - y_true.min("time").item()

            mask = np.isfinite(y_true) & np.isfinite(y_pred)

            if mask.sum() < 2:
                gof.append(np.nan)
                continue

            y_true = y_true[mask]
            y_pred = y_pred[mask]
            

            rmse = np.sqrt(np.mean((y_true.values - y_pred)**2))
            nrmse = rmse / norm if norm > 0 else np.nan

            gof.append(nrmse)
            accept.append(nrmse < GOF_THRESHOLD)

        rows.append(pd.DataFrame({"ensembl_gene_id": genes, "source": src, "nrmse": gof, "accepted": accept, }))

    return pd.concat(rows, ignore_index=True)



# ----------------------------------------------------------
# main
# ----------------------------------------------------------

if __name__ == "__main__":

    ds = xr.load_dataset("../data/genes_tpms_white_pauli_JN_BK_mean.nc")
    #ds = ds.sel(source = ['White et al.', ])
    #ds = ds.sel(ensembl_gene_id=ds.ensembl_gene_id.values[0:5])

    mask = ds.tpm.max(dim=["time", "source"], skipna=True) >= 5

    ds_filtered = ds.sel(ensembl_gene_id=mask).sel(time=slice(0, T_END))
    trajectories = fit_all_genes(ds_filtered)
    #print(trajectories)

    trajectories.to_netcdf(f"results/gene_trajectories_{T_END}.nc")