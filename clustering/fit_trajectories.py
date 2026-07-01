"""
fit_trajectories.py

Fit study-aware GAM trajectories for every gene
and evaluate them on the common time grid.

Input
-----
xarray.Dataset

Dimensions:
    gene
    time
    source

Output
------
trajectory matrix

(n_genes × n_timepoints)

"""

import numpy as np
import xarray as xr

from pygam import LinearGAM, s, f
from joblib import Parallel, delayed
from tqdm.auto import tqdm


# ----------------------------------------------------------
# configuration
# ----------------------------------------------------------

N_SPLINES = 12
LAM = 0.6
N_JOBS = -1

# ----------------------------------------------------------
# fit one gene
# ----------------------------------------------------------

def fit_gene(gene, ds, prediction_grid):

    g = ds.sel(ensembl_gene_id=gene)

    t_all = []
    y_all = []
    source_all = []

    sources = ds.source.values

    # ---------------------------------------------
    # collect observations from every study
    # ---------------------------------------------

    for source_index, source in enumerate(sources):

        y = g.tpm.sel(source=source).values
        t = ds.time.values

        mask = np.isfinite(y)

        if mask.sum() < 5:
            continue

        t_all.append(t[mask])
        y_all.append(y[mask])
        source_all.extend([source_index] * mask.sum())

    if len(t_all) == 0:
        return None

    t = np.concatenate(t_all)
    y = np.concatenate(y_all)

    source_all = np.asarray(source_all)

    # ---------------------------------------------
    # log transform & normalization
    # ---------------------------------------------

    y = np.log2(y + 1)
    #y = (y - y.mean()) / (y.std() + 1e-8)
    #y = (y - y.mean(dim=("time", "source"))) / y.std(dim=("time", "source"))

    # ---------------------------------------------
    # design matrix
    #
    # column 0 = time
    # column 1 = study
    # ---------------------------------------------

    X = np.column_stack([t, source_all])

    try:
        gam = LinearGAM(s(0,n_splines=N_SPLINES)+f(1))
        gam.gridsearch(X,y, lam=np.logspace(-3,3,8), progress=False)

    except:
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


# ----------------------------------------------------------
# main
# ----------------------------------------------------------

if __name__ == "__main__":

    t_end = 12
    ds = xr.open_dataset("../data/genes_tpms_white_pauli_JN_BK_mean.nc").sel(time=slice(0, t_end))
    #ds = ds.sel(ensembl_gene_id=ds.ensembl_gene_id.values[0:500])
    trajectories = fit_all_genes(ds)

    print(trajectories)

    trajectories.to_netcdf(f"results/gene_trajectories_{t_end}.nc")