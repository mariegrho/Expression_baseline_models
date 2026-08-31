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

# ----------------------------------------------------------
# configuration
# ----------------------------------------------------------

N_SPLINES = 15
N_JOBS = -1

from pygam import LinearGAM, s, f
from joblib import Parallel, delayed
from tqdm import tqdm

def fit_single_gene_numpy(y_flat, X_full, mask_full, prediction_matrix, n_sources, n_pred_points, n_splines):
    """
    Diese Funktion arbeitet NUR noch mit schnellen NumPy-Arrays.
    Kein xarray-Overhead mehr!
    """
    # 1. Maskierung blitzschnell auf NumPy-Ebene anwenden
    # (Da y_flat für das Gen individuelle NaNs enthalten kann)
    gene_mask = mask_full & np.isfinite(y_flat)
    
    X = X_full[gene_mask]
    y = y_flat[gene_mask]

    if len(y) == 0:
        return None

    try:
        # 2. GAM initialisieren und fitten
        gam = LinearGAM(s(0, n_splines=n_splines) + f(1))
        gam.gridsearch(X, y, lam=np.logspace(-3, 4, 15), progress=False)
        
        # 3. Vektorisierte Vorhersage für alle Quellen gleichzeitig
        # prediction_matrix hat das Shape (n_sources * n_pred_points, 2)
        all_preds = gam.predict(prediction_matrix)
        
        # Reshapen zu (n_sources, n_pred_points) und direkt den Achsenmittelwert berechnen
        mean_curve = all_preds.reshape(n_sources, n_pred_points).mean(axis=0)
        return mean_curve

    except Exception:
        # Fehler abfangen (z.B. zu wenige Datenpunkte für die Splines)
        return None

def fit_all_genes_efficient(ds, n_splines=N_SPLINES, n_jobs=-1):
    prediction_grid = ds.time.values
    genes = ds.ensembl_gene_id.values
    sources = np.atleast_1d(ds.source.values)
    
    n_genes = len(genes)
    n_sources = len(sources)
    n_times = len(prediction_grid)
    n_pred_points = len(prediction_grid)

    print("Prepare global NumPy arrays...")
    # 1. Konvertiere das gesamte Xarray-Dataset in ein großes NumPy Array
    # Erwartetes Shape nach dem Laden: (genes, sources, time)
    raw_tpm = ds.tpm.values 
    
    # 2. Erstelle das globale X-Array EINMALIG vorab
    # Wir bauen das Raster aus Source-Indizes und Zeitpunkten
    source_indices, time_mesh = np.meshgrid(np.arange(n_sources), prediction_grid, indexing='ij')
    
    # Flachklopfen zu Spalten für das GAM
    X_full = np.column_stack([time_mesh.ravel(), source_indices.ravel()]) # Shape: (n_sources * n_times, 2)
    
    # Maske für valide Zeiten/Sources auf Dataset-Ebene
    mask_full = np.isfinite(X_full[:, 0]) 

    # 3. Erstelle die prediction_matrix EINMALIG vorab für alle Quellen
    pred_source_indices, pred_time_mesh = np.meshgrid(np.arange(n_sources), prediction_grid, indexing='ij')
    prediction_matrix = np.column_stack([pred_time_mesh.ravel(), pred_source_indices.ravel()])

    # Reshape der Gen-Daten, sodass wir sie zeilenweise (flach) verarbeiten können
    # Neues Shape: (30000, n_sources * n_times)
    flat_gene_data = raw_tpm.reshape(n_genes, -1)

    print("Start parallel GAM-Fitting...")
    # 4. Parallele Schleife übergibt nur noch die flachen NumPy-Zeilen
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_single_gene_numpy)(
            flat_gene_data[i], 
            X_full, 
            mask_full, 
            prediction_matrix, 
            n_sources, 
            n_pred_points,
            n_splines
        )
        for i in tqdm(range(n_genes))
    )

    # 5. Ergebnisse einsammeln
    curves = []
    kept = []

    for gene, curve in zip(genes, results):
        if curve is None:
            continue
        curves.append(curve)
        kept.append(gene)

    # 6. Zurück in ein sauberes Xarray DataArray konvertieren
    trajectories = xr.DataArray(
        np.asarray(curves),
        dims=["ensembl_gene_id", "time"],
        coords=dict(
            ensembl_gene_id=kept,
            time=prediction_grid
        ),
        name="trajectory"
    )
    return trajectories


# ----------------------------------------------------------
# fit one gene
# ----------------------------------------------------------
def fit_gene(gene, ds, prediction_grid):

    g = ds.sel(ensembl_gene_id=gene)

    t_all = []
    y_all = []
    source_all = []

    sources = np.atleast_1d(ds.source.values)
    for source_index, source in enumerate(sources):

        y = g.tpm.sel(source=source).values
        y = np.atleast_1d(y)    
        t = ds.time.values

        mask = np.isfinite(y)

        t_all.append(t[mask])
        y_all.append(y[mask])
        source_all.extend([source_index] * mask.sum())

    if len(t_all) == 0:
        return None

    t = np.concatenate(t_all)
    y = np.concatenate(y_all)
    source_all = np.asarray(source_all)
    X = np.column_stack([t, source_all]) 

    n_obs = len(y)

    try:
        ''' 
        s(0) term: Smooth, continuous, shared shape over time
        f(1) term: Categorical, one constant offset per source (batch/study effect)
                    treats column index 1 of X as a categorical (factor) variable, and fits a separate constant level for each category (source).
                    --> each source gets its own baseline offset
        X column 0: time t
        X column 1: sources 
        '''
        gam = LinearGAM(s(0, n_splines=N_SPLINES) + f(1))
        #gam = GAM(s(0, n_splines=N_SPLINES) + f(1), distribution='normal', link='log') # untransformierte daten
        gam.gridsearch(X, y, lam=np.logspace(-3, 4, 15), progress=False)

    except Exception as e:
        print(f"[fit_gene] {gene}: {e}")
        return None
    
    #print(gam.terms[0].lam)
    #print(gam.statistics_['edof'])

    # predict on common time grid
    pred = np.zeros((len(sources), len(prediction_grid)))
    for source_index in range(len(sources)):
        Xpred = np.column_stack([prediction_grid, np.repeat(source_index, len(prediction_grid))])
        pred[source_index] = gam.predict(Xpred)

    mean_curve = pred.mean(axis=0) # averages the per-source offset 
    #mean_curve = pred.mean(axis=0) - 1 # undo offset
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
    using all data points from all sources together.
    Returns one row per gene with aggregated metrics.
    """

    # backtranform log2(TPM+1) to TPM -> comparable to ODE model fits

    trajectories = np.exp2(trajectories) -1
    genes = trajectories.ensembl_gene_id.values
    sources = ds.source.values
    rows = []

    for gene in tqdm(genes):

        # Collect all data points from all sources for this gene
        y_true_all = []
        y_pred_all = []

        for source in sources:
            try:
                y_true = ds.sel(ensembl_gene_id=gene, source=source, drop=True).tpm.values
                y_pred = trajectories.sel(ensembl_gene_id=gene, drop=True).values
                
                # Match timepoints between observed data and predictions
                if len(y_true) == len(y_pred):
                    mask = np.isfinite(y_true) & np.isfinite(y_pred)
                    if mask.sum() >= 2:
                        y_true_all.extend(y_true[mask])
                        y_pred_all.extend(y_pred[mask])
            except Exception:
                continue

        # Convert to numpy arrays
        y_true_all = np.asarray(y_true_all)
        y_pred_all = np.asarray(y_pred_all)

        # Calculate metrics using all data points together
        if len(y_true_all) >= 2:
            rmse = np.sqrt(np.mean((y_true_all - y_pred_all)**2))
            pearson = pearsonr(y_true_all, y_pred_all)[0]
            spearman = spearmanr(y_true_all, y_pred_all)[0]

            # Calculate NRMSE using the overall range of the gene's expression
            n_range = np.max(y_true_all) - np.min(y_true_all)
            nrmse = rmse / n_range if n_range > 0 else np.nan

            accept = bool((nrmse < 0.25) or (spearman > 0.5)) if np.isfinite(nrmse) else False
        else:
            rmse = np.nan
            pearson = np.nan
            spearman = np.nan
            nrmse = np.nan
            accept = False

        rows.append({
            "ensembl_gene_id": gene,
            "nrmse": nrmse,
            "pearson": pearson,
            "spearman": spearman,
            "accepted": accept
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"results/gof_trajectories_{t_end}.csv", index=False)

    return df


# ----------------------------------------------------------
# main
# ----------------------------------------------------------

if __name__ == "__main__":

    DATA = ["all", "avg", 'White', "Pauli", "BK", "JN"]
    data_sel = DATA[0]

    ds = xr.load_dataset("../data/genes_tpms_white_pauli_JN_BK_mean.nc")
    ds = ds.transpose("ensembl_gene_id", "source", "time")
    ds = ds.dropna(dim="time", how="all", subset=["tpm"])

    # Remove low expressed genes -> too noisy, no effective pattern
    mask = (ds.tpm.max(dim="time", skipna=True) > 0).all(dim="source") 
    ds_clean = ds.sel(ensembl_gene_id=mask)
    #ds_clean = ds_clean.sel(ensembl_gene_id=ds_clean.ensembl_gene_id.values[0:20]) # test

    #ds_clean = ds_clean.mean(dim="source").expand_dims({"source":["avg"]}) # for average fitting

    # Reduce variance by log scaling ->  for LinarGAMs application 
    ds_clean["tpm"] = np.log2(ds_clean.tpm + 1) 

    # # z-score
    # mean = ds_clean.tpm.mean(dim=("time", "source"))
    # std = ds_clean.tpm.std(dim=("time", "source"))
    # ds_clean["tpm"] = (ds_clean.tpm - mean) / std

    print(len(ds_clean.ensembl_gene_id))
    for T_END in [120]:

        ds_filtered = ds_clean.sel(time=slice(0, T_END))
        print(f"fitting over t={T_END} hpf")
        # trajectories = fit_all_genes_efficient(ds_filtered)
        # trajectories.to_netcdf(f"results/{data_sel}_gene_trajectories_{T_END}_log.nc")

        print(f"Calculating goodness of fit...")
        trajectories = xr.load_dataarray(f"results/{data_sel}_gene_trajectories_{T_END}_log.nc")
        gof_trajectories(ds, trajectories, T_END)
