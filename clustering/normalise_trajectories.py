"""
normalize_trajectories.py

Step 2.
Normalize fitted gene trajectories before clustering.

Input
-----
gene_trajectories.nc

Output
------
normalized_trajectories.nc
"""

import numpy as np
import xarray as xr
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

# =====================================================
# Configuration
# =====================================================

NORMALIZATION = "zscore"

# options:
#   "none"
#   "center"
#   "zscore"
#   "minmax"
#   "meanmax"

REMOVE_LOW_VARIANCE = True
VARIANCE_THRESHOLD = 0.001


import numpy as np
import xarray as xr

def normalize_dataset(da, normalization_method="zscore", 
                      remove_low_variance=REMOVE_LOW_VARIANCE, variance_threshold=VARIANCE_THRESHOLD):
    """
    Normalisiert das Dataset vektorisiert entlang der Zeitachse.
    Erwartet ein xarray.DataArray mit den Dimensionen (ensembl_gene_id, time).
    """
    curves = da.values  # Shape: (gene, time)
    genes = da.ensembl_gene_id.values

    # 1. VOR der Normalisierung: Low-Variance-Filter anwenden
    if remove_low_variance:
        # Berechne echte biologische Varianz pro Gen vorab
        raw_variances = np.var(curves, axis=1)
        keep = raw_variances >= variance_threshold
        
        # Sofort filtern
        curves = curves[keep]
        genes = genes[keep]

    print(curves.shape[0], "genes surviving variance_treshold")

    # Wenn nach dem Filter keine Gene übrig sind, abbrechen
    if curves.shape[0] == 0:
        raise ValueError("No genes survive VARIANCE_THRESHOLD. Too high?")

    # 2. Vektorisierte Normalisierung 
    method = normalization_method.lower()
    
    if method == "none":
        normalized = curves
        
    elif method == "center":
        # Zeilenweiser Mittelwert: Shape (n_genes, 1) für korrektes Broadcasting
        means = curves.mean(axis=1, keepdims=True)
        normalized = curves - means
        
    elif method == "zscore":
        means = curves.mean(axis=1, keepdims=True)
        stds = curves.std(axis=1, keepdims=True)
        
        # Vektorisierter Schutz: Wo std < 1e-8 ist, setzen wir die Kurve auf 0
        # (Da wir flache Gene oben filtern, betrifft das meist nur komplett tote Gene)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = (curves - means) / stds
        normalized[stds.squeeze() < 1e-8] = 0.0
        
    elif method == "minmax":
        mn = curves.min(axis=1, keepdims=True)
        mx = curves.max(axis=1, keepdims=True)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = (curves - mn) / (mx - mn)
        normalized[(mx - mn).squeeze() == 0] = 0.0
        
    elif method == "percentile":
        # Prozentile entlang axis=1 berechnen
        lower = np.percentile(curves, 5, axis=1, keepdims=True)
        upper = np.percentile(curves, 95, axis=1, keepdims=True)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = (curves - lower) / (upper - lower)
        normalized[(upper - lower).squeeze() == 0] = 0.0
        
    elif method == "meanmax":
        means = curves.mean(axis=1, keepdims=True)
        centered = curves - means
        # Maximaler absoluter Abstand vom Mittelwert pro Gen
        max_abs_dist = np.abs(centered).max(axis=1, keepdims=True)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = centered / max_abs_dist
        normalized[max_abs_dist.squeeze() == 0] = 0.0
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # 3. Neues xarray DataArray mit den gefilterten & transformierten Daten bauen
    out = xr.DataArray(
        normalized,
        dims=("ensembl_gene_id", "time"),
        coords={"ensembl_gene_id": genes, "time": da.time.values},
        name="trajectory"
    )
    return out



# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    DATA = ["all", "avg", 'White', "Pauli", "BK", "JN"]
    source = DATA[0]

    gof = pd.read_csv("results/gof_trajectories_120.csv")
    gene_sums = gof.groupby("ensembl_gene_id").sum("accepted")
    accepted_genes = gene_sums[gene_sums["accepted"] > 0].index.to_list()
    print(len(accepted_genes))

    for t_end in [120]:
        print(f"[Info] Normalise dataset -> {t_end} hpf")
        da = xr.load_dataarray(f"results/{source}_gene_trajectories_{t_end}_log.nc")
        da = da.sel(ensembl_gene_id = accepted_genes)

        normalized = normalize_dataset(da, normalization_method=NORMALIZATION)

        normalized.to_netcdf(f"results/{source}_normalized_trajectories_{t_end}_{NORMALIZATION}.nc")
        print(f"[Info] Saved under: ./results/{source}_normalized_trajectories_{t_end}_{NORMALIZATION}.nc")