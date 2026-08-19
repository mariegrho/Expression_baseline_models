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

NORMALIZATION = "minmax"

# options:
#   "none"
#   "center"
#   "zscore"
#   "minmax"
#   "meanmax"

REMOVE_LOW_VARIANCE = True
VARIANCE_THRESHOLD = 0.01

# =====================================================
# Normalization methods
# =====================================================

def normalize_curve(curve, method):

    curve = np.asarray(curve)

    if method == "none":
        return curve
    
    elif method == "center":
        return curve - curve.mean()
    
    elif method == "zscore":
        sd = curve.std()
        if sd < 1e-8:
            return np.zeros_like(curve)
        return (curve - curve.mean()) / sd
    
    elif method == "minmax":
        mn = curve.min()
        mx = curve.max()
        if mx == mn:
            return np.zeros_like(curve)
        return (curve - mn) / (mx - mn)
    
    elif method == "percentile":
        lower = np.percentile(curve, 5)
        upper = np.percentile(curve, 95)

        return (curve - lower) / (upper - lower)
    
    elif method == "meanmax":
        return (curve - curve.mean()) / abs(curve - curve.mean()).max()
    
    else:
        raise ValueError("Unknown normalization")


# =====================================================
# Normalize all trajectories
# =====================================================

def normalize_dataset(da):

    curves = da.values
    normalized = np.zeros_like(curves)
    keep = np.ones(curves.shape[0], dtype=bool)

    for i in range(curves.shape[0]):
        c = normalize_curve(curves[i], NORMALIZATION)
        if REMOVE_LOW_VARIANCE:
            if np.var(c) < VARIANCE_THRESHOLD:
                keep[i] = False
                continue
        normalized[i] = c

    normalized = normalized[keep]
    genes = da.ensembl_gene_id.values[keep]
    out = xr.DataArray(
        normalized,
        dims=("ensembl_gene_id", "time"),
        coords={"ensembl_gene_id": genes,"time": da.time.values},
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
        da = xr.load_dataarray(f"results/{source}_gene_trajectories_{t_end}.nc")
        da = da.sel(ensembl_gene_id = accepted_genes)
        normalized = normalize_dataset(da)
        normalized.to_netcdf(f"results/{source}_normalized_trajectories_{t_end}_{NORMALIZATION}.nc")
        print(f"[Info] Saved under: ./results/{source}_normalized_trajectories_{t_end}_{NORMALIZATION}.nc")