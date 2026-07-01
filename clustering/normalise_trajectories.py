"""
normalize_trajectories.py

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


# =====================================================
# Configuration
# =====================================================

NORMALIZATION = "minmax"
# options:
#   "none"
#   "center"
#   "zscore"
#   "minmax"
#   "l2"

REMOVE_LOW_VARIANCE = True
VARIANCE_THRESHOLD = 0.05


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
    
    elif method == "meanmax":
        return (curve - curve.mean()) / abs(curve - curve.mean()).max()
    
    elif method == "l2":
        norm = np.linalg.norm(curve)
        if norm < 1e-8:
            return np.zeros_like(curve)
        return curve / norm
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

    t_end = 12
    da = xr.open_dataarray(f"results/gene_trajectories_{t_end}.nc")

    print(da)
    normalized = normalize_dataset(da)
    print(normalized)
    normalized.to_netcdf(f"results/normalized_trajectories_{t_end}.nc")