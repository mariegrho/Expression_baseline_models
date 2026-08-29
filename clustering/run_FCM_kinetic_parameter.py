"""
hierarchical_clustering.py

Step 3.
Cluster the normalised trajectories using fuzzy c-means.
Calls fuzzy_clustering_params.py.

Input
-----
_normalized_trajectories_120-.nc

Output
------
_gene_cluster_annotation_.nc
"""

import xarray as xr
import numpy as np
import pandas as pd
from glob import glob
import FCM_kinetic_parameter as fc

NORMALISATION = "StandardScaler"
# options:
#   "RobustScaler"

DATA = ["joint", "all", 'White', "Pauli", "BK", "JN"]
source = DATA[0]

print(f"[Info] Starting hierarchical parameter clustering for dataset: {source}, normalization: {NORMALISATION}")

ds_params = xr.load_dataset("joint_params_results.nc")

ds_params["beta"] = ds_params.beta.clip(min=1e-5)
ds_params["alpha"] = ds_params.alpha.clip(min=1e-5)
ds_params["delta_m"] = ds_params.delta_m.clip(min=1e-5)
ds_params["t_zga"] = ds_params.t_zga.clip(max=120)
ds_params["t_reg"] = ds_params.t_reg.clip(max=120)

membership, labels, centers_std, centers_orig = fc.cluster_dataset(ds_params, f"results/{source}_param_superclusters", dataset_name="supercluster", k_range=range(3, 10))

# --- reindex superclusters by increasing peak time ---
cluster_peak_times = centers_std.sel(feature="peak_time").values  # peak_time value per cluster
new_order = np.argsort(cluster_peak_times)    # sort clusters by their peak_time values
rank = np.argsort(new_order)                  # rank[old_id] = new_id

print(f"Cluster peak times: {cluster_peak_times}")

# reorder centers
centers_std = centers_std.isel(cluster=new_order).assign_coords(cluster=np.arange(len(new_order)))
centers_std.to_netcdf(f"results/{source}_param_superclusters_centers.nc")

# reorder centers
centers_orig = centers_orig.isel(cluster=new_order).assign_coords(cluster=np.arange(len(new_order)))
centers_orig.to_netcdf(f"results/{source}_param_superclusters_centers_orig.nc")

labels = labels.copy(data=rank[labels.values])  # remap gene labels to new ids
labels.to_netcdf(f"results/{source}_param_superclusters_labels.nc")
#fc.plot_cluster_centers(centers_std, "Superclusters")

# reorder _membership
membership = membership.isel(cluster=new_order).assign_coords(cluster=np.arange(len(new_order)))
membership.to_netcdf(f"results/{source}_param_superclusters_membership.nc")

unique, counts = np.unique(labels.values, return_counts=True)
print(f"Post-remap cluster sizes: {dict(zip(unique, counts))}")

'''  ---- SUBCLUSTERING ---- '''
#ds_params_sub = xr.load_dataset("joint_params_results.nc")
ds_params_sub = ds_params.copy()

common_genes = list(set(ds_params.ensembl_gene_id.values) & set(ds_params_sub.ensembl_gene_id.values))
print(f"[Info] Number of common genes: {len(common_genes)}")
labels_sub = labels.sel(ensembl_gene_id=common_genes)

# Minimum sample threshold for subclustering
MIN_SAMPLES_FOR_SUBCLUSTERING = 10


for sc in np.unique(labels_sub.values):
    
    genes = labels_sub.ensembl_gene_id.where(labels_sub == sc,drop=True)
    subtraj = ds_params_sub.where(ds_params_sub.ensembl_gene_id.isin(genes.values), drop=True)

    print(f" --- Cluster {sc}, size: {len(genes)}")

    # Check if subcluster has enough samples before clustering
    if len(subtraj.ensembl_gene_id) < MIN_SAMPLES_FOR_SUBCLUSTERING:
        print(f"[Warning] Skipping supercluster {sc}: only {len(subtraj.ensembl_gene_id)} samples (minimum {MIN_SAMPLES_FOR_SUBCLUSTERING} required)")
        continue
    
    membership, labels, centers_std, centers_ori = fc.cluster_dataset(
        subtraj, f"results/{source}_param_supercluster_{sc}",
        dataset_name=f"subcluster_{sc}", k_range=range(2, 10),
        min_samples=MIN_SAMPLES_FOR_SUBCLUSTERING
    )
    
    # Only plot if clustering was successful
    #if centers_std is not None:
        #fc.plot_cluster_centers(centers_std, dataset_name=f"Supercluster {sc}")

### DataSet
super_labels = xr.load_dataarray(f"results/{source}_param_superclusters_labels.nc")
genes = common_genes

# initialize subcluster labels
subcluster = xr.DataArray(
    np.full(len(genes), -1, dtype=int),
    dims=("ensembl_gene_id",),
    coords={"ensembl_gene_id": genes},
    name="subcluster"
)

for sc in np.unique(super_labels.values):
    # Check if the subcluster file exists (might not exist if skipped)
    labels_file = f"results/{source}_param_supercluster_{sc}_labels.nc"
    try:
        labels = xr.load_dataarray(labels_file)
        subcluster.loc[dict(ensembl_gene_id=labels.ensembl_gene_id)] = labels
    except FileNotFoundError:
        print(f"[Warning] Subcluster file {labels_file} not found (likely skipped due to insufficient samples)")

# load expression data, restricted to the relevant genes
expr_data = xr.load_dataset("../data/genes_tpms_white_pauli_JN_BK_mean.nc")
expr_data = expr_data.sel(ensembl_gene_id=subcluster.ensembl_gene_id)

# build annotation dataset with supercluster/subcluster as coordinates
annotation = expr_data.assign_coords(
    supercluster=("ensembl_gene_id", super_labels.sel(ensembl_gene_id=genes).values),
    subcluster=("ensembl_gene_id", subcluster.values),
)

annotation.to_netcdf(f"results/{source}_gene_params_cluster_annotation.nc")
