"""
hierarchical_clustering.py

Step 3.
Cluster the normalised trajectories using fuzzy c-means.
Calls fuzzy_clustering.py.

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
import fuzzy_clustering as fc

NORMALISATION = "minmax"

# options:
#   "none"
#   "center"
#   "zscore"
#   "minmax"
#   "percentile"

DATA = ["all", 'White', "Pauli", "BK", "JN"]
source = DATA[0]

traj120 = xr.load_dataarray(f"results/{source}_normalized_trajectories_120_{NORMALISATION}.nc")
membership120, labels120, centers120 = fc.cluster_dataset(traj120, f"results/{source}_superclusters", k_range=range(3, 8))

# --- reindex superclusters by increasing peak time ---
peak_times = np.argmax(centers120.values, axis=1)      # peak time index per (old) cluster
new_order = np.argsort(peak_times)              # old cluster ids, sorted by peak time
rank = np.argsort(new_order)                    # rank[old_id] = new_id

# reorder centers
centers120 = centers120.isel(cluster=new_order).assign_coords(cluster=np.arange(len(new_order)))
centers120.to_netcdf(f"results/{source}_superclusters_centers.nc")

labels120 = labels120.copy(data=rank[labels120.values])  # remap gene labels to new ids
labels120.to_netcdf(f"results/{source}_superclusters_labels.nc")
fc.plot_clusters(centers120, "Superclusters")


'''  ---- SUBCLUSTERING ---- '''
traj_sub = xr.load_dataarray(f"results/{source}_normalized_trajectories_120_{NORMALISATION}.nc")

common_genes = list(set(traj120.ensembl_gene_id.values) & set(traj_sub.ensembl_gene_id.values))
labels120 = labels120.sel(ensembl_gene_id=common_genes)
 
for sc in np.unique(labels120.values):
    genes = labels120.ensembl_gene_id.where(labels120 == sc,drop=True)
    subtraj = traj_sub.sel(ensembl_gene_id=genes)
    membership, labels, centers =fc.cluster_dataset(subtraj,f"results/{source}_supercluster_{sc}", k_range=range(2, 6))
    fc.plot_clusters(centers, cluster=f"Supercluster {sc}")

### DataSet
super_labels = xr.load_dataarray(f"results/{source}_superclusters_labels.nc")
genes = common_genes

# initialize subcluster labels
subcluster = xr.DataArray(
    np.full(len(genes), -1, dtype=int),
    dims=("ensembl_gene_id",),
    coords={"ensembl_gene_id": genes},
    name="subcluster"
)

for sc in np.unique(super_labels.values):
    labels = xr.load_dataarray(f"results/{source}_supercluster_{sc}_labels.nc")
    subcluster.loc[dict(ensembl_gene_id=labels.ensembl_gene_id)] = labels

# load expression data, restricted to the relevant genes
expr_data = xr.load_dataset("../data/genes_tpms_white_pauli_JN_BK_mean.nc")
expr_data = expr_data.sel(ensembl_gene_id=subcluster.ensembl_gene_id)

# build annotation dataset with supercluster/subcluster as coordinates
annotation = expr_data.assign_coords(
    supercluster=("ensembl_gene_id", super_labels.sel(ensembl_gene_id=genes).values),
    subcluster=("ensembl_gene_id", subcluster.values),
)

#annotation.to_dataframe().reset_index().to_csv(f"results/{source}_gene_cluster_annotation_{NORMALISATION}.csv", index=False)
annotation.to_netcdf(f"results/{source}_gene_cluster_annotation_{NORMALISATION}.nc")
