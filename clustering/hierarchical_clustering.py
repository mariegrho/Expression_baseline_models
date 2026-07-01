import xarray as xr
import numpy as np

import fuzzy_clustering as fc

'''
Hierarchical fuzzy c-means clustering of gene trajectories to indentify subclusters within superclusters with focous on early dynamics.
'''

traj120 = xr.open_dataarray("results/normalized_trajectories_120.nc")
membership120, labels120, centers120 = fc.cluster_dataset(traj120, "results/superclusters")

traj24 = xr.open_dataarray("results/normalized_trajectories_24.nc")

for sc in labels120.cluster_label.values:
    genes = labels120.ensembl_gene_id.where(labels120 == sc,drop=True)
    subtraj = traj24.sel(ensembl_gene_id=genes)
    fc.cluster_dataset(subtraj,f"results/supercluster_{sc}")



# ---------------------------------------
# Load supercluster labels
# ---------------------------------------

super_labels = xr.open_dataarray("results/superclusters_labels.nc")

genes = super_labels.ensembl_gene_id.values

# initialize subcluster labels
subcluster = xr.DataArray(
    np.full(len(genes), -1, dtype=int),
    dims=("ensembl_gene_id",),
    coords={"ensembl_gene_id": genes},
    name="subcluster"
)

# ---------------------------------------
# Fill subcluster labels
# ---------------------------------------

for sc in np.unique(super_labels.values):

    labels = xr.open_dataarray(f"results/supercluster_{sc}_labels.nc")

    subcluster.loc[
        dict(ensembl_gene_id=labels.ensembl_gene_id)
    ] = labels

# ---------------------------------------
# Combine into one Dataset
# ---------------------------------------

annotation = xr.Dataset(
    data_vars=dict(
        supercluster=super_labels,
        subcluster=subcluster
    )
)

annotation.to_netcdf("results/gene_cluster_annotation.nc")