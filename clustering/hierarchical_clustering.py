import xarray as xr
import numpy as np
import pandas as pd
from glob import glob
import fuzzy_clustering as fc

# Hierarchical fuzzy c-means clustering of gene trajectories to indentify subclusters within superclusters with focous on early dynamics.

traj120 = xr.open_dataarray("results/normalized_trajectories_120.nc")
membership120, labels120, centers120 = fc.cluster_dataset(traj120, "results/superclusters", k_range=range(2, 7))
fc.plot_clusters(centers120, "Superclusters")

traj24 = xr.open_dataarray("results/normalized_trajectories_12.nc")

common_genes = list(set(traj120.ensembl_gene_id.values) & set(traj24.ensembl_gene_id.values))
labels120 = labels120.sel(ensembl_gene_id=common_genes)

for sc in np.unique(labels120.values):
    genes = labels120.ensembl_gene_id.where(labels120 == sc,drop=True)
    subtraj = traj24.sel(ensembl_gene_id=genes)
    membership, labels, centers =fc.cluster_dataset(subtraj,f"results/supercluster_{sc}", k_range=range(2, 4))
    fc.plot_clusters(centers, cluster=f"Supercluster {sc}")


### DataSet

super_labels = xr.open_dataarray("results/superclusters_labels.nc")
genes = super_labels.ensembl_gene_id.values

# initialize subcluster labels
subcluster = xr.DataArray(
    np.full(len(genes), -1, dtype=int),
    dims=("ensembl_gene_id",),
    coords={"ensembl_gene_id": genes},
    name="subcluster"
)

for sc in np.unique(super_labels.values):
    labels = xr.open_dataarray(f"results/supercluster_{sc}_labels.nc")
    subcluster.loc[dict(ensembl_gene_id=labels.ensembl_gene_id)] = labels

annotation = xr.Dataset(data_vars=dict(supercluster=super_labels,subcluster=subcluster))
annotation.to_netcdf("results/gene_cluster_annotation.nc")

## Dataframe 
super_labels = xr.load_dataarray("results/superclusters_labels.nc")
df = (
    super_labels.to_series()
    .to_frame("supercluster")
    .join(
        pd.concat(
            [xr.open_dataarray(f).to_series() for f in glob("results/supercluster_*_labels.nc")]
        ).rename("subcluster"),
        how="left"
    )
    .fillna({"subcluster": -1})
    .astype({"subcluster": int})
)
df.to_csv("results/gene_cluster_annotation.csv")
