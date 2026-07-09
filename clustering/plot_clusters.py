"""
plot_clusters.py

Visualization utilities for fuzzy gene expression clustering.

Inputs:
    - cluster_centers.nc
    - fuzzy_membership.nc (optional)
    - normalized trajectories (optional for overlays)

Outputs:
    - cluster plots
    - optional per-cluster gene overlays
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# =====================================================
# 1. Plot cluster prototype trajectories
# =====================================================

def plot_cluster_centers(centers_da):

    plt.figure(figsize=(12, 6))

    for k in centers_da.cluster.values:

        plt.plot(
            centers_da.time.values,
            centers_da.sel(cluster=k).values,
            linewidth=2,
            label=f"Cluster {k}"
        )

    plt.xlabel("Time")
    plt.ylabel("Normalized expression")
    plt.title("Fuzzy cluster prototype trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/cluster_centers.png")
    #plt.show()
    plt.close()


# =====================================================
# 2. Plot clusters with variability (if membership provided)
# =====================================================

def plot_cluster_with_variance(trajectories_da, labels, n_clusters=None):

    data = trajectories_da.values

    if n_clusters is None:
        n_clusters = labels.max() + 1

    plt.figure(figsize=(12, 6))

    for k in range(n_clusters):

        idx = labels == k
        if idx.sum() == 0:
            continue

        cluster_data = data[idx]

        mean = cluster_data.mean(axis=0)
        std = cluster_data.std(axis=0)

        plt.plot(trajectories_da.time.values,mean,label=f"Cluster {k}")

        plt.fill_between(
            trajectories_da.time.values,
            mean - std,mean + std,
            alpha=0.2
        )

    plt.xlabel("Time")
    plt.ylabel("Expression")
    plt.title("Cluster mean trajectories with variability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/cluster_with_variance.png")
    plt.show()
    plt.close()


# =====================================================
# 3. Plot fuzzy membership heatmap
# =====================================================

def plot_membership_heatmap(membership_da, sc, max_genes=1000):

    membership = membership_da.values
    membership = membership[:max_genes]

    dominant_cluster = np.argmax(membership, axis=1)
    order = np.argsort(dominant_cluster)
    membership_sorted = membership[order]
    n_clusters = membership.shape[1]


    plt.figure(figsize=(8, 8))
    plt.imshow(membership_sorted,aspect="auto",cmap="viridis")
    plt.colorbar(label="Membership strength")

    plt.xticks(np.arange(n_clusters), labels=[f"Cluster {k}" for k in range(n_clusters)])
    plt.ylabel("Genes (subset)")
    plt.title(f"Fuzzy membership heatmap, Supercluster {sc}")

    plt.tight_layout()
    plt.savefig(f"figs/membership_heatmap_Supercluster_{sc}.png")
    #plt.show()
    plt.close()


# =====================================================
# 4. Plot representative genes per cluster
# =====================================================

def plot_representative_genes(trajectories_da, labels, cluster_id, top_n=50):

    data = trajectories_da.values
    idx = np.where(labels == cluster_id)[0]

    if len(idx) == 0:
        print(f"No genes found for cluster {cluster_id}")
        return

    plt.figure(figsize=(10, 5))

    for i in idx[:top_n]:
        plt.plot(trajectories_da.time.values,data[i],alpha=0.4)

    mean_curve = data[idx].mean(axis=0)

    plt.plot(
        trajectories_da.time.values,
        mean_curve,
        color="black",
        linewidth=3,
        label="Mean"
    )

    plt.xlabel("Time")
    plt.ylabel("Expression")
    plt.title(f"Cluster {cluster_id} representative genes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/representative_genes_Cluster_{cluster_id}.png")
    #plt.show()
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_super_subcluster_grid1(trajectories_da, super_labels, sub_labels, max_super=None, max_sub=None, figsize=None):

    data = trajectories_da.values
    t = trajectories_da.time.values

    super_ids = np.unique(super_labels)
    # group subclusters per supercluster
    groups = {sc: np.unique(sub_labels[(super_labels == sc) & (sub_labels >= 0)]) for sc in super_ids}

    # estimate layout
    nrows = len(super_ids)
    ncols = max(len(v) for v in groups.values())

    if figsize is None:
        figsize = (3 * ncols, 2.5 * nrows)

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize,sharex=True,sharey=True)
    axes = np.atleast_2d(axes)

    for i, sc in enumerate(super_ids):
        sub_ids = groups[sc]

        for j, sb in enumerate(sub_ids):

            if j >= ncols:
                break

            ax = axes[i, j]
            idx = np.where((super_labels == sc) &(sub_labels == sb))[0]
            if len(idx) == 0:
                ax.set_visible(False)
                continue

            subset = data[idx]
            mean = subset.mean(axis=0)

            # plot individual trajectories (limit for readability)
            for g in range(min(50, len(subset))):
                ax.plot(t, subset[g], alpha=0.3, linewidth=0.8)
            ax.plot(t, mean, "k", linewidth=2)
            ax.set_title(f"S{sc}.C{sb} ({len(subset)})", fontsize=9)

        # hide unused columns in this row
        for j in range(len(sub_ids), ncols):
            axes[i, j].set_visible(False)

    fig.suptitle("Supercluster → Subcluster trajectories", fontsize=12)
    plt.tight_layout()
    plt.savefig("figs/super_subcluster_grid.png", dpi=300)
    plt.show()
    plt.close()

def plot_super_subcluster_grid(trajectories_da,super_labels,sub_labels,
                                max_super=None,max_sub=None,figsize=(12, 7)):
    """
    Plot trajectories grouped by (supercluster, subcluster)
    in a single subplot grid.
    """

    data = trajectories_da.values
    t = trajectories_da.time.values

    super_ids = np.unique(super_labels)
    sub_ids = np.unique(sub_labels[sub_labels >= 0])

    if max_super is not None:
        super_ids = super_ids[:max_super]
    if max_sub is not None:
        sub_ids = sub_ids[:max_sub]

    nrows = len(super_ids)
    ncols = len(sub_ids)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=True
    )

    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    for i, sc in enumerate(super_ids):
        for j, sb in enumerate(sub_ids):

            ax = axes[i, j]
            idx = np.where((super_labels == sc) &(sub_labels == sb))[0]

            if len(idx) == 0:
                ax.set_visible(False)
                continue

            subset = data[idx]
            mean = subset.mean(axis=0)

            # representative genes (up to 20)
            for g in idx[:20]:
                ax.plot(t, subset[g],alpha=0.4)

            ax.plot(t, mean, "k", linewidth=2 )
            ax.set_title(f"{sc}.{sb}", fontsize=9)

    fig.suptitle("Supercluster × Subcluster trajectories", fontsize=12)
    plt.tight_layout()
    plt.savefig("figs/super_subcluster_grid.png")
    plt.show()
    plt.close()


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    NORMALISATION = "minmax"
    t_end = 120
    #data = "White"
    data = "all"

    labels = xr.load_dataset(f"results/{data}_gene_cluster_annotation_{NORMALISATION}.nc")
    super_labels = labels.supercluster.values
    sub_labels = labels.subcluster.values

    trajectories_da = xr.load_dataarray(f"results/{data}_normalized_trajectories_{t_end}_{NORMALISATION}.nc")
    trajectories_da = trajectories_da.sel(ensembl_gene_id=labels.ensembl_gene_id)


    # Plot super × sub cluster grid
    plot_super_subcluster_grid1(trajectories_da,super_labels,sub_labels)
    
    plot_cluster_centers(xr.load_dataarray(f"results/{data}_superclusters_centers.nc"))
    plot_cluster_with_variance(trajectories_da, super_labels,n_clusters=None)

    for cluster_id in np.unique(super_labels):
        plot_representative_genes(trajectories_da, super_labels, cluster_id, top_n=50)
    
    membership_da = xr.load_dataarray(f"results/{data}_superclusters_membership.nc")
    plot_membership_heatmap(membership_da, "All", max_genes=500)
    
    for sc in np.unique(super_labels):
        membership_da = xr.load_dataarray(f"results/{data}_supercluster_{sc}_membership.nc")
        plot_membership_heatmap(membership_da, sc, max_genes=500)