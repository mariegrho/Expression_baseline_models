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
    plt.show()
    plt.close()


# =====================================================
# 2. Plot clusters with variability (if membership provided)
# =====================================================

def plot_cluster_with_variance(trajectories_da, labels_da, n_clusters=None):

    data = trajectories_da.values
    labels = labels_da.values

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

def plot_membership_heatmap(membership_da, max_genes=1000):

    membership = membership_da.values
    membership = membership[:max_genes]

    dominant_cluster = np.argmax(membership, axis=1)
    order = np.argsort(dominant_cluster)
    membership_sorted = membership[order]

    plt.figure(figsize=(8, 10))
    plt.imshow(membership_sorted,aspect="auto",cmap="viridis")
    plt.colorbar(label="Membership strength")

    plt.xlabel("Cluster")
    plt.ylabel("Genes (subset)")
    plt.title("Fuzzy membership heatmap")

    plt.tight_layout()
    plt.savefig(f"figs/membership_heatmap.png")
    plt.show()
    plt.close()


# =====================================================
# 4. Plot representative genes per cluster
# =====================================================

def plot_representative_genes(trajectories_da, labels_da, cluster_id, top_n=10):

    data = trajectories_da.values
    labels = labels_da.values

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
    plt.show()
    plt.close()


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    t_end = 24
    centers = xr.open_dataarray(f"results/cluster_centers_{t_end}.nc")
    labels = xr.open_dataarray(f"results/hard_labels_{t_end}.nc")
    membership = xr.open_dataarray(f"results/fuzzy_membership_{t_end}.nc")
    n_clusters = np.unique(labels).size

    # 1. cluster prototypes
    plot_cluster_centers(centers)

    # 2. variability
    # (requires original trajectories file if you want full accuracy)
    trajectories = xr.open_dataarray(f"results/normalized_trajectories_{t_end}.nc")
    plot_cluster_with_variance(trajectories, labels)

    # 3. membership heatmap
    plot_membership_heatmap(membership)

    # 4. representative genes
    for c in range(n_clusters):
        plot_representative_genes(trajectories_da=xr.open_dataarray(f"results/normalized_trajectories_{t_end}.nc"), 
                                  labels_da=labels,cluster_id=c)