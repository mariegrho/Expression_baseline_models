''' Selection metrics used to determine the optimal number of clusters c for fuzzy c-means
    - Xie-and-Beni index (XB)
    - Stability score (ARI)
    - Fuzzy Partitioning Coefficient (FCP)
    '''


import os
from itertools import combinations

import numpy as np
import xarray as xr
import skfuzzy as fuzz
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score


def plot_k_selection(ks, means, stds, fpc_scores, xb_scores, best_k, dataset_name):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    ax = axes[1]
    ax.errorbar(ks, means, yerr=stds, marker="o", linestyle="--", color="k", capsize=5 )
    ax.axvline(best_k, color="red", alpha=0.3, linestyle="-", linewidth=6, )
    ax.set_xlabel("Number of clusters (c)")
    ax.set_ylabel("ARI (1 is best)")
    ax.set_title("Cluster stability")
    ax = axes[2]
    ax.plot(ks, fpc_scores, marker="o", color="darkgreen")
    ax.plot(ks, np.ones(len(ks))/ks, color="grey", alpha=0.5)
    ax.axvline(best_k, color="red", alpha=0.3, linewidth=6)
    ax.set_xlabel("Number of clusters (c)")
    ax.set_ylabel("FPC (1 is best)")
    ax.set_title("Fuzzy Partition Coefficient")

    ax = axes[0]
    ax.plot(ks, xb_scores, marker="o", color="darkred")
    ax.axvline(best_k, color="red", alpha=0.3, linewidth=6, label=f"selected c={best_k}")
    ax.set_xlabel("Number of clusters (c)")
    ax.set_ylabel("XB-index (lower is better)")
    ax.set_title("Xie-Beni index")
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"Fuzzy clustering model selection ({dataset_name})")
    fig.tight_layout()
    fig.savefig(f"figs/k_selection_stability_{dataset_name}.png", dpi=300)
    plt.close(fig)


def xie_beni_index(X, cntr, u, m):
    """
    XB Index: S = J / (n * d_min²) 
    with d_min² = spearation, n = no. samples, J_2 = compactness

    The optimal number of clusters k is is such that the index takes the minimum value

    X: Data (features, samples)
    cntr: Cluster centers (clusters, features)
    u: Membership Degree (clusters, samples)
    """
    n_samples = X.shape[1]

    diff = cntr[:, :, None] - X[None, :, :]   # (k, features, n_samples)
    dist = np.sum(diff ** 2, axis=1)          # (k, n_samples)

    compactness = np.sum((u ** m) * dist)
    # cluster separation
    center_dist = np.linalg.norm(cntr[:, None, :] - cntr[None, :, :], axis=2) ** 2
    np.fill_diagonal(center_dist, np.inf)
    min_dist = np.min(center_dist)

    xb = compactness / (n_samples * min_dist)

    return xb

def plot_clusters(centers_da, cluster):

    plt.figure(figsize=(8, 4))

    for k in centers_da.cluster.values:
        plt.plot( centers_da.time.values,
            centers_da.sel(cluster=k), label=f"Cluster {k}" )

    plt.xlabel("Time")
    plt.ylabel("Expression (normalized trajectory)")
    plt.title(f"Fuzzy c-means cluster center trajectories ({cluster})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/FCM_cluster_centers_{cluster}.png")
    plt.close()


# =====================================================
# Plot cluster centers
# =====================================================

def plot_cluster_centers(centers_std_da, dataset_name):
    """
    Heatmap of standardized cluster centers: one row per cluster, one
    column per parameter. Replaces the trajectory line plot from the
    time-series version, since there's no time axis here.
    """

    import seaborn as sns

    # centers_std_da is already a DataArray, not a Dataset
    da = centers_std_da  # dims: (cluster, feature)
    k = centers_std_da.sizes["cluster"]
    features = centers_std_da.feature.values

    fig, ax = plt.subplots(figsize=(1.1 * len(features) + 2, 0.6 * k + 2))
    sns.heatmap(data=centers_std_da, cmap="RdBu_r", annot=True)

    ax.set_xticklabels(features, rotation=0, ha="right")
    ax.set_yticklabels([f"Cluster {c}" for c in range(k)], rotation=0)

    ax.set_title(f"Fuzzy c-means - parameter cluster ({dataset_name})")
    fig.tight_layout()
    fig.savefig(f"figs/FCM_param_cluster_center_{dataset_name}.png", dpi=300)
    plt.close()

