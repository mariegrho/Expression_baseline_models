"""
cluster_fuzzy.py

Fuzzy c-means clustering of gene expression trajectories.

Input:
    normalized_trajectories.nc
        dims: (ensembl_gene_id × time)

Output:
    - membership matrix (U)
    - hard cluster labels
    - cluster centers (prototype trajectories)
"""

# GLOBAL CONVENTION
# X = (genes, time)

import numpy as np
import xarray as xr
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import os

# =====================================================
# CONFIG
# =====================================================

K_RANGE = range(2, 5)
FUZZINESS = 1.8   # m parameter (1.5–2.5 typical)
MAX_ITER = 300
ERROR = 1e-5
N_CLUSTER = None

RNG_SEED = 1

rng = np.random.default_rng(seed=1)

# =====================================================
# Fuzzy clustering wrapper
# =====================================================

def run_fuzzy_cmeans(data, k, seed=None):
    """
    data shape: (n_timepoints, n_genes)
    """

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, c=k, 
        m=FUZZINESS, error=ERROR, maxiter=MAX_ITER, 
        init=None,seed=seed)
    return cntr, u, fpc



# =====================================================
# Select optimal K 
# =====================================================

from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score
import numpy as np

def run_once(data, k):
    cntr, u, _ = run_fuzzy_cmeans(data, k)
    labels = np.argmax(u, axis=0)
    return labels


from sklearn.metrics import adjusted_rand_score
import numpy as np
from sklearn.utils import resample

def stability_score(data, k, n_runs=20, sample_frac=0.8):

    scores = []
    n_genes = data.shape[1]

    for _ in range(n_runs):

        seed1 = rng.integers(0, 1_000_000)
        seed2 = rng.integers(0, 1_000_000)
        resample_seed1 = rng.integers(0, 1_000_000)
        resample_seed2 = rng.integers(0, 1_000_000)

        idx = resample(np.arange(n_genes),replace=True,n_samples=int(n_genes * sample_frac), random_state=resample_seed1)

        sub = data[:, idx]
        cntr1, u1, _ = run_fuzzy_cmeans(sub, k, seed1)

        # assign FULL dataset
        u_full1, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            data, cntr1, m=FUZZINESS, error=ERROR, maxiter=MAX_ITER)
        labels1 = np.argmax(u_full1, axis=0)

        # second run for comparison
        idx2 = resample(np.arange(n_genes),replace=True,n_samples=int(n_genes * sample_frac), random_state=resample_seed2)

        sub2 = data[:, idx2]
        cntr2, u2, _ = run_fuzzy_cmeans(sub2, k, seed2)

        # assign FULL dataset
        u_full2, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            data, cntr2, m=FUZZINESS, error=ERROR, maxiter=MAX_ITER)
        labels2 = np.argmax(u_full2, axis=0)

        scores.append(adjusted_rand_score(labels1, labels2))

    return {"mean": np.mean(scores),
            "std": np.std(scores),
            "scores": scores}


def select_best_k_stability(data, k_range, dataset_name):
    best_k = None
    best_score = -np.inf

    means = []
    stds = []

    for k in k_range:
        scores = stability_score(data, k)
        print(f"K={k} | stability={scores['mean']:.4f}")

        means.append(scores["mean"])
        stds.append(scores["std"])

        if scores['mean'] > best_score:
            best_score = scores['mean']
            best_k = k

    plt.figure(figsize=(6, 3))
    plt.errorbar(list(k_range), means, yerr=stds, marker="o", linestyle="--", capsize=5)
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Stability Score")
    plt.title("Fuzzy clustering model selection")
    plt.tight_layout()
    plt.savefig(f"figs/k_selection_stability_{dataset_name}.png")
    #plt.show()
    plt.close()

    return best_k


def xie_beni_index(X, cntr, u, m):
    """
    X: (features, samples)
    cntr: (clusters, features)
    u: (clusters, samples)
    """

    n_samples = X.shape[1]

    diff = cntr[:, :, None] - X[None, :, :]   # (k, features, n_samples)
    dist = np.sum(diff ** 2, axis=1)          # (k, n_samples)

    # fuzzy weighted compactness
    numerator = np.sum((u ** m) * dist)
    # cluster separation
    center_dist = np.linalg.norm(cntr[:, None, :] - cntr[None, :, :],axis=2) ** 2

    np.fill_diagonal(center_dist, np.inf)
    min_dist = np.min(center_dist)

    return numerator / (n_samples * min_dist)


def select_best_k_XB(data, k_range):

    X = data
    best_score = np.inf

    for k in k_range:
        cntr, u, fpc = run_fuzzy_cmeans(X, k)
        xb = xie_beni_index(X, cntr, u, FUZZINESS)

        print(f"K={k} | XB={xb:.4f}")

        if xb < best_score:
            best_score = xb
            best_k = k
    
    return best_k


def select_best_k_fpc(data, k_range):

    best_k = None
    best_fpc = -np.inf
    results = []

    for k in k_range:
        cntr, u, fpc = run_fuzzy_cmeans(data, k)
        results.append((k, fpc))
        print(f"K={k} | FPC={fpc:.4f}")

        if fpc > best_fpc:
            best_fpc = fpc
            best_k = k

    # plot selection curve
    ks, fpcs = zip(*results)

    plt.plot(ks, fpcs, marker='o')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Fuzzy Partition Coefficient (FPC)")
    plt.title("Fuzzy clustering model selection")
    plt.close()

    return best_k


# =====================================================
# Run final clustering
# =====================================================

def fuzzy_cmeans_clustering(da, dataset_name, k_range=K_RANGE):

    data = da.values.T

    if N_CLUSTER == None:
        print("Selecting best K...")
        #best_k = select_best_k_XB(data, k_range)
        #best_k = select_best_k_fpc(data, k_range)
        best_k = select_best_k_stability(data, k_range, dataset_name)
    else:
        best_k = N_CLUSTER
    print(f"Number of Clusters: {best_k}")
    centers, membership, fpc = run_fuzzy_cmeans(data, best_k, seed=1)

    labels = np.argmax(membership, axis=0)

    # -------------------------------------------------
    # package outputs
    # -------------------------------------------------

    genes = da.ensembl_gene_id.values

    membership_da = xr.DataArray(
        membership.T,
        dims=("ensembl_gene_id", "cluster"),
        coords={
            "ensembl_gene_id": genes,
            "cluster": np.arange(best_k)
        },
        name="membership"
    )

    labels_da = xr.DataArray(
        labels,
        dims=("ensembl_gene_id",),
        coords={"ensembl_gene_id": genes},
        name="cluster_label"
    )

    centers_da = xr.DataArray(
        centers,
        dims=("cluster", "time"),
        coords={
            "cluster": np.arange(best_k),
            "time": da.time.values
        },
        name="cluster_centers"
    )
    return membership_da, labels_da, centers_da, best_k, fpc


# =====================================================
# Plot cluster trajectories
# =====================================================

def plot_clusters(centers_da, cluster):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))

    for k in centers_da.cluster.values:
        plt.plot(
            centers_da.time.values,
            centers_da.sel(cluster=k),
            label=f"Cluster {k}"
        )

    plt.xlabel("Time")
    plt.ylabel("Expression (normalized trajectory)")
    plt.title(f"Fuzzy c-means cluster center trajectories ({cluster})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/FCM_cluster_centers_{cluster}.png")
    plt.close()


def cluster_dataset(da, output_prefix, dataset_name, k_range=K_RANGE):

    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    membership, labels, centers, best_k, fpc = fuzzy_cmeans_clustering(da, dataset_name, k_range)

    membership.to_netcdf(f"{output_prefix}_membership.nc")
    labels.to_netcdf(f"{output_prefix}_labels.nc")
    centers.to_netcdf(f"{output_prefix}_centers.nc")

    print(f"{output_prefix}: K={best_k}, FPC={fpc:.3f}")

    return membership, labels, centers

