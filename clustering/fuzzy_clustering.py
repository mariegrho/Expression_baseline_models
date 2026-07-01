"""
cluster_fuzzy.py

Fuzzy clustering of gene expression trajectories.

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


# =====================================================
# CONFIG
# =====================================================

K_RANGE = range(2, 10)
FUZZINESS = 1.8   # m parameter (1.5–2.5 typical)
MAX_ITER = 300
ERROR = 1e-5
N_CLUSTER = None


# =====================================================
# Fuzzy clustering wrapper
# =====================================================

def run_fuzzy_cmeans(data, k):
    """
    data shape: (n_genes, n_timepoints)
    """

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, c=k, 
        m=FUZZINESS, error=ERROR, maxiter=MAX_ITER, 
        init=None,seed=0)
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

    for _ in range(n_runs):

        idx = resample(
            np.arange(data.shape[0]),
            replace=False,
            n_samples=int(data.shape[0] * sample_frac)
        )

        sub = data[idx, :]

        cntr, u, _ = run_fuzzy_cmeans(sub, k)
        labels = np.argmax(u, axis=0)

        # second run for comparison
        idx2 = resample(
            np.arange(data.shape[0]),
            replace=False,
            n_samples=int(data.shape[0] * sample_frac)
        )

        sub2 = data[idx2, :]
        cntr2, u2, _ = run_fuzzy_cmeans(sub2, k)
        labels2 = np.argmax(u2, axis=0)

        # overlap genes between subsets
        common = np.intersect1d(idx, idx2)

        if len(common) < 5:
            continue

        map1 = np.searchsorted(idx, common)
        map2 = np.searchsorted(idx2, common)

        scores.append(adjusted_rand_score(labels[map1], labels2[map2]))

    return np.mean(scores)


def select_best_k_stability(data, k_range):
    best_k = None
    best_score = -np.inf

    for k in k_range:
        score = stability_score(data, k)

        print(f"K={k} | stability={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

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

def fuzzy_cmeans_clustering(da, k_range=K_RANGE):

    data = da.values.T

    if N_CLUSTER == None:
        print("Selecting best K...")
        #best_k = select_best_k_XB(data, k_range)
        #best_k = select_best_k_fpc(data, k_range)
        best_k = select_best_k_stability(data, k_range)
    else:
        best_k = N_CLUSTER
    print(f"Number of Clusters: {best_k}")
    centers, membership, fpc = run_fuzzy_cmeans(data, best_k)

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
    plt.show()


def cluster_dataset(da, output_prefix, k_range=K_RANGE):

    membership, labels, centers, best_k, fpc = fuzzy_cmeans_clustering(da, k_range)

    membership.to_netcdf(f"{output_prefix}_membership.nc")
    labels.to_netcdf(f"{output_prefix}_labels.nc")
    centers.to_netcdf(f"{output_prefix}_centers.nc")

    print(f"{output_prefix}: K={best_k}, FPC={fpc:.3f}")

    return membership, labels, centers


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    t_end = 24
    da = xr.open_dataarray(f"results/normalized_trajectories_{t_end}.nc")
    membership, labels, centers = cluster_dataset(da, f"results/{t_end}hpf", k_range=K_RANGE)

    plot_clusters(centers)