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

import numpy as np
import xarray as xr
import skfuzzy as fuzz
import matplotlib.pyplot as plt


# =====================================================
# CONFIG
# =====================================================

K_RANGE = range(2, 16)
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

    # scikit-fuzzy expects (features × samples)
    X = data.T

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X, c=k, 
        m=FUZZINESS, error=ERROR, maxiter=MAX_ITER, 
        init=None,seed=0)
    return cntr, u, fpc

# =====================================================
# Select optimal K using FPC
# =====================================================

def select_best_k(data, k_range):

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
    plt.show()

    return best_k


# =====================================================
# Run final clustering
# =====================================================

def fuzzy_cmeans_clustering(da, k_range=K_RANGE):

    data = da.values

    if N_CLUSTER == None:
        print("Selecting best K...")
        best_k = select_best_k(data, k_range)
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

def plot_clusters(centers_da):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    for k in centers_da.cluster.values:
        plt.plot(
            centers_da.time.values,
            centers_da.sel(cluster=k),
            label=f"Cluster {k}"
        )

    plt.xlabel("Time")
    plt.ylabel("Expression (normalized trajectory)")
    plt.title("Fuzzy cluster prototype trajectories")
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

    t_end = 120
    da = xr.open_dataarray(f"results/normalized_trajectories_{t_end}.nc")
    membership, labels, centers = cluster_dataset(da, f"results/{t_end}hpf", k_range=K_RANGE)

    plot_clusters(centers)