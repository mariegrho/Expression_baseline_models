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

import os
from itertools import combinations

import numpy as np
import xarray as xr
import skfuzzy as fuzz
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except ImportError:
    _HAVE_JOBLIB = False

# =====================================================
# CONFIG
# =====================================================

K_RANGE = range(2, 10)
FUZZINESS = 1.5   # m parameter: higher -> fuzzier cluster (1.5-2.5 typical)
MAX_ITER = 500
ERROR = 1e-4
N_CLUSTER = None

RNG_SEED = 1  # top-level seed; all randomness below is derived from this,
              # deterministically, via numpy SeedSequence spawning -- so
              # results are reproducible even when K's are evaluated in
              # parallel (see select_best_k_stability).

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
        init=None, seed=seed)
    return cntr, u, fpc


def _project_full(data, cntr, seed=None):
    """Project a fitted set of cluster centers onto the full dataset."""
    u_full, *_ = fuzz.cluster.cmeans_predict(
        data, cntr, m=FUZZINESS, error=ERROR, maxiter=MAX_ITER, seed=seed)
    return np.argmax(u_full, axis=0)


def _shuffle_trajectories(data, seed_seq):
    """
    Build a null-model dataset for the stability baseline: independently
    permute the time-order of each gene's trajectory. This destroys the
    trajectory *shape* while preserving each gene's own marginal distribution of values.
    A genuinely informative K should be far more stable than this.
    """
    local_rng = np.random.default_rng(seed_seq)
    shuffled = data.copy()
    n_time, n_genes = data.shape
    for t in range(n_time):
        shuffled[t, :] = data[t, local_rng.permutation(n_genes)]
    return shuffled


# =====================================================
# Select optimal K -- stability-based (improved)
# =====================================================

def stability_score(data, k, seed_seq, n_runs=20, sample_frac=0.7):
    """
    Estimate clustering stability for a given K via bootstrap resampling.
    Also returns the Fuzzy Partition Coefficient (FPC) for each run as a
    complementary diagnostic.
    seed_seq: a numpy.random.SeedSequence
    """
    n_genes = data.shape[1]
    labels_per_run = []
    fpcs = []

    child_seeds = seed_seq.spawn(n_runs)
    for child in child_seeds:
        local_rng = np.random.default_rng(child)
        fit_seed = int(local_rng.integers(0, 1_000_000))
        resample_seed = int(local_rng.integers(0, 1_000_000))

        idx = local_rng.choice(n_genes, size=int(n_genes), replace=True) # Bootstrap sampling -> with replacement
        #idx = local_rng.choice(n_genes, size=int(n_genes*sample_frac), replace=False) # subsampling  -> without replacement, only fraction

        cntr, _, fpc = run_fuzzy_cmeans(data[:, idx], k, seed=fit_seed)
        fpcs.append(fpc)
        labels_per_run.append(_project_full(data, cntr, seed=fit_seed))

    pair_scores = [
        adjusted_rand_score(labels_per_run[i], labels_per_run[j])
        for i, j in combinations(range(n_runs), 2)
    ]

    return {
        "mean": float(np.mean(pair_scores)),
        "std": float(np.std(pair_scores)),
        "scores": pair_scores,
        "fpc_mean": float(np.mean(fpcs)),
        "fpc_std": float(np.std(fpcs)),
    }


def _plot_k_selection(ks, means, stds, null_means, fpc_scores, xb_scores,
                       best_k, dataset_name):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    ax = axes[0]
    ax.errorbar(ks, means, yerr=stds, marker="o", linestyle="--", color="k",
                capsize=5, label="stability (bootstrap ARI)")
    if not np.all(np.isnan(null_means)):
        ax.plot(ks, null_means, marker="x", linestyle=":", color="gray",
                label="null baseline (shuffled trajectories)")
    ax.axvline(best_k, color="red", alpha=0.3, linestyle="-", linewidth=6,
               label=f"selected K={best_k}")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Stability (ARI)")
    ax.set_title("Bootstrap stability")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(ks, fpc_scores, marker="o", color="darkgreen")
    ax.axvline(best_k, color="red", alpha=0.3, linewidth=6)
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("FPC (full-data fit)")
    ax.set_title("Fuzzy Partition Coefficient")

    ax = axes[2]
    ax.plot(ks, xb_scores, marker="o", color="darkred")
    ax.axvline(best_k, color="red", alpha=0.3, linewidth=6)
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Xie-Beni (lower is better)")
    ax.set_title("Xie-Beni index")

    fig.suptitle(f"Fuzzy clustering model selection ({dataset_name})")
    fig.tight_layout()
    fig.savefig(f"figs/k_selection_stability_{dataset_name}.png", dpi=300)
    plt.close(fig)


def select_best_k_stability(data, k_range, dataset_name, n_runs=15, sample_frac=0.7,
                             use_one_se_rule=False, compute_null=False, n_jobs=1, master_seed=RNG_SEED):
    """
    Choose K via bootstrap clustering stability.

    Returns
    -------
    best_k : int
    diagnostics : dict mapping k -> {"stability": ..., "null": ..., "fpc": ..., "xb": ...}
    """
    ks = list(k_range)
    root_seq = np.random.SeedSequence(master_seed)
    k_seed_seqs = root_seq.spawn(len(ks))

    def _worker(k, seed_seq):
        real_seq, null_seq = seed_seq.spawn(2)
        real = stability_score(data, k, real_seq, n_runs=n_runs, sample_frac=sample_frac)

        null = None
        if compute_null:
            shuffle_seq, null_fit_seq = null_seq.spawn(2)
            null_data = _shuffle_trajectories(data, shuffle_seq)
            null = stability_score(null_data, k, null_fit_seq,
                                    n_runs=n_runs, sample_frac=sample_frac)

        fit_seed = int(seed_seq.generate_state(1)[0])
        cntr, u, fpc_full = run_fuzzy_cmeans(data, k, seed=fit_seed)
        xb = xie_beni_index(data, cntr, u, FUZZINESS)

        return k, real, null, xb, fpc_full

    if n_jobs != 1 and _HAVE_JOBLIB:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_worker)(k, seed_seq) for k, seed_seq in zip(ks, k_seed_seqs))
    else:
        results = [_worker(k, seed_seq) for k, seed_seq in zip(ks, k_seed_seqs)]

    means = np.array([r[1]["mean"] for r in results])
    stds = np.array([r[1]["std"] for r in results])
    n_pairs = len(results[0][1]["scores"])
    null_means = np.array([r[2]["mean"] if r[2] is not None else np.nan for r in results])
    xb_scores = np.array([r[3] for r in results])
    fpc_scores = np.array([r[4] for r in results])

    diagnostics = {}
    for k, real, null, xb, fpc_full in results:
        diagnostics[k] = {"stability": real, "null": null, "fpc": fpc_full, "xb": xb}
        null_str = f", null={null['mean']:.4f}" if null is not None else ""
        print(f"K={k} | stability={real['mean']:.4f} +/- {real['std']:.4f}, {null_str} | FPC={fpc_full:.4f} | XB={xb:.4f}")

    best_idx = int(np.argmax(means))
    if use_one_se_rule:
        se_best = stds[best_idx] / np.sqrt(max(n_pairs, 1))
        threshold = means[best_idx] - se_best
        candidates = [i for i in range(len(ks)) if means[i] >= threshold]
        chosen_idx = min(candidates)  # smallest K within 1 SE of the best -> simplest adequate model
    else:
        chosen_idx = best_idx

    #select = 0.7 *(1-means) + 0.3 * xb_scores
    #chosen_idx = np.argmin(select)
    best_k = ks[chosen_idx]
    #best_k = xb_scores[np.argmin(xb_scores)]

    print("best by stability score :", ks[best_idx])
    print("best by XB index:", ks[np.argmin(xb_scores)]) # best k by XB index

    _plot_k_selection(ks, means, stds, null_means, fpc_scores, xb_scores,
                       best_k, dataset_name)

    return best_k, diagnostics


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

    # fuzzy weighted compactness
    numerator = np.sum((u ** m) * dist)
    # cluster separation
    center_dist = np.linalg.norm(cntr[:, None, :] - cntr[None, :, :], axis=2) ** 2

    np.fill_diagonal(center_dist, np.inf)
    min_dist = np.min(center_dist)

    return numerator / (n_samples * min_dist)


# =====================================================
# Run final clustering
# =====================================================

def fuzzy_cmeans_clustering(da, dataset_name, k_range=K_RANGE):

    data = da.values.T

    if N_CLUSTER is None:
        print("Selecting best K...")
        best_k, _ = select_best_k_stability(data, k_range, dataset_name)
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
        coords={ "ensembl_gene_id": genes, "cluster": np.arange(best_k) },
        name="membership"
    )

    labels_da = xr.DataArray(
        labels, dims=("ensembl_gene_id",),
        coords={"ensembl_gene_id": genes},
        name="cluster_label"
    )

    centers_da = xr.DataArray(
        centers, dims=("cluster", "time"),
        coords={ "cluster": np.arange(best_k), "time": da.time.values }, 
        name="cluster_centers"
    )
    return membership_da, labels_da, centers_da, best_k, fpc


# =====================================================
# Plot cluster trajectories
# =====================================================

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


def cluster_dataset(da, output_prefix, dataset_name="dataset", k_range=K_RANGE):

    membership, labels, centers, best_k, fpc = fuzzy_cmeans_clustering(da, dataset_name, k_range)

    membership.to_netcdf(f"{output_prefix}_membership.nc")
    labels.to_netcdf(f"{output_prefix}_labels.nc")
    centers.to_netcdf(f"{output_prefix}_centers.nc")

    print(f"{output_prefix}: K={best_k}, FPC={fpc:.3f}")

    return membership, labels, centers


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    t_end = 120
    da = xr.load_dataarray(f"results/normalized_trajectories_{t_end}.nc")
    membership, labels, centers = cluster_dataset(
        da, f"results/{t_end}hpf", dataset_name=f"{t_end}hpf", k_range=K_RANGE)

    plot_clusters(centers, "All")
