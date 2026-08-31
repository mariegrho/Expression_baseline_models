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

from FCM_cluster_selection import plot_k_selection, xie_beni_index, plot_clusters


try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except ImportError:
    _HAVE_JOBLIB = False

# =====================================================
# CONFIG
# =====================================================

K_RANGE = range(2, 10)
FUZZINESS = 1.6  # m parameter: higher -> fuzzier cluster (1.5-2.5 typical) m=1: hard cluster assignment
MAX_ITER = 500
ERROR = 1e-3
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

def _project_full(data, cntr):
    """
    Superschnelle, vektorisierte Zuweisung der dominanten Cluster-Labels.
    Ersetzt cmeans_predict für den ARI-Check.
    data: (n_features, n_genes) -> (6, 30000)
    cntr: (k, n_features) -> (k, 6)
    """
    # Berechne den euklidischen Abstand jedes Gens zu jedem Clusterzentrum (k)
    # Shape von dist: (k, n_genes)
    dist = np.linalg.norm(data[np.newaxis, :, :] - cntr[:, :, np.newaxis], axis=1)
    # Der kleinste Abstand liefert das harte Cluster-Label
    return np.argmin(dist, axis=0)

# def _project_full(data, cntr, seed=None):
#     """Project a fitted set of cluster centers onto the full dataset."""
#     u_full, *_ = fuzz.cluster.cmeans_predict(
#         data, cntr, m=FUZZINESS, error=ERROR, maxiter=MAX_ITER, seed=seed)
#     return np.argmax(u_full, axis=0)


def _shuffle_trajectories(data, seed_seq):
    """
    Build a null-model dataset for the stability baseline: independently
    permute the time-order of each gene's trajectory. This destroys the
    trajectory *shape*. A genuinely informative K should be far more stable than this.
    """
    local_rng = np.random.default_rng(seed_seq)
    shuffled = data.copy()
    n_time, n_genes = data.shape
    for t in range(n_time):
        shuffled[t, :] = data[t, local_rng.permutation(n_genes)]
    return shuffled


# =====================================================
# Select optimal K 
# =====================================================

def stability_score(data, k, seed_seq, n_runs=15, sample_frac=0.7):
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

        #idx = local_rng.choice(n_genes, size=int(n_genes), replace=True) # Bootstrap sampling -> with replacement
        idx = local_rng.choice(n_genes, size=int(n_genes*sample_frac), replace=False) # subsampling  -> without replacement, only fraction

        cntr, _, fpc = run_fuzzy_cmeans(data[:, idx], k, seed=fit_seed)
        fpcs.append(fpc)

        labels_per_run.append(_project_full(data, cntr))

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


def select_best_k_stability(data, k_range, dataset_name, n_runs=15, sample_frac=0.7,
                             use_one_se_rule=False, compute_null=False, n_jobs=-1, master_seed=RNG_SEED):
    """
    Choose K via clustering stability.
    Calculates validation metrics (XB, FCP)

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
        return k, real, null,

    if n_jobs != 1 and _HAVE_JOBLIB:
        worker_results = Parallel(n_jobs=n_jobs)(
            delayed(_worker)(k, seed_seq) for k, seed_seq in zip(ks, k_seed_seqs))
    else:
        worker_results = [_worker(k, seed_seq) for k, seed_seq in zip(ks, k_seed_seqs)]

    # Nachbereitung: XB und FPC einmalig und sauber auf dem Hauptprozess berechnen
    means = []
    stds = []
    null_means = []
    xb_scores = []
    fpc_scores = []
    diagnostics = {}

    # Generiere feste Seeds für den finalen Fit pro K
    master_rng = np.random.default_rng(master_seed)
    final_seeds = master_rng.integers(0, 1_000_000, size=len(ks))

    for idx, (k, real, null) in enumerate(worker_results):
        # Einmaliger, exakter Fit auf den vollen Daten
        cntr, u, fpc_full = run_fuzzy_cmeans(data, k, seed=int(final_seeds[idx]))
        xb = xie_beni_index(data, cntr, u, FUZZINESS)

        means.append(real["mean"])
        stds.append(real["std"])
        null_means.append(null["mean"] if null is not None else np.nan)
        xb_scores.append(xb)
        fpc_scores.append(fpc_full)

        diagnostics[k] = {"stability": real, "null": null, "fpc": fpc_full, "xb": xb}
        
        null_str = f", null={null['mean']:.4f}" if null is not None else ""
        print(f"K={k} | stability={real['mean']:.4f} +/- {real['std']:.4f}"
              f"{null_str} | FPC={fpc_full:.4f} | XB={xb:.4f}")

    means = np.array(means)
    stds = np.array(stds)
    null_means = np.array(null_means)
    xb_scores = np.array(xb_scores)
    fpc_scores = np.array(fpc_scores)
    n_pairs = len(worker_results[0][1]["scores"])

    # Bestimmung des besten K (One-Standard-Error-Rule oder Max-ARI)
    best_idx = int(np.argmax(means))
    if use_one_se_rule:
        se_best = stds[best_idx] / np.sqrt(max(n_pairs, 1))
        threshold = means[best_idx] - se_best
        candidates = [i for i in range(len(ks)) if means[i] >= threshold]
        chosen_idx = min(candidates)
    else:
        chosen_idx = best_idx

    #best_k = ks[chosen_idx]
    best_k = ks[np.argmin(xb_scores)]

    print("-> Best by stability score:", ks[best_idx])
    print("-> Best by XB index:", ks[np.argmin(xb_scores)])



    plot_k_selection(ks, means, stds, fpc_scores, xb_scores, best_k, dataset_name+"_traj")

    return best_k, diagnostics


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
    da = xr.load_dataarray(f"results/all_normalized_trajectories_{t_end}_minmax.nc")
    membership, labels, centers = cluster_dataset(
        da, f"results/{t_end}hpf", dataset_name=f"{t_end}hpf", k_range=K_RANGE)

    plot_clusters(centers, "All")
