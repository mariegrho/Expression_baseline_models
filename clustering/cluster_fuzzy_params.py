"""
cluster_fuzzy_params.py

Fuzzy c-means clustering of genes based on inferred kinetic parameters
(e.g. from an mRNA/protein production-degradation model), rather than
raw expression trajectories.

Input:
    params_results.nc
        an xarray.Dataset, dims: (ensembl_gene_id,)
        data_vars: alpha, beta, delta_m, sigma_y, t_rep, t_zga, t_reg

Output:
    - membership matrix (U)
    - hard cluster labels
    - cluster centers (prototype parameter profiles, in both
      standardized and original units)

"""
# GLOBAL CONVENTION
# X = (genes, features)   

import os
from itertools import combinations

import numpy as np
import xarray as xr
import skfuzzy as fuzz
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
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

# Parameters to cluster on
FEATURES = ["alpha", "beta", "delta_m", "delta_z", "t_zga", "t_reg"]

# log transformed features
LOG_FEATURES = ["alpha", "beta", "delta_m", "delta_z", "t_zga", "t_reg"]

K_RANGE = range(3, 8)
FUZZINESS = 1.8   # m parameter (1.5-2.5 typical)
MAX_ITER = 300
ERROR = 1e-5
N_CLUSTER = 4
RNG_SEED = 1 

# =====================================================
# Build feature matrix from the parameter dataset
# =====================================================

def build_feature_matrix(ds, features=FEATURES, log_features=LOG_FEATURES):
    """
    ds: xarray.Dataset with dims (ensembl_gene_id,) and one data_var
        per kinetic parameter.

    Returns
    -------
    X : (n_genes, n_features) standardized float array
    genes : (n_genes,) array of gene ids, in the same row order as X.
    scaler : fitted sklearn StandardScaler (post-log) 
    raw : (n_genes, n_features) array of the unstandardized, but log-transformed values
    """
    genes = ds.ensembl_gene_id.values
    n_genes = genes.shape[0]
    n_features = len(features)

    raw = np.empty((n_genes, n_features), dtype=np.float64)
    for j, feat in enumerate(features):
        vals = ds[feat].values.astype(np.float64)
        if feat in log_features:
            if np.any(vals <= 0):
                n_bad = int(np.sum(vals <= 0))
                raise ValueError(
                    f"Feature '{feat}' has {n_bad} non-positive value(s); "
                    "can't log-transform. Either filter those genes out "
                    "upstream or remove this feature from LOG_FEATURES.")
            vals = np.log(vals)
        raw[:, j] = vals

    # z-score data
    scaler = StandardScaler()
    X = scaler.fit_transform(raw)

    return X, genes, scaler, raw


def cluster_centers_to_original_units(cntr, scaler, features=FEATURES,
                                       log_features=LOG_FEATURES):
    """
    cntr: (k, n_features) cluster centers in standardized (z-scored,
          post-log) space, as returned by fuzz.cluster.cmeans.

    Returns a (k, n_features) array back in the original parameter
    units (undoing z-score, then undoing log where applicable).
    """
    centers_log = scaler.inverse_transform(cntr)
    centers_orig = centers_log.copy()
    for j, feat in enumerate(features):
        if feat in log_features:
            centers_orig[:, j] = np.exp(centers_log[:, j]) 
    return centers_orig


# =====================================================
# Fuzzy clustering wrapper
# =====================================================

def run_fuzzy_cmeans(data, k, seed=None):
    """
    data shape: (n_features, n_genes)
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


def _permute_features(data, seed_seq):
    """
    Build a null-model dataset for the stability baseline: independently
    permute each *feature* across genes. This destroys cross-feature
    (i.e. cross-parameter) correlation structure -- the thing that
    would actually create clusters -- while preserving each feature's
    own marginal distribution across the gene population.

    data shape: (n_features, n_genes)
    """
    local_rng = np.random.default_rng(seed_seq)
    shuffled = data.copy()
    n_features, n_genes = data.shape
    for f in range(n_features):
        shuffled[f, :] = data[f, local_rng.permutation(n_genes)]
    return shuffled


# =====================================================
# Select optimal K -- stability-based (improved)
# =====================================================

def stability_score(data, k, seed_seq, n_runs=15, sample_frac=0.8):
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

        idx = resample(
            np.arange(n_genes), replace=True,
            n_samples=int(n_genes * sample_frac),
            random_state=resample_seed)

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
                label="null baseline (permuted features)")
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


def select_best_k_stability(data, k_range, dataset_name,
                             n_runs=15, sample_frac=0.8,
                             use_one_se_rule=True,
                             compute_null=True,
                             n_jobs=1, master_seed=RNG_SEED):
    """
    Choose K via bootstrap clustering stability.

    data: (n_features, n_genes)

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
            perm_seq, null_fit_seq = null_seq.spawn(2)
            null_data = _permute_features(data, perm_seq)
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
        print(f"K={k} | stability={real['mean']:.4f} +/- {real['std']:.4f}"
              f"{null_str} | FPC={fpc_full:.4f} | XB={xb:.4f}")

    best_idx = int(np.argmax(means))
    if use_one_se_rule:
        se_best = stds[best_idx] / np.sqrt(max(n_pairs, 1))
        threshold = means[best_idx] - se_best
        candidates = [i for i in range(len(ks)) if means[i] >= threshold]
        chosen_idx = min(candidates)  # smallest K within 1 SE of the best -> simplest adequate model
    else:
        chosen_idx = best_idx
    best_k = ks[chosen_idx]

    _plot_k_selection(ks, means, stds, null_means, fpc_scores, xb_scores, best_k, dataset_name)

    return best_k, diagnostics


def xie_beni_index(X, cntr, u, m):
    """
    XB Index
    The optimal number of clusters k is is such that the index takes the minimum value
    ---------
    X: Data (features, samples)
    cntr: (clusters, features)
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

def fuzzy_cmeans_clustering(ds, dataset_name, features=FEATURES,
                             log_features=LOG_FEATURES, k_range=K_RANGE):

    X, genes, scaler, raw = build_feature_matrix(ds, features, log_features)
    data = X.T  # (n_features, n_genes), what skfuzzy expects

    if N_CLUSTER is None:
        print("Selecting best K...")
        best_k, _ = select_best_k_stability(data, k_range, dataset_name)
    else:
        best_k = N_CLUSTER
    print(f"Number of Clusters: {best_k}")
    centers, membership, fpc = run_fuzzy_cmeans(data, best_k, seed=1)

    labels = np.argmax(membership, axis=0)
    centers_orig = cluster_centers_to_original_units(
        centers, scaler, features, log_features)

    # -------------------------------------------------
    # package outputs
    # -------------------------------------------------

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

    # standardized (z-scored, post-log) centers -- what the clustering
    # actually operated on; useful for comparing relative parameter
    # importance across clusters.
    centers_std_da = xr.DataArray(
        centers,
        dims=("cluster", "feature"),
        coords={
            "cluster": np.arange(best_k),
            "feature": features
        },
        name="cluster_centers_standardized"
    )

    # centers mapped back to original (natural) parameter units --
    # what you'd actually report/interpret.
    centers_orig_da = xr.DataArray(
        centers_orig,
        dims=("cluster", "feature"),
        coords={
            "cluster": np.arange(best_k),
            "feature": features
        },
        name="cluster_centers_original_units"
    )

    return membership_da, labels_da, centers_std_da, centers_orig_da, best_k, fpc


# =====================================================
# Plot cluster centers
# =====================================================

def plot_cluster_centers(centers_std_da, dataset_name):
    """
    Heatmap of standardized cluster centers: one row per cluster, one
    column per parameter. Replaces the trajectory line plot from the
    time-series version, since there's no time axis here.
    """
    k = centers_std_da.sizes["cluster"]
    features = centers_std_da.feature.values
    mat = centers_std_da.values  # (k, n_features)

    fig, ax = plt.subplots(figsize=(1.1 * len(features) + 2, 0.6 * k + 2))
    vmax = np.max(np.abs(mat))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Cluster {c}" for c in range(k)])

    for i in range(k):
        for j in range(len(features)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="standardized value (log & z-score)")
    ax.set_title(f"Fuzzy c-means cluster parameter profiles ({dataset_name})")
    fig.tight_layout()
    fig.savefig(f"figs/FCM_param_cluster_center_{dataset_name}.png", dpi=300)
    plt.close(fig)


def cluster_dataset(ds, output_prefix, dataset_name="dataset",
                     features=FEATURES, log_features=LOG_FEATURES,
                     k_range=K_RANGE):

    membership, labels, centers_std, centers_orig, best_k, fpc = fuzzy_cmeans_clustering(ds, dataset_name, features, log_features, k_range)

    membership.to_netcdf(f"{output_prefix}_membership.nc")
    labels.to_netcdf(f"{output_prefix}_labels.nc")
    centers_std.to_netcdf(f"{output_prefix}_centers_standardized.nc")
    centers_orig.to_netcdf(f"{output_prefix}_centers_original_units.nc")

    print(f"{output_prefix}: K={best_k}, FPC={fpc:.3f}")

    return membership, labels, centers_std, centers_orig


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    models = ["Rep_M", "Rep_Z"]
    for model in models:
        print(model)
        input_file = (f"../results/results_summary/{model}/params_results.nc")
        ds = xr.load_dataset(input_file)

        membership, labels, centers_std, centers_orig = cluster_dataset(
            ds, f"results/{model}_model_params", dataset_name=model, k_range=K_RANGE)

        plot_cluster_centers(centers_std, model)
