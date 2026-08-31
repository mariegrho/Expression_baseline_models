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

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,QuantileTransformer
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score

from FCM_cluster_selection import plot_k_selection, xie_beni_index, plot_cluster_centers

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except ImportError:
    _HAVE_JOBLIB = False

# =====================================================
# CONFIG
# =====================================================

# Parameters to cluster on
FEATURES = ["repression_ratio", "maternal_half_life", "t_zga", "t_reg", "peak_time", "y0"]# "decay_mode", ]

# decay mode need to be outside z-scoring
# tune: >1 to let mechanism choice drive clusters more, <1 to soften it
feature_weights = {"decay_mode": 0.1, "t_zga":1.5, "repression_ratio":1.5}

# log transformed features
LOG_FEATURES = ["repression_ratio", "maternal_half_life", "t_zga", "t_reg", "peak_time", "y0"]

K_RANGE = range(2, 10)
FUZZINESS = 1.6   # m parameter: higher -> fuzzier
MAX_ITER = 500
ERROR = 1e-3
N_CLUSTER = None
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

    # Check for empty dataset
    if n_genes == 0:
        raise ValueError(f"Cannot cluster empty dataset: {n_genes} samples, {n_features} features")

    raw = np.empty((n_genes, n_features), dtype=np.float64)
    for j, feat in enumerate(features):
        vals = ds[feat].values.astype(np.float64)

        n_nonfinite = int(np.sum(~np.isfinite(vals)))
        if n_nonfinite > 0:
            bad_genes = genes[~np.isfinite(vals)]
            raise ValueError(f"Parameter '{feat}' has {n_nonfinite} non-finite values (NaN/inf) ")

        if feat in log_features:
            if feat == "repression_ratio":
                vals = np.log(vals)  
            else:
                vals = np.log1p(vals)  # temporal values +1 before log transform (avoid zeros)

        if feat not in log_features and set(np.unique(vals)) <= {0.0, 1.0}:
            p = float(np.mean(vals))
            sep = np.inf if p in (0.0, 1.0) else 1.0 / np.sqrt(p * (1 - p))
            w = feature_weights.get(feat, 1.0)
            print(f"[Info] '{feat}' looks binary (p={p:.3f}); unweighted "
                  f"z-scored class separation = {sep:.2f}; "
                  f"applying weight={w} -> effective separation = {sep * w:.2f}")
            
        raw[:, j] = vals

    # z-score data
    scaler = StandardScaler()
    #scaler = RobustScaler()
    X = scaler.fit_transform(raw)

    # apply explicit per-feature weights on top of the z-scored data
    weights = np.array([feature_weights.get(feat, 1.0) for feat in features])
    X = X * weights[None, :]

    return X, genes, scaler, raw, weights


def cluster_centers_to_original_units(cntr, scaler, weights, features=FEATURES, log_features=LOG_FEATURES):
    """
    cntr: (k, n_features) cluster centers in standardized (z-scored,
          post-log) space, as returned by fuzz.cluster.cmeans.

    Returns a (k, n_features) array back in the original parameter
    units (undoing z-score, then undoing log where applicable).
    """
    centers_unweighted = cntr / weights[None, :]
    centers_log = scaler.inverse_transform(centers_unweighted)
    centers_orig = centers_log.copy()
    for j, feat in enumerate(features):
        if feat in log_features:
            if feat == "repression_ratio":
                centers_orig[:, j] = np.exp(centers_log[:, j]) 
            else:
                centers_orig[:, j] = np.expm1(centers_log[:, j])
            
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
            perm_seq, null_fit_seq = null_seq.spawn(2)
            null_data = _permute_features(data, perm_seq)
            null = stability_score(null_data, k, null_fit_seq, n_runs=n_runs, sample_frac=sample_frac)
        return k, real, null

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

    plot_k_selection(ks, means, stds, fpc_scores, xb_scores, best_k, dataset_name+"_params")

    return best_k, diagnostics


# =====================================================
# Run final clustering
# =====================================================

def fuzzy_cmeans_clustering(ds, dataset_name, features=FEATURES,
                             log_features=LOG_FEATURES, k_range=K_RANGE,
                             min_samples=5):

    # Check for empty or very small datasets early
    n_genes = len(ds.ensembl_gene_id.values)
    if n_genes < min_samples:
        print(f"[Warning] Dataset '{dataset_name}' has only {n_genes} samples (minimum {min_samples} required). Skipping clustering.")
        return None, None, None, None, None, None

    try:
        X, genes, scaler, raw, weights = build_feature_matrix(ds, features, log_features)
    except ValueError as e:
        print(f"[Warning] Failed to build feature matrix for '{dataset_name}': {e}")
        return None, None, None, None, None, None

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
        centers, scaler, weights, features, log_features)

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


def cluster_dataset(ds, output_prefix, dataset_name="dataset",
                     features=FEATURES, log_features=LOG_FEATURES,
                     k_range=K_RANGE, min_samples=5):

    membership, labels, centers_std, centers_orig, best_k, fpc = fuzzy_cmeans_clustering(ds, dataset_name, features, log_features, k_range, min_samples)

    # Check if clustering was successful
    if membership is None:
        print(f"[Warning] Clustering failed for {dataset_name}, skipping {output_prefix}")
        return None, None, None, None

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
    input_file = ("joint_params_results.nc")
    ds = xr.load_dataset(input_file)

    ds["beta"] = ds.beta.clip(min=1e-5)
    ds["alpha"] = ds.alpha.clip(min=1e-5)
    ds["delta_m"] = ds.delta_m.clip(min=1e-5)
    ds["t_zga"] = ds.t_zga.clip(max=120)
    ds["t_reg"] = ds.t_reg.clip(max=120)

    membership, labels, centers_std, centers_orig = cluster_dataset(
        ds, f"results/joint_params_results", dataset_name="joint", k_range=K_RANGE)

    
    # --- reindex superclusters by increasing peak time ---
    cluster_peak_times = centers_std.sel(feature="peak_time").values  # peak_time value per cluster
    new_order = np.argsort(cluster_peak_times)    # sort clusters by their peak_time values
    rank = np.argsort(new_order)                  # rank[old_id] = new_id

    # reorder centers
    centers_std = centers_std.isel(cluster=new_order).assign_coords(cluster=np.arange(len(new_order)))
    centers_std.to_netcdf(f"results/joint_param_superclusters_centers.nc")
    # reorder centers
    centers_orig = centers_orig.isel(cluster=new_order).assign_coords(cluster=np.arange(len(new_order)))
    centers_orig.to_netcdf(f"results/joint_param_superclusters_centers_orig.nc")
    # reorder _membership
    membership = membership.isel(cluster=new_order).assign_coords(cluster=np.arange(len(new_order)))
    membership.to_netcdf(f"results/joint_param_superclusters_membership.nc")

    labels = labels.copy(data=rank[labels.values])  # remap gene labels to new ids
    labels.to_netcdf(f"results/joint_param_superclusters_labels.nc")

    plot_cluster_centers(centers_std, "joint")


