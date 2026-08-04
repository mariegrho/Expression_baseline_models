"""
Build a per-gene fit-quality scorecard from ArviZ InferenceData results.

Assumes you have one InferenceData object per gene (e.g. loaded from
24,000 NetCDF files, or held in a dict {gene_id: idata}). If instead you
have a SINGLE InferenceData with a shared "gene" dimension across groups,
see the note at the bottom — the per-gene loop collapses to `.sel(gene=g)`
slicing instead of iterating over separate objects.

Each InferenceData is expected to have groups:
    posterior, posterior_predictive, log_likelihood,
    sample_stats, observed_data, posterior_model_fits, posterior_residuals

Requires: arviz, numpy, pandas
"""

import numpy as np
import pandas as pd
import arviz as az


def score_gene(gene_id: str, idata: az.InferenceData, obs_var: str = "y") -> dict:
    """Compute one row of fit-quality metrics for a single gene's InferenceData."""

    row = {"gene": gene_id}

    # ---- 1. Convergence diagnostics (posterior + sample_stats) --------
    try:
        summ = az.summary(idata, group="posterior", var_names=None)
        row["rhat_max"] = summ["r_hat"].max()
        row["ess_bulk_min"] = summ["ess_bulk"].min()
        row["ess_tail_min"] = summ["ess_tail"].min()
    except Exception as e:
        row["rhat_max"] = np.nan
        row["ess_bulk_min"] = np.nan
        row["ess_tail_min"] = np.nan
        row["convergence_error"] = str(e)

    try:
        divergences = idata.sample_stats["diverging"].values
        row["n_divergences"] = int(divergences.sum())
        row["frac_divergences"] = float(divergences.mean())
    except Exception:
        row["n_divergences"] = np.nan
        row["frac_divergences"] = np.nan

    # ---- 2. PSIS-LOO and WAIC (log_likelihood) -------------------------
    try:
        loo_res = az.loo(idata, pointwise=True)
        row["loo"] = loo_res.elpd_loo
        row["loo_se"] = loo_res.se
        row["p_loo"] = loo_res.p_loo
        pareto_k = loo_res.pareto_k.values
        row["max_pareto_k"] = float(np.max(pareto_k))
        row["frac_pareto_k_high"] = float(np.mean(pareto_k > 0.7))
    except Exception as e:
        row["loo"] = np.nan
        row["max_pareto_k"] = np.nan
        row["loo_error"] = str(e)

    try:
        waic_res = az.waic(idata)
        row["waic"] = waic_res.elpd_waic
        row["waic_se"] = waic_res.se
    except Exception:
        row["waic"] = np.nan

    # ---- 3. Point-estimate fit quality (posterior_predictive vs obs) --
    try:
        obs = idata.observed_data[obs_var].values.astype(float)  # (time,)
        # posterior_predictive dims: (chain, draw, time)
        pp = idata.posterior_predictive[obs_var]
        pp_mean = pp.mean(dim=("chain", "draw")).values

        mask = ~np.isnan(obs)
        resid = obs[mask] - pp_mean[mask]

        rmse = np.sqrt(np.mean(resid ** 2))
        mae = np.mean(np.abs(resid))
        obs_range = obs[mask].max() - obs[mask].min()
        obs_mean = obs[mask].mean()

        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((obs[mask] - obs_mean) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        row["rmse"] = rmse
        row["mae"] = mae
        row["nrmse_range"] = rmse / obs_range if obs_range > 0 else np.nan
        row["nrmse_mean"] = rmse / obs_mean if obs_mean != 0 else np.nan
        row["r2"] = r2

        # ---- 4. Posterior predictive coverage --------------------------
        for cred, (lo_q, hi_q) in {"50": (0.25, 0.75), "90": (0.05, 0.95)}.items():
            lo = pp.quantile(lo_q, dim=("chain", "draw")).values
            hi = pp.quantile(hi_q, dim=("chain", "draw")).values
            inside = (obs[mask] >= lo[mask]) & (obs[mask] <= hi[mask])
            row[f"coverage_{cred}"] = float(np.mean(inside))

    except Exception as e:
        row["rmse"] = np.nan
        row["r2"] = np.nan
        row["fit_error"] = str(e)

    # ---- 5. Residual autocorrelation (using precomputed posterior_residuals) ----
    # A quick lag-1 autocorrelation of the mean residual trace flags
    # systematic (non-white-noise) misfit, which flat R2/RMSE can hide.
    try:
        resid_da = idata.posterior_residuals[obs_var]  # adjust var name if different
        resid_mean = resid_da.mean(dim=("chain", "draw")).values
        resid_mean = resid_mean[~np.isnan(resid_mean)]
        if len(resid_mean) > 2:
            r_lag1 = np.corrcoef(resid_mean[:-1], resid_mean[1:])[0, 1]
            row["resid_autocorr_lag1"] = r_lag1
    except Exception:
        row["resid_autocorr_lag1"] = np.nan

    return row


def build_scorecard(idata_dict: dict, obs_var: str = "y") -> pd.DataFrame:
    """
    idata_dict: {gene_id: az.InferenceData}
    Returns a DataFrame with one row per gene.

    NOTE: for large gene sets, prefer build_scorecard_from_files() below,
    which streams one file at a time instead of requiring every
    InferenceData to already be loaded in memory.
    """
    rows = []
    for gene_id, idata in idata_dict.items():
        rows.append(score_gene(gene_id, idata, obs_var=obs_var))
    return pd.DataFrame(rows).set_index("gene")


def _score_one_file(args):
    """Worker function: load one .nc file, score it, return the row dict.
    Any failure is captured in the row rather than raised, so one bad
    file doesn't kill the whole batch."""
    gene_id, path, obs_var = args
    try:
        idata = az.from_netcdf(path)
        row = score_gene(gene_id, idata, obs_var=obs_var)
    except Exception as e:
        row = {"gene": gene_id, "load_error": str(e)}
    return row


def build_scorecard_from_files(paths: dict, obs_var: str = "y",
                                out_csv: str = "fit_scorecard.csv",
                                n_workers: int = 4,
                                checkpoint_every: int = 200) -> pd.DataFrame:
    """
    paths: {gene_id: path_to_netcdf_file}
    Streams and scores one InferenceData at a time (parallelized across
    n_workers processes), writing to out_csv incrementally so a crash
    partway through doesn't lose completed work. Re-running will skip
    genes already present in out_csv.

    For 24,000 genes, start with a small subset (e.g. 50-100 files) to
    sanity-check timing before committing to a full run, and set
    n_workers based on available CPU cores / memory per idata.
    """
    import os
    import multiprocessing as mp

    if not paths:
        print("No paths given -- nothing to score. Check discover_gene_paths() output above.")
        if os.path.exists(out_csv):
            return pd.read_csv(out_csv, index_col="gene")
        return pd.DataFrame()

    done_genes = set()
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv, index_col="gene")
        done_genes = set(existing.index.astype(str))
        print(f"Resuming: {len(done_genes)} genes already scored.")

    todo = [(gid, p, obs_var) for gid, p in paths.items() if str(gid) not in done_genes]
    print(f"{len(todo)} genes remaining out of {len(paths)}.")

    if not todo:
        print("Nothing left to score -- all requested genes already in out_csv.")
        return pd.read_csv(out_csv, index_col="gene")

    write_header = not os.path.exists(out_csv)
    buffer = []

    with mp.Pool(n_workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_score_one_file, todo), start=1):
            buffer.append(row)
            if i % checkpoint_every == 0 or i == len(todo):
                df_chunk = pd.DataFrame(buffer)
                df_chunk.to_csv(out_csv, mode="a", header=write_header, index=False)
                write_header = False
                buffer = []
                print(f"  scored {i}/{len(todo)} genes")

    return pd.read_csv(out_csv, index_col="gene")


def flag_genes(scorecard: pd.DataFrame,
               rhat_thresh: float = 1.01,
               ess_thresh: float = 400,
               pareto_k_thresh: float = 0.7,
               r2_thresh: float = 0.7,
               coverage90_bounds=(0.80, 0.98)) -> pd.DataFrame:
    """
    Add boolean triage columns to the scorecard for quick filtering.
    Thresholds are starting points -- inspect the metric distributions
    (histograms/ECDFs) first and adjust to your data before trusting these.
    """
    sc = scorecard.copy()
    sc["converged"] = (
        (sc["rhat_max"] <= rhat_thresh) &
        (sc["ess_bulk_min"] >= ess_thresh) &
        (sc["n_divergences"] == 0)
    )
    sc["good_fit"] = sc["r2"] >= r2_thresh
    sc["reliable_loo"] = sc["max_pareto_k"] < pareto_k_thresh
    lo, hi = coverage90_bounds
    sc["well_calibrated"] = sc["coverage_90"].between(lo, hi)

    sc["status"] = "good"
    sc.loc[~sc["converged"], "status"] = "non_converged"
    sc.loc[sc["converged"] & ~sc["good_fit"], "status"] = "converged_poor_fit"
    sc.loc[sc["converged"] & sc["good_fit"] & ~sc["reliable_loo"], "status"] = "outlier_influenced"

    return sc


def discover_gene_paths(results_root: str = "results/120_hpf/Rep_M/all",
                         filename: str = "numpyro_posterior.nc") -> dict:
    """
    Walk the fixed directory structure:
        results/120_hpf/Rep_M/all/<ENSDARG...>/numpyro_posterior.nc
    and return {gene_id: full_path} for every gene found.
    """
    import glob
    import os

    root_abs = os.path.abspath(results_root)
    pattern = os.path.join(results_root, "*", filename)

    if not os.path.isdir(results_root):
        print(f"WARNING: results_root does not exist as a directory: {root_abs}")
        print(f"  Current working directory is: {os.getcwd()}")
        print(f"  Pass an absolute path, or run the script from the directory "
              f"where 'results/...' is relative to.")
        return {}

    paths = {}
    for p in glob.glob(pattern):
        gene_id = os.path.basename(os.path.dirname(p))  # the ENSDARG... folder
        paths[gene_id] = p

    if not paths:
        # Diagnostic: show what's actually in results_root so it's obvious
        # whether the mismatch is the filename, an extra nesting level, etc.
        subdirs = [d for d in os.listdir(results_root)
                   if os.path.isdir(os.path.join(results_root, d))][:5]
        print(f"WARNING: 0 files matched pattern: {os.path.abspath(pattern)}")
        print(f"  results_root resolved to: {root_abs}")
        print(f"  First few subdirectories found there: {subdirs}")
        if subdirs:
            example_dir = os.path.join(results_root, subdirs[0])
            print(f"  Contents of {example_dir}: {os.listdir(example_dir)}")

    print(f"Found {len(paths)} gene fits under {results_root}")
    return paths


if __name__ == "__main__":
    # Example usage for the fixed path structure:
    #   results/120_hpf/Rep_M/all/<gene_id>/numpyro_posterior.nc
    
    paths = discover_gene_paths("results/120_hpf/Rep_M/all")
    
    scorecard = build_scorecard_from_files(
        paths, obs_var="y",
        out_csv="fit_scorecard.csv",
        n_workers=4,            # tune to your CPU count / memory
        checkpoint_every=200,
    )
    
    scorecard = flag_genes(scorecard)
    scorecard.to_csv("fit_scorecard_flagged.csv")
    print(scorecard["status"].value_counts())
    pass
