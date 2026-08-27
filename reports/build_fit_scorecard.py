"""
Build a per-gene fit-quality scorecard from ArviZ InferenceData results.

Assumes you have one InferenceData object per gene (e.g. loaded from NetCDF files, 
or held in a dict {gene_id: idata}). 
If instead you have a SINGLE InferenceData with a shared "gene" dimension across groups,
see the note at the bottom — the per-gene loop collapses to `.sel(gene=g)`
slicing instead of iterating over separate objects.

Each InferenceData is expected to have groups:
    posterior, posterior_predictive, log_likelihood,
    sample_stats, observed_data, posterior_model_fits, posterior_residuals

Requires: arviz, numpy, pandas
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import numpy as np
import pandas as pd
import arviz as az
import xarray as xr


SCORECARD_COLUMNS = [
    "gene",
    "rhat_max", "ess_bulk_min", "ess_tail_min", "convergence_error",
    "n_divergences", "frac_divergences",
    "loo", "loo_se", "p_loo", "max_pareto_k", "frac_pareto_k_high", "loo_error",
    "waic", "waic_se",
    "r2", "spearman_rho", "pearson_r",
    "pp_coverage_50", "pp_coverage_90", "pmf_coverage_95", "fit_error",
    "resid_autocorr_lag1",
    "load_error",
]

def score_gene(gene_id: str, idata: az.InferenceData, obs_var: str = "y") -> dict:
    """Compute one row of fit-quality metrics for a single gene's InferenceData."""

    row = {"gene": gene_id}

    ll = idata.log_likelihood["y"]  # dims: (chain, draw, time, source)
    grouped_ll = ll.sum(dim="source")   # -> dims: (chain, draw, time)

    # Wrap into a fresh InferenceData for WAIC/LOO
    idata_grouped = az.InferenceData(
        posterior=idata.posterior,      
        log_likelihood=xr.Dataset({"y": grouped_ll}),
    )

    # ---- 1. Convergence diagnostics (posterior + sample_stats) --------
    try:
        rhat_ds = az.rhat(idata, method="rank")
        ess_bulk_ds = az.ess(idata, method="bulk")
        ess_tail_ds = az.ess(idata, method="tail")

        row["rhat_max"] = max(float(np.nanmax(v.values)) for v in rhat_ds.data_vars.values())
        row["ess_bulk_min"] = min(float(np.nanmin(v.values)) for v in ess_bulk_ds.data_vars.values())
        row["ess_tail_min"] = min(float(np.nanmin(v.values)) for v in ess_tail_ds.data_vars.values())
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
        loo_res = az.loo(idata_grouped, pointwise=True)
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
        waic_res = az.waic(idata_grouped)
        row["waic"] = waic_res.elpd_waic
        row["waic_se"] = waic_res.se
    except Exception:
        row["waic"] = np.nan

    # ---- 3. Point-estimate fit quality (posterior_predictive vs obs) --
    try:
        obs = idata.observed_data[obs_var]
        model_fit = idata.posterior_model_fits[obs_var]
        pred = model_fit.mean(dim=("chain", "draw", "source"))

        obs_mean = obs.mean(dim="source")

        ss_res = ((obs_mean - pred) ** 2).sum(dim="time")
        ss_tot = ((obs_mean - obs.mean(dim=("time", "source"))) ** 2).sum(dim="time")
        r2_per_source = (1 - ss_res / ss_tot).where(ss_tot > 0)
        r2_mean = r2_per_source.mean().item()

        row["r2"] = r2_mean
        row["spearman_rho"] = spearmanr(obs_mean, pred)[0]
        row["pearson_r"] = pearsonr(obs_mean, pred)[0]

    # ---- 4. Posterior predictive coverage --------------------------
        pp = idata.posterior_predictive[obs_var]  # posterior_predictive dims: (chain, draw, time, source)
        pp = pp.isel(draw=slice(None, None, 4))   # every 4th draw
        mask = obs.notnull()
        for cred, (lo_q, hi_q) in { "50": (0.25, 0.75), "90": (0.05, 0.95),}.items():
            lo = pp.quantile( lo_q, dim=("chain", "draw"))
            hi = pp.quantile(hi_q, dim=("chain", "draw"))
            inside = ((obs >= lo) & (obs <= hi) & mask )
            row[f"pp_coverage_{cred}"] = (inside.sum() / mask.sum()).item()

        ## posterior-model-fit coverage
        for cred, (lo_q, hi_q) in { "95": (0.025, 0.975),}.items():
            lo = model_fit.quantile( lo_q, dim=("chain", "draw"))
            hi = model_fit.quantile(hi_q, dim=("chain", "draw"))
            inside = ((obs >= lo) & (obs <= hi) & mask )
            row[f"pmf_coverage_{cred}"] = (inside.sum() / mask.sum()).item()

    except Exception as e:
        row["r2"] = np.nan
        row["spearman_rho"] = np.nan
        row["pearson_r"] = np.nan
        row["fit_error"] = str(e)

    # ---- 5. Residual autocorrelation (using precomputed posterior_residuals) ----
    try:
        resid_da = idata.posterior_residuals[obs_var]
        resid_mean = resid_da.mean(dim=("chain", "draw"))

        if "source" in resid_mean.dims:
            acf_per_source = []

            for source in resid_mean.source:
                x = resid_mean.sel(source=source).values
                x = x[~np.isnan(x)]

                if len(x) > 2:
                    acf_per_source.append( np.corrcoef(x[:-1], x[1:])[0, 1] )
            row["resid_autocorr_lag1"] = (float(np.nanmean(acf_per_source)) if acf_per_source else np.nan )
        else:
            x = resid_mean.values
            x = x[~np.isnan(x)]
            row["resid_autocorr_lag1"] = ( np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else np.nan )
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


def build_scorecard_from_files(paths: dict, obs_var: str = "y", out_csv: str = "fit_scorecard.csv",
                                n_workers: int = 4, checkpoint_every: int = 200) -> pd.DataFrame:
    """
    paths: {gene_id: path_to_netcdf_file}
    Streams and scores one InferenceData at a time (parallelized across
    n_workers processes), writing to out_csv. 
    Re-running will skip genes already present in out_csv.
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
        #pbar = tqdm(pool.imap_unordered(_score_one_file, todo), total=len(todo), desc="Scoring genes", mininterval=30)
        #for i, row in enumerate(pool.imap_unordered(_score_one_file, todo), start=1):
        for i, row in enumerate(pool.imap_unordered(_score_one_file, todo, chunksize=4), start=1):
        #for i, row in enumerate(pbar, start=1):
            buffer.append(row)
            if i % checkpoint_every == 0 or i == len(todo):
                df_chunk = pd.DataFrame(buffer).reindex(columns=SCORECARD_COLUMNS)
                df_chunk.to_csv(out_csv, mode="a", header=write_header, index=False)
                write_header = False
                buffer = []
                #pbar.set_postfix_str(f"checkpoint @ {i}")
                print(f"  scored {i}/{len(todo)} genes", flush=True)

    return pd.read_csv(out_csv, index_col="gene")


def flag_genes(scorecard: pd.DataFrame,
               rhat_thresh: float = 1.05,
               ess_thresh: float = 400,
               pareto_k_thresh: float = 0.7,
               r2_thresh: float = 0.7,
               spearman_thresh: float = 0.7,
               coverage90_bounds=(0.80, 0.98)) -> pd.DataFrame:
    """
    Add boolean triage columns to the scorecard for quick filtering.
    Thresholds are starting points -- inspect the metric distributions
    (histograms/ECDFs) first and adjust to your data before trusting these.
    """
    sc = scorecard.copy()

    sc["converged"] = (
        (sc["rhat_max"] <= rhat_thresh) &
        (sc["ess_bulk_min"] >= ess_thresh)
    )

    sc["good_fit_r2"] = sc["r2"] >= r2_thresh
    sc["good_pattern"] = sc["spearman_rho"] >= spearman_thresh

    sc["reliable_loo"] = sc["max_pareto_k"] < pareto_k_thresh
    lo, hi = coverage90_bounds
    sc["well_calibrated"] = sc["pp_coverage_90"].between(lo, hi)

    sc["status"] = "good"
    sc.loc[~sc["converged"], "status"] = "non_converged"
    sc.loc[sc["converged"] & ~sc["good_fit_r2"] & ~sc["good_pattern"], "status"] = "converged_poor_fit"
    sc.loc[sc["converged"] & sc["good_fit_r2"] & sc["good_pattern"] & ~sc["reliable_loo"], "status"] = "outlier_influenced"

    return sc

def discover_gene_paths(
    results_root: str,
    filename: str = "numpyro_posterior.nc",
    gene_list_path: str = "data/genes.txt",
) -> dict:
    """
    Walk the fixed directory structure:
        results/120_hpf/Rep_M/all/<ENSDARG...>/numpyro_posterior.nc
    and return {gene_id: full_path} for every gene found in results_root
    that is also listed in gene_list_path.
    """
    import glob
    import os

    root_abs = os.path.abspath(results_root)
    pattern = os.path.join(results_root, "*", filename)

    if not os.path.isdir(results_root):
        print(f"WARNING: results_root does not exist as a directory: {root_abs}")
        print(f"  Current working directory is: {os.getcwd()}")
        print(f"  Pass an absolute path, or run the script from the directory where 'results/...' is relative to.")
        return {}

    # --- load the gene whitelist ---
    if not os.path.isfile(gene_list_path):
        print(f"WARNING: gene_list_path does not exist: {os.path.abspath(gene_list_path)}")
        return {}

    with open(gene_list_path) as f:
        # handles genes separated by whitespace and/or newlines
        wanted_genes = set(f.read().split())

    print(f"Loaded {len(wanted_genes)} gene IDs from {gene_list_path}")

    # --- walk results_root and keep only genes in the whitelist ---
    paths = {}
    for p in glob.glob(pattern):
        gene_id = os.path.basename(os.path.dirname(p))  # the ENSDARG... folder
        if gene_id in wanted_genes:
            paths[gene_id] = p

    if not paths:
        # Diagnostic: show what's actually in results_root so it's obvious
        # whether the mismatch is the filename, an extra nesting level,
        # or a whitelist/results_root mismatch.
        subdirs = [d for d in os.listdir(results_root)
                   if os.path.isdir(os.path.join(results_root, d))][:5]
        print(f"WARNING: 0 matching genes found under: {os.path.abspath(pattern)}")
        print(f"  results_root resolved to: {root_abs}")
        print(f"  First few subdirectories found there: {subdirs}")
        if subdirs:
            example_dir = os.path.join(results_root, subdirs[0])
            print(f"  Contents of {example_dir}: {os.listdir(example_dir)}")
            if subdirs[0] not in wanted_genes:
                print(f"  Note: '{subdirs[0]}' is NOT in the gene whitelist -- "
                      f"check that gene_list_path and results_root match up.")

    missing = wanted_genes - paths.keys()
    if missing:
        print(f"WARNING: {len(missing)} genes from {gene_list_path} "
              f"were not found under {results_root} (e.g. {sorted(missing)[:5]})")

    print(f"Found {len(paths)} gene fits under {results_root} matching {gene_list_path}")
    return paths

 
if __name__ == "__main__":
    import click
 
    @click.command()
    @click.option("--results-root", type=click.Path(), required=True,help="Directory containing <gene_id>/numpyro_posterior.nc subfolders, ")
    @click.option("--filename", type=str, default="numpyro_posterior.nc", show_default=True,help="NetCDF filename inside each gene subfolder.",)
    @click.option("--obs-var", type=str, default="y", show_default=True, help="Name of the observed variable in observed_data/posterior_predictive.",)
    @click.option("--out-csv", type=click.Path(), default="fit_scorecard.csv", show_default=True, help="Path to write the scorecard CSV.",)
    @click.option("--n-workers", type=int, default=8, show_default=True, help="Number of parallel worker processes.",)
    @click.option("--checkpoint-every", type=int, default=200, show_default=True, help="Write to out_csv every N scored genes.",)
    def main(results_root, filename, obs_var, out_csv, n_workers, checkpoint_every):
        """Build a per-gene fit-quality scorecard from ArviZ NetCDF results."""
        
        paths = discover_gene_paths(results_root, filename=filename)
 
        scorecard = build_scorecard_from_files(paths, obs_var=obs_var, out_csv=out_csv, n_workers=n_workers, checkpoint_every=checkpoint_every, )

        # create flagged csv from existing file
        #scorecard = pd.read_csv(out_csv, index_col="gene")
 
        if not scorecard.empty:
            scorecard = flag_genes(scorecard)
            flagged_csv = out_csv.replace(".csv", "_flagged.csv")
            scorecard.to_csv(flagged_csv)
            if "status" in scorecard:
                print(scorecard["status"].value_counts())
            else:
                print(scorecard.head())
            print(f"Wrote flagged scorecard to {flagged_csv}")
 
    main()
 