"""
Recompute ONLY the LOO/WAIC block from build_fit_scorecard.py and patch
those columns into an already-written fit_scorecard.csv, without redoing
convergence diagnostics, posterior-predictive coverage, or residual
autocorrelation (which don't depend on log_likelihood).

Usage:
    python recompute_loo_waic.py \
        --results-root results/120_hpf/Rep_M/all \
        --out-csv fit_scorecard.csv \
        --only-errors        # optional: only redo rows where loo_error was set

Writes the patched columns back into the same CSV (in place, after a
".bak" backup), or to --out-csv-new if you'd rather not overwrite.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import arviz as az


LOO_WAIC_COLS = ["loo", "loo_se", "p_loo", "max_pareto_k",
                  "frac_pareto_k_high", "loo_error", "waic", "waic_se"]


def _compute_loo_waic_one(gene_id: str, path: str, obs_var: str = "y") -> dict:
    """Load one netcdf, sum log_likelihood over 'source', compute loo/waic.
    Returns only the loo/waic-related fields (plus gene_id)."""
    row = {"gene": gene_id}
    try:
        idata = az.from_netcdf(path)

        ll = idata.log_likelihood[obs_var]          # (chain, draw, time, source)
        grouped_ll = ll.sum(dim="source")            # (chain, draw, time)

        idata_grouped = az.InferenceData(
            posterior=idata.posterior,
            log_likelihood=xr.Dataset({obs_var: grouped_ll}),
        )

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

    except Exception as e:
        # netcdf failed to load at all
        row["load_error"] = str(e)

    return row


def _worker(args):
    """Module-level worker so it can be pickled by multiprocessing.Pool.
    args: (gene_id, path, obs_var)"""
    gene_id, path, obs_var = args
    return _compute_loo_waic_one(gene_id, path, obs_var=obs_var)


def recompute_loo_waic(paths: dict, out_csv: str, obs_var: str = "y",
                        only_errors: bool = False, n_workers: int = 4,
                        checkpoint_every: int = 200, out_csv_new: str = None,
                        task_timeout: int = 300) -> pd.DataFrame:
    """
    paths: {gene_id: path_to_netcdf_file} -- same dict discover_gene_paths() returns.
    out_csv: existing scorecard to patch.
    only_errors: if True, only recompute rows where loo_error/load_error was
                 previously set (or loo is NaN) -- useful if you fixed a bug
                 and want to retry only the genes that failed before.
    task_timeout: seconds to wait for any single gene's loo/waic computation
                 before giving up on it and recording a timeout error. This
                 is what prevents one pathological file from wedging the
                 whole pool -- with plain imap_unordered, a single hung
                 worker blocks all queued-but-unstarted tasks behind it
                 forever, which is why runs can appear to "freeze" a few
                 hundred genes short of completion.
    """
    import multiprocessing as mp
 
    if not os.path.exists(out_csv):
        raise FileNotFoundError(f"{out_csv} does not exist -- nothing to patch.")
 
    scorecard = pd.read_csv(out_csv, index_col="gene")
 
    # Decide which genes to recompute
    if only_errors:
        had_error = scorecard.get("loo_error", pd.Series(dtype=object)).notna()
        was_nan = scorecard.get("loo", pd.Series(dtype=float)).isna()
        target_genes = set(scorecard.index[had_error | was_nan].astype(str))
        todo = [(gid, p, obs_var) for gid, p in paths.items() if str(gid) in target_genes]
    else:
        todo = [(gid, p, obs_var) for gid, p in paths.items() if str(gid) in set(scorecard.index.astype(str))]
 
    print(f"Recomputing LOO/WAIC for {len(todo)} genes "
          f"({'errors/NaN only' if only_errors else 'all rows present in CSV'}). "
          f"Per-task timeout: {task_timeout}s")
 
    # Make sure the columns we're about to patch exist
    for col in LOO_WAIC_COLS:
        if col not in scorecard.columns:
            scorecard[col] = np.nan
 
    results = []
    with mp.Pool(n_workers) as pool:
        pending = [(task[0], pool.apply_async(_worker, (task,))) for task in todo]
 
        for i, (gid, ar) in enumerate(pending, start=1):
            try:
                row = ar.get(timeout=task_timeout)
            except mp.TimeoutError:
                row = {"gene": gid, "loo": np.nan, "max_pareto_k": np.nan,
                       "loo_error": f"timed out after {task_timeout}s"}
                print(f"  [TIMEOUT] gene={gid} exceeded {task_timeout}s -- "
                      f"recorded as error, continuing")
            except Exception as e:
                row = {"gene": gid, "loo": np.nan, "max_pareto_k": np.nan,
                       "loo_error": f"worker exception: {e}"}
                print(f"  [ERROR] gene={gid}: {e}")
 
            results.append(row)
            if i % checkpoint_every == 0 or i == len(todo):
                print(f"  recomputed {i}/{len(todo)}")
 
    # Patch results back into the scorecard, row by row, column by column
    # (so we never wipe out other unrelated columns for that gene)
    for row in results:
        gene_id = row.pop("gene")
        if gene_id not in scorecard.index:
            continue
        for col, val in row.items():
            if col not in scorecard.columns:
                scorecard[col] = np.nan
            scorecard.loc[gene_id, col] = val
 
    dest = out_csv_new if out_csv_new else out_csv
    if dest == out_csv:
        backup = out_csv + ".bak"
        scorecard_before = pd.read_csv(out_csv, index_col="gene")
        scorecard_before.to_csv(backup)
        print(f"Backed up original to {backup}")
 
    scorecard.to_csv(dest)
    print(f"Wrote patched scorecard to {dest}")
    return scorecard


if __name__ == "__main__":
    import click
    from build_fit_scorecard import discover_gene_paths  # reuse your existing helper

    @click.command()
    @click.option("--results-root", type=click.Path(), required=True)
    @click.option("--filename", type=str, default="numpyro_posterior.nc", show_default=True)
    @click.option("--obs-var", type=str, default="y", show_default=True)
    @click.option("--out-csv", type=click.Path(), required=True,
                  help="Existing scorecard CSV to patch.")
    @click.option("--out-csv-new", type=click.Path(), default=None,
                  help="Write patched result here instead of overwriting --out-csv.")
    @click.option("--only-errors", is_flag=True, default=False,
                  help="Only recompute rows where loo previously failed / was NaN.")
    @click.option("--n-workers", type=int, default=4, show_default=True)
    @click.option("--checkpoint-every", type=int, default=200, show_default=True)
    @click.option("--task-timeout", type=int, default=300, show_default=True, help="Seconds to wait for a single gene's loo/waic before giving up on it and recording a timeout error.")
    def main(results_root, filename, obs_var, out_csv, out_csv_new, only_errors,
             n_workers, checkpoint_every, task_timeout):
        paths = discover_gene_paths(results_root, filename=filename)
        recompute_loo_waic(
            paths, out_csv=out_csv, obs_var=obs_var, only_errors=only_errors,
            n_workers=n_workers, checkpoint_every=checkpoint_every,
            out_csv_new=out_csv_new, task_timeout=task_timeout, )

    main()
