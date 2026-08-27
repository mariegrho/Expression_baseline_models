from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import click
import xarray as xr
import arviz as az

MODE_CONFIG = {
    "simulation": {"group": "posterior_model_fits", "out_name": "simulation_results.nc"},
    "params": {"group": "posterior", "out_name": "params_results.nc"},
}


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def process_batch(args):
    gene_batch, res_dir, group = args
    datasets = []

    for g in gene_batch:
        in_path = Path(res_dir) / g / "numpyro_posterior.nc"

        if not in_path.exists():
            print(f"[Warning] Filepath path {in_path} not found {g}")
            continue

        try:
            ds = (
                getattr(az.from_netcdf(in_path), group)
                .mean(dim=("chain", "draw"))
                .expand_dims(ensembl_gene_id=[g])
                .load()  # force compute now, in the worker process
            )
            datasets.append(ds)
        except Exception as e:
            print(f"[ERROR] {g}: {e}")

    if not datasets:
        return None

    return xr.concat(datasets, dim="ensembl_gene_id")


@click.command()
@click.argument("res_dir")
@click.argument("gene_file")
@click.argument("out_dir")
@click.option("--mode", type=click.Choice(MODE_CONFIG.keys()), default="simulation", show_default=True,
              help="Which dataset to extract: 'simulation' uses posterior_model_fits -> simulation_results.nc, "
                   "'params' uses posterior -> params_results.nc.")
@click.option("--batch-size", type=int, default=100, show_default=True,
              help="Number of genes processed per worker task.")
@click.option("--max-workers", type=int, default=4, show_default=True,
              help="Number of parallel worker processes.")
def collect_results_concurrent(res_dir, gene_file, out_dir, mode, batch_size, max_workers):

    config = MODE_CONFIG[mode]
    group = config["group"]
    out_name = config["out_name"]

    with open(gene_file) as f:
        gene_list = [line.strip() for line in f]

    print("[INFO] Number of genes:", len(gene_list))
    print(f"[INFO] Mode: {mode} (group={group}, out_name={out_name})")

    batches = list(chunked(gene_list, batch_size))
    tasks = [(batch, res_dir, group) for batch in batches]

    print(f"[INFO] Processing {len(gene_list)} genes in {len(batches)} batches "
          f"(batch_size={batch_size}, max_workers={max_workers})...")

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        results = list(exe.map(process_batch, tasks))

    datasets = [d for d in results if d is not None]

    if not datasets:
        raise RuntimeError("[ERROR] No datasets were produced — check per-gene input paths and logs above.")

    print("[INFO] Concatenating datasets...")
    ds_res = xr.concat(datasets, dim="ensembl_gene_id")

    out_path = Path(out_dir) / out_name
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Writing merged dataset to {out_path}...")
    ds_res.to_netcdf(out_path, engine="netcdf4")

    genes = ds_res.ensembl_gene_id.size
    print(f"[INFO] {genes} genes found and saved to {out_path}")


if __name__ == "__main__":
    collect_results_concurrent()