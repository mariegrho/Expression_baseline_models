from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import click
import xarray as xr
import arviz as az

def process_gene(args):
    g, res_dir, tmp_dir = args
    in_path = Path(res_dir) / g / "numpyro_posterior.nc"

    if not in_path.exists():
        print(f"[Warning] Filepath path {in_path} not found {g}")
        return None

    try:
       # ds = (az.from_netcdf(in_path).posterior_model_fits.mean(dim=("chain", "draw")).expand_dims(ensembl_gene_id=[g]))
        ds = (az.from_netcdf(in_path).posterior(dim=("chain", "draw")).expand_dims(ensembl_gene_id=[g]))

        out_path = Path(tmp_dir) / f"{g}.nc"
        ds.to_netcdf(out_path)
        return str(out_path)

    except Exception as e:
        print(f"[ERROR] {g}: {e}")
        return None


@click.command()
@click.argument("res_dir")
@click.argument("gene_file")
@click.argument("out_dir")
def collect_results_concurrent(res_dir, gene_file, out_dir):
    with open(gene_file) as f:
        gene_list = [line.strip() for line in f]

    print("[INFO] Number of genes:", len(gene_list))

    tmp_dir = Path(res_dir) / "_tmp_gene_results"
    tmp_dir.mkdir(exist_ok=True)

    tasks = [(gene, res_dir, tmp_dir) for gene in gene_list]

    with ProcessPoolExecutor(max_workers=4) as exe:
        out_files = list(exe.map(process_gene, tasks, chunksize=20))

    out_files = [f for f in out_files if f is not None]

    print("[INFO] Opening reduced datasets...")
    ds_res = xr.open_mfdataset(out_files, combine="by_coords", parallel=True)
    #out_path = Path(out_dir) / "simulation_results.nc"
    out_path = Path(out_dir) / "params_results.nc"
    ds_res.to_netcdf(out_path)

    genes = ds_res.ensembl_gene_id.size
    print(f"[INFO] {genes} found and saved to {out_path}")


if __name__ == "__main__":
    collect_results_concurrent()