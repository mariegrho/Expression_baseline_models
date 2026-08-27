from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import click
import xarray as xr
import arviz as az

MODE_CONFIG = {
    "simulation": {"group": "posterior_model_fits", "out_name": "simulation_results.nc"},
    "params": {"group": "posterior", "out_name": "params_results.nc"},
}

def process_gene(args):
    g, res_dir, tmp_dir, group = args
    in_path = Path(res_dir) / g / "numpyro_posterior.nc"
 
    if not in_path.exists():
        print(f"[Warning] Filepath path {in_path} not found {g}")
        return None
 
    try:
        ds = (
            getattr(az.from_netcdf(in_path), group)
            .mean(dim=("chain", "draw"))
            .expand_dims(ensembl_gene_id=[g])
        )
 
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
@click.option( "--mode", type=click.Choice(MODE_CONFIG.keys()), default="simulation", show_default=True,
                help="Which dataset to extract: 'simulation' uses posterior_model_fits -> simulation_results.nc, 'params' uses posterior -> params_results.nc.",)
def collect_results_concurrent(res_dir, gene_file, out_dir, mode):

    config = MODE_CONFIG[mode]
    group = config["group"]
    out_name = config["out_name"]

    with open(gene_file) as f:
        gene_list = [line.strip() for line in f]

    print("[INFO] Number of genes:", len(gene_list))
    print(f"[INFO] Mode: {mode} (group={group}, out_name={out_name})")

    tmp_dir = Path(res_dir) / f"_tmp_gene_results_{mode}"
    tmp_dir.mkdir(exist_ok=True)

    tasks = [(gene, res_dir, tmp_dir, group) for gene in gene_list]

    with ProcessPoolExecutor(max_workers=4) as exe:
        out_files = list(exe.map(process_gene, tasks, chunksize=20))

    out_files = [f for f in out_files if f is not None]

    print("[INFO] Opening reduced datasets...")
    ds_res = xr.open_mfdataset(out_files, combine="by_coords", parallel=True, engine="netcdf4")
    out_path = Path(out_dir) / out_name
    ds_res.to_netcdf(out_path, engine="netcdf4")

    genes = ds_res.ensembl_gene_id.size
    print(f"[INFO] {genes} found and saved to {out_path}")


if __name__ == "__main__":
    collect_results_concurrent()