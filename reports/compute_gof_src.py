import xarray as xr
import arviz as az
import os
import click
from report import *
import warnings

def gof_evaluation(idata, gene_id, model, out_path):

    row = []

    for src in idata.observed_data.source.values:
        ds = idata.sel(source=src)
        non_nan_times = ds.observed_data.y.time.values[np.isfinite(ds.observed_data.y.values)]
        obs = ds.observed_data.y.sel(time=non_nan_times).values
        pred = idata.posterior_model_fits.y.mean(dim=("chain","draw","source")).sel(time=non_nan_times).values

        ll = idata.log_likelihood["y"]  # dims: (chain, draw, time, source)
        grouped_ll = ll.sum(dim="source")   # -> dims: (chain, draw, time)

        # Wrap into a fresh InferenceData for WAIC/LOO
        idata_grouped = az.InferenceData(
            posterior=idata.posterior,          # reuse original posterior
            log_likelihood=xr.Dataset({"y": grouped_ll}),
        )

        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            waic = az.waic(idata_grouped, pointwise=True).elpd_waic
            loo = az.loo(idata_grouped, pointwise=True).elpd_loo

        rho = spearman_correlation(obs, pred)
        pearsonr = pearson_correlation(obs, pred)
        nrmse = calc_nrmse(obs, pred)[0]         # by Range
        accepted = (rho > 0.7) & (nrmse < 0.2)

        row.append({
            "gene_id":gene_id,
            "model":model,
            "source": src,
            "BIC": calc_bic(ds),
            "AIC": calc_AIC(ds),
            "WAIC": waic,
            "LOO": loo,
            "rho": rho,
            "pearsonr": pearsonr,
            "NRMSE": nrmse, 
            "MASE": calc_mase(obs, pred),
            "accepted": accepted,
        })

    pd.DataFrame(row).to_csv(os.path.join(out_path, "gof_metrics.csv"), index=False)
    
    
@click.command() 
@click.option("--gene_id", type=str, default=None,    help="Run a single gene ID (used for array jobs)")
@click.option("--model", type=str, default=None,    help="Model version: Basic, Rep_M, Rep_Z")
@click.option("--t_end",   type=int, default=120,     help="Simulation Endpoint")
def main(gene_id, model, t_end):

    file_path = os.path.join("results", f"{t_end}_hpf", model, "all", gene_id, )
    idata = az.from_netcdf(os.path.join(file_path, "numpyro_posterior.nc"))

    gof_evaluation(idata, gene_id, model, file_path)


if __name__ == "__main__":
    main()