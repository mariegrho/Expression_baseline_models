import os
import click
import xarray as xr
import arviz as az

from model.initialise import *
from model.models import *
from reports.report import *

from pymob.simulation import SimulationBase
from pymob.sim.config import DataVariable
from pymob.sim.parameters import Param
from pymob.solvers.diffrax import JaxSolver
from pymob.sim.plot import SimulationPlot

''' Wrapper to simulate the different model versions (ZGA, Rep) and datasets '''

def regulator_activity(t, t_on, t_off=4.0):
    t = np.asarray(t)
    slope = 1/(t_off - t_on)
    return np.where(t <= t_on, 0.0, np.where(t <= t_off, slope * (t - t_on), 1.0))


def prepare_dataset(gene_id, model_version, t_end, scale="tpm"):

    try:
        transcript_data = xr.load_dataset("data/genes_tpms_white_pauli_JN_BK_mean.nc").sel(time=slice(0, t_end))
        obs = transcript_data.sel(ensembl_gene_id=gene_id).tpm.to_dataset(name="y") 
    except Exception as e:
        raise ValueError(f"gene id {gene_id} not found in dataset.\n {e}") 

    if scale == "log2":
        obs = np.log2(obs+1)
        #cond = (obs["y"] >= -3) | obs["y"].isnull()
        #obs["y"] = obs["y"].where(cond, -3)

    t = np.linspace(0, t_end, 1001)
    r = regulator_activity(t, t_on=3, t_off=4.0)
    rep = xr.Dataset(data_vars= dict(repression = ("time_rep", r)), coords=dict(time_rep=t ))

    rep_on_obs = rep.interp(time_rep=obs.time).drop_vars("time_rep")
    combined_ds = xr.merge([obs, rep_on_obs])

    if model_version in ["Basic", "ZGA_M", "Rep_M", "Rep_V"]:
        return obs ## M decay version without regulator
    else:
        return combined_ds


def gof_evaluation(idata, gene_id, model, out_path):

    row = []

    for src in idata.observed_data.source.values:
        ds = idata.sel(source=src)
        non_nan_times = ds.observed_data.y.time.values[np.isfinite(ds.observed_data.y.values)]
        obs = ds.observed_data.y.sel(time=non_nan_times)
        pred = idata.posterior_model_fits.y.mean(dim=("chain","draw","source")).sel(time=non_nan_times)

        ll = idata.log_likelihood["y"]  # dims: (chain, draw, time, source)
        grouped_ll = ll.sum(dim="source")   # -> dims: (chain, draw, time)

        # Wrap into a fresh InferenceData for WAIC/LOO
        idata_grouped = az.InferenceData(
            posterior=idata.posterior,          # reuse original posterior
            log_likelihood=xr.Dataset({"y": grouped_ll}),
        )

        waic = az.waic(idata_grouped, pointwise=True).elpd_waic
        loo = az.loo(idata_grouped, pointwise=True).elpd_loo

        rho = spearman_correlation(obs, pred)
        pearsonr = pearson_correlation(obs, pred)
        nrmse = calc_nrmse(obs, pred, norm="range")

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
        })

    pd.DataFrame(row).to_csv(os.path.join(out_path, "gof_metrics.csv"), index=False)
    
@click.command() 
@click.option("--gene_id", type=str, default=None,    help="Run a single gene ID (used for array jobs)")
@click.option("--model_version", type=str, default=None,    help="Model version: Basic, ZGA_M, ZGA_Z, Rep_M, Rep_Z")
@click.option("--t_end",   type=int, default=120,     help="Simulation Endpoint")
@click.option("--kernel",  type=str, default="nuts",  help="Inference kernel to use: svi or nuts")
@click.option("--plot",    is_flag=True,              help="generate plots of results")
@click.option("--smooth",  is_flag=True,              help="Produce smoother trajectories with higher time resolution")
@click.option("--skip_duplicates", is_flag=True,      help="Skip duplicate gene IDs for processing")
@click.option("--seed",    type=int, default=1,       help="Random seed for reproducibility")
def main(gene_id, model_version, kernel="nuts", t_end=120, plot=True, smooth=False, skip_duplicates=True, seed=1):

    sim = SimulationBase()
    model = {
        "ZGA_M": ZGA_Model_M(),
        "ZGA_Z": ZGA_Model_Z(),
        "Rep_M": Repression_M(),
        "Rep_Z": Repression_Z(),
        "Rep_V": Repression_V(),
        }[model_version]

    sim.model = model._rhs_jax

    # simulation setup
    sim.config.case_study.name = f"{t_end}_hpf/{model.name}/full"
    sim.config.case_study.scenario = f"{gene_id}"

    # output directories
    out_path = os.getenv("RESULTS_DIR", "./results")
    os.makedirs(out_path, exist_ok=True)
    gene_output_dir = os.path.join(out_path, sim.config.case_study.name , sim.config.case_study.scenario)

    if os.path.exists(os.path.join(gene_output_dir, "numpyro_posterior.nc")) and skip_duplicates:
        print(f"[SKIP] Gene {gene_id} already processed — skipping.")
        return

    os.makedirs(gene_output_dir, exist_ok=True)
    sim.config.case_study.output_path = gene_output_dir
    sim.config.create_directory("scenario", force=True)

    # --- prepare Data ---
    obs = prepare_dataset(gene_id, model_version, t_end, scale="tpm")
    sim.observations = obs

    sim.config.simulation.n_ode_states = 2
    sim.config.simulation.x_dimension = "time"
    sim.config.simulation.seed = seed
    sim.config.report.goodness_of_fit_use_predictions = True

    jax.config.update("jax_enable_x64", True)
    
    sim.solver = JaxSolver
    sim.config.jaxsolver.throw_exception = False
    sim.config.jaxsolver.diffrax_solver = "Dopri5" # Tsit5
    sim.config.jaxsolver.pcoeff = 0.2
    sim.config.jaxsolver.icoeff = 0.4
    sim.config.jaxsolver.icoeff = 0.1
    sim.config.jaxsolver.rtol = 1e-06
    sim.config.jaxsolver.atol = 1e-08
    sim.config.jaxsolver.max_steps = int(1e6)
    sim.solver_post_processing = model._solver_post_processing
    
    # --- Report settings ---
    sim.config.report.diagnostics = True     #  trace/pair plots
    sim.config.report.model = True           #  model code and DAG image
    sim.config.report.plot_trace = False   
    sim.config.report.plot_parameter_pairs = False        
    sim.config.report.table_parameter_estimates = True
    sim.config.report.goodness_of_fit = True

    ## initialise model parameter
    sim, model = {
        "ZGA_M": init_ZGA_M,
        "ZGA_Z": init_ZGA_Z,
        "Rep_M": init_Rep_M,
        "Rep_Z": init_Rep_Z,
        "Rep_V": init_Rep_V,
        }[model_version](sim, model)

    # Simulate
    sim.dispatch_constructor()
    sim.set_inferer("numpyro")
    sim.config.inference_numpyro.kernel = kernel
    sim.config.inference_numpyro.init_strategy= "init_to_median"

    sim.config.inference_numpyro.svi_iterations = 20000
    sim.config.inference_numpyro.svi_learning_rate = 0.001
    sim.config.inference_numpyro.gaussian_base_distribution = True

    sim.config.inference_numpyro.warmup = 1500
    sim.config.inference_numpyro.draws = 2000
    sim.config.inference_numpyro.chains = 4
    sim.config.inference_numpyro.nuts_step_size = 0.1
    sim.config.inference_numpyro.nuts_target_accept_prob = 0.8
    sim.config.inference_numpyro.nuts_dense_mass = True
    sim.config.inference_numpyro.nuts_adapt_step_size = True
    sim.config.inference_numpyro.nuts_adapt_mass_matrix = True

    #sim.prior_predictive_checks(pred_mode="draws")
    sim.dispatch_constructor()

    try:
        sim.inferer.run()
    except Exception as e:
        print(f"[ERROR] Gene {gene_id} failed: {e}")
        with open(os.path.join(out_path, "failed_genes.txt"), "a") as f:
            f.write(f"{sim.config.case_study.name}, {gene_id}, {e}\n")
        return

    if smooth:
        sim.coordinates["time"]= np.linspace(0, t_end, 1000)
        sim.dispatch_constructor()
        p_pred = sim.inferer.posterior_predictions(n=1000, seed=10)
        p_pred.to_netcdf(f"{gene_output_dir}/posterior_predictive.nc")

    idata = sim.inferer.idata
    if model_version in ["Rep_M", "Rep_Z", "Rep_V"]:
        sim.config.model_parameters.t_reg = Param(value=20, free=True, prior=f"lognorm(scale=20, s=1.0)")
        sim.inferer.idata.posterior["t_reg"] = idata.posterior["t_zga"] + idata.posterior["t_rep"]

    sim.inferer.store_results()
    sim.posterior_predictive_checks(pred_mode="mean+hdi", pred_hdi_style={"color": "#7034b1", "alpha": .15})

    sim.report()
    sim.config.save(force=True)

    # evaluation of results
    gof_evaluation(sim.inferer.idata, gene_id, model_version, out_path=gene_output_dir)

    if plot:
        from model.plots import plot_model_results
        plot_model_results(sim.inferer.idata, gene_id, model_version, path=gene_output_dir)


if __name__ == "__main__":
    main()



