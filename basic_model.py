from pymob.sim.solvetools import solve_analytic_1d
from pymob.solvers.diffrax import JaxSolver
from pymob.sim.config import DataVariable
from pymob.simulation import SimulationBase
from pymob.sim.config import Param

from model.plots import plot_model_results
from model.initialise import init_Basic
from reports.report import *

import jax.numpy as jnp
import os
import click
import xarray as xr
import numpy as np
import pandas as pd


def prepare_dataset(gene_id, t_end):

    try:
        transcript_data = xr.load_dataset("data/genes_tpms_white_pauli_JN_BK_mean.nc").sel(time=slice(0, t_end))
        obs = transcript_data.sel(ensembl_gene_id=gene_id).tpm.to_dataset(name="y") 
    except Exception as e:
        raise FileNotFoundError(f"gene id {gene_id} not found in dataset. \n {e}") 

    return obs


def gof_evaluation(idata, gene_id, model, out_path):

    row = []

    for src in idata.observed_data.source.values:
        ds = idata.sel(source=src)
        non_nan_times = ds.observed_data.y.time.values[np.isfinite(ds.observed_data.y.values)]
        obs = ds.observed_data.y.sel(time=non_nan_times).values
        pred = idata.posterior_model_fits.y.mean(dim=("chain","draw","source")).sel(time=non_nan_times).values

        metrics = pd.DataFrame(columns=["gene_id", "model", "BIC", "rho", "NRMSE", "MASE"])

        rho = spearman_correlation(obs, pred)
        nrmse = calc_nrmse(obs, pred)[0]         # by Range

        row.append({
            "gene_id":gene_id,
            "model":model,
            "source": src,
            "BIC": calc_bic(ds),
            "rho": rho,
            "NRMSE": nrmse, 
            "MASE": calc_mase(obs, pred),
        })

    pd.DataFrame(row).to_csv(os.path.join(out_path, "gof_metrics.csv"), index=False)

def make_basic_1s(n_source):
    def basic_1s(t, M0, beta, delta):
        '''
        beta: transcription rate
        delta: degradation rate
        '''
        y = M0 * jnp.exp(-delta * t) + beta/delta * (1 - jnp.exp(-delta * t))
        return jnp.tile(y[:, None], (1, n_source))  # (time,) -> (time, n_source)
    return basic_1s

@click.command() 
@click.option("--gene_id", type=str, default=None,    help="Run a single gene ID (used for array jobs)")
@click.option("--t_end",   type=int, default=120,     help="Simulation Endpoint")
@click.option("--kernel",  type=str, default="nuts",  help="Inference kernel to use: svi or nuts")
@click.option("--plot",    is_flag=True,              help="generate plots of results")
@click.option("--smooth",  is_flag=True,              help="Produce smoother trajectories with higher time resolution")
@click.option("--skip_duplicates", is_flag=True,      help="Skip duplicate gene IDs for processing")
@click.option("--seed",    type=int, default=1,       help="Random seed for reproducibility")
def main(gene_id, kernel="nuts", t_end=120, plot=True, smooth=False, skip_duplicates=True, seed=1):

    sim = SimulationBase()
    model = "Basic"

    sim.config.case_study.name = f"{t_end}_hpf/{model}/all"
    sim.config.case_study.scenario = gene_id
    
    # --- Create output directories -
    out_path = os.getenv("RESULTS_DIR", "./results")
    os.makedirs(out_path, exist_ok=True)
    gene_output_dir = os.path.join(out_path, sim.config.case_study.name , gene_id)

    if os.path.exists(os.path.join(gene_output_dir, "numpyro_posterior.nc")) and skip_duplicates:
        print(f"[SKIP] Gene {gene_id} already processed — skipping.")
        return

    # --- Create output directories ---
    os.makedirs(gene_output_dir, exist_ok=True)
    sim.config.case_study.output_path = gene_output_dir
    sim.config.create_directory("scenario", force=True)

    # --- Prepare the Simulation --
    obs = prepare_dataset(gene_id, t_end)
    sim.observations = obs 
    n_source = obs.sizes["source"]
    sim.model = make_basic_1s(n_source)

    # --- Config Settings ---
    sim.config.simulation.x_dimension = "time"
    sim.config.simulation.n_ode_states = 1
    sim.config.simulation.seed = seed
    sim.config.report.goodness_of_fit_use_predictions = True

    sim.solver = solve_analytic_1d
    sim.config.jaxsolver.throw_exception = False
    sim.config.jaxsolver.diffrax_solver = "Tsit5"
    sim.config.jaxsolver.pcoeff = 0.2
    sim.config.jaxsolver.icoeff = 0.4
    sim.config.jaxsolver.rtol = 1e-04
    sim.config.jaxsolver.atol = 1e-06

    # --- Report settings ---
    sim.config.report.diagnostics = True     # skips trace/pair plots
    sim.config.report.model = True           # skips model code and DAG image
    sim.config.report.plot_trace = False   
    sim.config.report.plot_parameter_pairs = False        
    sim.config.report.table_parameter_estimates = True
    sim.config.report.goodness_of_fit = True

    # --- Parameterize ---
    sim = init_Basic(sim)

    # --- Parameter Estimation & Inferer Settings ---~
    sim.dispatch_constructor()
    sim.set_inferer("numpyro")
    sim.inferer.config.inference_numpyro.kernel = kernel
    sim.config.inference_numpyro.init_strategy = "init_to_median"

    sim.config.inference_numpyro.svi_iterations = 15000
    sim.config.inference_numpyro.svi_learning_rate = 0.001
    sim.config.inference_numpyro.gaussian_base_distribution = True

    sim.config.inference_numpyro.warmup = 1000
    sim.config.inference_numpyro.draws = 2000
    sim.config.inference_numpyro.chains = 4
    sim.config.inference_numpyro.nuts_step_size = 0.1
    sim.config.inference_numpyro.nuts_target_accept_prob = 0.95
    sim.config.inference_numpyro.nuts_dense_mass = True
    sim.config.inference_numpyro.nuts_adapt_step_size = True
    sim.config.inference_numpyro.nuts_adapt_mass_matrix = True

    sim.dispatch_constructor()

    try:
        sim.inferer.run()
    except Exception as e:
        print(f"[ERROR] Gene {sim.config.case_study.scenario} failed: {e}")
        with open(os.path.join(out_path, "failed_genes.txt"), "a") as f:
            f.write(f"{sim.config.case_study.name}, {gene_id}, {e}\n")
        return
    
    # --- Plots and Results
    sim.inferer.store_results()
    sim.posterior_predictive_checks(pred_mode="mean+hdi", pred_hdi_style={"color": "#7034b1", "alpha": .15}) 
    sim.report()
    sim.config.save(force=True)

    # evaluation of results
    gof_evaluation(sim.inferer.idata, gene_id, "Basic", out_path=gene_output_dir)

    if smooth:
        sim.coordinates["time"]= np.linspace(0, t_end, 1000)
        sim.dispatch_constructor()
        p_pred = sim.inferer.posterior_predictions(n=1000, seed=10)
        p_pred.to_netcdf(f"{gene_output_dir}/posterior_predictive.nc")

    if plot:
        plot_model_results(sim.inferer.idata, gene_id, model_version="Basic", path=gene_output_dir)


if __name__ == "__main__":
    main()

