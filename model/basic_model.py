from pymob.sim.solvetools import solve_analytic_1d
from pymob.solvers.diffrax import JaxSolver
from pymob.sim.config import DataVariable
from pymob.simulation import SimulationBase
from pymob.sim.config import Param

import xarray as xr
import pandas as pd
import jax.numpy as jnp
import numpy as np
import jax
import os
import click

from model.plots import plot_model_results
from model.dataset import prepare_dataset, tpm_genedata_white

def basic_1s(t, M0, beta, delta):
    '''
    beta: transcription rate
    delta:  degradation rate
    '''
    return M0 * jnp.exp(-delta * t) + beta/delta * (1 - jnp.exp(-delta * t))


@click.command()
@click.option("--gene_id",     type=str, default=None,     help="Run a single gene ID (used for array jobs)")
@click.option("--model",       type=str, default=None,     help="Chose Model: zga, impulse, degradation, basic")
@click.option("--kernel",      type=str, default="nuts",   help="Inference kernel to use: svi or nuts")
@click.option("--t_end",       type=int, default=120,      help="end point of simulation: 120 or 24 hpf")
def main(gene_id, model, kernel, t_end):

    sim = SimulationBase()
    sim.config.case_study.name = f"{model}_{kernel}_{t_end}"
    sim.config.case_study.scenario = gene_id
    sim.config.simulation.x_dimension = "time"
    sim.config.simulation.seed = 2

    # --- Create output directories -
    output = os.getenv("RESULTS_DIR", "./results")
    os.makedirs(output, exist_ok=True)
    gene_output_dir = os.path.join(output, sim.config.case_study.name , gene_id)

    #if os.path.exists(os.path.join(gene_output_dir, "numpyro_posterior.nc")):
    #    print(f"[SKIP] Gene {gene_id} already processed — skipping.")
    #    return

    # --- Prepare the Simulation --
    if model=="basic_1s":
        sim.solver = solve_analytic_1d
    else:
        sim.solver = JaxSolver

    sim.model = basic_1s
    obs = tpm_genedata_white(gene_id).sel(time=slice(0, t_end))
    sim.observations = obs 

    # --- Config Settings ---
    sim.config.data_structure.y = DataVariable(dimensions=["time"], observed=True)
    sim.config.simulation.n_ode_states = 1

    # --- Create output directories ---
    os.makedirs(gene_output_dir, exist_ok=True)
    sim.config.case_study.output_path = gene_output_dir
    sim.config.create_directory("scenario", force=True)
    os.makedirs(output, exist_ok=True)


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
    M0 = obs.sel(time=0).y.item()
    Z_ss0 = sim.observations.y[-3:].mean().item()  # steady-state TPM -> beta

    delta_0 = 0.1 # 6 hours halflife
    beta0 = Z_ss0 * delta_0 + 1e-8

    sim.config.model_parameters.M0 =    Param(value=M0, free=False)
    sim.config.model_parameters.beta =  Param(value=beta0, free=True,     prior=f"lognorm(scale={beta0}, s=1.0)")
    sim.config.model_parameters.delta = Param(value=delta_0, free=True,     prior=f"lognorm(scale={delta_0}, s=1.0)")

    # Error Model
    sim.config.model_parameters.sigma_y = Param(value=0.1,  free=True, prior="lognorm(scale=0.5, s=0.5)")
    sim.config.error_model.y = "normal(loc=0, scale=sigma_y, obs=jnp.log1p(obs) - jnp.log1p(y), obs_inv=jnp.expm1(res + jnp.log1p(y)))"

    sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict
    print("model_parameters:", sim.config.model_parameters.value_dict)
    sim.dispatch_constructor()

    # --- Parameter Estimation & Inferer Settings ---~
    sim.set_inferer("numpyro")
    sim.inferer.config.inference_numpyro.kernel = kernel
    sim.config.jaxsolver.throw_exception = False
    sim.config.inference_numpyro.init_strategy = "init_to_median"

    if kernel == "svi":
        sim.config.inference_numpyro.svi_iterations = 15000
        sim.config.inference_numpyro.svi_learning_rate = 0.001
        sim.config.inference_numpyro.gaussian_base_distribution = True
    elif kernel == "nuts":
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
        with open(os.path.join(output, "failed_genes.txt"), "a") as f:
            f.write(f"{sim.config.case_study.name}, {gene_id}, {e}\n")
        return e
    
    # --- Plots and Results
    sim.inferer.store_results()
    sim.posterior_predictive_checks(pred_mode="mean+hdi", pred_hdi_style={"color": "#7034b1", "alpha": .15}) 
    
    # --- Report and save results
    sim.report()
    sim.save_observations(force=True)
    sim.config.save(force=True)

    ## plot smoother trajectories with higher time resolution
    #sim.coordinates["time"]= np.linspace(0, t_end, 1000)
    #sim.dispatch_constructor()
    #p_pred = sim.inferer.posterior_predictions(n=1000, seed=10)
    #p_pred.to_netcdf(f"{gene_output_dir}/posterior_predictive.nc")


    return f"Finished {gene_id} ({sim.config.case_study.name}). Saved to: {gene_output_dir}"



if __name__ == "__main__":
    main()
