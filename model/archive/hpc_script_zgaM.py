import os
import click
import arviz as az
import xarray as xr
import numpy as np
import pandas as pd
import jax.numpy as jnp

from model.models import ZGA_Model_M

from pymob.simulation import SimulationBase
from pymob.sim.config import DataVariable
from pymob.sim.parameters import Param
from pymob.solvers.diffrax import JaxSolver
from pymob.sim.plot import SimulationPlot

def prepare_dataset(gene_id, t_end):
    transcript_data = xr.load_dataset("data/white_dataset_mean.nc").sel(time=slice(0, t_end))
    try:
        obs = transcript_data.y.sel(ensembl_gene_id=gene_id).to_dataset(name="y") 
    except KeyError:
        raise ValueError(f"gene id {gene_id} not found in dataset.") 
    return obs


def _simulate(gene_id, kernel="nuts", t_end=120, seed=1):

    output = "results"
    os.makedirs(output, exist_ok=True)

    mod = ZGA_Model_M() # M-decay

    # --- Initialize simulation ---
    sim = SimulationBase()
    sim.model = mod._rhs_jax
    sim.config.case_study.name = f"{mod.name}_{kernel}_{t_end}"
    sim.config.case_study.scenario = gene_id

    # --- Output Directory ---
    gene_output_dir = os.path.join(output, sim.config.case_study.name, gene_id)
    #if os.path.exists(os.path.join(gene_output_dir, "numpyro_posterior.nc")):
    #    print(f"[SKIP] Gene {gene_id} already processed — skipping.")
    #    return
        
    os.makedirs(gene_output_dir, exist_ok=True)
    sim.config.case_study.output_path = gene_output_dir
    sim.config.create_directory("scenario", force=True)

    # --- prepare Data ---
    obs = prepare_dataset(gene_id, t_end)
    sim.observations = obs

    sim.config.simulation.n_ode_states = 2
    sim.config.simulation.x_dimension = "time"
    sim.config.report.goodness_of_fit_use_predictions = True
    sim.config.simulation.seed = seed
    
    sim.solver = JaxSolver
    sim.config.jaxsolver.throw_exception = False
    sim.config.jaxsolver.diffrax_solver = "Tsit5"
    sim.config.jaxsolver.pcoeff = 0.2
    sim.config.jaxsolver.icoeff = 0.4
    sim.config.jaxsolver.rtol = 1e-04
    sim.config.jaxsolver.atol = 1e-06
    sim.solver_post_processing = mod._solver_post_processing
    
    # --- Report settings ---
    sim.config.report.diagnostics = True     #  trace/pair plots
    sim.config.report.model = True           #  model code and DAG image
    sim.config.report.plot_trace = False   
    sim.config.report.plot_parameter_pairs = False        
    sim.config.report.table_parameter_estimates = True
    sim.config.report.goodness_of_fit = True

    # --- Initialize parameters ---
    sim.config.data_structure.M = DataVariable(dimensions=("time",), observed=False)
    sim.config.data_structure.Z = DataVariable(dimensions=("time",), observed=False)

    # inital condiion
    M0 =  sim.observations.sel(time=0).y.item()
    mod.state_variables["M"]["y0"] = M0
    mod.state_variables["Z"]["y0"] = 0.0

    sim.config.simulation.y0 = [f"{k}={v['y0']}" for k, v in mod.state_variables.items() if "y0" in v]
    sim.model_parameters["y0"] = sim.parse_input("y0", sim.observations, drop_dims="time")

    Z_ss0 = sim.observations.y[-3:].mean().item()  # steady-state TPM -> beta
    print("Mean TPM", Z_ss0) 

    delta_m = 0.35 # t1/2 = 3h
    t1 = 3.0
    delta_z0 = 0.1 # t12 = 6h
    beta0 = Z_ss0 * delta_z0 + 1e-8

    sim.config.model_parameters.delta_z = Param(value=delta_z0, free=True,     prior=f"lognorm(scale={delta_z0}, s=1.0)")
    sim.config.model_parameters.beta =    Param(value=beta0, free=True,        prior=f"lognorm(scale={beta0}, s=0.5)")

    sim.config.model_parameters.t_zga =   Param(value=t1, free=True,      prior=f"lognorm(scale={t1}, s=1.0)")
    sim.config.model_parameters.delta_m = Param(value=delta_m, free=True,    prior=f"lognorm(scale={delta_m}, s=1.0)")
    sim.config.model_parameters.s =   Param(value=5, free=False)

    # Error Model
    sim.config.model_parameters.sigma_y = Param(value=0.3,  free=True, prior="lognorm(scale=0.5, s=0.5)")
    sim.config.error_model.y = "normal(loc=0, scale=sigma_y, obs=jnp.log1p(obs) - jnp.log1p(y), obs_inv=jnp.expm1(res + jnp.log1p(y)))"
    sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict

    print("model_parameters:", sim.config.model_parameters.value_dict)

    sim.dispatch_constructor()
    sim.set_inferer("numpyro")
    sim.config.inference_numpyro.kernel = kernel
    sim.config.inference_numpyro.init_strategy= "init_to_median"
    #sim.config.inference_numpyro.init_strategy= "init_to_value"

    sim.config.inference_numpyro.svi_iterations = 20000
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

    #sim.prior_predictive_checks(pred_mode="draws")
    sim.dispatch_constructor()

    try:
        sim.inferer.run()
    except Exception as e:
        print(f"[ERROR] Gene {gene_id} failed: {e}")
        with open(os.path.join(output, "failed_genes.txt"), "a") as f:
            f.write(f"{sim.config.case_study.name}, {gene_id}, {e}\n")
        return 

    sim.inferer.store_results()
    sim.posterior_predictive_checks(pred_mode="mean+hdi", pred_hdi_style={"color": "#7034b1", "alpha": .15})
    sim.report()
    sim.config.save(force=True)

    ## plot smoother trajectories with higher time resolution
    #sim.coordinates["time"]= np.linspace(0, t_end, 1000)
    #sim.dispatch_constructor()
    #p_pred = sim.inferer.posterior_predictions(n=1000, seed=10)
    #p_pred.to_netcdf(f"{gene_output_dir}/posterior_predictive.nc")

    # --- Plotting ---
    # from model.plots import plot_model_results
    #plot_model_results(sim.inferer.idata, gene_id, mod.name, path=gene_output_dir)
