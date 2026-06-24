import numpy as np
import pandas as pd
import jax.numpy as jnp

from pymob.sim.config import DataVariable
from pymob.sim.parameters import Param
from pymob.solvers.diffrax import JaxSolver

def init_Basic(sim, model):

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

    return sim, model

def init_ZGA_M(sim, model):

    # --- Initialize parameters ---
    sim.config.data_structure.M = DataVariable(dimensions=("time",), observed=False)
    sim.config.data_structure.Z = DataVariable(dimensions=("time",), observed=False)

    # inital condiion
    M0 =  sim.observations.sel(time=0).y.item()
    model.state_variables["M"]["y0"] = M0
    model.state_variables["Z"]["y0"] = 0.0

    sim.config.simulation.y0 = [f"{k}={v['y0']}" for k, v in model.state_variables.items() if "y0" in v]
    sim.model_parameters["y0"] = sim.parse_input("y0", sim.observations, drop_dims="time")

    Z_ss0 = sim.observations.y[-3:].mean().item()  # steady-state TPM -> beta
    print("Mean TPM (Z_ss0): ", Z_ss0) 

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
    
    return sim, model

def init_ZGA_Z(sim, model):
    # --- Initialize parameters ---
    sim.config.data_structure.M = DataVariable(dimensions=("time",), observed=False)
    sim.config.data_structure.Z = DataVariable(dimensions=("time",), observed=False)

    # inital condiion
    M0 =  sim.observations.sel(time=0).y.item()
    model.state_variables["M"]["y0"] = M0
    model.state_variables["Z"]["y0"] = 0.0

    sim.config.simulation.y0 = [f"{k}={v['y0']}" for k, v in model.state_variables.items() if "y0" in v]
    sim.model_parameters["y0"] = sim.parse_input("y0", sim.observations, drop_dims="time")
    
    # input data - x_in
    sim.config.simulation.x_in = ["repression=repression"]
    sim.model_parameters["x_in"] = sim.parse_input(input="x_in", reference_data=sim.observations, drop_dims=[])
    sim.config.data_structure.repression.observed = False

    Z_ss0 = sim.observations.y[-3:].mean().item()  # steady-state TPM -> beta
    print("Mean TPM", Z_ss0) 

    delta_r0 = 1.4  # t12 = 30 min
    t1 = 3.0
    delta_z0 = 0.1 # t12 = 6h
    beta0    = Z_ss0 * delta_z0 + 1e-8
   
    sim.config.model_parameters.delta_z = Param(value=delta_z0, free=True,     prior=f"lognorm(scale={delta_z0}, s=1.0)")
    sim.config.model_parameters.beta =    Param(value=beta0, free=True,        prior=f"lognorm(scale={beta0}, s=0.5)")

    sim.config.model_parameters.t_zga =   Param(value=t1, free=True,      prior=f"lognorm(scale={t1}, s=1.0)")
    sim.config.model_parameters.s =   Param(value=5, free=False)
    sim.config.model_parameters.delta_m = Param(value=delta_r0, free=True, prior=f"lognorm(scale={delta_r0}, s=0.5)")

    # Error Model
    sim.config.model_parameters.sigma_y = Param(value=0.1,  free=True, prior="lognorm(scale=0.5, s=0.5)")
    sim.config.error_model.y = "normal(loc=0, scale=sigma_y, obs=jnp.log1p(obs) - jnp.log1p(y), obs_inv=jnp.expm1(res + jnp.log1p(y)))"
    sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict

    print("model_parameters:", sim.config.model_parameters.value_dict)

    return sim, model


def init_Rep_M(sim, model):

    # --- Initialize parameters ---
    sim.config.data_structure.M = DataVariable(dimensions=("time",), observed=False)
    sim.config.data_structure.Z = DataVariable(dimensions=("time",), observed=False)

    # inital condiion
    M0 =  sim.observations.sel(time=0).y.item()
    model.state_variables["M"]["y0"] = M0
    model.state_variables["Z"]["y0"] = 0.0
    model.state_variables["M"]["y0"] = sim.observations.y[0].item()
    
    sim.config.simulation.y0 = [f"{k}={v['y0']}" for k, v in model.state_variables.items() if "y0" in v]
    sim.model_parameters["y0"] = sim.parse_input("y0", sim.observations, drop_dims="time")
    
    Z_max0 = sim.observations.y[2:-2].max().item() # steady-state TPM -> alpha
    Z_ss0 = sim.observations.y[-3:].mean().item()  # steady-state TPM -> beta
    print("Mean TPM", Z_max0, Z_ss0) 

    delta_m = 0.35 # t1/2 = 3h
    t1 = 3.0
    dt2 = 6.0
    delta_z0 = 0.1 # t12 = 6h
    alpha0    = Z_max0 * delta_z0 + 1e-8  # first rate
    beta0    = Z_ss0 * delta_z0 + 1e-8  # final rate

    sim.config.model_parameters.alpha = Param(value=alpha0, free=True, prior=f"lognorm(scale={alpha0}, s=0.5)"    )
    sim.config.model_parameters.beta = Param(value=beta0, free=True, prior=f"lognorm(scale={beta0}, s=0.5)"    )

    sim.config.model_parameters.delta_z = Param(value=delta_z0, free=True, prior=f"lognorm(scale={delta_z0}, s=1.0)"    )
    sim.config.model_parameters.delta_m = Param(value=delta_m, free=True,    prior=f"lognorm(scale={delta_m}, s=1.0)")

    sim.config.model_parameters.t_zga =   Param(value=t1, free=True,      prior=f"lognorm(scale={t1}, s=1.0)")
    sim.config.model_parameters.dt_rep =   Param(value=dt2,  free=True,      prior=f"lognorm(scale={dt2}, s=1.0)")
    sim.config.model_parameters.s =   Param(value=5, free=False)

    # Error Model
    sim.config.model_parameters.sigma_y = Param(value=0.3,  free=True, prior="lognorm(scale=0.5, s=0.5)")
    sim.config.error_model.y = "normal(loc=0, scale=sigma_y, obs=jnp.log1p(obs) - jnp.log1p(y), obs_inv=jnp.expm1(res + jnp.log1p(y)))"
    sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict

    print("model_parameters:", sim.config.model_parameters.value_dict)
    
    return sim, model


def init_Rep_Z(sim, model):

    # --- Initialize parameters ---
    sim.config.data_structure.M = DataVariable(dimensions=("time",), observed=False)
    sim.config.data_structure.Z = DataVariable(dimensions=("time",), observed=False)

    # inital condiion
    M0 =  sim.observations.sel(time=0).y.item()
    model.state_variables["M"]["y0"] = M0
    model.state_variables["Z"]["y0"] = 0.0
    model.state_variables["M"]["y0"] = sim.observations.y[0].item()
    
    sim.config.simulation.y0 = [f"{k}={v['y0']}" for k, v in model.state_variables.items() if "y0" in v]
    sim.model_parameters["y0"] = sim.parse_input("y0", sim.observations, drop_dims="time")

    # input data - x_in
    sim.config.simulation.x_in = ["repression=repression"]
    sim.model_parameters["x_in"] = sim.parse_input(input="x_in", reference_data=sim.observations, drop_dims=[])
    sim.config.data_structure.repression.observed = False
    
    Z_max0 = sim.observations.y[2:-2].max().item() # steady-state TPM -> alpha
    Z_ss0 = sim.observations.y[-3:].mean().item()  # steady-state TPM -> beta
    print("Mean TPM", Z_max0, Z_ss0) 

    delta_r = 1.4  # t12 = 30 min

    t1 = 3.0
    dt2 = 6.0
    delta_z0 = 0.1 # t12 = 6h
    alpha0   = Z_max0 * delta_z0 + 1e-8 # first rate
    beta0    = Z_ss0 * delta_z0 + 1e-8 # final rate

    sim.config.model_parameters.alpha = Param(value=alpha0, free=True, prior=f"lognorm(scale={alpha0}, s=0.5)")
    sim.config.model_parameters.beta = Param(value=beta0, free=True, prior=f"lognorm(scale={beta0}, s=0.5)")

    sim.config.model_parameters.delta_z = Param(value=delta_z0, free=True, prior=f"lognorm(scale={delta_z0}, s=1.0)")
    sim.config.model_parameters.delta_m = Param(value=delta_r, free=True, prior=f"lognorm(scale={delta_r}, s=0.5)")

    sim.config.model_parameters.t_zga =   Param(value=t1, free=True,      prior=f"lognorm(scale={t1}, s=1.0)")
    sim.config.model_parameters.dt_rep =   Param(value=dt2,  free=True,      prior=f"lognorm(scale={dt2}, s=1.0)")
    sim.config.model_parameters.s =   Param(value=5, free=False)

    # Error Model
    sim.config.model_parameters.sigma_y = Param(value=0.3,  free=True, prior="lognorm(scale=0.5, s=0.5)")
    sim.config.error_model.y = "normal(loc=0, scale=sigma_y, obs=jnp.log1p(obs) - jnp.log1p(y), obs_inv=jnp.expm1(res + jnp.log1p(y)))"
    sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict

    print("model_parameters:", sim.config.model_parameters.value_dict)

    return sim, model


