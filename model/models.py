''' 
Baseline gene expression models
Author: Marie gr. Holthaus
'''

import jax
import jax.numpy as jnp
from pymob.simulation import SimulationBase

class ZGA_Model_Z(SimulationBase):

    def __init__(self):
        super().__init__()

        self.name = "ZGA_Z" 

        # model parameters
        self.params_info = {
            "beta":    {"name": "beta",    "initial": 2.0,  "vary": True,   "prior": "lognorm(scale=2, s=1)"},
            "delta_r": {"name": "delta_r", "initial": 1.4,  "vary": False,  },
            "delta_z": {"name": "delta_z", "initial": 0.5,  "vary": True,   "prior": "lognorm(scale=0.5, s=1)"},
            "t_zga":   {"name": "t_zga",   "initial": 3.0,  "vary": True,   "prior": "lognorm(scale=3, s=0.2)"},
        }

        # model states
        self.state_variables = {
            "M":          {"dimensions": ["time",], "observed": False, "y0": 1.0},  # maternal
            "Z":          {"dimensions": ["time",], "observed": False, "y0": 0.0},  # zygotic
            "y":          {"dimensions": ["time",], "observed": True},              # M + Z
            "repression": {"dimensions": ["time",], "observed": False},
        }

    # right-hand side ODE
    @staticmethod
    def _rhs_jax(t, y, x_in, beta, delta_m, delta_z, t_zga, s):

        M, Z = y
        # interpolate repressor profile from x_in
        R_t = x_in.evaluate(t)
        dM_dt = -(R_t * delta_m) * M

        on = jax.nn.sigmoid(s * (t - t_zga))
        dZ_dt = beta * on - delta_z * Z

        return dM_dt, dZ_dt

    @staticmethod
    def _solver_post_processing(results, time, interpolation):
        # add total transcript = maternal + zygotic
        results["y"] = results["M"] + results["Z"]
        # track interpolated input (repressor profile)
        results["repression"] = jax.vmap(interpolation.evaluate)(time)

        return results

class ZGA_Model_M(SimulationBase):

    def __init__(self):
        super().__init__()

        self.name = "ZGA_M" 

        # model parameters
        self.params_info = {
            "beta":    {"name": "beta",    "initial": 2.0,  "vary": True,   "prior": "lognorm(scale=2, s=1)"},
            "delta_m": {"name": "delta_r", "initial": 0.28,  "vary": False,  },
            "delta_z": {"name": "delta_z", "initial": 0.5,  "vary": True,   "prior": "lognorm(scale=0.5, s=1)"},
            "t_zga":   {"name": "t_zga",   "initial": 3.0,  "vary": True,   "prior": "lognorm(scale=3, s=0.2)"},
        }

        # model states
        self.state_variables = {
            "M":          {"dimensions": ["time",], "observed": False, "y0": 1.0},  # maternal
            "Z":          {"dimensions": ["time",], "observed": False, "y0": 0.0},  # zygotic
            "y":          {"dimensions": ["time",], "observed": True},              # M + Z
            "repression": {"dimensions": ["time",], "observed": False},
        }

    # right-hand side ODE
    @staticmethod
    def _rhs_jax(t, y, beta, delta_z, delta_m, t_zga, s):

        M, Z = y

        dM_dt = - delta_m * M
        on = jax.nn.sigmoid(s * (t - t_zga))
        dZ_dt = beta * on - delta_z * Z
        
        return dM_dt, dZ_dt

    @staticmethod
    def _solver_post_processing(results, time, interpolation):
        # add total transcript = maternal + zygotic
        results["y"] = results["M"] + results["Z"]
        return results

class Repression_Z():

    def __init__(self):
        self.name = "Rep_Z" 

        # model parameters
        self.params_info = {
            "alpha":    {"name": "alpha", "initial": 1.0, "vary": True,   "prior": "lognorm(scale=1, s=2)"},
            "beta":      {"name": "beta",  "initial": 3.0, "vary": True,   "prior": "lognorm(scale=3, s=2)"},

            "delta_z":   {"name": "delta_z", "initial": 0.126, "vary": True,   "prior": "lognorm(scale=0.1, s=1)"},
            "delta_r":   {"name": "delta_r", "initial": 1.4, "vary": False,},

            "t_zga":    {"name": "t_zga",  "initial": 3.0, "vary": True,   "prior": "lognorm(scale=3, s=1)"},
            "dt_rep":        {"name": "t_rep",   "min": 1e-3,  "max": 50,  "initial": 15.0, "vary": True,   "prior": "lognorm(scale=15, s=1)"},
        }

        # model states
        self.state_variables = {
            "M":          {"dimensions": ["time",], "observed": False, "y0": 1.0},  # maternal
            "Z":          {"dimensions": ["time",], "observed": False, "y0": 0.0},  # zygotic
            "y":          {"dimensions": ["time",], "observed": True},              # M + Z
            "repression": {"dimensions": ["time",], "observed": False},

        }

    # right-hand side ODE
    @staticmethod
    def _rhs_jax(t, y, x_in, alpha, beta, delta_m, delta_z, t_zga, dt_rep, s):

        M, Z = y
        R_t = x_in.evaluate(t)

        dM_dt = - R_t * delta_m * M

        t_rep = t_zga + dt_rep
        on = jax.nn.sigmoid(s * (t - t_zga))
        off =  jax.nn.sigmoid(s * (t - t_rep))
        beta_on = alpha * on * (1 - off) + beta * off

        dZ_dt = beta_on - delta_z * Z

        return dM_dt, dZ_dt

    @staticmethod
    def _solver_post_processing(results, time, interpolation):
        results["y"] = results["M"] + results["Z"]
        results["repression"] = jax.vmap(interpolation.evaluate)(time)
        return results


class Repression_M():

    def __init__(self):
        self.name = "Rep_M" 

        # model parameters
        self.params_info = {
            "alpha":    {"name": "alpha", "initial": 1.0, "vary": True,   "prior": "lognorm(scale=1, s=2)"},
            "beta":      {"name": "beta",  "initial": 3.0, "vary": True,   "prior": "lognorm(scale=3, s=2)"},

            "delta_z":   {"name": "delta_z", "initial": 0.126, "vary": True,   "prior": "lognorm(scale=0.1, s=1)"},
            "delta_m":   {"name": "delta_r", "initial": 0.7, "vary": False,},

            "t_zga":    {"name": "t_zga",  "initial": 3.0, "vary": True,   "prior": "lognorm(scale=3, s=1)"},
            "dt_rep":        {"name": "t_rep",   "min": 1e-3,  "max": 50,  "initial": 15.0, "vary": True,   "prior": "lognorm(scale=15, s=1)"},
        }

        # model states
        self.state_variables = {
            "M":          {"dimensions": ["time",], "observed": False, "y0": 1.0},  # maternal
            "Z":          {"dimensions": ["time",], "observed": False, "y0": 0.0},  # zygotic
            "y":          {"dimensions": ["time",], "observed": True},              # M + Z
        }

    # right-hand side ODE
    @staticmethod
    def _rhs_jax(t, y, alpha, beta, delta_z, delta_m, t_zga, dt_rep, s):

        M, Z = y

        dM_dt = - delta_m * M

        t_rep = t_zga + dt_rep
        on = jax.nn.sigmoid(s * (t - t_zga))
        off =  jax.nn.sigmoid(s * (t - t_rep))

        beta_on = alpha * on * (1 - off) + beta * off
        dZ_dt = beta_on - delta_z * Z

        return dM_dt, dZ_dt

    @staticmethod
    def _solver_post_processing(results, time, interpolation):
        results["y"] = results["M"] + results["Z"]
        return results
        