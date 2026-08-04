''' 
Baseline gene expression models
Author: Marie gr. Holthaus
'''

import jax
import jax.numpy as jnp
from pymob.simulation import SimulationBase


class Basic_1s(SimulationBase):

    def __init__(self):
        super().__init__()

        self.name = "Basic_1s"

        # model parameters
        self.params_info = {
            "beta":    {"name": "beta",    "initial": 2.0,  "vary": True,   "prior": "lognorm(scale=2, s=1)"},
            "delta": {"name": "delta", "initial": 1.4,  "vary": True,  },
        }

        # model states
        self.state_variables = {
            "y":          {"dimensions": ["time","source"], "observed": True},              # M + Z
        }

        @staticmethod
        def _rhs_jax(t, M0, beta, delta):
            '''
            beta: transcription rate
            delta:  degradation rate
            '''
            y = M0 * jnp.exp(-delta * t) + beta/delta * (1 - jnp.exp(-delta * t))
            return y


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
            "s": {"name": "s", "initial": 5, "vary": False,},
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
            "s": {"name": "s", "initial": 5, "vary": False,},
        }

        # model states
        self.state_variables = {
            "M":          {"dimensions": ["time",], "observed": False, "y0": 1.0},  # maternal
            "Z":          {"dimensions": ["time",], "observed": False, "y0": 0.0},  # zygotic
            "y":          {"dimensions": ["time",], "observed": True},              # M + Z
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
            "t_rep":        {"name": "t_rep",   "min": 1.0,  "max": 50,  "initial": 15.0, "vary": True,   "prior": "lognorm(scale=15, s=1)"},
            "s": {"name": "s", "initial": 5, "vary": False,},
        }

        # model states
        self.state_variables = {
            "M":          {"dimensions": ["time","source"], "observed": False, "y0": 1.0},  # maternal
            "Z":          {"dimensions": ["time","source"], "observed": False, "y0": 0.0},  # zygotic
            "y":          {"dimensions": ["time","source"], "observed": True},              # M + Z
            "repression": {"dimensions": ["time","source"], "observed": False},

        }

    # right-hand side ODE
    @staticmethod
    def _rhs_jax(t, y, x_in, alpha, beta, delta_m, delta_z, t_zga, t_rep, s):

        M, Z = y
        R_t = x_in.evaluate(t)

        dM_dt = - R_t * delta_m * M

        t_reg = t_zga + t_rep
        on = jax.nn.sigmoid(s * (t - t_zga))
        off =  jax.nn.sigmoid(s * (t - t_reg))
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
            "t_rep":    {"name": "t_rep",   "min": 1.0,  "max": 50,  "initial": 15.0, "vary": True,   "prior": "lognorm(scale=15, s=1)"},
            "s": {"name": "s", "initial": 5, "vary": False,},
        }

        # model states
        self.state_variables = {
            "M":          {"dimensions": ["time","source"], "observed": False, "y0": 1.0},  # maternal
            "Z":          {"dimensions": ["time","source"], "observed": False, "y0": 0.0},  # zygotic
            "y":          {"dimensions": ["time","source"], "observed": True},              # M + Z
        }

    # right-hand side ODE
    @staticmethod
    def _rhs_jax(t, y, alpha, beta, delta_z, delta_m, t_zga, t_rep, s):

        M, Z = y

        dM_dt = - delta_m * M

        t_reg = t_zga + t_rep
        on = jax.nn.sigmoid(s * (t - t_zga))
        off =  jax.nn.sigmoid(s * (t - t_reg))

        beta_on = alpha * on * (1 - off) + beta * off
        dZ_dt = beta_on - delta_z * Z

        return dM_dt, dZ_dt

    @staticmethod
    def _solver_post_processing(results, time, interpolation):
        results["y"] = results["M"] + results["Z"]
        return results
        

class Repression_V():

    def __init__(self):
        self.name = "Rep_V" 

        # model parameters
        self.params_info = {
            "alpha":   {"name": "alpha", "initial": 1.0, "vary": True,   "prior": "lognorm(scale=1, s=2)"},
            "beta":    {"name": "beta",  "initial": 3.0, "vary": True,   "prior": "lognorm(scale=3, s=2)"},

            "delta_z": {"name": "delta_z", "initial": 0.126, "vary": True,   "prior": "lognorm(scale=0.1, s=1)"},
            "delta_r": {"name": "delta_r", "initial": 1.4, "vary": False,},

            "t_deg":   {"name": "t_deg",   "min": 0.0,  "max": 10,  "initial": 3.0, "vary": True,   "prior": "lognorm(scale=3, s=1)"},
            "t_zga":   {"name": "t_zga",  "initial": 3.0, "vary": True,   "prior": "lognorm(scale=3, s=1)"},
            "t_rep":   {"name": "t_rep",   "min": 1.0,  "max": 50,  "initial": 15.0, "vary": True,   "prior": "lognorm(scale=15, s=1)"},
            "s": {"name": "s", "initial": 5, "vary": False,},
        }

        # model states
        self.state_variables = {
            "M":          {"dimensions": ["time","source"], "observed": False, "y0": 1.0},  # maternal
            "Z":          {"dimensions": ["time","source"], "observed": False, "y0": 0.0},  # zygotic
            "y":          {"dimensions": ["time","source"], "observed": True},              # M + Z
        }

    # right-hand side ODE
    @staticmethod
    def _rhs_jax(t, y, alpha, beta, delta_m, delta_z, t_deg, t_zga, t_rep, s):

        M, Z = y

        switch = jax.nn.sigmoid(s * (t - t_deg))
        dM_dt = - switch * delta_m * M

        t_reg = t_zga + t_rep
        on = jax.nn.sigmoid(s * (t - t_zga))
        off =  jax.nn.sigmoid(s * (t - t_reg))
        beta_on = alpha * on * (1 - off) + beta * off

        dZ_dt = beta_on - delta_z * Z

        return dM_dt, dZ_dt

    @staticmethod
    def _solver_post_processing(results, time, interpolation):
        results["y"] = results["M"] + results["Z"]
        return results