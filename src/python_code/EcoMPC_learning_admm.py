# mpc_admm_cvxpy.py
# -----------------------------------------------------------------------------
# ADMM for linear MPC implemented with **CVXPY**.
# - Prime step: per-stage QP solved by CVXPY (OSQP/ECOS)
# - Aux step: global projection QP solved by CVXPY
# - Includes `lti_parameter()` and cost utilities akin to your Julia files.
#
# Usage:
#   pip install cvxpy osqp numpy
#   python mpc_admm_cvxpy.py
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cvxpy as cp
import os
import pickle
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import mICNN as mICNN  # assumes myICNN_JAX.py is in the same directory 



# ====================== System parameters (from system.jl example) ============

def lti_parameter():
    nu = 6
    N = 24
    au = np.array([4, 3, 5, 6, 4, 5])
    tau = np.array([23, 26, 25, 23, 20, 16, 17, 23, 23, 22, 21, 20,
                  21, 22, 21, 20, 25, 30, 33, 32, 28, 27, 25, 21])
    u_min = np.full(nu, 0.0)
    return nu, N, u_min, au, tau


# ====================== Cost functions ====================

def economic_cost(u: np.ndarray, nu: int, k: int, au: np.ndarray, tau: np.ndarray, eta: float = 5) -> float:
    u_i = u[:nu]
    au_i = au[:nu]
    tau_k = tau[k]
    return np.sum(1.0/np.minimum(u_i, au_i) - 1.0/au_i) + eta * np.maximum(np.sum(u_i) - tau_k, 0) + float((np.sum(u_i) - 2.)**2)


# ====================== Class ==============================================

@dataclass
class MPCData:
    u_min: np.ndarray
    N: int
    nu: int
    au: np.ndarray
    tau: np.ndarray
    rho: float = 1.0
    max_iter: int = 100
    tol: float = 1e-5
    solver: str = "OSQP"  # or "ECOS"
    solver_opts: Optional[Dict[str, Any]] = None


# ====================== Prime subproblem (CVXPY) ==============================

class ME:
    def __init__(self, model):
        self.model = model
    def solve(self, query:np.ndarray):
        result = mICNN.grad_wrt_x(self.model , jnp.asarray(query))
        return result



# ====================== Aux subproblem (CVXPY) ================================

class AuxSolverCVX:
    """
    Global projection:
      min_v  ||v - p||^2
      s.t.   x_{k+1} = A x_k + B u_k,   u0 fixed,
             x_min <= x_k <= x_max (k=1..N),
             u_min <= u_k <= u_max (k=0..N-1)
    Variable v stacks [u0; u0; x1; u1; ...; xN; uN] (length (N+1)*nz).
    """
    def __init__(self, u_bounds: np.ndarray,
                 N: int, nu: int,
                 solver: str = "OSQP",
                 solver_opts: Optional[Dict[str, Any]] = None):
        self.N = N
        self.nu = nu
        self.solver = solver.upper()
        self.solver_opts = solver_opts or {}
        u_min= u_bounds

        nvar = N * self.nu
        self.v = cp.Variable(nvar)
        self.p = cp.Parameter(nvar)

        obj = cp.sum_squares(self.v - self.p)

        def idx_u(k):
            return slice(k * nu, (k + 1) * nu)

        # Constraints: u_k >= u_min for all k
        constraints = []
        for k in range(N):
            constraints.append(self.v[idx_u(k)] >= u_min)

        # Objective
        obj = cp.sum_squares(self.v - self.p)
        self.prob = cp.Problem(cp.Minimize(obj), constraints)

    def solve(self, p: np.ndarray) -> np.ndarray:
        self.p.value = p.ravel()  # ensure 1D
        self.prob.solve(solver=self.solver, warm_start=True, **self.solver_opts)
        # Return as (N, nu) for convenience
        return self.v.value.reshape(self.N, self.nu)


# ====================== ADMM loop ============================================

def admm_iter(data: MPCData, params, verbose: bool = True):

    nu = data.nu
    N  = data.N
    au = data.au
    tau = data.tau


    prime = ME(params)
    aux = AuxSolverCVX(data.u_min,
                       N, nu,
                       solver=data.solver,
                       solver_opts=(data.solver_opts or {"eps_abs":1e-6, "eps_rel":1e-6, "verbose": False}))

    z = np.zeros((N, nu))
    w = np.zeros((N, nu))
    beta = np.zeros((N, nu))

    hist_r = []
    J_ADMM = 0.0

    for it in range(1, data.max_iter + 1):
        # z-update
        for k in range(N):
            qk = w[k, :] + beta[k, :]
            me = prime.solve(qk)
            z[k, :] = qk -  np.array(1/data.rho) * me

        # w-update
        p = z - beta
        w = aux.solve(p)

        # dual update
        beta = beta + (w - z)

        # residuals
        r = np.linalg.norm(w - z)
        hist_r.append(r)

        if verbose and (it % 5 == 0 or it == 1):
            print(f"[it {it}] r={r:.3e}")

        if r < 1e-5:
            if verbose:
                print(f"Converged at it={it}: r={r:.3e}")
            break
        

    for k in range(N):
        J_ADMM += economic_cost(z[k, :], nu, k, au, tau)


    return {"z": z, "w": w, "beta": beta, "r": np.array(hist_r), "J_ADMM": J_ADMM}


# ====================== Demo ==================================================
if __name__ == "__main__":
    model_path = os.path.join('model', 'Economic-rho=2.0.pkl')
    with open(model_path, 'rb') as f:
        params_loaded = pickle.load(f)

    nu, N, u_min, au, tau = lti_parameter()

    data = MPCData(u_min=u_min,
                   N=N, nu=nu, au=au, tau=tau, rho=2., max_iter=200, tol=1e-6,
                   solver="OSQP",
                   solver_opts={"eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 1000, "verbose": False})
    out = admm_iter(data, params_loaded)
    J_ADMM = out["J_ADMM"]
    print(f"\nFinal ADMM cost: J_ADMM = {J_ADMM}")