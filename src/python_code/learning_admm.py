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
import jax
import jax.numpy as jnp
import time

# from src.ADMM_learning.systems import LTI_2D
from systems import LTI_2D
# from juliacall import Main as jl
# jl.seval('include("src/data_gen/system.jl")')


@jax.jit
def icnn_forward(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    U, W, b = params["U"], params["W"], params["b"]
    act = jax.nn.softplus  # convex & nondecreasing

    z = act(jnp.dot(U[0], x) + b[0])  # first layer (no state W)
    for i in range(1, len(U)):
        z = act(jnp.dot(W[i], z) + jnp.dot(U[i], x) + b[i])

    f = jnp.dot(params["v"], z) + jnp.dot(params["a"], x) + params["c"]
    return f 

@jax.jit
def d_softplus(x: jnp.ndarray) -> jnp.ndarray:
    # derivative of softplus is sigmoid
    return jax.nn.sigmoid(x)

@jax.jit
def icnn_forward_grad(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    """
    Manual gradient df/dx (no autodiff).
    """
    U, W, b, v, a = params["U"], params["W"], params["b"], params["v"], params["a"]

    # ---------- Forward pass (save pre-activations & activations) ----------
    s0 = jnp.dot(U[0], x) + b[0]
    z  = jax.nn.softplus(s0)
    s_list = [s0]

    for i in range(1, len(U)):
        si = jnp.dot(W[i], z) + jnp.dot(U[i], x) + b[i]
        z  = jax.nn.softplus(si)
        s_list.append(si)

    # ---------- Backward pass (manual backprop) ----------
    dL_dz = v           # start from last layer
    grad_x = jnp.zeros_like(x)

    for i in range(len(U) - 1, -1, -1):
        dL_ds = dL_dz * d_softplus(s_list[i])
        grad_x += jnp.dot(U[i].T, dL_ds)

        if i >= 1:
            dL_dz = jnp.dot(W[i].T, dL_ds)

    grad_x += a  # add linear term derivative
    return grad_x



# ====================== System parameters (from system.jl example) ============




# ====================== Class ==============================================

@dataclass
class MPCData:
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray
    x0: np.ndarray
    N: int
    rho: float = 1.0
    max_iter: int = 100
    tol: float = 1e-5
    solver: str = "Clarabel"  # or "OSQP"
    solver_opts: Optional[Dict[str, Any]] = None


# ====================== Prime subproblem (CVXPY) ==============================

class ME:
    def __init__(self, model):
        self.model = model
    def solve(self, query:np.ndarray):
        result = icnn_forward_grad(self.model , jnp.asarray(query))
        return result



# ====================== Aux subproblem (CVXPY) ================================

class AuxSolverCVX:
    """
    Global projection:
      min_v  ||v - p||^2
      s.t.   x_{k+1} = A x_k + B u_k,   x0 fixed,
             x_min <= x_k <= x_max (k=1..N),
             u_min <= u_k <= u_max (k=0..N-1)
    Variable v stacks [x0; u0; x1; u1; ...; xN; uN] (length (N+1)*nz).
    """
    def __init__(self, A: np.ndarray, B: np.ndarray,
                 x_bounds: Tuple[np.ndarray, np.ndarray],
                 u_bounds: Tuple[np.ndarray, np.ndarray],
                 x0: np.ndarray, N: int, nx: int, nu: int,
                 solver: str = "Ipopt",
                 solver_opts: Optional[Dict[str, Any]] = None):
        self.N = N
        self.nz = nx + nu
        self.solver = solver.upper()
        self.solver_opts = solver_opts or {}

        xmin, xmax = x_bounds
        umin, umax = u_bounds

        nvar = (N + 1) * self.nz
        self.v = cp.Variable(nvar)
        self.p = cp.Parameter(nvar)

        obj = cp.sum_squares(self.v - self.p)
        cons = []

        # index helpers
        def idx_x(k): return slice(k*self.nz, k*self.nz + nx)
        def idx_u(k): return slice(k*self.nz + nx, (k+1)*self.nz)

        # x0 fixed
        cons += [self.v[idx_x(0)] == x0]

        # dynamics
        for k in range(N):
            xk = self.v[idx_x(k)]
            uk = self.v[idx_u(k)]
            xkp1 = self.v[idx_x(k+1)]
            cons += [xkp1 == A @ xk + B @ uk]

        # bounds
        for k in range(1, N + 1):
            cons += [self.v[idx_x(k)] >= xmin,
                     self.v[idx_x(k)] <= xmax]
        for k in range(0, N):
            cons += [self.v[idx_u(k)] >= umin,
                     self.v[idx_u(k)] <= umax]

        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def solve(self, p: np.ndarray, init: np.ndarray) -> np.ndarray:
        self.p.value = p.reshape(-1,)
        self.v.value = init
        self.prob.solve(solver=self.solver, warm_start=True, **self.solver_opts)
        return self.v.value, self.prob.solver_stats.solve_time


# ====================== ADMM loop ============================================

def admm_iter(data: MPCData, params, verbose: bool = True):
    
    nx = data.A.shape[0]
    nu = data.B.shape[1]


    nz = nx + nu
    N  = data.N


    prime = ME(params)
    aux = AuxSolverCVX(data.A, data.B,
                       (data.x_min, data.x_max),
                       (data.u_min, data.u_max),
                       data.x0, N, nx, nu,
                       solver=data.solver,
                       solver_opts=(data.solver_opts or {"eps_abs":1e-6, "eps_rel":1e-6, "verbose": False}))

    z = np.zeros((N + 1, nz))
    w = np.zeros((N + 1, nz))
    beta = np.zeros((N + 1, nz))

    hist_r = []
    init = np.zeros((N + 1)*nz)

    total_time = 0.0

    for it in range(1, data.max_iter + 1):
        # z-update
        start = time.time()   
        for k in range(N + 1):
            qk = w[k, :] + beta[k, :]
            me = prime.solve(qk)
            z[k, :] = qk -  np.array(1/data.rho) * me

        total_time += (time.time() - start)

        # w-update
        sol, time_sol = aux.solve(z - beta, init)
        total_time += time_sol
        init = sol
        w    = sol.reshape(N + 1, nz)

        # dual update
        beta += (w - z)

        # residuals
        r = jnp.max(jnp.abs(w - z))

        if r < data.tol:
            if verbose:
                print(f"Learning ADMM converges at it={it}: r={r:.3e}")
            break
        
    print(f"Learning ADMM is solved {total_time} seconds")
    return {"z": z, "w": w, "beta": beta, "r": np.array(hist_r)}


# ====================== Demo ==================================================
if __name__ == "__main__":

    # sys_params = jl.LTI_2D()
    sys_params = LTI_2D()

    data_train = np.load(os.path.join("data","logsum-rho=1-LTI_2D-train.npz"))

    with open(os.path.join('model','logsum-rho=1-LTI_2D-ICNN.pkl'), 'rb') as f:
        params_loaded = pickle.load(f)

    params_sys = MPCData(A=sys_params["A"], B=sys_params["B"], Q=sys_params["Q"], R=sys_params["R"],
                   x_min=sys_params["x_min"], x_max=sys_params["x_max"],
                   u_min=sys_params["u_min"], u_max=sys_params["u_max"],
                   x0=sys_params["x0"], 
                   N=sys_params["N"],  
                   rho = data_train["rho"], 
                   max_iter=100, tol=1e-5,
                   solver="OSQP",
                   solver_opts={"eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 1000, "verbose": False})
    
    out = admm_iter(params_sys, params_loaded)

    z, w = out["z"], out["w"]
    np.set_printoptions(suppress=True)
    print("uz =", np.round(z[:,-1],4))
    print("uw =", np.round(w[:,-1],4))