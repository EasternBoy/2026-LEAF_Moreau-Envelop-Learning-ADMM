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
from typing import Optional, Tuple, Dict, Any, NamedTuple
import numpy as np
import cvxpy as cp
import os
import pickle
import jax

jax.config.update('jax_platform_name', 'cpu') # or 'gpu'

import jax.numpy as jnp
import jax.lax as lax


import numpy as np
import time
import timeit
from plum import dispatch


from systems import LTI_2D


@jax.jit
def icnn_forward(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    U, W, b = params["U"], params["W"], params["b"]
    act = jax.nn.softplus  # convex & nondecreasing

    z = act(jnp.dot(U[0], x) + b[0])  # first layer (no state W)
    for i in range(1, len(U)):
        z = act(jnp.dot(W[i], z) + jnp.dot(U[i], x) + b[i])

    f = jnp.dot(params["v"], z) + jnp.dot(params["a"], x) + params["c"]
    return f 


def d_softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(x)

@dispatch
@jax.jit
def icnn_forward_grad(x: jnp.ndarray) -> jnp.ndarray:
    """
    Manual gradient df/dx (no autodiff).
    """    
    U, W, b, v, a = params_loaded["U"],params_loaded["W"], params_loaded["b"], params_loaded["v"], params_loaded["a"]

    # ---------- Forward pass (save pre-activations & activations) ----------
    s0 = lax.add(lax.dot(U[0], x), b[0])
    s1 = lax.dot(W[1], jax.nn.softplus(s0)) + lax.add(lax.dot(U[1], x), b[1])

    # ---------- Backward pass (manual backprop) ----------
    dL_ds  = jax.lax.mul(v, d_softplus(s1))
    grad_x = lax.dot(U[1].T, dL_ds)
    dL_dz  = lax.dot(W[1].T, dL_ds)

    dL_ds   = lax.mul(dL_dz, d_softplus(s0))
    grad_x  = lax.add(grad_x, jnp.dot(U[0].T, dL_ds))

    return lax.add(grad_x,a)




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
    max_iter: int = 1000
    tol: float = 1e-5
    solver: str = "Clarabel"  # or "OSQP"
    solver_opts: Optional[Dict[str, Any]] = None


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
                 solver: str = "OSQP",
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
        # for k in range(1, N + 1):
        #     cons += [self.v[idx_x(k)] >= xmin,
        #              self.v[idx_x(k)] <= xmax]
        # for k in range(0, N):
        #     cons += [self.v[idx_u(k)] >= umin,
        #              self.v[idx_u(k)] <= umax]

        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def solve(self, p: np.ndarray) -> jnp.ndarray:
        self.p.value = p.reshape(-1,)
        # self.prob.solve(solver=self.solver, warm_start=True, **self.solver_opts)
        self.prob.solve(solver=self.solver, warm_start=True)
        return self.v.value, self.prob.solver_stats.solve_time

@jax.jit
def prime_solve(q: jnp.ndarray) -> jnp.ndarray:
    global irho

    icnn_forward_grad(q)
    
    return q - jnp.dot(irho, icnn_forward_grad(q))

batch_prime = jax.jit(jax.vmap(prime_solve))





class PrimeSolver:
    def __init__(self, A: np.ndarray, B: np.ndarray,
                 x_min: float,
                 x_min: float,
                 u_max: float,
                 u_max: float,
                 x0: np.ndarray, N: int, nx: int, nu: int):
        self.N = N
        self.nx = nx
        self.nu = nu
        self.A = A
        self.B = B
        self.x_min
        self.u_max


    def dynamics_projection(x0: jnp.ndarray):
        A  = self.A
        Bm = self.B
        nx = self.nx
        nu = self.nu
        N  = self.N
        nz = nx + nu

        # Blocks
        M1 = jnp.hstack([jnp.eye(nx, dtype=jnp.float64), jnp.zeros((nx, nu), dtype=jnp.float64)])  # (nx, nz)
        M2 = jnp.hstack([A, Bm])  # (nx, nz)

        # Build big block matrix M of size ((nx*(N+1)) x (nz*(N+1)))
        rows = nx * (N + 1)
        cols = nz * (N + 1)
        M = jnp.zeros((rows, cols), dtype=jnp.float64)

        def put(M, block, br, bc):
            r0, c0 = br * nx, bc * nz
            return M.at[r0:r0+nx, c0:c0+nz].set(block)

        M = put(M, M1, 0, 0)
        for i in range(1, N + 1):
            M = put(M, -M2, i, i - 1)  # sub-diagonal
            M = put(M,  M1, i, i)      # diagonal

        # Nullspace basis and projector P = B B^T
        Bns = _nullspace_jax(M)                    # (cols, nullity)
        P = (Bns @ Bns.T) if Bns.size else jnp.zeros((cols, cols), dtype=jnp.float64)

        # Particular solution of M w = [x0; 0; ...; 0]
        rhs = jnp.concatenate([jnp.asarray(x0, dtype=jnp.float64).reshape(nx), jnp.zeros(nx * N, dtype=jnp.float64)])
        # Least-squares for robustness
        # lstsq returns (solution, residuals, rank, singular_values) in NumPy,
        # but JAX provides only the solution via jnp.linalg.lstsq
        wstar = jnp.linalg.lstsq(M, rhs, rcond=None)[0]  # (cols,)

        # Prepare shapes to reshape back
        out_shape = (N + 1, nz)

        @jit  # you can JIT the projection; P and wstar are closed-over constants
        def proj(q):
            qv = jnp.asarray(q, dtype=jnp.float64).reshape(-1)   # (cols,)
            w = P @ (qv - wstar) + wstar
            # enforce final control u_N = 0 (last nu entries of w)
            w = w.at[-nu:].set(0.0)
            return w.reshape(out_shape), 0.0

        return proj

# ====================== ADMM loop ============================================
def admm_iter(data: MPCData,  verbose: bool = True):
    
    nx = data.A.shape[0]
    nu = data.B.shape[1]


    nz = nx + nu
    N  = data.N


    aux = AuxSolverCVX(data.A, data.B,
                       (data.x_min, data.x_max),
                       (data.u_min, data.u_max),
                       data.x0, N, nx, nu,
                       solver=data.solver,
                       solver_opts=(data.solver_opts or {"eps_abs":1e-6, "eps_rel":1e-6, "verbose": False}))

    global irho 
    irho = jnp.array(1/data.rho)

    z    = np.array([np.zeros(nz) for i in range(N+1)])
    w    = np.array([np.zeros(nz) for i in range(N+1)])
    beta = np.array([np.zeros(nz) for i in range(N+1)])

    batch_prime(z) #prebuild
    batch_prime(w)

    hist_r = []

    total_time_for_learn = 0.0
    total_time_for_project = 0.0

    for it in range(1, data.max_iter + 1):
        # z-update
        start = time.time()
        z = batch_prime(w + beta)
        # print(time.time() - start)
        total_time_for_learn += time.time() - start

        z = z.at[:,0:nx].set(jnp.clip(z[:,0:nx], data.x_min[0], data.x_max[0]))
        z = z.at[:,nx:nz].set(jnp.clip(z[:,nx:nz], data.u_min[0], data.u_max[0]))

        

        # w-update
        sol, time_sol = aux.solve(np.array(z  - beta))
        total_time_for_project += time_sol
        w    = sol.reshape(N + 1, nz)

        # dual update
        beta += (w - z)

        # residuals
        r = jnp.max(jnp.abs(w - z))

        if r < data.tol:
            if verbose:
                print(f"Learning ADMM converges at it={it}: r={r:.3e}")
            break
    total_time = total_time_for_project + total_time_for_learn    
    print(f"Time to learn: {total_time_for_learn} seconds")
    print(f"Time to project: {total_time_for_project} seconds")
    print(f"Learning ADMM is solved {total_time} seconds")


    return {"z": z, "w": w, "beta": beta, "r": np.array(hist_r)}


# ====================== Demo ==================================================
if __name__ == "__main__":

    sys_params = LTI_2D()

    data_train = np.load(os.path.join("data","logsum-rho=1-LTI_2D-train.npz"))

    with open(os.path.join('model','logsum-rho=1-LTI_2D-ICNN-cstr.pkl'), 'rb') as f:
        params_loaded = pickle.load(f)
    
    icnn_forward_grad(jnp.array([0., 0., 0.], dtype=jnp.float32)) #prebuild

    x = jnp.array([3., 2., 1.], dtype=jnp.float32)

    sample = 1000
    setup_globals = {
        "icnn_forward_grad": icnn_forward_grad,
        "x": x    
    }
    code = "icnn_forward_grad(x)"

    execution_time =  timeit.timeit(stmt=code, globals=setup_globals, number = sample)
    print(f"Average Execution time for icnn_forward_grad with {sample} samples: {execution_time/sample} seconds \n")

   
    params_sys = MPCData(A=sys_params["A"], B=sys_params["B"], Q=sys_params["Q"], R=sys_params["R"],
                   x_min=sys_params["x_min"], x_max=sys_params["x_max"],
                   u_min=sys_params["u_min"], u_max=sys_params["u_max"],
                   x0=sys_params["x0"], 
                   N=sys_params["N"],  
                   rho = data_train["rho"], 
                   max_iter=1000, tol=1e-5,
                   solver="OSQP",
                   solver_opts={"eps_abs": 1e-6, "eps_rel": 1e-6, "max_iter": 1000, "verbose": False})
    
    out = admm_iter(params_sys, icnn_forward_grad)

    z, w = out["z"], out["w"]
    np.set_printoptions(suppress=True)
    print("uz =", np.round(z[:,-1],4))
    print("uw =", np.round(w[:,-1],4))
    print("\n")