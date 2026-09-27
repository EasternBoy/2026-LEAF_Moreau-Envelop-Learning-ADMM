from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cvxpy as cp


# ====================== System parameters (from system.jl example) ============

def lti_parameter():
    """
    Example system (from your system.jl):
      A = [[2, -1],
           [1,  0.2]]
      B = [[1],
           [0]]
      Q = I, R = 2, N = 7
    """
    A = np.array([[2.0, -1.0],
                  [1.0,  0.2]], dtype=float)
    B = np.array([[1.0],
                  [0.0]], dtype=float)
    nx = 2
    nu = 1
    Q = np.eye(nx)
    R = np.array([[2.0]])
    N = 7
    x_min = np.full(nx, -5.0)
    x_max = np.full(nx, +5.0)
    u_min = np.full(nu, -1.0)
    u_max = np.full(nu, +1.0)
    x0 = [3., 1.]
    return A, B, Q, R, x_min, x_max, u_min, u_max, x0, N


# ====================== Cost functions ====================

def logsum_cost(x: np.ndarray, u: np.ndarray) -> float:
    return np.log(np.sum(np.exp(x))) + float(u.T @ u)

def quadratic_cost(x: np.ndarray, u: np.ndarray, Q: np.ndarray, R: np.ndarray) -> float:
    return x.T @ Q @ x + u.T @ R @ u

def huber_nonsmooth_cost(x: np.ndarray, u: np.ndarray, delta: float = 1.0) -> float:
    u = np.atleast_1d(u)
    absu = np.abs(u)
    out = np.where(absu < delta, u**2, delta*(absu - delta/2.0))
    return np.sum(out) + x.T @ x


# ====================== Data Class ==============================================
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
    max_iter: int = 200
    tol: float = 1e-5
    solver: str = "OSQP"  # or "ECOS"
    solver_opts: Optional[Dict[str, Any]] = None


# ====================== Prime subproblem (CVXPY) ==============================

class PrimeSolverCVX:
    """
    Stage-wise QP:
      min_z  1/2 z^T (H + rho I) z + (-rho q)^T z,  z = [x_k; u_k]
      s.t.   x_min <= x_k <= x_max,  (and u bounds if k < N)
    We build two problems: "normal" (with u-bounds) and "terminal" (only x-bounds).
    """
    def __init__(self, nx: int, nu: int, N: int,
                 rho: float,
                 solver: str = "Clarabel",
                 solver_opts: Optional[Dict[str, Any]] = None):
        self.nx, self.nu, self.N = nx, nu, N
        self.nz = nx + nu
        self.rho = rho
        self.solver = solver.upper()
        self.solver_opts = solver_opts or {}

        # Build two problem templates sharing the same parameter q
        self.z = cp.Variable(self.nz)
        self.q = cp.Parameter(self.nz)

        obj = cp.log_sum_exp(self.z[0:nx]) + cp.quad_form(self.z[nx:self.nz], np.eye(self.nu)) + (rho/2) * cp.quad_form(self.z - self.q, np.eye(self.nz))

        self.prob = cp.Problem(cp.Minimize(obj))

    def solve_stage(self, q: np.ndarray, k: int) -> np.ndarray:
        self.q.value = q.reshape(-1,)
        # warm start
        self.prob.solve(solver=self.solver, warm_start=True, **self.solver_opts)
        return np.asarray(self.z.value).reshape(-1)



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
        for k in range(1, N + 1):
            cons += [self.v[idx_x(k)] >= xmin,
                     self.v[idx_x(k)] <= xmax]
        for k in range(0, N):
            cons += [self.v[idx_u(k)] >= umin,
                     self.v[idx_u(k)] <= umax]

        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def solve(self, p: np.ndarray) -> np.ndarray:
        self.p.value = p.reshape(-1,)
        self.prob.solve(solver=self.solver, warm_start=True, **self.solver_opts)
        return np.asarray(self.v.value).reshape(self.N + 1, self.nz)


# ====================== ADMM loop ============================================

def admm_iter(data: MPCData, verbose: bool = True) -> Dict[str, Any]:
    nx = data.A.shape[0]
    nu = data.B.shape[1]
    nz = nx + nu
    N  = data.N

    prime = PrimeSolverCVX(nx, nu, N, data.rho,
                           solver = data.solver,
                           solver_opts=(data.solver_opts or {"verbose": False}))
    
    aux = AuxSolverCVX(data.A, data.B,
                       (data.x_min, data.x_max),
                       (data.u_min, data.u_max),
                       data.x0, N, nx, nu,
                       solver="OSQP",
                       solver_opts=(data.solver_opts or {"verbose": False}))

    z = np.zeros((N + 1, nz))
    w = np.zeros((N + 1, nz))
    beta = np.zeros((N + 1, nz))

    hist_r = []

    for it in range(1, data.max_iter + 1):
        # z-update
        for k in range(N + 1):
            qk = w[k, :] + beta[k, :]
            z[k, :] = prime.solve_stage(qk, k)

        # w-update
        p = z - beta
        w = aux.solve(p)

        # dual update
        beta = beta + (w - z)

        # residuals
        r = np.linalg.norm(w - z)
        hist_r.append(r)

        if verbose and (it % 5 == 0 or it == 1):
            # print(f"[it {it:04d}] r={r:.3e}  s={s:.3e}")
            print(f"[it {it}] r={r:.3e}")

        if r < data.tol:
            if verbose:
                print(f"Converged at itr. = {it}: r={r:.3e}")
            break

    return {"z": z, "w": w, "beta": beta, "r": np.array(hist_r)}


# ====================== Demo ==================================================
if __name__ == "__main__":
    A, B, Q, R, x_min, x_max, u_min, u_max, x0, N = lti_parameter()
    data = MPCData(A=A, B=B, Q=Q, R=R,
                   x_min=x_min, x_max=x_max,
                   u_min=u_min, u_max=u_max,
                   x0=x0, N=N, rho=1.0, max_iter=300, tol=1e-6,
                   solver="Clarabel",
                   solver_opts={ "max_iter": 10000, "verbose": False})
    
    out = admm_iter(data, verbose=True)
    z, w = out["z"], out["w"]
    
    print("uz =", np.round(z[:,-1],4))
    print("uw =", np.round(w[:,-1],4))
    # Cost utilities on first stage