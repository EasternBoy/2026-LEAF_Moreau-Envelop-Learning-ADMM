"""Reference solvers for the maximum-entropy cone program.

`cvxpy` + Clarabel solves the exponential-cone form

    min sum_i w_i log w_i   s.t.  1^T w = 1,  A w <= b,  w >= w_floor

which is the same problem the Julia scripts hand to Ipopt / Clarabel.  The
Julia-side Ipopt reference (the repository's own ground truth, `tol = 1e-8`) is
produced by ``DC3/julia/reference_cone.jl`` on the *same* saved instances; the
benchmark cross-checks the two.
"""

from __future__ import annotations

import time

import numpy as np


def solve_instance(A: np.ndarray, b: np.ndarray, w_floor: float = 0.0,
                   solver: str = "CLARABEL", tol: float = 1e-9, verbose: bool = False):
    import cvxpy as cp

    n = A.shape[1]
    w = cp.Variable(n, nonneg=True)
    obj = cp.Minimize(-cp.sum(cp.entr(w)))          # entr(w) = -w log w
    cons = [cp.sum(w) == 1, A @ w <= b]
    if w_floor > 0:
        cons.append(w >= w_floor)
    prob = cp.Problem(obj, cons)
    kwargs = {}
    if solver == "CLARABEL":
        kwargs = dict(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
    elif solver == "ECOS":
        kwargs = dict(abstol=tol, reltol=tol, feastol=tol)
    t0 = time.perf_counter()
    prob.solve(solver=solver, verbose=verbose, **kwargs)
    t = time.perf_counter() - t0
    if w.value is None:
        return None, t, float("nan"), prob.status
    wv = np.asarray(w.value, dtype=float)
    J = float(np.sum(np.where(wv > 0, wv * np.log(np.clip(wv, 1e-300, None)), 0.0)))
    return wv, t, J, prob.status


def solve_batch(A: np.ndarray, b: np.ndarray, **kw):
    sols, times, objs, status = [], [], [], []
    for i in range(A.shape[0]):
        w, t, J, st = solve_instance(A[i], b[i], **kw)
        sols.append(np.full(A.shape[2], np.nan) if w is None else w)
        times.append(t)
        objs.append(J)
        status.append(st)
    return np.array(sols), np.array(times), np.array(objs), status
