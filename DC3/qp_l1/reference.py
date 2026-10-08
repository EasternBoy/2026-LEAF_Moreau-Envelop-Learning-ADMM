"""CVXPY reference solves of QP + L1, following DC3/qp/reference.py."""

from __future__ import annotations

import time

import numpy as np

from .data import LAMBDA, fixed_data


def solve_instance(x, solver="CLARABEL", tol=1e-9, verbose=False):
    import cvxpy as cp

    data = fixed_data()
    y = cp.Variable(data["p"].size)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(y, data["Q"]) + data["p"] @ y + LAMBDA * cp.norm1(y)),
                      [data["A"] @ y == x, data["G"] @ y <= data["h"]])
    kwargs = {}
    if solver == "CLARABEL":
        kwargs = dict(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
    elif solver == "OSQP":
        kwargs = dict(eps_abs=tol, eps_rel=tol)
    t0 = time.perf_counter()
    prob.solve(solver=solver, verbose=verbose, **kwargs)
    elapsed = time.perf_counter() - t0
    if y.value is None:
        return None, elapsed, float("nan"), prob.status
    value = np.asarray(y.value, dtype=float)
    objective = float(0.5 * value @ data["Q"] @ value + data["p"] @ value + LAMBDA * np.abs(value).sum())
    return value, elapsed, objective, prob.status


def solve_batch(X, **kwargs):
    sols, times, objs, status = [], [], [], []
    for x in X:
        y, t, J, st = solve_instance(x, **kwargs)
        sols.append(np.full(100, np.nan) if y is None else y)
        times.append(t)
        objs.append(J)
        status.append(st)
    return np.array(sols), np.array(times), np.array(objs), status
