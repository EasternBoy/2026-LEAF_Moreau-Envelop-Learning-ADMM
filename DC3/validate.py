"""Correctness checks for the DC3 implementation.

Run with::

    python -m DC3.validate                 # both applications
    python -m DC3.validate --app cone      # one of {cone, power, both}

What is checked
---------------
1. **Formulation parity** - the Python objective / residuals agree with the
   Julia model on identical data.  For both problems the cvxpy reference
   solution is verified to satisfy the Python constraints and to reproduce the
   Python objective.
2. **Completion** - ``A_eq complete(Z, b_eq) - b_eq == 0``; round-trip
   ``complete(partial_of(Y*), b_eq) == Y*`` for the reference solution; the
   generic linear solve equals the hand-derived closed form (power grid).
3. **Affinity of the completion** -
   ``complete(Z - s) == complete(Z) - scatter_step(s)``, i.e. the z-space
   correction used here is identical to DC3's full-space update.
4. **Gradients** - autograd through completion matches central finite
   differences; the closed-form ``ineq_partial_grad`` matches autograd; the
   gradient of the training loss through ``corr_train_steps`` correction steps
   matches finite differences of the same computation.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .common.completion import LinearCompletion
from .common.dc3 import DC3Config, DC3Solver

TOL = dict(loose=1e-6, tight=1e-9)
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def rel_err(a, b):
    """Max absolute difference normalised by the global magnitude of the pair.

    Element-wise relative error is useless for gradient checks: entries that are
    exactly zero in one of the two vectors blow it up to 1 even when the
    absolute discrepancy is 1e-17.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b)) / scale)


# ---------------------------------------------------------------------------
def validate_cone(n=60, m=8, n_inst=4, seed=0):
    from .entr_max.data import make_split
    from .entr_max.problem import MaxEntropyProblem
    from .entr_max import reference as ref

    print(f"\n=== entr_max (n={n}, m={m}, {n_inst} instances) ===")
    dev, dt = torch.device("cpu"), torch.float64
    prob = MaxEntropyProblem(n=n, m=m, dtype=dt, device=dev)
    params = make_split(n, m, n_inst, seed, "test", dev, dt)
    comp = LinearCompletion(prob.A_eq, strategy="explicit", other_vars=[n - 1])
    comp.check(strict=True)
    print("  " + comp.info.summary())

    # --- 1. reference solution satisfies the Python model -----------------
    A = params.A.numpy(); b = params.b.numpy()
    W, _, Jref, status = ref.solve_batch(A, b, w_floor=prob.w_floor, tol=1e-10)
    Y = torch.as_tensor(W, dtype=dt)
    J_py = prob.obj_fn(params, Y).numpy()
    check("cvxpy status all optimal", all("optimal" in s for s in status), str(set(status)))
    check("objective parity (python vs cvxpy)", rel_err(J_py, Jref) < 1e-8,
          f"max rel err {rel_err(J_py, Jref):.2e}")
    eqmax = float(prob.eq_resid(params, Y).abs().max())
    inmax = float(prob.ineq_dist(params, Y).max())
    check("reference satisfies equalities", eqmax < 1e-8, f"max |1'w - 1| = {eqmax:.2e}")
    check("reference satisfies inequalities", inmax < 1e-7, f"max viol = {inmax:.2e}")

    _completion_and_gradient_checks(prob, params, comp, Y, dt,
                                    perturb=0.2 / n, round_trip_tol=1e-8)
    return prob, params, comp


# ---------------------------------------------------------------------------
def validate_power(N=96, n_inst=4, seed=0):
    from .power_grid.data import make_split
    from .power_grid.problem import EcoMPCProblem
    from .power_grid import reference as ref

    print(f"\n=== power_grid (N={N}, {n_inst} instances) ===")
    dev, dt = torch.device("cpu"), torch.float64
    prob = EcoMPCProblem(N=N, dtype=dt, device=dev)
    params = make_split(N, n_inst, seed, "test", dev, dt, split_strategy="legacy_offsets")
    comp = LinearCompletion(prob.A_eq, strategy="explicit", other_vars=prob.default_other_vars())
    comp.check(strict=True)
    print("  " + comp.info.summary())

    x0 = params.x0.numpy(); load = params.load.numpy(); gen = params.gen.numpy()
    Ysol, _, Jref, status = ref.solve_batch(x0, load, gen, tol=1e-10)
    Y = torch.as_tensor(Ysol, dtype=dt)
    J_py = prob.obj_fn(params, Y).numpy()
    check("cvxpy status all optimal", all("optimal" in s for s in status), str(set(status)))
    check("objective parity (python vs cvxpy)", rel_err(J_py, Jref) < 1e-8,
          f"max rel err {rel_err(J_py, Jref):.2e}")
    eqmax = float(prob.eq_resid(params, Y).abs().max())
    inmax = float(prob.ineq_dist(params, Y).max())
    check("reference satisfies equalities", eqmax < 1e-6, f"max |h| = {eqmax:.2e}")
    check("reference satisfies inequalities", inmax < 1e-6, f"max viol = {inmax:.2e}")

    # nominal instance must reproduce the hard-coded Jopt from problems/power_grid/setup.jl
    if N == 96:
        check("nominal instance reproduces Jopt = 36479.1",
              abs(J_py[0] - 36479.1) < 0.5, f"J = {J_py[0]:.4f}")

    # --- closed-form completion vs generic linear solve -------------------
    Z = comp.partial_of(Y)
    Yg = comp.complete(Z, prob.eq_rhs(params))
    Yc = _closed_form_completion(prob, params, Z)
    check("generic solve == hand-derived closed form",
          float((Yg - Yc).abs().max()) < 1e-8, f"max abs diff {float((Yg-Yc).abs().max()):.2e}")

    # cond(A_D) ~ 5.6e4, so the ~1e-12 equality residual of the conic solver is
    # amplified to ~1e-7 when the solution is re-completed from its partial part.
    _completion_and_gradient_checks(prob, params, comp, Y, dt,
                                    perturb=1e-2, round_trip_tol=1e-5)

    return prob, params, comp


def _closed_form_completion(prob, params, Z):
    """x_k = x0 + B sum_{i<=k} u_i ; u_N = -sum_{i<N} u_i ; m = load-gen-u+p."""
    N = prob.N
    u_head = Z[:, : N - 1]
    p = Z[:, N - 1 :]
    u_last = -u_head.sum(dim=1, keepdim=True) * (1.0)          # A = 1  =>  sum u = 0
    u = torch.cat([u_head, u_last], dim=1)
    x = params.x0.unsqueeze(1) + prob.Bd * torch.cumsum(u, dim=1)
    m = params.load - params.gen - u + p
    return torch.cat([m, u, p, x], dim=1)


# ---------------------------------------------------------------------------
def _completion_and_gradient_checks(prob, params, comp, Yref, dt,
                                    perturb: float = 1e-2,
                                    round_trip_tol: float = 1e-8):
    B = Yref.shape[0]
    b_eq = prob.eq_rhs(params)
    g = torch.Generator().manual_seed(1234)

    # round trip on the reference solution
    Zr = comp.partial_of(Yref)
    err = float((comp.complete(Zr, b_eq) - Yref).abs().max())
    check("completion round-trip on reference solution", err < round_trip_tol,
          f"max abs {err:.2e} (tol {round_trip_tol:.0e}, cond(A_D)={comp.info.cond_A_other:.2e})")

    # equality residual is zero by construction for perturbed Z
    Z = torch.randn(B, comp.n_partial, generator=g, dtype=dt) * perturb + Zr
    Y = comp.complete(Z, b_eq)
    r = float(prob.eq_resid(params, Y).abs().max())
    scale = float(b_eq.abs().max().clamp(min=1.0))
    check("completion zeroes the equality residual", r / scale < 1e-10, f"max |h| = {r:.2e}")

    # affinity: complete(Z - s) == complete(Z) - scatter_step(s)
    s = torch.randn(B, comp.n_partial, generator=g, dtype=dt) * 0.05
    lhs = comp.complete(Z - s, b_eq)
    rhs = comp.complete(Z, b_eq) - comp.scatter_step(s)
    d = float((lhs - rhs).abs().max())
    check("z-space step == DC3 full-space step", d < 1e-9, f"max abs {d:.2e}")

    # autograd through completion vs finite differences (objective)
    Zg = Z.clone().requires_grad_(True)
    f = prob.obj_fn(params, comp.complete(Zg, b_eq)).sum()
    (ga,) = torch.autograd.grad(f, Zg)
    idx = torch.randint(0, comp.n_partial, (12,), generator=g)
    h = 1e-6
    fd = torch.zeros(B, idx.numel(), dtype=dt)
    for j, k in enumerate(idx):
        e = torch.zeros_like(Z); e[:, k] = h
        fp = prob.obj_fn(params, comp.complete(Z + e, b_eq))
        fm = prob.obj_fn(params, comp.complete(Z - e, b_eq))
        fd[:, j] = (fp - fm) / (2 * h)
    e_fd = rel_err(ga[:, idx].detach().numpy(), fd.numpy())
    check("autograd(objective o completion) == finite differences", e_fd < 1e-5,
          f"max rel err {e_fd:.2e}")

    # closed-form correction gradient vs autograd
    cf = prob.ineq_partial_grad(params, Z, comp)
    if cf is not None:
        Zg = Z.clone().requires_grad_(True)
        viol = (prob.ineq_dist(params, comp.complete(Zg, b_eq), margin=True) ** 2).sum()
        (ag,) = torch.autograd.grad(viol, Zg)
        e = rel_err(cf.detach().numpy(), ag.numpy())
        check("closed-form ineq_partial_grad == autograd", e < 1e-8, f"max rel err {e:.2e}")

    # closed form with row scaling (the path DC3 actually uses)
    rs = prob.ineq_row_scale(comp)
    if rs is not None:
        cf = prob.ineq_partial_grad(params, Z, comp, rs)
        Zg = Z.clone().requires_grad_(True)
        viol = ((prob.ineq_dist(params, comp.complete(Zg, b_eq), margin=True) * rs) ** 2).sum()
        (ag,) = torch.autograd.grad(viol, Zg)
        e = rel_err(cf.detach().numpy(), ag.numpy())
        check("row-scaled closed-form ineq_partial_grad == autograd", e < 1e-8,
              f"max rel err {e:.2e}; scale range [{float(rs.min()):.3g}, {float(rs.max()):.3g}]")

    # gradients through completion AND correction
    cfg = DC3Config(corr_train_steps=3, corr_lr=1e-6, corr_momentum=0.5,
                    dtype="float64", device="cpu", batch_norm=False, dropout=0.0,
                    soft_loss_power=2)
    solver = DC3Solver(prob, comp, cfg).to(torch.float64)

    def fd_check(fn, name, tol, h=1e-6):
        Zg = Z.clone().requires_grad_(True)
        (ga,) = torch.autograd.grad(fn(Zg), Zg)
        fd = torch.zeros(idx.numel(), dtype=dt)
        for j, k in enumerate(idx):
            e = torch.zeros_like(Z); e[:, k] = h
            fd[j] = (fn(Z + e) - fn(Z - e)) / (2 * h)
        e_fd = rel_err(ga[:, idx].sum(dim=0).detach().numpy(), fd.numpy())
        check(name, e_fd < tol, f"max rel err {e_fd:.2e} (tol {tol:.0e})")

    # (a) strictly smooth probe: ||y||^2 downstream of correction.  relu(.)^2 is
    #     C^1, so central differences must agree to ~1e-8 here.
    def smooth_loss(Zin):
        Zc = solver.correct_train(params, Zin)
        return (solver.complete(params, Zc) ** 2).sum()

    fd_check(smooth_loss, "autograd through completion AND correction (smooth probe)", 1e-7)

    # (b) the real DC3 training loss.  The eco-MPC objective (|u|, max(m,0),
    #     max(a/p-1,0)) and the cone objective at the domain boundary are only
    #     piecewise smooth, so central differences are first-order accurate at
    #     best near a kink; a looser tolerance is used and documented.
    def real_loss(Zin):
        Zc = solver.correct_train(params, Zin)
        return solver.total_loss(params, solver.complete(params, Zc)).sum()

    fd_check(real_loss, "autograd of the DC3 training loss (piecewise-smooth)", 1e-2)

    # a correction step must not break the equalities
    Zc, steps, conv, _ = solver.correct_test(params, Z)
    r = float(prob.eq_resid(params, comp.complete(Zc, b_eq)).abs().max())
    check("correction preserves equality feasibility", r / scale < 1e-10,
          f"max |h| = {r:.2e} after {steps} steps")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", choices=["cone", "power", "both"], default="both")
    ap.add_argument("--cone-n", type=int, default=60)
    ap.add_argument("--cone-m", type=int, default=8)
    ap.add_argument("--power-N", type=int, default=96)
    ap.add_argument("--n-inst", type=int, default=4)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    if a.app in ("cone", "both"):
        validate_cone(a.cone_n, a.cone_m, a.n_inst)
    if a.app in ("power", "both"):
        validate_power(a.power_N, a.n_inst)

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
