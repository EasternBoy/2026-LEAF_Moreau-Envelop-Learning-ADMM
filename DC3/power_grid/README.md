# DC3 for the economic-MPC microgrid problem

DC3 (*Deep Constraint Completion and Correction*, [Donti, Rolnick & Kolter,
ICLR 2021](https://arxiv.org/abs/2104.12225), code
[locuslab/DC3](https://github.com/locuslab/DC3)) applied to the problem in
`problems/power_grid`.

**This is the repository's economic MPC, not the AC-OPF example of the DC3
paper.**  The formulation below is a transcription of `problems/power_grid/problem.jl` and
`jump_solver.jl`; nothing about the optimisation problem is changed.

---

## 1. The problem (source of truth: the Julia code)

Constants from `energy_mag()`: `N = 96`, `dT = 0.25 h`, `A = 1`,
`B = −dT/BESS = −5·10⁻⁴` (`BESS = 500`), `r_ec = 0.1`, `r_df = 10`,
`r_op = 19.19`, `η = 0.8`, `a = 50`, `x∈[0.2, 0.8]`, `x_N ≥ 0.5`, `u∈[−700, 700]`.

| | |
|---|---|
| decision variables | `m_k` grid import, `u_k` BESS power, `p_k` delivered power (`k = 1..N`), `x_k` state of charge (`k = 0..N`) |
| instance parameters | `x0`, `load_{1..N}`, `gen_{1..N}` |
| objective | `Σ_k r_ec·dT·(m_k + (1−η)/(2√η)·\|u_k\|) + r_op·max(m_k,0) + r_df·max(a/p_k − 1, 0)` |
| equalities | `x_k = A x_{k−1} + B u_k`; `x_0 = x0`; `u_k + m_k + gen_k − load_k − p_k = 0` |
| inequalities | `u_min ≤ u_k ≤ u_max`, `p_k ≥ 0`, `x_min ≤ x_k ≤ x_max`, `x_N ≥ x_end_min = 0.5` (`m` is free) |
| domain | `p_k > 0` (needed by `a/p_k`) |

The JuMP model writes the three non-smooth terms in epigraph form
(`su ≥ ±u`, `sm ≥ 0, sm ≥ m`, `sd ≥ 0, sd ≥ a/p − 1`), which makes it an SOCP and
equals the closed form above at the optimum; that closed form is also the
`model === nothing` branch of `(obj::eco_mpc)(m,u,p,model)`.

**Verified:** Ipopt on the nominal instance (`x0 = 0.5`, first 96 CSV samples)
gives `J = 36479.1113`, matching the hard-coded `Jopt = 36479.1` in
`problems/power_grid/setup.jl`, and the Python objective reproduces it to
1e-16 relative.  Because the battery need not return to `x0`, the optimum
depends on `x0` (it is `46085.39` at `x0 = 0.25` and `26873.01` at `x0 = 0.75`);
`Jopt` is valid for the nominal `x0 = 0.5` only.

## 2. DC3 adaptation

**Decision vector.** `x_0` is a known parameter, so

```
y = [ m_1..m_N | u_1..u_N | p_1..p_N | x_1..x_N ]      n_y  = 4N = 384
A_eq y = b_eq(x0, load, gen)                            n_eq = 2N = 192
g(y) ≤ 0                                                n_ineq = 5N+1 = 481
```

`A_eq` is exactly the matrix `M` assembled by
`problems/power_grid/utils.jl::dynamics_projection` (same row order: `N` dynamics
rows, then `N` power-flow rows).  The terminal bound `x_N ≥ 0.5` is the last row
of `g`.

**Variable partition** (`n_y − n_eq = 2N = 192` predicted variables):

```
P = { u_1 … u_N } ∪ { p_1 … p_N }
D = { x_1 … x_N } ∪ { m_1 … m_N }
```

**Completion** is the generic linear solve `y_D = A_D⁻¹(b_eq − A_P y_P)`, but the
partition was chosen so that it is block-triangular and interpretable:

```
x_k = x0 + B·Σ_{i≤k} u_i        (k = 1..N)       forward recursion
m_k = load_k − gen_k − u_k + p_k                  power-flow rows
```

`validate.py` asserts that the generic solve reproduces this closed form to
2·10⁻¹³.

**Assumption check.** `rank(A_eq) = 192 = n_eq` (full row rank) and `A_D` is
invertible with `cond(A_D) = 1.23·10²`, `log|det A_D| = 0`, error amplification
`‖A_D⁻¹A_P‖₂ = 1.41`.  It is reported in every result file, and
`partition.cond_warn` makes it a hard failure above a configurable threshold.

**Correction and why row scaling is needed.**  All inequalities are affine, so
`G_eff = G_P − G_D·(A_D⁻¹A_P)` is a constant `481 × 192` matrix and the
correction gradient `2·relu(g)ᵀ G_eff` is closed form.  However the row norms of
`G_eff` span four orders of magnitude:

| rows | `‖G_eff,i‖` | meaning |
|---|---|---|
| `u` bounds | 0.10 – 1.0 (after normalisation) | `∂u/∂u = 1` |
| `p ≥ 0` | 1.0 | `p` is a partial variable |
| `x` bounds (incl. `x_N ≥ 0.5`) | up to **2000** | `∂x/∂u = B = −5·10⁻⁴` |

With a single `corr_lr`, DC3's plain gradient step therefore either diverges on
the power rows or makes no progress at all on the state-of-charge rows (measured:
≈2·10⁻⁷ movement per step, i.e. ~10⁶ steps to fix a 0.2 violation).  The
implementation therefore rescales the rows of the *internal* residual by
`1/‖G_eff,i‖` (normalised to a median of 1) — equivalent to measuring the SOC in
kWh rather than as a fraction.  A row with an identically zero reduced gradient
(none in the default partition) would be given weight 1 instead of `1/0`.
**All reported metrics use the unscaled, original constraints.**
Set `dc3.ineq_row_scale = "none"` to reproduce the unscaled behaviour.

**Objective domain.** `a/p_k` needs `p_k > 0`; a randomly initialised network
gives `p ≈ 0` and an objective of ~10¹³.  Two measures:

* the read-out is `p = p_margin + softplus(·)` and `u = u_min + σ(·)(u_max − u_min)`
  (DC3's ACOPF read-out generalised to one-sided bounds), with the bias
  initialised so the first prediction is `u = 0`, `p = a = 50` — an idle battery
  at the comfort setpoint;
* DC3's internal constraint set uses `p ≥ p_margin` (default `10⁻³`) instead of
  `p ≥ 0`. Tightening can only make DC3 more conservative, so feasibility
  reported against the original `p ≥ 0` remains valid.  `domain_max = max(−p)` is
  reported separately and `p_safe_eps` clamps the objective argument so a
  violating point yields a large finite number rather than `inf`.

**Network input.** `[x0, load_{1..N}, gen_{1..N}]`, `x_dim = 2N+1 = 193`
(`feature_mode = "x0_netload"` uses `[x0, load−gen]` instead).

## 3. Data — how a *family* of instances is obtained

`energy_mag()` defines a **single** instance.  DC3 is a parametric solver, so
`data.py` varies only the quantities the JuMP model already exposes as
`MOI.Parameter`:

* `x0 ~ U(0.25, 0.75)` — brackets the pool used for the ADMM training data in
  the former `experiments/power_grid/data_gen.jl` (`train_pool = [1/2, 2/3, 3/4]`, `test_pool = [3/5]`);
* `(load, gen)` = the length-`N` window of the two CSVs starting at offset `s`,
  with `s` drawn from an offset pool.  The 97 admissible offsets are split
  **disjointly** 60/20/20 between train/validation/test, so no forecast window is
  shared between splits.

The **nominal benchmark instance** (`x0 = 0.5`, `s = 0`) is forced to be test
instance 0, so the published `Jopt = 36479.1` can be checked directly.

The plant, the cost function and every constant are untouched.

> The `.npz` files under `data/training_data/eco_mpc-*` are **not** instances of
> this problem: they are `(q, Moreau-envelope value, gradient)` samples of the
> *per-timestep prox subproblem* used to fit the ICNN of LME-ADMM.  They are
> unrelated to DC3's training data and are not used here.

## 4. Reference solver

`reference.py` solves the SOCP with cvxpy + **Clarabel** at `tol = 1e-9`
(`a/p` enters as `cp.pos(a*cp.inv_pos(p) − 1)`).  Ipopt on the same instances is
produced by `experiments/dc3/baselines_power.jl`; `validate.py` checks that the
Python objective and the cvxpy solution agree, and the report compares both with Ipopt.

## 5. Commands

```bash
# from the repository root, with DC3/.venv activated (see DC3/README.md)
python -m DC3.validate --app power                      # formulation + gradient checks
python -m DC3.power_grid.train     --tag default
python -m DC3.power_grid.benchmark --tag default

# repository baselines on the same test instances (Ipopt + LME-ADMM split)
julia --project=. experiments/dc3/baselines_power.jl DC3/results/power_grid-default
python -m DC3.report --app power_grid --tag default

# hyper-parameter search on the validation split only
python -m DC3.tune --app power_grid \
    --grid '{"dc3.corr_lr":[1e-3,1e-2],"dc3.obj_scale":[100,1000]}' --set dc3.epochs=80
```

## 6. Deviations from the paper / official code

| | |
|---|---|
| Problem | the repository's economic MPC, **not** the paper's AC-OPF. |
| Completion | `h` is affine, so DC3's Newton completion collapses to one linear solve. |
| Correction space | steps in `z` (partial) space; identical to DC3's full-space update for affine completion (asserted in `validate.py`). |
| Row scaling | `1/‖G_eff,i‖` on the internal residuals (see §2). Not in DC3; without it the correction cannot fix the SOC bounds. Disable with `dc3.ineq_row_scale="none"`. |
| Read-out | `σ`-box on `u`, `softplus` on `p`, bias initialised at `(u,p) = (0, a)`. DC3 uses a `sigmoid` read-out for ACOPF; the one-sided variant is new. |
| Constraint margin | `p ≥ p_margin = 10⁻³` internally (tightening only). |
| Soft loss | `soft_loss_power` = norm (official code, default) or squared norm (paper); `obj_scale` divides the objective term — needed here because `J ≈ 3.6·10⁴` while violations are O(1). |
| `soft_weight_eq_frac` | defaults to 0: with completion the equality residual is ~10⁻¹⁴, so the equality term of the soft loss carries no signal. |
| Inconsistency in the Julia code (documented, not changed) | `aux_solver_eco` in `eMPC_ADMM.jl` declares `p[1:N] .>= 1` while `mpc_eco_solver` and `aux_solver_eco_data` use `p .>= 0`. DC3 uses `p ≥ 0`, matching the model that defines `Jopt`. |

## 7. Results and protocol

The former numerical tables have been withdrawn because they used tolerant
objective-domain checks and mixed whole-batch quality with single-instance
timing. The power-grid baseline also required residual and state-reset fixes.
See [the benchmark protocol](../README.md#benchmark-protocol-version-2) and
regenerated `results/` reports for current measurements. Saved checkpoints can
be reevaluated without retraining; their original training history is retained.

Objective values outside the mathematical domain are undefined, even when a
small negative coordinate passes the ordinary constraint tolerance. Such points
are not counted as feasible, and clamped training surrogates are not reported
as optimization objectives or optimality gaps.
