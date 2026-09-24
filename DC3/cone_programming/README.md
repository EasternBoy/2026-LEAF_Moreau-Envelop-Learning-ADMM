# DC3 for the maximum-entropy cone program

DC3 (*Deep Constraint Completion and Correction*, [Donti, Rolnick & Kolter,
ICLR 2021](https://arxiv.org/abs/2104.12225), code
[locuslab/DC3](https://github.com/locuslab/DC3)) applied to the problem in
`examples/cone_programming`.

---

## 1. The problem (source of truth: the Julia code)

`examples/cone_programming/maxEntropy.jl` and `JuMPsolver.jl` build, with
`scale = 2n`:

```julia
@variable(model,   x[1:n] >= 1e-8)
@constraint(model, sum(x) == scale)
@constraint(model, A*x .<= b*scale)
@objective(model,  Min, sum(x[i]*(log(x[i]) - log(scale)) for i in 1:n))
# returns  x/scale  and  objective_value/scale
```

Substituting `w = x/scale` this is exactly

| | |
|---|---|
| decision variable | `w ∈ R^n` |
| instance parameters | `A ∈ R^{m×n}`, `b ∈ R^m` |
| objective | `min Σᵢ wᵢ log wᵢ` (negative entropy; the reported `J` is negative) |
| equality | `1ᵀw = 1`  (`n_eq = 1`) |
| inequalities | `A w ≤ b` (`m` rows), `wᵢ ≥ 1e-8/scale` (`n` rows) |
| domain | `wᵢ > 0` (needed by `w log w`; `0 log 0 := 0`) |

Instance distribution, from the two-argument `data_opt(n, m)` constructor that
every benchmark script uses:

```julia
A = rand(Uniform(0,1), m, n)
b = [sum(A[i,:]) / (1.06*n) for i in 1:m]
```

(The `@kwdef` default inside the struct divides by `1.1` instead; that variant is
never exercised by a benchmark and is not used here.)

Default dimensions `n = 1000`, `m = 100` are those of
`examples/cone_programming/benchmarkOG.jl`; `configs/small.json` uses
`n = 100, m = 10`, which is the other size the repository benchmarks
(`data/solving_data/maxEntropy-n=100m=10-*.npz`).

## 2. DC3 adaptation

**Variable partition.**  `n_eq = 1`, so the network predicts `n-1` partial
variables and one variable is completed:

```
P = {1, …, n-1}      (predicted)
D = {n}              (completed)
w_n = 1 − Σ_{i∈P} w_i
```

**Assumption check.** Completion needs `A_D = A_eq[:, D]` to be invertible.
Here `A_eq = 1ᵀ` so `A_D = [1]` for *any* single index: `rank(A_eq) = 1 = n_eq`,
`cond(A_D) = 1`.  This is asserted at start-up (`LinearCompletion.check`) and the
numbers are printed and stored in every result file.  Column-pivoted QR
(`partition.strategy = "auto_qr"`) returns an equivalent choice.

**Correction.** `g(w) = [A w − b ; w_floor − w]` is affine, so
`G_eff = G_P − G_D·(A_D⁻¹A_P)` and

```
∇_z ‖relu(g)‖²  =  2 · relu(g)ᵀ G_eff
```

is evaluated in closed form (validated against autograd in `DC3/validate.py`).
The row norms of `G_eff` differ by only about one order of magnitude
(`‖A_i‖ ≈ √(n/3)` versus `‖e_i‖ ≈ √2`), so no row scaling is applied
(`ineq_row_scale` returns `None`).

**Objective domain.** `w log w` is undefined for `w < 0`.  Two things are done:

* the network output is passed through `w = w_floor + softplus(·)` (the
  generalisation of DC3's ACOPF `sigmoid` read-out to a one-sided bound) and
  initialised at the uniform distribution `w = 1/n`, as a bias target; random output weights and completion can still leave
  the domain;
* the completed coordinate `w_n = 1 − Σ w_i` can still go negative.  The
  objective is then evaluated as `w·log(max(w, 1e-30))` and the *domain
  violation* `max(−w)` is reported as a separate column, so an out-of-domain
  point is never presented as a solution.

**Network input.** `A` (flattened) concatenated with `b`, i.e. `x_dim = m·n + m`.
`b` is a deterministic function of `A`, so it is redundant in principle;
including it is a cheap, explicit summary statistic.  `feature_mode` also
accepts `"A_flat"` and `"b"` (the latter as an ablation).
Note `x_dim = 100_100` at the default size — the entire instance parameter is a
100k-dimensional object, which is far larger than anything in the DC3 paper and
is the dominant difficulty of this benchmark.

## 3. Data

`data.py` draws instances from the distribution above with NumPy's PCG64, using
disjoint `SeedSequence` spawn keys for train / validation / test, so the three
sets are guaranteed disjoint and reproducible from `data.seed` alone.  The
**test** split is written to `results/<tag>/test_instances.npz` and is what both
the Python reference solver and the Julia baselines read, so every method sees
identical instances.

> **Deviation.** Instances are drawn with NumPy rather than Julia's RNG (they must
> be shared across three tool-chains).  The distribution is identical; the
> particular realisations are not the ones a Julia run would produce.

> The `.npz` files under `data/training_data/maxEntropy-*` are **not** instances
> of this problem: they are `(q, Moreau-envelope value, gradient)` samples of the
> *scalar prox subproblem* used to fit the ICNN of LME-ADMM.  They are unrelated
> to DC3's training data and are not used here.

## 4. Reference solver

`reference.py` solves the exponential-cone form with cvxpy + **Clarabel** at
`tol = 1e-9`.  The repository's own ground truth, Ipopt at `tol = 1e-8`
(`JuMP_solver("Ipopt", para, 1e-8)`), is produced on the same instances by
`DC3/julia/baselines_cone.jl`; `DC3/validate.py` checks that the Python
objective and residuals agree with both.

## 5. Commands

```bash
# from the repository root, with DC3/.venv activated (see DC3/README.md)
python -m DC3.validate --app cone                      # formulation + gradient checks

# small configuration (n=100, m=10)
python -m DC3.cone_programming.train     --config DC3/cone_programming/configs/small.json --tag small
python -m DC3.cone_programming.benchmark --config DC3/cone_programming/configs/small.json --tag small

# headline configuration (n=1000, m=100)
python -m DC3.cone_programming.train     --tag default
python -m DC3.cone_programming.benchmark --tag default

# repository baselines on the same test instances (Ipopt + sLME-ADMM)
julia --project=. DC3/julia/baselines_cone.jl DC3/results/cone_programming-small
python -m DC3.report --app cone_programming --tag small

# hyper-parameter search on the validation split only
python -m DC3.tune --app cone_programming --config DC3/cone_programming/configs/small.json \
    --grid '{"dc3.corr_lr":[3e-3,1e-2],"dc3.lr":[1e-4,3e-4]}' --set dc3.epochs=40
```

Any config entry can be overridden on the command line, e.g.
`--set dc3.epochs=500 problem.n=200 data.n_test=200`.

## 6. Deviations from the paper / official code

| | |
|---|---|
| Completion | `h` is affine, so DC3's Newton completion collapses to one linear solve; the solve is done directly with a cached `A_D⁻¹`. |
| Correction space | steps are taken in `z` (partial) space instead of writing them into the full `y`. For affine completion the two are algebraically identical — asserted in `validate.py`. |
| Read-out | `w = w_floor + softplus(·)`, the one-sided analogue of DC3's ACOPF `sigmoid` read-out, plus a bias initialised at `w = 1/n`. Disable with `dc3.output_transform="none"`, `dc3.output_init_target=false`. |
| Soft loss | `soft_loss_power` selects the *norm* used by `method.py::total_loss` (`1`, default) or the *squared* norm written in the paper (`2`). |
| Objective scaling | `obj_scale` divides the objective term of the soft loss (`1.0` here — the objective is already O(10)). |
| Partition search | DC3's random search accepts a block when `|det| > 1e-4`; that test underflows for large `n_eq`, so `log|det| > log(1e-4)` is used instead. |

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
