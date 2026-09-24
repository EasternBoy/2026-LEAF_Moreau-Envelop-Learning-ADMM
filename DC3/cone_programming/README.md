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
  initialised at the uniform distribution `w = 1/n`, so training starts inside
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

## 7. Executed results

Both configurations below were run end to end on this machine (Apple M5 Pro,
CPU, float64, torch 2.14.0).  Full tables: `results/cone_programming-small/REPORT.md`
and `results/cone_programming-default/REPORT.md`.

### `small.json` — n = 100, m = 10, 500 test instances

| method | obj (mean) | gap % mean | gap % max | feasible rate | max violation | latency (ms) |
|---|---|---|---|---|---|---|
| Clarabel (reference) | −4.55105 | 0 | 0 | 1.000 | 0 | 2.99* |
| Ipopt, tol 1e-8 (repo ground truth) | −4.55105 | 0 | 0 | 1.000 | — | 1.45 |
| Ipopt, early stop @0.1 % | −4.54911 | 0.043 | 0.093 | 1.000 | 1.0e-15 | 0.86 |
| sLME-ADMM (`LME-ADMM.jl`) | −4.55104 | 0.00019 | 0.00072 | 1.000 | 4.9e-6 | 1.36 |
| **DC3 + correction** | −4.51568 | **0.777** | 8.70 | **1.000** | 9.8e-5 | **0.73** |

\* the cvxpy row re-canonicalises on every call; use the Ipopt rows for solver speed.

Ipopt (tol 1e-8) and Clarabel (tol 1e-9) agree to `2.3e-10` relative, so the
reference is sound.

**Correction iterations.**  Training unrolls `corr_train_steps = 10`
differentiable correction steps.  At test time correction runs until every
instance in the batch satisfies `corr_eps = 1e-4` (cap `corr_test_max_steps =
1000`): it stopped after **119 steps**, all 500 instances converged, and the
mean / max step at which an instance first became feasible was 25.0 / 119.

**sLME-ADMM stopping rule.**  `sLME_ADMM` stops only when *both* the optimality
gap (`sLME_ADMM_callback`, < `max_opt_gap` = 0.1 % against the Ipopt optimum)
*and* the ADMM residual `max(‖w−v‖∞, ‖v−z‖∞) < tol = 1e-3` hold, or at 1000
iterations.  The gap is computed on the `n` decision variables only; an
earlier version also summed `s log s` over the `m` slack variables of the ADMM
iterate `[x; s]`, which biased the gap and left some instances running to the
1000-iteration cap.  It used 19–27 iterations (median 21) here; the residual is
the binding condition (the iteration counts are identical with a 0.01 % gap
threshold).  With the earlier rule (gap *or* residual) the gap test fired after
a median of 7 iterations and only 2 % of the returned points were feasible.

**Ipopt early-stop rule.**  `Ipopt_callback_BM` stops when the gap of the
current iterate is below 0.1 % *and* Ipopt's primal infeasibility
`inf_pr < 1e-4·scale` (a violation below 1e-4 in `w`).  Stopping on the gap
alone returned infeasible interior-point iterates (up to 94 % infeasible at
n = 1000, m = 100 with a 1 % threshold).

The trade-off: DC3 + correction and sLME-ADMM are both feasible on every
instance.  DC3 + correction is the fastest end to end (0.73 ms vs 1.36 ms), but
its objective gap (0.78 %) is about four orders of magnitude worse than
sLME-ADMM's (0.00019 %).

### `default.json` — n = 1000, m = 100, 100 test instances — **DC3 fails here**

| method | obj (mean) | gap % mean | feasible rate | max violation | latency (ms) |
|---|---|---|---|---|---|
| Clarabel (reference) | −6.33571 | 0 | 1.000 | 0 | 1138* |
| Ipopt, tol 1e-8 | −6.33571 | 0 | 1.000 | — | 97.2 |
| Ipopt, early stop @0.1 % | −6.33262 | 0.049 | 1.000 | 9.4e-5 | 51.6 |
| sLME-ADMM | −6.33551 | 0.0032 | 1.000 | 5.5e-8 | 40.2 |
| **DC3 + correction** | −1.5639 | **75.4** | **0.000** | 2.2e-3 | 138.8 |

sLME-ADMM (same stopping rule as above) is feasible on every instance and
stops after 59–104 iterations (median 91.5); no instance reaches the
1000-iteration cap.  It is faster than Ipopt's early stop (40 vs 52 ms median).
These baselines run Julia single-threaded; with `--threads=auto`
(`script/entropy_max/CP_table.jl`) sLME-ADMM takes about 27 ms here.

This is a negative result and is reported as such.  Two causes, both measured:

1. **The completion amplifies error by `‖A_D⁻¹A_P‖₂ = √(n−1)`.**  For the simplex
   equality `1ᵀw = 1` this is `31.6` at n = 1000 (and `9.95` at n = 100), and it
   is the *same for every choice of dependent variable* — the amplification is a
   property of the constraint, not of the partition.  The single completed
   coordinate absorbs the aggregate error of the 999 predicted ones: measured, it
   comes out at `w_D ≈ 0.19` where a typical coordinate is `1e-3`.
2. **The correction cannot recover.**  Training unrolls 10 correction steps; at
   test time the loop ran the full cap of **2000 steps** (`corr_test_max_steps`)
   and no instance reached `corr_eps = 1e-4`.  Pushing `A w ≤ b` moves the 999
   partial variables, each unit of which is absorbed 1:1 by `w_D`; after the
   correction 162 of 1000 coordinates are negative (outside the domain of
   `w log w`) and the objective ends at −1.56 *without* reaching feasibility.  `corr_lr` was swept over `{1e-5, 3e-5, 1e-4, 3e-4}` on the
   validation split — every value gives a 0 % feasible rate, and `1e-2` (the value
   that is optimal at n = 100) diverges to NaN.

A third, independent difficulty: the instance parameter `A` is a
100 000-dimensional object, far larger than anything in the DC3 paper, and the
training set that fits in memory is 800 instances, so the network essentially
learns an instance-averaged solution.

No attempt was made to "rescue" this configuration with problem-specific tricks
(for example a softmax read-out that would enforce the simplex directly), because
that would no longer be DC3.
