# DC3 benchmark

An implementation of **DC3 — Deep Constraint Completion and Correction**
([Donti, Rolnick & Kolter, ICLR 2021](https://arxiv.org/abs/2104.12225);
official code [locuslab/DC3](https://github.com/locuslab/DC3), Apache-2.0)
applied to the two applications of this repository:

| folder | problem | source of truth |
|---|---|---|
| [`cone_programming/`](cone_programming/README.md) | maximum-entropy cone program | `examples/cone_programming` |
| [`power_grid/`](power_grid/README.md) | economic MPC of a PV + BESS microgrid | `examples/power_grid` |

Nothing outside `DC3/` is modified.  The Julia drivers in `DC3/julia/` `include`
the existing example files so that the repository's own solvers
(Ipopt, sLME-ADMM, LME-ADMM) are benchmarked on **the same instances** as DC3.

Attribution and the upstream license are in
[`common/NOTICE.md`](common/NOTICE.md) and
[`common/LICENSE-Apache-2.0-DC3`](common/LICENSE-Apache-2.0-DC3).

## Layout

```
DC3/
  common/            problem API, equality completion, DC3 solver + training,
                     metrics, timing, experiment runner
  cone_programming/  problem, data, reference solver, train / benchmark, configs
  power_grid/        idem
  julia/             drivers that run the repository's own baselines on DC3's
                     exported test instances
  validate.py        formulation / completion / gradient checks
  tune.py            validation-only hyper-parameter search
  report.py          tables + plots from benchmark.json
  results/           generated (checkpoints, json, csv, pdf)
```

## Install

```bash
uv venv --python 3.12 DC3/.venv            # or: python3.12 -m venv DC3/.venv
DC3/.venv/bin/pip install -r DC3/requirements.txt
```

All Python commands are run **from the repository root** so that `DC3` is
importable as a package:

```bash
DC3/.venv/bin/python -m DC3.validate
```

For the Julia baselines use the repository's own project:

```bash
julia --project=. DC3/julia/baselines_power.jl DC3/results/power_grid-default
```

## End-to-end

```bash
PY=DC3/.venv/bin/python

$PY -m DC3.validate                                        # 27 correctness checks

# economic MPC
$PY -m DC3.power_grid.train     --tag default
$PY -m DC3.power_grid.benchmark --tag default
julia --project=. DC3/julia/baselines_power.jl DC3/results/power_grid-default
$PY -m DC3.report --app power_grid --tag default

# cone program (small configuration)
$PY -m DC3.cone_programming.train     --config DC3/cone_programming/configs/small.json --tag small
$PY -m DC3.cone_programming.benchmark --config DC3/cone_programming/configs/small.json --tag small
julia --project=. DC3/julia/baselines_cone.jl DC3/results/cone_programming-small
$PY -m DC3.report --app cone_programming --tag small
```

Every config key can be overridden from the command line:
`--set dc3.epochs=500 dc3.corr_lr=1e-2 data.n_test=200`.

## What the implementation does

1. **Partial-variable prediction** — an MLP (`Linear → BatchNorm → ReLU →
   Dropout`, Kaiming init, as in `method.py::NNSolver`) maps the instance
   parameters to `n_y − n_eq` partial variables.  An optional bounded read-out
   (DC3's ACOPF `sigmoid` device, generalised to one-sided bounds) keeps the
   first prediction inside the objective's domain.
2. **Equality completion** — the remaining `n_eq` variables solve
   `A_eq y = b_eq`.  Both problems have affine equalities, so DC3's Newton
   completion collapses to one cached linear solve `y_D = A_D⁻¹(b_eq − A_P y_P)`.
   The invertibility and conditioning of `A_D` are checked, reported and stored.
3. **Inequality correction** — gradient descent with momentum on
   `‖relu(g)‖²` in the partial-variable space, so equalities stay satisfied by
   construction.  Closed form for affine `g`, autograd otherwise.
4. **Training through completion and correction** — `corr_train_steps`
   differentiable correction steps inside the graph, then DC3's soft loss.

Everything that DC3 exposes as a flag is exposed here (`common/dc3.py::DC3Config`):
`use_compl`, `use_train_corr`, `use_test_corr`, `corr_mode` (`partial`/`full`),
`corr_train_steps`, `corr_test_max_steps`, `corr_eps`, `corr_lr`,
`corr_momentum`, `soft_weight`, `soft_weight_eq_frac`, plus the additions
documented per application (`obj_scale`, `soft_loss_power`, `ineq_row_scale`,
`output_transform`).

## How results are reported

* **Objective** in the same convention as the Julia code, and the relative gap
  `100·|J − J_ref|/|J_ref|` used by `examples/*/benchmark*.jl`.  Because that
  absolute value hides the direction of the error, the **signed** gap and the gap
  **restricted to feasible instances** are reported next to it — an infeasible
  point that undercuts the optimum is never presented as a better solution.
* **Feasibility** — max/mean equality residual, max/mean inequality violation,
  objective-**domain** violation, number of violated rows, and the feasible-instance
  rate at several thresholds (`1e-6 … 1e-2`; the headline threshold is DC3's
  `corr_eps = 1e-4`).  Residuals are always measured on the *original*
  constraints, never on the internally scaled or margin-tightened ones.
* **Correction** — steps actually used, how many instances reached `corr_eps`,
  and the number of **correction failures**.  A finite step budget is never
  assumed to imply feasibility.
* **Latency** — single-instance (`batch = 1`) separately from batched
  throughput, after warm-up and with device synchronisation, plus a per-stage
  breakdown (predict / complete / correct).  Training and data-generation time
  are reported separately and are never folded into the latency column.

Timing boundary: the measured region starts at the network input (parameters
already resident on the device) and ends after the final completion — i.e. what a
deployment pays per query.  This matches how the Julia baselines are timed
(`JuMP.solve_time` / `time_ns` around the solve, with parameters already bound).

## Results obtained on this machine

Apple M5 Pro, CPU, float64, torch 2.14.0 (6 torch threads), Julia 1.x with Ipopt.
Feasibility threshold 1e-4 on the equality residual, the inequality violation and
the objective-domain violation simultaneously.  Reference cross-checks:
Julia/Ipopt and Python/Clarabel agree to `9e-9` (eco-MPC) and `4.5e-9` (cone,
n=1000) relative.

| experiment | method | gap % mean | feasible rate | latency ms | training s |
|---|---|---|---|---|---|
| eco-MPC, N=96, 500 inst. | Ipopt (early stop) | 0.0047 | 1.000 | 10.80 | — |
| | LME-ADMM (split) | 0.563 | 0.506 | 3.10 | — |
| | **DC3** | 5.83 | **1.000** | **0.735** | 635 |
| | DC3, no correction | 267 | 0.000 | 0.044 | — |
| cone, n=100, m=10, 500 inst. | Ipopt (early stop) | 0.0035 | 1.000 | 1.09 | — |
| | sLME-ADMM | 0.0032 | 0.020 | 0.38 | — |
| | **DC3** | 0.777 | **1.000** | **0.73** | 610 |
| | DC3, no correction | 3.675 | 0.000 | 0.053 | — |
| cone, n=1000, m=100, 100 inst. | Ipopt (early stop) | 0.0042 | 1.000 | 58.6 | — |
| | sLME-ADMM | 0.0094 | 0.990 | 12.9 | — |
| | **DC3** | **75.4** | **0.000** | 138.8 | 480 |
| | DC3, no correction | 5.14 | 0.000 | 1.51 | — |

Headlines:

* **DC3 buys feasibility with optimality.**  On both problems where it works it
  is the only method with a 100 % feasible rate and the lowest latency, but its
  objective gap is 1–3 orders of magnitude worse than the repository's learned
  ADMM.  The ADMM baselines stop on an *optimality-gap* test and consequently
  return points that violate the inequalities (50.6 % feasible for the eco-MPC,
  2.0 % for the small cone program).  Reporting gap without feasibility, or
  feasibility without gap, would misrepresent either method.
* **Correction does the heavy lifting.**  Network + completion alone is feasible
  on 0 % of instances in every configuration.
* **DC3 does not scale to the n=1000 cone program** — the completion amplifies
  prediction error by `‖A_D⁻¹A_P‖₂ = √(n−1) = 31.6`, which is intrinsic to the
  simplex constraint and independent of the partition.  Details and the
  correction-learning-rate sweep are in
  [`cone_programming/README.md`](cone_programming/README.md#7-executed-results).
* **Gurobi-dependent baselines could not be run** (no license on this machine);
  they are reported as `not_run` rather than silently substituted.

Per-experiment tables, plots and machine-readable output live in
`results/<app>-<tag>/` (`REPORT.md`, `benchmark.json`, `summary.csv`,
`per_instance.csv`, `dc3_gap_violation.pdf`, `latency_comparison.pdf`,
`julia_baselines.json`).
