# DC3 benchmark

An implementation of **DC3 — Deep Constraint Completion and Correction**
([Donti, Rolnick & Kolter, ICLR 2021](https://arxiv.org/abs/2104.12225);
official code [locuslab/DC3](https://github.com/locuslab/DC3), Apache-2.0)
applied to the two applications of this repository:

| folder | problem | source of truth |
|---|---|---|
| [`entr_max/`](entr_max/README.md) | maximum-entropy cone program | `problems/entr_max` |
| [`power_grid/`](power_grid/README.md) | economic MPC of a PV + BESS microgrid | `problems/power_grid` |

The Julia drivers in `DC3/julia/` `include`
the shared problem files so that the repository's own solvers
(Ipopt, sLME-ADMM, LME-ADMM) are benchmarked on **the same instances** as DC3.

Attribution and the upstream license are in
[`common/NOTICE.md`](common/NOTICE.md) and
[`common/LICENSE-Apache-2.0-DC3`](common/LICENSE-Apache-2.0-DC3).

## Layout

```
DC3/
  common/            problem API, equality completion, DC3 solver + training,
                     metrics, timing, experiment runner
  entr_max/  problem, data, reference solver, train / benchmark, configs
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
$PY -m DC3.entr_max.train     --config DC3/entr_max/configs/small.json --tag '(n,m)=(100,10)'
$PY -m DC3.entr_max.benchmark --config DC3/entr_max/configs/small.json --tag '(n,m)=(100,10)'
julia --project=. DC3/julia/baselines_cone.jl 'DC3/results/entr_max-(n,m)=(100,10)'
$PY -m DC3.report --app entr_max --tag '(n,m)=(100,10)'
```

Every config key can be overridden from the command line:
`--set dc3.epochs=500 dc3.corr_lr=1e-2 data.n_test=200`.

## Power-grid comparison table

The publication-style power-grid workflow compares IPOPT, MadNLP, ADMM,
MEL-ADMM, sMEL-ADMM, and DC3 on the same saved instances. Run every command
from the repository root. The Julia benchmark uses Gurobi for ADMM and
MEL-ADMM, so those methods require a working Gurobi installation and license.

Before running a horizon, set the same `N` in both:

- `experiments/power_grid/table.jl`
- `experiments/power_grid/table_benchmark.py`

Also set `g_opt` in `table.jl` to the desired target optimality gap
in percent (`0.1` means `0.1%`). Then run, for example, the `N = 96` case:

```bash
PY=DC3/.venv/bin/python

# Create the shared inputs once. Reuse the file if it already exists.
$PY experiments/power_grid/generate_table_instances.py \
    --N 96 --samples 1000 --seed 20262309

# Train and evaluate DC3 on those inputs.
$PY experiments/power_grid/train_table.py --tag table-N96
$PY experiments/power_grid/table_benchmark.py \
    --checkpoint DC3/results/power_grid-table-N96/checkpoint.pt

# Run the Julia solvers on the same inputs.
julia --project=. --threads=8 experiments/power_grid/table.jl
```

To run `N = 192`, change both `N` constants to `192`, generate the `N = 192`
inputs, and use a distinct checkpoint tag such as `table-N192`:

```bash
$PY experiments/power_grid/generate_table_instances.py \
    --N 192 --samples 1000 --seed 20262309
$PY experiments/power_grid/train_table.py --tag table-N192
$PY experiments/power_grid/table_benchmark.py \
    --checkpoint DC3/results/power_grid-table-N192/checkpoint.pt
julia --project=. --threads=8 experiments/power_grid/table.jl
```

Julia results are written to
`results/power_grid/table/gap=<g_opt>/`. A rerun replaces the matching
case-specific Julia files. Input generation and DC3 training do not overwrite
existing files; reuse existing inputs or choose a new checkpoint tag.

After both horizons are complete, set `G_OPT` in
`experiments/power_grid/boxplot.py` equal to Julia's `g_opt` and generate the figures:

```bash
$PY experiments/power_grid/boxplot.py
```

The PDFs are saved in `results/power_grid/figures/`.

## What the implementation does

1. **Partial-variable prediction** — an MLP (`Linear → BatchNorm → ReLU →
   Dropout`, Kaiming init, as in `method.py::NNSolver`) maps the instance
   parameters to `n_y − n_eq` partial variables.  An optional bounded read-out
   (DC3's ACOPF `sigmoid` device, generalised to one-sided bounds) bounds the
   partial prediction; completion can still leave the objective domain.
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
  `100·|J − J_ref|/|J_ref|` used by `experiments/*/benchmark*.jl`.  Because that
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

## Benchmark protocol (version 2)

Feasibility requires equality and inequality residuals within the configured
threshold **and exact objective-domain membership**: `w >= 0` for entropy
(with `0 log 0 = 0`) and `p > 0` for MPC. Clamped objectives remain training
surrogates only. Undefined objectives and gaps are reported as missing;
all-instance aggregates are undefined when any objective is undefined, and
feasible-subset gaps are reported separately.

Headline DC3 quality, per-instance correction counts, and latency now come
from the same batch=1 solve calls on every test instance after warm-up.
Additional batched timings are throughput diagnostics only.

Julia baseline drivers default to **deployment** stopping, without access to
the reference optimum for termination. To measure time to a known target,
pass `oracle` as the third argument after the output directory and gap (%).
Oracle-assisted rows are explicitly labeled; reference-solving cost is excluded.
All baseline feasibility rates are measured from returned solutions.

The power-grid LME split solver cold-starts every call, checks absolute
consensus residuals, and requires returned-solution feasibility. A callback
cannot bypass those conditions. A step cap can still return an infeasible
point, which is reported as a failure rather than assumed feasible.

The previous headline tables used the old protocol and have been withdrawn.
Use regenerated `results/<app>-<tag>/REPORT.md` files. Version-1 JSON results
are rejected by the report generator; rerun both benchmark and Julia drivers.
The entropy CP-table scripts similarly require version-2 metric artifacts.
