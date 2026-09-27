# Repository layout

| folder | contents |
|---|---|
| `src/` | code shared by every problem: `metrics.jl` (optimality gap, constraint violation, feasibility), `icnn.jl` (the learned Moreau envelope: ICNN, `load_model`, `gradient_struct`, `mini_batch`), `kkt.jl` (the sLME-ADMM v-step: projection onto `{v : M v = b}` through one LDLᵀ-factored KKT system) |
| `problems/<p>/` | one problem's library code, included by the experiments, never run directly |
| `experiments/<p>/` | runnable scripts: benchmarks, data generation, ICNN training, tables and figures |
| `python/` | `micnn.py`: ICNN (Moreau-envelope model) training used by every `experiments/*/train.py`; `make_icnn(weight_act, keep_best)` selects each problem's settings |
| `models/<p>/` | trained ICNNs (`.json` read by Julia, `.pkl` from JAX); `models/legacy/` holds the former `model/` folder |
| `data/<p>/` | training data (`training/`) and input data (`micro_grid/` for the power grid) |
| `results/<p>/` | benchmark outputs, tables and figures |
| `DC3/` | the DC3 + correction baseline (Python package, see DC3/README.md) |
| `archive/` | code that nothing runs any more (former `src/`, `script/`), kept for reference |

Problems `<p>`: `entr_max` (maximum-entropy cone program), `mpc`, `power_grid`
(economic MPC of a PV + BESS microgrid), `mvee` (minimum-volume enclosing ellipsoid).
Every `problems/<p>/` uses the same file names:

| file | contents |
|---|---|
| `problem.jl` | problem data and objective (`data_opt`, `energy_mag`, ...) |
| `setup.jl` | loads the trained ICNN, defines the stopping callbacks and `pick_solver` |
| `jump_solver.jl` | the JuMP/IPOPT baseline |
| `admm.jl` | the ADMM baseline |
| `lme_admm.jl` | LME-ADMM and sLME-ADMM (each problem keeps its own loop: the variants differ in their updates and stopping rules) |
| `utils.jl` | includes `src/icnn.jl`, loads this problem's ICNN, and problem-specific helpers |

`entr_max` and `mpc` load their ICNN in `utils.jl` (`mpc` has no `setup.jl`); `mvee` also has
`convex_solver.jl` (a Convex.jl formulation).

Run every script from the repository root (the scripts call `Pkg.activate(".")`
and use paths relative to the root).

# Run scripts
## Whole table
Solving-time / optimality-gap table for the maximum-entropy cone program:
the IPOPT and sLME-ADMM columns.  The DC3 + correction column is filled by
experiments/entr_max/table.py.

  julia --project=. experiments/entr_max/table.jl            # use stored data, run what is missing
  julia --project=. experiments/entr_max/table.jl --force    # recompute everything

Internal mode (one (n, m) per process, because `n` is a `const` in
problems/entr_max/problem.jl):

  julia --project=. --threads=auto experiments/entr_max/table.jl worker n m [--instances-only] [--force]

Data and the rendered table live in results/entr_max/table (see README.md there).


## Each row of table
One (n, m) row block of the maximum-entropy table (table.jl) over
N_SAMPLES = 1000 instances: solving time (g_opt ≤ 0.1%), solving time
(g_opt ≤ 1%), Constr. viol. and Opt. gap (%) for IPOPT and sLME-ADMM.

  julia --project=. experiments/entr_max/row.jl n m            # use stored data, run what is missing
  julia --project=. experiments/entr_max/row.jl n m --force    # recompute this (n, m)

The DC3 + correction column is read from results/entr_max/table/DC3-n=..-m=...npz, which
experiments/entr_max/table.py writes; it shows — when that file does not exist.
The rows are printed and written to results/entr_max/table/entr_max_row-n=..-m=...md.
