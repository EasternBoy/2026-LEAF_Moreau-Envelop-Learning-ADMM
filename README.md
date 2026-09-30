# Repository layout

| folder | contents |
|---|---|
| `src/` | the `LMEADMM` package (`src/LMEADMM.jl`; this repository's `Project.toml` is the package, so scripts run with `--project=.` load it with `using LMEADMM`), shared by every problem: `metrics.jl` (optimality gap, constraint violation, feasibility), `icnn.jl` (the learned Moreau envelope: ICNN, `load_model`, `gradient_struct`, `mini_batch`), `kkt.jl` (the sLME-ADMM v-step: projection onto `{v : M v = b}` through one LDLᵀ-factored KKT system), `solvers.jl` (`solver_model`: JuMP models for Ipopt, MadNLP, OSQP, Gurobi, Clarabel, ECOS, Mosek; `callback_struct`) |
| `problems/<p>/` | one problem's library code, included by the experiments, never run directly |
| `experiments/<p>/` | runnable scripts: benchmarks, data generation, tables and figures |
| `python/` | ICNN (Moreau-envelope model) training: `python python/train.py <config>` trains with the settings in `python/configs/<config>.json` (data, output, `make_icnn` choices, `train_icnn` arguments) using `micnn.py`; `<config>` is `<p>` or a variant `<p>-<name>` |
| `models/<p>/` | trained ICNNs (`.json` read by Julia, `.pkl` from JAX, or a single `.npz` when the config's output ends in `.npz`; `load_model` reads both); `models/legacy/` holds the former `model/` folder |
| `data/<p>/` | training data (`training/`) and input data (`micro_grid/` for the power grid) |
| `results/<p>/` | benchmark outputs, tables and figures |
| `DC3/` | the DC3 + correction baseline (Python package, see DC3/README.md) |

Problems `<p>`: `entr_max` (maximum-entropy cone program), `mpc`, `power_grid`
(economic MPC of a PV + BESS microgrid), `mvee` (minimum-volume enclosing ellipsoid).
Every `problems/<p>/` uses the same file names:

| file | contents |
|---|---|
| `problem.jl` | problem data and objective (`data_opt`, `energy_mag`, ...) |
| `setup.jl` | loads the trained ICNN, defines the stopping callbacks (they differ per problem) and `pick_solver` on top of `src/solvers.jl` |
| `jump_solver.jl` | the JuMP/IPOPT baseline |
| `admm.jl` | the ADMM baseline |
| `lme_admm.jl` | LME-ADMM and sLME-ADMM (each problem keeps its own loop: the variants differ in their updates and stopping rules) |
| `utils.jl` | `using LMEADMM`, loads this problem's ICNN, and problem-specific helpers |

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

Internal mode (one (n, m) per process, so that the timings of different sizes do
not share a Julia session; the problem code itself takes n and m from `data_opt`):

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


## ICNN for the table
The table's sLME-ADMM uses the ICNN loaded in problems/entr_max/utils.jl,
models/entr_max/mEntropy-selfsupME-lw10-admm20-1144-rho=1-16.npz.  It and the supervised
alternative are trained on Moreau-envelope data collected in the ADMM loop (the prox inputs of
the first 20% of the iterations, sizes (100,1) (100,10) (1000,10) (1000,100) in shares 1:1:4:4):

  julia --project=. experiments/entr_max/data_gen_admm.jl 1,1,4,4    # data/entr_max/training/maxEntropy-admm20-1144-*
  python python/train.py entr_max-selfsupME-lw10                      # the default: mean T(q̂) + 10 · (ICNN(q) − ME(q))²
  python python/train.py entr_max                                     # value + 5 · gradient labels

The default uses no gradient labels: with q̂ = q − ∇ICNN(q)/ρ and
T(p) = f(p) + ρ/2 ‖p − q‖², mean T(q̂) is smallest when q̂ = prox(q) (T(prox(q)) = ME(q)),
and the ME labels fix the value.  To benchmark another ICNN in place of the default one:

  julia --project=. experiments/entr_max/table.jl --model=models/entr_max/mEntropy-admm20-1144-rho=1-16.npz --tag=admm20-1144

Its sLME-ADMM runs and table go to results/entr_max/table/<tag>/; IPOPT, DC3 and the ground
truth are read from results/entr_max/table.  row.jl takes the same --model and --tag.
