# Julia baseline drivers

These scripts read the **exact** test instances exported by the Python
benchmark (`DC3/results/<app>/test_instances.npz`) and run the solvers that
already exist in `examples/`, so that DC3 and the repository's own methods are
compared on identical data.  They write `julia_baselines.json` next to the
instances; `DC3/report.py` picks that file up automatically.

Run them from the **repository root**:

```bash
julia --threads=auto --project=. DC3/julia/baselines_cone.jl  DC3/results/entr_max
julia --threads=auto --project=. DC3/julia/baselines_power.jl DC3/results/power_grid
```

`--threads=auto` matters: `examples/entr_max/benchmarkOG.jl` sizes its
mini-batch as `s_mb = max(div(n, nthreads())+1, 50)`, which exceeds `n` when Julia
runs single-threaded and makes `utils.jl::mini_batch` index out of bounds.  The
cone driver additionally clamps `s_mb` to `n` so it also works with one thread.

## Environment caveats (this machine)

* `Ipopt` works.
* **`Gurobi` has no license here.**  `examples/*/preprocess.jl` calls
  `Gurobi.Env()` at load time, so these scripts predefine `GUROBI_ENV = nothing`
  to skip that (the call is guarded by `if !(@isdefined(GUROBI_ENV))`).
  Consequently the baselines that need a Gurobi QP subproblem
  (`eMPC_ADMM.jl::ADMM_eco_iter` + `aux_solver_eco("Gurobi", ...)`, and
  `LME-ADMM.jl::LME_ADMM` for the cone program) **cannot be run** and are
  reported as `not_run` rather than silently replaced by another solver.
  The Gurobi-free learned baselines - `sLME_ADMM` (cone) and `LME_ADMM_split`
  (power grid) - do run.

## Stopping modes and measured feasibility

The optional arguments are `OUTPUT_DIR GAP_PERCENT MODE`. `MODE` defaults to
`deployment`; Ipopt uses its own tolerance and learned ADMM uses computable
residuals. The accurate reference is used only for post-solve scoring.

`oracle` enables a known-optimum target in addition to feasibility/residual
checks. These results are labeled oracle-assisted and exclude reference-solve
cost. For example:

```bash
julia --project=. DC3/julia/baselines_power.jl DC3/results/power_grid-default 0.01 oracle
```

Use different output directories to retain both modes. Feasibility and exact
objective-domain membership are measured for every returned solution, including
Ipopt references. The power driver requests the actual state trajectory from
JuMP. No feasible rate is hardcoded. Version-1 baseline files must be regenerated.
