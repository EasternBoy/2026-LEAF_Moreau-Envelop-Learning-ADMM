# QP table results

Run from the repository root:

```sh
julia --project=. experiments/qp/table.jl
```

Add the DC3 + correction results and render a three-method table, following
the entropy workflow:

```sh
python3 experiments/qp/table.py
python3 experiments/qp/boxplot.py
```

Save an entropy-style row summary from the existing result files:

```sh
julia --project=. experiments/qp/row.jl
```

This writes `qp_row-*-tol=*-gopt=*.md`, showing mean (maximum) time in ms,
constraint violation, and objective gap. As in entropy, a timing cell is marked
"unable to achieve" unless all saved solutions meet the gap target and feasibility
threshold c_v = 1e-2. Missing DC3 results are shown as —. Missing Julia results are
generated; `--force` reruns the Julia solvers. DC3 is not trained or evaluated by
`row.jl`; use `table.py` first to include its column.

`table.py` uses `DC3/results/qp-default/checkpoint.pt`, training it only if missing
or `--retrain` is supplied. Use `--force` to re-evaluate DC3. It evaluates the
same saved instances at batch size 1, warms up, and synchronizes the device
around each solve. `evaluate.jl` reuses `table.jl`'s scoring and rendering for
all three methods, with gaps measured against the saved OSQP optimum.
DC3 times include prediction, completion, and correction with inputs already
on-device; Julia uses its solvers' internal times, so setup exclusions differ.

The scripts use the default dimensions, 833 samples, sLME tolerance 1e-3, and
gap target 1.0%; keep their filename constants aligned if changing these settings.
The combined table keeps the existing `qp_table-*.md` filename and reports
mean per-instance seconds. `DC3-n=100-neq=50-m=50-samples=833.npz` stores DC3
solutions, times, correction counts, and the same scored metrics as the Julia
methods. The box plots are `qp_gap_boxplot.pdf` and `qp_viol_boxplot.pdf` in
this folder. Like entropy's plot script, rendering uses Matplotlib and LaTeX.

Edit `N_SAMPLES`, `SLME_TOL`, `MAX_ITER`, and `max_opt_gap` directly in
`table.jl`. Their values are `833`, `1e-3`, `1000`, and `1.0` (percent).

The fixed QP matrices are generated with DC3's NumPy seed 17. Instance right-hand
sides use Julia seed 20260923, skipping one warmup draw to match `benchmark.jl`.
The script saves and reuses those instances under `instances/`. It reruns the
solvers and writes these files, with dimensions and sample count in their names:

- `ground_truth-*.npz`: OSQP solutions (`W`), optimal objectives (`J_opt`),
  solve times (`time_ms`), seed, and tolerance.
- `OSQP-*-tol=1.0e-8.npz` and `sLME-ADMM-*-tol=*.npz`: returned solutions (`W`),
  objectives (`objective`), times (`time_ms`), objective gaps (`gap_pct`),
  per-instance maximum and mean equality/inequality violations (`max_eq`,
  `mean_eq`, `max_ineq`, `mean_ineq`), and combined maximum violations (`max_viol`).
- `qp_table-*-tol=*.md`: DC3-style columns for both solvers.

The table averages each metric across instances, including each instance's
maximum violation. Equality violations use absolute residuals; inequality
violations use positive parts. All violations are in original coordinates.
Saved times are in milliseconds; table times are mean seconds per instance.
Compilation is excluded. sLME-ADMM's default projection setup occurs before its
internal timer and is excluded from the reported per-instance solving time.
Existing saved tables were produced with the earlier optimized solver; rerun the
script to replace their results with measurements of the restored solver.

sLME-ADMM stops when its consensus residual is below `SLME_TOL` and its
objective gap is at most `max_opt_gap` percent, or reaches the iteration cap.
The callback uses the OSQP optimum, so this is oracle-assisted stopping.
The table includes objective gap as mean (maximum) percent. sLME-ADMM files and
the Markdown table include `-gopt=1.0` in their names. Results are reported
without a feasibility gate. Parenthesized standard deviations across repeated experiments
are omitted because the script performs a single benchmark run.
