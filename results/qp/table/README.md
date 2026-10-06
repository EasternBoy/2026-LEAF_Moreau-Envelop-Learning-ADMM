# QP table results

Run from the repository root:

```sh
julia --project=. experiments/qp/table.jl
```

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
Compilation is excluded; sLME-ADMM's per-solve projection factorization is included.
Existing saved tables were produced with the earlier optimized solver; rerun the
script to replace their results with measurements of the restored solver.

sLME-ADMM stops when its consensus residual is below `SLME_TOL` and its
objective gap is at most `max_opt_gap` percent, or reaches the iteration cap.
The callback uses the OSQP optimum, so this is oracle-assisted stopping.
The table includes objective gap as mean (maximum) percent. sLME-ADMM files and
the Markdown table include `-gopt=1.0` in their names. Results are reported
without a feasibility gate. Parenthesized standard deviations across repeated experiments
are omitted because the script performs a single benchmark run.
