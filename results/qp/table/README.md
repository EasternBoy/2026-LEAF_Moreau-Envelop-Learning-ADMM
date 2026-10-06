# QP table results

Run from the repository root:

```sh
julia --project=. experiments/qp/table.jl
```

Optional positional arguments are the sample count, sLME-ADMM consensus tolerance,
and iteration cap; defaults are `1000 1e-3 1000`:

```sh
julia --project=. experiments/qp/table.jl 833 1e-3 1000
```

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
Compilation and sLME-ADMM's reusable projection setup are excluded.

sLME-ADMM stops only on its consensus residual or iteration cap. The OSQP
objective is used for scoring, not stopping. Results are reported without a
feasibility gate. Parenthesized standard deviations across repeated experiments
are omitted because the script performs a single benchmark run.
