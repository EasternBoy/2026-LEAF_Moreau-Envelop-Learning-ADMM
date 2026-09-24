> Protocol update: regenerate CP-table artifacts with both scripts (`--force`).
> Version-1 results used tolerant domain checks. Feasibility now requires w >= 0;
> invalid entropy objectives/gaps are undefined. IPOPT and sLME rows are
> oracle-assisted time-to-target measurements, excluding reference-solving cost.

# Cone-program benchmark data (maximum-entropy problem)

This folder holds the raw per-instance data behind [entr_max_table.md](entr_max_table.md),
the solving-time / optimality-gap table for
`min Σ wᵢ log wᵢ  s.t.  1ᵀw = 1,  A w ≤ b` (`examples/entr_max`).

## Reproducing

Run from the repository root:

```bash
julia --project=. script/entropy_max/entr_max_table.jl    # IPOPT and sLME-ADMM columns
python script/entropy_max/entr_max_table.py               # DC3 column (DC3/.venv, see DC3/README.md)
```

Both scripts read the data in this folder and only compute what is missing.
Either one rewrites `entr_max_table.md` from all the data present, so the table is
complete after both have run.  `--force` recomputes a script's data;
`entr_max_table.py --retrain` also retrains the DC3 networks.

`entr_max_table.jl` handles each (n, m) in a separate Julia process started with
`--threads=auto`, because `n` is a `const` in
`examples/entr_max/maxEntropy.jl`.

## Instances and ground truth

* **Instances:** 1000 per (n, m), drawn with the repository's own
  `data_opt(n, m)`: `A ~ U(0,1)^{m×n}`, `bᵢ = Σⱼ Aᵢⱼ / (1.06 n)`.  The Julia RNG
  is seeded with `Random.seed!(20260923)`.  All three methods see the same
  instances.
* **Ground truth:** `J_opt` is Ipopt with `tol = 1e-8`.  Every optimality gap is
  `g = 100 |J − J_opt| / |J_opt|` (in %), where `J = Σ wᵢ log wᵢ` is evaluated
  on the point the method returns.

## Methods and stopping rules

| method | stops when | solving time |
|---|---|---|
| IPOPT | `Ipopt_callback_BM`: the gap of the current iterate is below `g_opt` **and** Ipopt's primal infeasibility `inf_pr < FEAS_TOL·scale` with `FEAS_TOL = 1e-5` (a violation below 1e-5 in `w`; `ipopt_feas_tol` in `preprocess.jl`, default 1e-4, set by `entr_max_table.jl`); Ipopt `tol` 1e-4 otherwise | `JuMP.solve_time` |
| sLME-ADMM | `sLME_ADMM_callback`: gap < `g_opt` **and** ADMM residual `max(‖w−v‖∞, ‖v−z‖∞) < tol = SLME_TOL·scale` with `SLME_TOL = 1e-5` (the residual is in `x = scale·w`, so this bounds it by 1e-5 in `w`: tol = 2e-3 at n = 100, 2e-2 at n = 1000), or 1000 iterations.  The gap is computed on the `n` decision variables only (the iterate is `[x; s]` with `m` slacks; including the slacks biased the gap and left some runs at the cap) | wall time of `sLME_ADMM`, including the LDLᵀ factorisation |
| DC3 + correction | a single forward pass; correction runs until the violation is ≤ `corr_eps = 1e-4` or the step cap is hit | wall time of `DC3Solver.solve` for one instance (batch = 1): predict + complete + correct |

In every run the first solve is a warm-up and is not recorded.

**DC3 has no gap target.**  A DC3 time cell shows a number only if DC3 meets
the row's `g_opt` on **every** instance (max gap ≤ `g_opt`) **and** is feasible
on every instance (max violation ≤ 1e-4).  Otherwise it shows *unable to
achieve*.  DC3 networks: (100, 10) and (1000, 100) reuse
`DC3/results/entr_max-small` and `-default`.  (100, 1) and (1000, 10) are
trained by `entr_max_table.py`, with the settings of `small.json` (n = 100) or
`default.json` (n = 1000), into `DC3/results/entr_max-table-n{n}-m{m}`.

**Opt. gap rows.**  IPOPT is the ground truth, so its gap is 0.  sLME-ADMM
shows the gap of its `g_opt = 0.1 %` run.  DC3 shows the gap of its only run,
with its feasible rate when that is below 100 %.

**Feasibility is not in the table.**  Every file stores `max_viol` per
instance.  IPOPT and sLME-ADMM return a feasible point (`max_viol ≤ 1e-4`) on
every instance, because both stopping rules require feasibility as well as the
gap.  For IPOPT this was added to the callback: stopping on the gap alone
returned infeasible interior-point iterates (at n = 1000, m = 100,
`g_opt = 1 %`, 94 % of them violated a constraint by more than 1e-4).

## Files

All arrays have one entry per instance (length 1000, same order as the
instance file).  Time is in ms, gap in %.

| file | arrays |
|---|---|
| `instances/instances-n={n}-m={m}.npz` | `A` (1000, m, n), `b` (1000, m).  Git-ignored (up to 800 MB); regenerated deterministically by `entr_max_table.jl` |
| `ground_truth-n={n}-m={m}.npz` | `J_opt`, `time_ms` (Ipopt, tol 1e-8), `seed` |
| `IPOPT-gopt={1,0.1}-n={n}-m={m}.npz` | `time_ms`, `gap_pct`, `max_viol`, `iterations` (unused, 0) |
| `sLME-ADMM-gopt={1,0.1}-n={n}-m={m}.npz` | `time_ms`, `gap_pct`, `max_viol`, `iterations` |
| `DC3-n={n}-m={m}.npz` | `time_ms`, `gap_pct`, `max_viol`, `feasible`, `corr_steps` |
| `entr_max_table.md` | the rendered table |

`max_viol = max(max(A w − b), max(−w), |1ᵀw − 1|, 0)`.  For DC3 it is the
largest of the equality residual, the inequality violation and the
objective-domain violation `max(−w)`.
