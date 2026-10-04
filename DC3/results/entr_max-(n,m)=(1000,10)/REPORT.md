# DC3 benchmark - entr_max ((n,m)=(1000,10))

* instances: 100 test instances (`test_instances.npz`)
* feasibility: exact objective-domain membership and max |h|, max relu(g) <= 0.0001
* device `cpu`, dtype `torch.float64`, arm, torch 2.14.0 (6 threads)
* completion: `n_partial` = 999, cond(A_D) = 1.000e+00, error amplification ||A_D⁻¹A_P||₂ = 3.161e+01
* network parameters: 710,055; training time 15080.4 s (excluded from the latency column)

Quality and headline latency come from the same batch=1 calls on every test instance.
Oracle-labeled rows use the known reference optimum for stopping; their reference-solve cost is excluded.
Undefined objective-domain values are never clamped for reporting. All-instance means/gaps are
undefined if any sample has an invalid objective; feasible-subset gaps are reported separately.

## Objective, optimality gap and feasibility

```
method            obj (mean)  gap% mean  gap% max  gap% mean(feas)  feas rate  domain valid  max |h|   max viol  latency ms
----------------  ----------  ---------  --------  ---------------  ---------  ------------  --------  --------  ----------
CLARABEL(cvxpy)   -6.85924    3.884e-15  2.59e-14  -                1.000      1.000         1.54e-10  0.00e+00  23.5
DC3 + correction  -           -          -         0.11             0.400      0.400         3.33e-16  1.00e-04  0.719
```

The `CLARABEL(cvxpy)` row is the **reference**: its gap is 0 by definition.  Its
latency includes cvxpy canonicalisation on every call, so it is *not* a fair
solver-speed comparison - use the Julia `Ipopt` rows, which report
`JuMP.solve_time` on a pre-built parametric model, exactly as `problems/` does.

Gap is `100*|J - J_ref|/|J_ref|` against the reference solver, matching the
convention of `experiments/*/benchmark*.jl`.  `gap% mean(feas)` restricts the
average to instances that pass the feasibility test, so an infeasible point that
undercuts the optimum is not reported as a better solution.

## Correction

* correction steps per timed single-instance solve: mean 40.96, max 368 (cap 2000)
* instances within `corr_eps` at the end: 100.0% (**0 correction failures**)
* first step at which an instance became feasible: mean 40.96, max 368

`correction failures` counts instances that did not reach `corr_eps` on DC3's
*internal* criterion (margin-tightened and, for the eco-MPC, row-scaled).
`feas rate` in the table above is measured on the **original** constraints, so
the two numbers can differ. Internal convergence does not certify objective-domain
membership; domain validity is checked separately. A finite step budget never implies
imply feasibility - both numbers are reported.

## Latency (warm-up + device synchronisation, timing excludes host->device transfer)

DC3's test-time correction loop is **batch-global**: it keeps stepping until every
instance in the batch is within `corr_eps`.  Large batches therefore pay for their
worst instance, which is why `ms/instance` is *not* monotone in the batch size.

```
stage              median ms  mean ms   p90 ms    ms/instance  inst/s
-----------------  ---------  --------  --------  -----------  ------
batch_1            0.7215     0.7206    0.7474    0.7215       1386
batch_8            30.15      30.15     30.46     3.769        265.31
batch_32           99.45      99.58     99.95     3.108        321.76
batch_100          426        426.7     432.2     4.26         234.77
stage_predict_b1   0.0705     0.07102   0.07307   -            -
stage_complete_b1  0.006333   0.006588  0.007099  -            -
stage_correct_b1   0.5906     0.589     0.601     -            -
single_instance    0.719      2.778     10.21     -            -
```

## Julia baselines

`julia_baselines.json` not present - the LME-ADMM / Ipopt baselines from
`experiments/` were **not executed** for this run.  Produce them with
`julia --project=. experiments/dc3/baselines_power.jl DC3/results/entr_max-(n,m)=(1000,10)`.

