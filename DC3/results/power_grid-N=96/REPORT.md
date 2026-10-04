# DC3 benchmark - power_grid (N=96)

* instances: 500 test instances (`test_instances.npz`)
* feasibility: exact objective-domain membership and max |h|, max relu(g) <= 0.0001
* device `cpu`, dtype `torch.float64`, arm, torch 2.14.0 (6 threads)
* completion: `n_partial` = 191, cond(A_D) = 5.571e+04, error amplification ||A_D⁻¹A_P||₂ = 1.384e+01
* network parameters: 462,015; training time 635.0 s (excluded from the latency column)

Quality and headline latency come from the same batch=1 calls on every test instance.
Oracle-labeled rows use the known reference optimum for stopping; their reference-solve cost is excluded.
Undefined objective-domain values are never clamped for reporting. All-instance means/gaps are
undefined if any sample has an invalid objective; feasible-subset gaps are reported separately.

## Objective, optimality gap and feasibility

```
method            obj (mean)  gap% mean  gap% max   gap% mean(feas)  feas rate  domain valid  max |h|   max viol  latency ms
----------------  ----------  ---------  ---------  ---------------  ---------  ------------  --------  --------  ----------
CLARABEL(cvxpy)   37499.1     5.24e-15   3.972e-14  -                1.000      1.000         6.03e-09  7.74e-09  9.614
DC3 + correction  39505.7     5.354      37.07      5.354            1.000      1.000         1.07e-13  9.07e-06  0.7288
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

* correction steps per timed single-instance solve: mean 21.18, max 500 (cap 500)
* instances within `corr_eps` at the end: 97.6% (**12 correction failures**)
* first step at which an instance became feasible: mean 9.40, max 194

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
stage              median ms  mean ms  p90 ms   ms/instance  inst/s
-----------------  ---------  -------  -------  -----------  ------
stage_predict_b1   0.04783    0.04785  0.04917  -            -
stage_complete_b1  0.01385    0.01505  0.01604  -            -
stage_correct_b1   0.4817     0.4862   0.5051   -            -
single_instance    0.7288     1.713    0.8909   -            -
```

## Julia baselines

`julia_baselines.json` not present - the LME-ADMM / Ipopt baselines from
`experiments/` were **not executed** for this run.  Produce them with
`julia --project=. experiments/dc3/baselines_power.jl DC3/results/power_grid-N=96`.

