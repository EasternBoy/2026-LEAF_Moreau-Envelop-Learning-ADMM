# DC3 benchmark - entr_max (small)

* instances: 500 test instances (`test_instances.npz`)
* feasibility: exact objective-domain membership and max |h|, max relu(g) <= 0.0001
* device `cpu`, dtype `torch.float64`, arm, torch 2.14.0 (6 threads)
* completion: `n_partial` = 99, cond(A_D) = 1.000e+00, error amplification ||A_D⁻¹A_P||₂ = 9.950e+00
* network parameters: 833,123; training time 609.6 s (excluded from the latency column)

Quality and headline latency come from the same batch=1 calls on every test instance.
Oracle-labeled rows use the known reference optimum for stopping; their reference-solve cost is excluded.
Undefined objective-domain values are never clamped for reporting. All-instance means/gaps are
undefined if any sample has an invalid objective; feasible-subset gaps are reported separately.

## Objective, optimality gap and feasibility

```
method             obj (mean)  gap% mean  gap% max   gap% mean(feas)  feas rate  domain valid  max |h|   max viol  latency ms
-----------------  ----------  ---------  ---------  ---------------  ---------  ------------  --------  --------  ----------
CLARABEL(cvxpy)    -4.55105    1.229e-14  3.913e-14  -                1.000      1.000         3.55e-11  0.00e+00  2.988
DC3 + correction   -           -          -          0.7695           0.740      0.740         4.44e-16  1.00e-04  0.7171
Ipopt(deployment)  -4.55105    9.729e-08  1.712e-07  9.729e-08        1.000      1.000         8.88e-16  0.00e+00  1.395
sLME-ADMM          -4.551      0.001199   0.004      0.001199         1.000      1.000         9.99e-16  4.99e-05  0.9876
LME-ADMM           -           -          -          -                -          -             -         -         -
Ipopt(tol=1e-8)    -4.55105    3.216e-14  1.363e-13  3.216e-14        1.000      1.000         8.88e-16  4.03e-11  1.506
```

The `CLARABEL(cvxpy)` row is the **reference**: its gap is 0 by definition.  Its
latency includes cvxpy canonicalisation on every call, so it is *not* a fair
solver-speed comparison - use the Julia `Ipopt` rows, which report
`JuMP.solve_time` on a pre-built parametric model, exactly as `examples/` does.

Gap is `100*|J - J_ref|/|J_ref|` against the reference solver, matching the
convention of `examples/*/benchmark*.jl`.  `gap% mean(feas)` restricts the
average to instances that pass the feasibility test, so an infeasible point that
undercuts the optimum is not reported as a better solution.

## Correction

* correction steps per timed single-instance solve: mean 25.00, max 119 (cap 1000)
* instances within `corr_eps` at the end: 100.0% (**0 correction failures**)
* first step at which an instance became feasible: mean 25.00, max 119

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
stage_predict_b1   0.05129    0.05158  0.05234  -            -
stage_complete_b1  0.005333   0.0054   0.00565  -            -
stage_correct_b1   2.506      2.483    2.56     -            -
single_instance    0.7171     1.185    3.123    -            -
```

## Julia baselines

Measured original-constraint feasibility and exact objective domain. Deployment stopping; no optimum oracle. Gurobi-dependent baselines not run.

