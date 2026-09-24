# DC3 benchmark - cone_programming (default)

* instances: 100 test instances (`test_instances.npz`)
* feasibility: exact objective-domain membership and max |h|, max relu(g) <= 0.0001
* device `cpu`, dtype `torch.float64`, arm, torch 2.14.0 (6 threads)
* completion: `n_partial` = 999, cond(A_D) = 1.000e+00, error amplification ||A_D⁻¹A_P||₂ = 3.161e+01
* network parameters: 25,949,415; training time 480.0 s (excluded from the latency column)

Quality and headline latency come from the same batch=1 calls on every test instance.
Oracle-labeled rows use the known reference optimum for stopping; their reference-solve cost is excluded.
Undefined objective-domain values are never clamped for reporting. All-instance means/gaps are
undefined if any sample has an invalid objective; feasible-subset gaps are reported separately.

## Objective, optimality gap and feasibility

```
method             obj (mean)  gap% mean  gap% max   gap% mean(feas)  feas rate  domain valid  max |h|   max viol  latency ms
-----------------  ----------  ---------  ---------  ---------------  ---------  ------------  --------  --------  ----------
CLARABEL(cvxpy)    -6.33571    6.028e-15  2.812e-14  -                1.000      1.000         2.31e-10  0.00e+00  1138
DC3 + correction   -           -          -          -                0.000      0.000         4.44e-16  2.15e-03  137.3
Ipopt(deployment)  -6.33571    6.946e-08  7.238e-08  6.946e-08        1.000      1.000         1.78e-15  0.00e+00  99.84
sLME-ADMM          -6.33552    0.003071   0.02186    0.003071         1.000      1.000         3.11e-15  1.71e-06  26.76
LME-ADMM           -           -          -          -                -          -             -         -         -
Ipopt(tol=1e-8)    -6.33571    9.876e-14  2.98e-13   9.876e-14        1.000      1.000         2.00e-15  4.61e-12  100.8
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

* correction steps per timed single-instance solve: mean 2000.00, max 2000 (cap 2000)
* instances within `corr_eps` at the end: 0.0% (**100 correction failures**)
* first step at which an instance became feasible: mean nan, max -1

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
stage_predict_b1   1.686      1.704     1.821     -            -
stage_complete_b1  0.006375   0.006438  0.006646  -            -
stage_correct_b1   134.2      134.9     135.4     -            -
single_instance    137.3      137.4     138.4     -            -
```

## Julia baselines

Measured original-constraint feasibility and exact objective domain. Deployment stopping; no optimum oracle. Gurobi-dependent baselines not run.

