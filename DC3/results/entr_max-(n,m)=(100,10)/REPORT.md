# DC3 benchmark - entr_max ((n,m)=(100,10))

* instances: 500 test instances (`test_instances.npz`)
* feasibility: exact objective-domain membership and max |h|, max relu(g) <= 0.0001
* device `cpu`, dtype `torch.float64`, arm, torch 2.14.0 (6 threads)
* completion: `n_partial` = 99, cond(A_D) = 1.000e+00, error amplification ||A_D⁻¹A_P||₂ = 9.950e+00
* network parameters: 833,123; training time 5459.9 s (excluded from the latency column)

Quality and headline latency come from the same batch=1 calls on every test instance.
Oracle-labeled rows use the known reference optimum for stopping; their reference-solve cost is excluded.
Undefined objective-domain values are never clamped for reporting. All-instance means/gaps are
undefined if any sample has an invalid objective; feasible-subset gaps are reported separately.

## Objective, optimality gap and feasibility

```
method            obj (mean)  gap% mean  gap% max   gap% mean(feas)  feas rate  domain valid  max |h|   max viol  latency ms
----------------  ----------  ---------  ---------  ---------------  ---------  ------------  --------  --------  ----------
CLARABEL(cvxpy)   -4.55105    1.229e-14  3.913e-14  -                1.000      1.000         3.55e-11  0.00e+00  2.872
DC3 + correction  -           -          -          0.7001           0.758      0.758         4.44e-16  1.00e-04  0.7101
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

* correction steps per timed single-instance solve: mean 25.06, max 115 (cap 1000)
* instances within `corr_eps` at the end: 100.0% (**0 correction failures**)
* first step at which an instance became feasible: mean 25.06, max 115

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
batch_1            3.636      3.635     3.738     3.636        275
batch_8            5.14       5.133     5.222     0.6426       1556.3
batch_32           14.43      14.45     14.55     0.451        2217.1
batch_100          28.74      28.77     28.97     0.2874       3479.9
stage_predict_b1   0.0495     0.05028   0.05254   -            -
stage_complete_b1  0.004875   0.004897  0.005083  -            -
stage_correct_b1   3.483      3.494     3.564     -            -
single_instance    0.7101     1.162     3.033     -            -
```

## Julia baselines

`julia_baselines.json` not present - the LME-ADMM / Ipopt baselines from
`experiments/` were **not executed** for this run.  Produce them with
`julia --project=. experiments/dc3/baselines_power.jl DC3/results/entr_max-(n,m)=(100,10)`.

