# DC3 benchmark - entr_max ((n,m)=(100,1))

* instances: 500 test instances (`test_instances.npz`)
* feasibility: exact objective-domain membership and max |h|, max relu(g) <= 0.0001
* device `cpu`, dtype `torch.float64`, arm, torch 2.14.0 (6 threads)
* completion: `n_partial` = 99, cond(A_D) = 1.000e+00, error amplification ||A_D⁻¹A_P||₂ = 9.950e+00
* network parameters: 367,715; training time 3958.0 s (excluded from the latency column)

Quality and headline latency come from the same batch=1 calls on every test instance.
Oracle-labeled rows use the known reference optimum for stopping; their reference-solve cost is excluded.
Undefined objective-domain values are never clamped for reporting. All-instance means/gaps are
undefined if any sample has an invalid objective; feasible-subset gaps are reported separately.

## Objective, optimality gap and feasibility

```
method             obj (mean)  gap% mean  gap% max   gap% mean(feas)  feas rate  domain valid  max |h|   max viol  latency ms
-----------------  ----------  ---------  ---------  ---------------  ---------  ------------  --------  --------  ----------
CLARABEL(cvxpy)    -4.60024    1.216e-14  5.792e-14  -                1.000      1.000         1.31e-10  0.00e+00  2.09
DC3 + correction   -           -          -          0.03462          0.996      0.996         5.55e-16  9.82e-05  0.3262
Ipopt(deployment)  -4.60024    9.609e-09  9.614e-09  9.609e-09        1.000      1.000         8.88e-16  0.00e+00  0.9959
sLME-ADMM          -4.60024    5.439e-05  0.0001963  5.439e-05        1.000      1.000         1.55e-15  2.04e-05  0.6453
LME-ADMM           -           -          -          -                -          -             -         -         -
Ipopt(tol=1e-8)    -4.60024    3.17e-14   1.159e-13  3.17e-14         1.000      1.000         8.88e-16  2.37e-11  1.072
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

* correction steps per timed single-instance solve: mean 4.81, max 11 (cap 1000)
* instances within `corr_eps` at the end: 100.0% (**0 correction failures**)
* first step at which an instance became feasible: mean 4.81, max 11

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
batch_1            0.3463     0.3494    0.3626    0.3463       2887.9
batch_8            0.6511     0.6426    0.6931    0.08139      12286
batch_32           1.328      1.332     1.361     0.04149      24104
batch_100          2.517      2.52      2.606     0.02517      39732
stage_predict_b1   0.04037    0.04118   0.04271   -            -
stage_complete_b1  0.004729   0.004759  0.004875  -            -
stage_correct_b1   0.2559     0.2584    0.2632    -            -
single_instance    0.3262     0.2935    0.4319    -            -
```

## Julia baselines

Measured original-constraint feasibility and exact objective domain. Deployment stopping; no optimum oracle. Gurobi-dependent baselines not run.

