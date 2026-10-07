# DC3 benchmark - qp (default)

* instances: 833 test instances (`test_instances.npz`)
* feasibility: exact objective-domain membership and max |h|, max relu(g) <= 0.0001
* device `cpu`, dtype `torch.float64`, arm, torch 2.14.0 (8 threads)
* completion: `n_partial` = 50, cond(A_D) = 3.557e+02, error amplification ||A_D⁻¹A_P||₂ = 1.674e+02
* network parameters: 316,466; training time 2506.2 s (excluded from the latency column)

Quality and headline latency come from the same batch=1 calls on every test instance.
Oracle-labeled rows use the known reference optimum for stopping; their reference-solve cost is excluded.
Undefined objective-domain values are never clamped for reporting. All-instance means/gaps are
undefined if any sample has an invalid objective; feasible-subset gaps are reported separately.

## Objective, optimality gap and feasibility

```
method            obj (mean)  gap% mean  gap% max  gap% mean(feas)  feas rate  domain valid  max |h|   max viol  latency ms
----------------  ----------  ---------  --------  ---------------  ---------  ------------  --------  --------  ----------
CLARABEL(cvxpy)   -15.0682    1.232e-14  5.69e-14  -                1.000      1.000         2.03e-13  0.00e+00  7.13
DC3 + correction  -14.2214    5.625      6.876     5.625            1.000      1.000         1.43e-13  0.00e+00  0.8117
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

* correction steps per timed single-instance solve: mean 9.93, max 10 (cap 500)
* instances within `corr_eps` at the end: 100.0% (**0 correction failures**)
* first step at which an instance became feasible: mean 9.93, max 10

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
batch_1            0.808      0.8371    0.8876    0.808        1237.6
batch_8            1.204      1.234     1.428     0.1506       6641.9
batch_32           1.339      1.353     1.481     0.04185      23897
batch_100          2.015      2.135     2.453     0.02015      49639
batch_833          14.04      14.19     15.51     0.01685      59344
stage_predict_b1   0.03117    0.03118   0.03155   -            -
stage_complete_b1  0.009542   0.009577  0.009879  -            -
stage_correct_b1   0.7286     0.7136    0.7418    -            -
single_instance    0.8117     0.8092    0.8437    -            -
```

## Julia baselines

Run `julia --project=. experiments/qp/table.jl` for the
same saved QP inputs. Its table reports mean per-instance solver time;
use the DC3 batch-1 timing for a per-instance comparison.

