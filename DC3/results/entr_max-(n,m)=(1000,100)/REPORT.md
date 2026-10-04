# DC3 benchmark - entr_max ((n,m)=(1000,100))

* instances: 100 test instances (`test_instances.npz`)
* feasibility: exact objective-domain membership and max |h|, max relu(g) <= 0.0001
* device `cpu`, dtype `torch.float64`, arm, torch 2.14.0 (6 threads)
* completion: `n_partial` = 999, cond(A_D) = 1.000e+00, error amplification ||A_D⁻¹A_P||₂ = 3.161e+01
* network parameters: 6,475,815; training time 57443.4 s (excluded from the latency column)

Quality and headline latency come from the same batch=1 calls on every test instance.
Oracle-labeled rows use the known reference optimum for stopping; their reference-solve cost is excluded.
Undefined objective-domain values are never clamped for reporting. All-instance means/gaps are
undefined if any sample has an invalid objective; feasible-subset gaps are reported separately.

## Objective, optimality gap and feasibility

```
method            obj (mean)  gap% mean  gap% max   gap% mean(feas)  feas rate  domain valid  max |h|   max viol  latency ms
----------------  ----------  ---------  ---------  ---------------  ---------  ------------  --------  --------  ----------
CLARABEL(cvxpy)   -6.33571    4.339e-15  2.804e-14  -                1.000      1.000         2.26e-10  0.00e+00  1234
DC3 + correction  -           -          -          -                0.000      0.000         4.44e-16  2.03e-04  135.5
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

* correction steps per timed single-instance solve: mean 1993.73, max 2000 (cap 2000)
* instances within `corr_eps` at the end: 7.0% (**93 correction failures**)
* first step at which an instance became feasible: mean 1910.43, max 1988

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
batch_1            134.3      134.3     134.8     134.3        7.4461
batch_8            423.6      428.1     452.4     52.96        18.884
batch_32           2022       2030      2050      63.2         15.823
batch_100          5738       5738      5740      57.38        17.428
stage_predict_b1   0.4975     0.501     0.5081    -            -
stage_complete_b1  0.007041   0.007077  0.007212  -            -
stage_correct_b1   136.9      136.9     137.3     -            -
single_instance    135.5      135.2     136.6     -            -
```

## Julia baselines

`julia_baselines.json` not present - the LME-ADMM / Ipopt baselines from
`experiments/` were **not executed** for this run.  Produce them with
`julia --project=. experiments/dc3/baselines_power.jl DC3/results/entr_max-(n,m)=(1000,100)`.

