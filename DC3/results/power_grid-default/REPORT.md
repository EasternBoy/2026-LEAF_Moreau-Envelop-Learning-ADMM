# DC3 benchmark - power_grid (default)

* instances: 500 test instances (`test_instances.npz`)
* feasibility threshold: max |h| , max relu(g) and max domain violation all <= 0.0001
* device `cpu`, dtype `torch.float64`, arm, torch 2.14.0 (6 threads)
* completion: `n_partial` = 191, cond(A_D) = 5.571e+04, error amplification ||A_D⁻¹A_P||₂ = 1.384e+01
* network parameters: 462,015; training time 635.0 s (excluded from the latency column)

## Objective, optimality gap and feasibility

```
method             obj (mean)  gap% mean  gap% max  gap% mean(feas)  feas rate  max |h|   max viol  latency ms
-----------------  ----------  ---------  --------  ---------------  ---------  --------  --------  ----------
CLARABEL(cvxpy)    37499.1     0          0         -                1.000      6.03e-09  7.74e-09  9.614     
DC3 + correction   39687.2     5.834      33.6      5.834            1.000      7.11e-14  9.07e-06  0.7354    
ADMM(Gurobi aux)   -           -          -         -                -          -         -         -         
Ipopt(tol=1e-10)   37499.1     0          0         -                1.000      -         -         13.66     
Ipopt(early-stop)  37499.1     0.004652   0.009938  -                1.000      -         -         10.8      
LME-ADMM(split)    37278.8     0.5628     9.536     -                0.506      1.60e-14  3.06e-01  3.102     
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

* correction steps used for the whole test batch: 500 (cap 500)
* instances within `corr_eps` at the end: 97.6% (**12 correction failures**)
* first step at which an instance became feasible: mean 9.40, max 194

`correction failures` counts instances that did not reach `corr_eps` on DC3's
*internal* criterion (margin-tightened and, for the eco-MPC, row-scaled).
`feas rate` in the table above is measured on the **original** constraints, so
the two numbers differ: an instance can fail the stricter internal test and still
be feasible for the problem as stated.  A finite step budget is never assumed to
imply feasibility - both numbers are reported.

## Latency (warm-up + device synchronisation, timing excludes host->device transfer)

DC3's test-time correction loop is **batch-global**: it keeps stepping until every
instance in the batch is within `corr_eps`.  Large batches therefore pay for their
worst instance, which is why `ms/instance` is *not* monotone in the batch size.

```
stage              median ms  mean ms  p90 ms   ms/instance  inst/s
-----------------  ---------  -------  -------  -----------  ------
single_instance    0.7354     0.9916   0.8741   -            -     
batch_1            0.5824     0.5812   0.5909   0.5824       1717.1
batch_8            1.878      1.893    1.953    0.2348       4259.6
batch_32           3.264      3.283    3.363    0.102        9803.4
batch_100          422.5      422.9    424.7    4.225        236.67
batch_500          1290       1289     1294     2.58         387.56
stage_predict_b1   0.04421    0.04532  0.04836  -            -     
stage_complete_b1  0.01279    0.01286  0.01313  -            -     
stage_correct_b1   0.4695     0.4686   0.4801   -            -     
```

## Julia baselines

produced by DC3/julia/baselines_power.jl on DC3's exported test instances; Gurobi-based baselines (standard ADMM and LME_ADMM with aux_solver_eco) were not run - no license.

