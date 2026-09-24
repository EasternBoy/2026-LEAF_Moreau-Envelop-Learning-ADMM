# DC3 benchmark - cone_programming (default)

* instances: 100 test instances (`test_instances.npz`)
* feasibility threshold: max |h| , max relu(g) and max domain violation all <= 0.0001
* device `cpu`, dtype `torch.float64`, arm, torch 2.14.0 (6 threads)
* completion: `n_partial` = 999, cond(A_D) = 1.000e+00, error amplification ||A_D⁻¹A_P||₂ = 3.161e+01
* network parameters: 25,949,415; training time 480.0 s (excluded from the latency column)

## Objective, optimality gap and feasibility

```
method             obj (mean)  gap% mean  gap% max  gap% mean(feas)  feas rate  max |h|   max viol  latency ms
-----------------  ----------  ---------  --------  ---------------  ---------  --------  --------  ----------
CLARABEL(cvxpy)    -6.33571    0          0         -                1.000      2.31e-10  0.00e+00  1138      
DC3 + correction   -1.5639     75.35      89.09     -                0.000      1.11e-16  2.15e-03  138.8     
sLME-ADMM          -6.33551    0.003176   0.01668   -                1.000      3.11e-15  5.50e-08  40.23     
LME-ADMM           -           -          -         -                -          -         -         -         
Ipopt(early-stop)  -6.33262    0.04884    0.09958   -                1.000      -         9.39e-05  51.55     
Ipopt(tol=1e-8)    -6.33571    0          0         -                1.000      -         -         97.23     
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

* correction steps used for the whole test batch: 2000 (cap 2000)
* instances within `corr_eps` at the end: 0.0% (**100 correction failures**)
* first step at which an instance became feasible: mean nan, max -1

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
stage              median ms  mean ms   p90 ms    ms/instance  inst/s
-----------------  ---------  --------  --------  -----------  ------
single_instance    138.8      138.8     139.8     -            -     
batch_1            139.4      139.2     139.9     139.4        7.1715
batch_8            548.6      547.5     552.7     68.58        14.582
batch_32           2444       2444      2451      76.37        13.094
batch_100          5696       6287      7320      56.96        17.556
stage_predict_b1   1.514      1.517     1.527     -            -     
stage_complete_b1  0.006791   0.006855  0.007129  -            -     
stage_correct_b1   129.1      129.2     129.6     -            -     
```

## Julia baselines

produced by DC3/julia/baselines_cone.jl on DC3's exported test instances; Gurobi-based baselines (LME_ADMM with aux_solver_gen) were not run - no license.

