# DC3 benchmark - cone_programming (small)

* instances: 500 test instances (`test_instances.npz`)
* feasibility threshold: max |h| , max relu(g) and max domain violation all <= 0.0001
* device `cpu`, dtype `torch.float64`, arm, torch 2.14.0 (6 threads)
* completion: `n_partial` = 99, cond(A_D) = 1.000e+00, error amplification ||A_D⁻¹A_P||₂ = 9.950e+00
* network parameters: 833,123; training time 609.6 s (excluded from the latency column)

## Objective, optimality gap and feasibility

```
method              obj (mean)  gap% mean  gap% max  gap% mean(feas)  feas rate  max |h|   max viol  latency ms
------------------  ----------  ---------  --------  ---------------  ---------  --------  --------  ----------
CLARABEL(cvxpy)     -4.55105    0          0         -                1.000      3.55e-11  0.00e+00  2.988     
DC3                 -4.51568    0.7771     8.698     0.7771           1.000      1.11e-16  9.77e-05  0.729     
DC3(no correction)  -4.44072    3.675      62.47     -                0.000      2.22e-16  6.05e-02  0.05275   
sLME-ADMM           -4.55107    0.003186   0.05345   -                0.020      1.11e-15  2.01e-03  0.3798    
LME-ADMM            -           -          -         -                -          -         -         -         
Ipopt(early-stop)   -4.55089    0.003504   0.009897  -                1.000      -         -         1.091     
Ipopt(tol=1e-8)     -4.55105    0          0         -                1.000      -         -         1.57      
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

* correction steps used for the whole test batch: 119 (cap 1000)
* instances within `corr_eps` at the end: 100.0% (**0 correction failures**)
* first step at which an instance became feasible: mean 25.00, max 119

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
single_instance    0.729      1.149     2.751     -            -     
batch_1            2.426      2.42      2.496     2.426        412.17
batch_8            4.666      4.67      4.826     0.5833       1714.4
batch_32           13.78      13.96     14.67     0.4307       2321.7
batch_100          27.16      27.38     27.64     0.2716       3682.3
batch_500          155.6      156.1     158.6     0.3112       3213.6
stage_predict_b1   0.05275    0.05266   0.05325   -            -     
stage_complete_b1  0.005083   0.005135  0.005213  -            -     
stage_correct_b1   2.29       2.286     2.341     -            -     
```

## Julia baselines

produced by DC3/julia/baselines_cone.jl on DC3's exported test instances; Gurobi-based baselines (LME_ADMM with aux_solver_gen) were not run - no license.

