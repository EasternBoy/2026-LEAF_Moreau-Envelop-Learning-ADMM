# QP + L1: n = 100, neq = 50, m = 50, λ = 1.0

min ½y'Qy + p'y + λ‖y‖₁ s.t. Ay = x, Gy ≤ h, with dense Q (problems/qp_l1/problem.jl),
over 1000 instances (seed 20260923, the x of the QP benchmark). Time in ms, gap in %,
as mean (maximum). Constraint violation is the maximum equality/inequality violation per
instance. Zeros: share of components with |y_i| ≤ 0.0001.

OSQP solves the slack form in (y, t) with -t ≤ y ≤ t (200 variables) at tolerance 1.0e-8
and is the reference optimum. sLME-ADMM works on y directly with the learned Moreau envelope
(consensus tolerance 0.01, feasibility tolerance 1.0e-6, at most 1000
iterations) and is rerun for each gap target, using the known optimum; violation, gap and
zeros are from the 1.0% run. DC3 + correction also works on y directly.
A timing cell needs every instance within the gap target and violation ≤ 0.0001;
**bold** marks the lowest mean time among those. — means results are missing.

|  | OSQP (slack form) mean (max) | sLME-ADMM 5000ep mean (max) | sLME-ADMM 10000ep mean (max) | DC3 + correction mean (max) |
|---|---|---|---|---|
| Solving time (g_opt ≤ 1.0%) | **48.43 (642.56)** | unable to achieve | unable to achieve | unable to achieve |
| Solving time (g_opt ≤ 11.0%) | **48.43 (642.56)** | unable to achieve | unable to achieve | unable to achieve |
| Constr. viol. | 9.4e-12 (5.3e-11) | 2.8e-01 (1.7e+01) | 2.8e-01 (1.7e+01) | 1.8e-14 (4.0e-14) |
| Opt. gap (%) | 0 (0) | 2.76e+05 (1.25e+06) | 2.76e+05 (1.25e+06) | 21.1 (38.4) |
| Zeros (%) | 48.3 (51.0) | 0.0 (1.0) | 0.0 (1.0) | 0.4 (3.0) |
