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

|  | OSQP (slack form) mean (max) | sLME-ADMM 5000ep mean (max) | DC3 + correction mean (max) |
|---|---|---|---|
| Solving time (g_opt ≤ 1.0%) | 39.68 (464.66) | **4.26 (7.96)** | — |
| Constr. viol. | 9.7e-12 (5.3e-11) | 1.9e-15 (4.1e-15) | — |
| Opt. gap (%) | 0 (0) | 0.968 (1) | — |
| Zeros (%) | 48.3 (51.0) | 4.8 (14.0) | — |
