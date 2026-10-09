# QP + L1: n = 100, neq = 50, m = 50, λ = 1.0

min ½y'Qy + p'y + λ‖y‖₁ s.t. Ay = x, Gy ≤ h, with dense Q (problems/qp_l1/problem.jl),
over 1000 instances (seed 20260923, the x of the QP benchmark). Time in ms, gap in %,
as mean (maximum). Constraint violation is the maximum equality/inequality violation per
instance. Zeros: share of components with |y_i| ≤ 0.0001.

OSQP solves the slack form in (y, t) with -t ≤ y ≤ t (200 variables) at tolerance 1.0e-8
and is the reference optimum. sLME-ADMM works on y directly with the learned Moreau envelope
(consensus tolerance 0.01, feasibility tolerance 1.0e-6, at most 2000
iterations) and is rerun for each gap target, using the known optimum; violation, gap and
zeros are from the 0.01% run. For the tighter targets sLME-ADMM follows the learned prox
with exact proximal-gradient (ISTA) correction steps on the prox subproblem (number in the row
label), each contracting the prox error by 1 - (λ_min(Q) + ρ)/(λ_max(Q) + ρ) ≈ 0.09. Clarabel (interior
point) solves the same slack form as OSQP at its default tolerances (1e-8). ADMM is
the standard splitting: the exact prox of f, then the projection onto {Ay = x, Gy ≤ h} by OSQP
(tolerance 1e-10), stopping at ‖w - z‖∞ < 0.01 with the same feasibility and gap tests.
DC3 + correction also works on y directly.
A timing cell needs every instance within the gap target and violation ≤ 0.0001;
**bold** marks the lowest mean time among those. — means results are missing.

|  | OSQP (slack form) mean (max) | ADMM mean (max) | Clarabel mean (max) | sLME-ADMM 64x64+100H-5000ep mean (max) | DC3 + correction mean (max) |
|---|---|---|---|---|---|
| Solving time (g_opt ≤ 1.0%) | 39.43 (463.47) | 29.87 (52.32) | 9.34 (210.79) | **2.83 (5.72)** | unable to achieve |
| Solving time (g_opt ≤ 0.1%; sLME-ADMM + 1 correction step) | 39.43 (463.47) | 83.77 (210.51) | 9.34 (210.79) | **7.58 (15.93)** | unable to achieve |
| Solving time (g_opt ≤ 0.01%; sLME-ADMM + 2 correction steps) | 39.43 (463.47) | 160.07 (814.76) | **9.34 (210.79)** | 15.44 (45.95) | unable to achieve |
| Constr. viol. | 9.7e-12 (5.3e-11) | 7.3e-11 (2.9e-10) | 9.1e-14 (8.9e-13) | 1.9e-15 (4.2e-15) | 2.1e-14 (6.0e-14) |
| Opt. gap (%) | 0 (0) | 0.00948 (0.01) | 5.52e-06 (2.38e-05) | 0.00966 (0.01) | 20.5 (58.3) |
| Zeros (%) | 48.3 (51.0) | 48.3 (52.0) | 48.3 (51.0) | 48.2 (52.0) | 0.4 (3.0) |
