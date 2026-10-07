# QP: n = 100, neq = 50, m = 50

Solving time in ms and optimality gap in %, as mean (maximum) over
833 instances. Constraint violation is the maximum original-coordinate
equality/inequality violation per instance, also shown as mean (maximum).
**Bold** marks the lowest mean time among methods whose every saved solution
meets the 1.0% objective-gap target and feasibility tolerance 0.0001.
"unable to achieve" means a target was missed; — means results are missing.
Timings retain the boundaries described in README.md; this script does not retrain DC3.

|  | n | neq | m | OSQP mean (max) | sLME-ADMM mean (max) | DC3 + correction mean (max) |
|---|---|---|---|---|---|---|
| Solving time (g_opt ≤ 1.0%) | 100 | 50 | 50 | **3.493 (5.959)** | unable to achieve | unable to achieve |
| Constr. viol. | 100 | 50 | 50 | 1.6e-08 (1.8e-07) | 4.1e-03 (5.4e-03) | 1.1e-13 (1.4e-13) |
| Opt. gap (%) | 100 | 50 | 50 | 3.94e-14 (1.85e-13) | 0.374 (40.5) | 5.62 (6.88) |
