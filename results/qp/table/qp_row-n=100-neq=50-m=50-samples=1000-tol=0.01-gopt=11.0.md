# QP: n = 100, neq = 50, m = 50

Solving time in ms and optimality gap in %, as mean (maximum) over
1000 instances. Constraint violation is the maximum original-coordinate
equality/inequality violation per instance, also shown as mean (maximum).
**Bold** marks the lowest mean time among methods whose every saved solution
meets the 11.0% objective-gap target and feasibility threshold c_v = 0.0001.
"unable to achieve" means a target was missed; — means results are missing.
Timings retain the boundaries described in README.md; this script does not retrain DC3.

|  | n | neq | m | OSQP mean (max) | sLME-ADMM mean (max) | DC3 + correction mean (max) |
|---|---|---|---|---|---|---|
| Solving time (g_opt ≤ 11.0%) | 100 | 50 | 50 | **3.202 (4.517)** | 47.935 (126.018) | — |
| Constr. viol. | 100 | 50 | 50 | 1.6e-08 (1.8e-07) | 1.0e-06 (1.6e-06) | — |
| Opt. gap (%) | 100 | 50 | 50 | 4.15e-14 (1.71e-13) | 0.132 (0.809) | — |
